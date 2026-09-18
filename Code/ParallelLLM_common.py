"""
单模型并行调用的共享逻辑。
各模型入口脚本仅需提供 ModelConfig 并调用 run_all_datasets。
"""
import csv
import json
import logging
import os
import queue
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import backoff
import numpy as np
import pandas as pd
import requests
from ratelimit import limits, sleep_and_retry
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm

from BasicConfig import ModelConfig, ThreadSafeCounter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("llm_parallel_processing.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# 三种输出数量设定：最相关知识点分别只输出 1 / 3 / 5 个
PROMPT_SETTINGS: Dict[str, int] = {
    "top1": 1,
    "top3": 3,
    "top5": 5,
}


class ParallelLLMProcessor:
    """单模型多线程并行处理系统"""

    def __init__(self, model_config: ModelConfig, top_k: int = 1):
        self.model_config = model_config
        self.model_name = model_config.name
        self.top_k = top_k
        self.mlb = MultiLabelBinarizer()
        self.stats = defaultdict(ThreadSafeCounter)
        self.result_queue = queue.Queue()

    def construct_unified_prompt(self, problem_text: str, knowledge_vocab: List[str]) -> str:
        k = self.top_k
        if k == 1:
            count_rule = "2. 只输出 1 个最相关的知识点标签，不能多也不能少"
            sep_rule = "3. 只输出标签名称本身，不要编号和解释"
            ending = "请输出最相关的 1 个知识点标签："
        else:
            count_rule = f"2. 只输出 {k} 个最相关的知识点标签，不能多也不能少"
            sep_rule = "3. 多个标签用英文分号分隔；只输出标签名称，不要编号和解释"
            ending = f"请按相关性从高到低输出最相关的 {k} 个知识点标签："

        return f"""作为数学教育专家，请分析以下数学题目并识别最相关的知识点标签。
        题目：{problem_text}

        可用知识点标签（按字母顺序排列）：
            {', '.join(sorted(knowledge_vocab))}

        要求：
        1. 从上述标签中选择最相关的知识点
        {count_rule}
        {sep_rule}
        4. 如果可用标签不足 {k} 个相关项，仍尽量选出最相关的标签，但总数不得超过 {k} 个

        {ending}"""

    @sleep_and_retry
    @limits(calls=300, period=60)
    def call_model_api_with_rate_limit(self, prompt: str):
        return self._call_model_api_internal(prompt)

    @backoff.on_exception(
        backoff.expo,
        (requests.exceptions.RequestException, requests.exceptions.HTTPError),
        jitter=backoff.full_jitter,
    )
    def _call_model_api_internal(self, prompt: str):
        headers = {
            "Authorization": f"Bearer {self.model_config.api_key}",
            "Content-Type": "application/json",
        }
        if self.model_config.extra_headers:
            headers.update(self.model_config.extra_headers)

        payload = {
            "model": self.model_config.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 300,
        }

        response = requests.post(
            self.model_config.api_endpoint,
            headers=headers,
            json=payload,
            timeout=None,
        )
        if response.status_code == 429:
            logger.warning(f"[{self.model_name}] 429 Too Many Requests, retrying...")
            raise requests.exceptions.HTTPError("429 Too Many Requests")

        response_json = response.json()
        print(response_json)
        text = response_json["choices"][0]["message"]["content"]
        usage = response_json.get("usage", {})
        return text, usage

    def process_single_problem(
        self, problem_data: Tuple[int, str], knowledge_vocab: List[str]
    ) -> Dict:
        idx, problem_text = problem_data
        result = {
            "idx": idx,
            "model": self.model_name,
            "problem": problem_text,
            "raw_output": "",
            "labels": [],
            "success": False,
            "error": None,
            "response_time": 0,
            "usage": {},
            "top_k": self.top_k,
        }

        start_time = time.time()
        try:
            prompt = self.construct_unified_prompt(problem_text, knowledge_vocab)
            response_text, usage = self.call_model_api_with_rate_limit(prompt)
            raw_output = response_text.strip() if response_text else ""
            parsed_labels = self.parse_model_response(raw_output, knowledge_vocab)

            result["usage"] = usage
            result["raw_output"] = raw_output
            result["labels"] = parsed_labels
            result["success"] = True
            self.stats[f"{self.model_name}_success"].increment()
        except Exception as e:
            result["error"] = str(e)
            logger.error(f"Model {self.model_name} failed on problem {idx}: {str(e)}")
            self.stats[f"{self.model_name}_errors"].increment()

        result["response_time"] = time.time() - start_time
        self.result_queue.put(result)
        return result

    def process_in_parallel(
        self,
        problems: List[Tuple[int, str]],
        knowledge_vocab: List[str],
        max_workers: int = 5,
    ) -> List[Dict]:
        logger.info(
            f"Starting parallel processing for {self.model_name} "
            f"(top_k={self.top_k}) with {max_workers} workers"
        )
        results = []
        with ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix=f"{self.model_name}_worker"
        ) as executor:
            future_to_idx = {
                executor.submit(self.process_single_problem, (idx, problem), knowledge_vocab): idx
                for idx, problem in problems
            }
            with tqdm(
                total=len(problems),
                desc=f"Processing {self.model_name}[top{self.top_k}]",
                unit="problem",
                ncols=100,
            ) as pbar:
                for future in as_completed(future_to_idx):
                    try:
                        results.append(future.result(timeout=600))
                    except Exception as e:
                        logger.error(f"Task failed: {str(e)}")
                    finally:
                        pbar.update(1)
        return sorted(results, key=lambda x: x["idx"])

    def parse_model_response(self, response: str, knowledge_vocab: List[str]) -> List[str]:
        if not response:
            return []

        separators = [";", "；", ",", "，", "\n", " "]
        for sep in separators:
            if sep in response:
                labels = [label.strip() for label in response.split(sep) if label.strip()]
                break
        else:
            labels = [response.strip()]

        # 保序去重，并截断到 top_k
        valid_labels = []
        seen = set()
        for label in labels:
            label = label.lower().strip()
            for vocab in knowledge_vocab:
                if self._similarity(label, vocab.lower()) > 0.6 and vocab not in seen:
                    valid_labels.append(vocab)
                    seen.add(vocab)
                    break
            if len(valid_labels) >= self.top_k:
                break

        return valid_labels

    def _similarity(self, s1: str, s2: str) -> float:
        set1, set2 = set(s1), set(s2)
        if not set1 or not set2:
            return 0.0
        return len(set1 & set2) / len(set1 | set2)

    def evaluate_model_performance(
        self, y_true: List[List[str]], y_pred: List[List[str]]
    ) -> Dict[str, float]:
        if not y_true or not y_pred:
            return {}

        y_true_bin = self.mlb.transform(y_true)
        y_pred_bin = self.mlb.transform(y_pred)
        return {
            "samples_precision": precision_score(
                y_true_bin, y_pred_bin, average="samples", zero_division=0
            ),
            "samples_recall": recall_score(
                y_true_bin, y_pred_bin, average="samples", zero_division=0
            ),
            "samples_f1": f1_score(
                y_true_bin, y_pred_bin, average="samples", zero_division=0
            ),
            "macro_precision": precision_score(
                y_true_bin, y_pred_bin, average="macro", zero_division=0
            ),
            "macro_recall": recall_score(
                y_true_bin, y_pred_bin, average="macro", zero_division=0
            ),
            "macro_f1": f1_score(y_true_bin, y_pred_bin, average="macro", zero_division=0),
            "micro_precision": precision_score(
                y_true_bin, y_pred_bin, average="micro", zero_division=0
            ),
            "micro_recall": recall_score(
                y_true_bin, y_pred_bin, average="micro", zero_division=0
            ),
            "micro_f1": f1_score(y_true_bin, y_pred_bin, average="micro", zero_division=0),
            "hamming_loss": np.mean(np.not_equal(y_true_bin, y_pred_bin)),
            "subset_accuracy": accuracy_score(y_true_bin, y_pred_bin),
        }


def load_dataset_from_excel(xlsx_path: str):
    df = pd.read_excel(xlsx_path)
    records, problems, true_labels = [], [], []

    for idx, row in df.iterrows():
        row_dict = row.to_dict()
        records.append(row_dict)
        problem_text = row_dict.get("original_text") or row_dict.get("raw_text")
        problems.append((idx, str(problem_text)))
        kp = row_dict.get("knowledge_point", "")
        labels = [x.strip() for x in str(kp).split(" ") if x.strip()]
        true_labels.append(labels)

    return records, problems, true_labels


def save_full_results_to_csv(records: List[Dict], results: List[Dict], save_path: str):
    assert len(records) == len(results)
    fieldnames = list(records[0].keys()) + [
        "model",
        "top_k",
        "raw_model_output",
        "parsed_labels",
        "success",
        "response_time",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "error",
    ]

    with open(save_path, mode="w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for raw, pred in zip(records, results):
            usage = pred.get("usage", {})
            row = dict(raw)
            row.update(
                {
                    "model": pred["model"],
                    "top_k": pred.get("top_k"),
                    "raw_model_output": pred.get("raw_output", ""),
                    "parsed_labels": ";".join(pred.get("labels", [])),
                    "success": pred["success"],
                    "response_time": round(pred["response_time"], 3),
                    "input_tokens": usage.get("input_tokens") or usage.get("prompt_tokens"),
                    "output_tokens": usage.get("output_tokens") or usage.get("completion_tokens"),
                    "total_tokens": usage.get("total_tokens"),
                    "error": pred["error"],
                }
            )
            writer.writerow(row)


def save_metrics_to_json(metrics: Dict, save_path: str):
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=4)


def _natural_sort_key(path: str):
    """按文件名中的数字自然排序，如 dataset_part_2 < dataset_part_10。"""
    name = os.path.basename(path)
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def discover_datasets(datasets_dir: str = "split_30_equal_parts") -> List[str]:
    """自动发现目录下全部 .xlsx 数据集，按自然顺序排序。"""
    if not os.path.isdir(datasets_dir):
        raise FileNotFoundError(f"数据集目录不存在: {datasets_dir}")

    paths = [
        os.path.join(datasets_dir, name)
        for name in os.listdir(datasets_dir)
        if name.lower().endswith(".xlsx") and not name.startswith("~$")
    ]
    paths.sort(key=_natural_sort_key)

    if not paths:
        raise FileNotFoundError(f"目录中未找到 .xlsx 文件: {datasets_dir}")

    logger.info(f"发现 {len(paths)} 个数据集: {[os.path.basename(p) for p in paths]}")
    return paths


def run_single_model(
    model_config: ModelConfig,
    dataset_path: str,
    root_output_dir: Optional[str] = None,
    prompt_name: str = "top1",
    top_k: Optional[int] = None,
) -> Dict[str, float]:
    """运行单个模型对单个数据集、单个 prompt 设定的完整流水线。"""
    if top_k is None:
        if prompt_name not in PROMPT_SETTINGS:
            raise ValueError(
                f"未知 prompt 设定: {prompt_name}，可选: {list(PROMPT_SETTINGS.keys())}"
            )
        top_k = PROMPT_SETTINGS[prompt_name]

    records, problems, true_labels = load_dataset_from_excel(dataset_path)
    all_labels = sorted(list(set(label for labels in true_labels for label in labels)))
    print(all_labels)

    processor = ParallelLLMProcessor(model_config, top_k=top_k)
    processor.mlb.fit([all_labels])
    knowledge_vocab = all_labels

    model_name = model_config.name
    dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
    logger.info(
        f"[{model_name}][{prompt_name}] Processing dataset: {dataset_name} (top_k={top_k})"
    )

    if root_output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        root_output_dir = f"{dataset_name}_experiment_results_{timestamp}"
    os.makedirs(root_output_dir, exist_ok=True)

    results = processor.process_in_parallel(
        problems=problems,
        knowledge_vocab=knowledge_vocab,
        max_workers=model_config.max_workers,
    )

    pred_labels = [item["labels"] for item in results]
    metrics = processor.evaluate_model_performance(
        y_true=true_labels, y_pred=pred_labels
    )
    metrics["top_k"] = top_k
    metrics["prompt_name"] = prompt_name

    # 输出目录: root / prompt_name / dataset_name
    model_output_dir = os.path.join(root_output_dir, prompt_name, dataset_name)
    os.makedirs(model_output_dir, exist_ok=True)

    csv_path = os.path.join(model_output_dir, f"{model_name}_predictions.csv")
    save_full_results_to_csv(records=records, results=results, save_path=csv_path)

    json_path = os.path.join(model_output_dir, f"{model_name}_metrics.json")
    save_metrics_to_json(metrics, json_path)

    logger.info(f"[{model_name}][{prompt_name}][{dataset_name}] CSV saved to {csv_path}")
    logger.info(f"[{model_name}][{prompt_name}][{dataset_name}] Metrics saved to {json_path}")

    print(f"\n===== {model_name} | {prompt_name} | {dataset_name} Metrics =====")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")

    return metrics


def run_all_datasets(
    model_config: ModelConfig,
    datasets_dir: str = "split_30_equal_parts",
    prompt_settings: Optional[Dict[str, int]] = None,
) -> Dict[str, Dict]:
    """
    自动读取 datasets_dir 下全部 .xlsx，并对每种 prompt 设定逐个完成推理与评估。

    默认三种设定：
      - top1: 只输出 1 个最相关知识点
      - top3: 只输出 3 个最相关知识点
      - top5: 只输出 5 个最相关知识点

    结果目录: {model_name}_all_parts_{timestamp}/{prompt_name}/{dataset_name}/
    """
    if prompt_settings is None:
        prompt_settings = PROMPT_SETTINGS

    dataset_paths = discover_datasets(datasets_dir)
    model_name = model_config.name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root_output_dir = f"{model_name}_all_parts_{timestamp}"
    os.makedirs(root_output_dir, exist_ok=True)

    all_metrics: Dict[str, Dict] = {}
    total_datasets = len(dataset_paths)
    total_prompts = len(prompt_settings)

    for p_i, (prompt_name, top_k) in enumerate(prompt_settings.items(), start=1):
        all_metrics[prompt_name] = {}
        logger.info(
            f"######## Prompt [{p_i}/{total_prompts}] {prompt_name} "
            f"(只输出 {top_k} 个最相关知识点) ########"
        )

        for d_i, dataset_path in enumerate(dataset_paths, start=1):
            dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
            logger.info(
                f"========== [{prompt_name}][{d_i}/{total_datasets}] "
                f"{model_name} → {dataset_name} =========="
            )
            try:
                metrics = run_single_model(
                    model_config=model_config,
                    dataset_path=dataset_path,
                    root_output_dir=root_output_dir,
                    prompt_name=prompt_name,
                    top_k=top_k,
                )
                all_metrics[prompt_name][dataset_name] = metrics
            except Exception as e:
                logger.error(f"[{model_name}][{prompt_name}][{dataset_name}] 失败: {e}")
                all_metrics[prompt_name][dataset_name] = {"error": str(e)}

        # 每个 prompt 设定单独一份汇总
        prompt_summary_path = os.path.join(
            root_output_dir, prompt_name, f"{model_name}_{prompt_name}_metrics_summary.json"
        )
        os.makedirs(os.path.dirname(prompt_summary_path), exist_ok=True)
        save_metrics_to_json(all_metrics[prompt_name], prompt_summary_path)

    summary_path = os.path.join(root_output_dir, f"{model_name}_all_metrics_summary.json")
    save_metrics_to_json(all_metrics, summary_path)
    logger.info(f"全部 prompt × 数据集 处理完成，汇总指标: {summary_path}")
    logger.info(f"结果目录: {root_output_dir}")
    return all_metrics
