"""
检索 deepseek-v3.2_all_parts_* 下 top1/top3/top5 各 CSV，
按 ParallelLLM_common.evaluate_model_performance 计算指标，
并汇总为与 nlu4mwps_all_models_metrics.csv 同结构的 Excel。
"""
import os
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import MultiLabelBinarizer

ROOT_DIR = "deepseek-v3.2_all_parts_20260806_221544"
PROMPT_FOLDERS = ["top1", "top3", "top5"]
OUTPUT_XLSX = os.path.join(ROOT_DIR, "deepseek-v3.2_all_parts_metrics.xlsx")
OUTPUT_CSV = os.path.join(ROOT_DIR, "deepseek-v3.2_all_parts_metrics.csv")

METRIC_COLS = [
    "samples_precision",
    "samples_recall",
    "samples_f1",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "micro_precision",
    "micro_recall",
    "micro_f1",
    "hamming_loss",
    "subset_accuracy",
]


def _natural_sort_key(text: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", text)]


def parse_true_labels(value) -> List[str]:
    """knowledge_point: 空格分隔。"""
    if pd.isna(value):
        return []
    return [x.strip() for x in str(value).split(" ") if x.strip()]


def parse_pred_labels(value) -> List[str]:
    """parsed_labels: 英文分号分隔；空值视为无预测。"""
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return []
    return [x.strip() for x in text.split(";") if x.strip()]


def evaluate_model_performance(
    y_true: List[List[str]], y_pred: List[List[str]]
) -> Dict[str, float]:
    """与 ParallelLLM_common.ParallelLLMProcessor.evaluate_model_performance 一致。"""
    if not y_true or not y_pred:
        return {}

    all_labels = sorted(set(label for labels in y_true for label in labels))
    mlb = MultiLabelBinarizer()
    mlb.fit([all_labels])

    y_true_bin = mlb.transform(y_true)
    y_pred_bin = mlb.transform(y_pred)
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
        "hamming_loss": float(np.mean(np.not_equal(y_true_bin, y_pred_bin))),
        "subset_accuracy": accuracy_score(y_true_bin, y_pred_bin),
    }


def find_prediction_csvs(root_dir: str) -> List[Tuple[str, str, str]]:
    """返回 (prompt, dataset_part, csv_path) 列表。"""
    items = []
    for prompt in PROMPT_FOLDERS:
        prompt_dir = os.path.join(root_dir, prompt)
        if not os.path.isdir(prompt_dir):
            print(f"[警告] 未找到: {prompt_dir}")
            continue
        for name in os.listdir(prompt_dir):
            part_dir = os.path.join(prompt_dir, name)
            if not os.path.isdir(part_dir):
                continue
            for fname in os.listdir(part_dir):
                if fname.lower().endswith(".csv") and "predictions" in fname.lower():
                    items.append((prompt, name, os.path.join(part_dir, fname)))
    items.sort(key=lambda x: (_natural_sort_key(x[0]), _natural_sort_key(x[1])))
    return items


def evaluate_one_csv(csv_path: str) -> Dict[str, float]:
    df = pd.read_csv(csv_path)
    if "knowledge_point" not in df.columns or "parsed_labels" not in df.columns:
        raise ValueError(f"缺少必要列 knowledge_point / parsed_labels: {csv_path}")

    y_true = [parse_true_labels(v) for v in df["knowledge_point"]]
    y_pred = [parse_pred_labels(v) for v in df["parsed_labels"]]
    model = str(df["model"].iloc[0]) if "model" in df.columns and len(df) else "unknown"
    metrics = evaluate_model_performance(y_true, y_pred)
    metrics["model"] = model
    metrics["n_samples"] = len(df)
    metrics["n_empty_pred"] = sum(1 for labels in y_pred if not labels)
    return metrics


def main():
    if not os.path.isdir(ROOT_DIR):
        raise FileNotFoundError(f"结果目录不存在: {ROOT_DIR}")

    csv_items = find_prediction_csvs(ROOT_DIR)
    print(f"发现 {len(csv_items)} 个 predictions CSV")

    rows = []
    for prompt, dataset_part, csv_path in csv_items:
        rel = os.path.relpath(csv_path, ROOT_DIR)
        try:
            metrics = evaluate_one_csv(csv_path)
            row = {
                "dataset_part": dataset_part,
                "prompt": prompt,
                "model": metrics["model"],
            }
            for col in METRIC_COLS:
                row[col] = metrics.get(col, np.nan)
            row["n_samples"] = metrics.get("n_samples")
            row["n_empty_pred"] = metrics.get("n_empty_pred")
            rows.append(row)
            print(
                f"[OK] {rel} | micro_f1={row['micro_f1']:.4f} | "
                f"empty_pred={row['n_empty_pred']}"
            )
        except Exception as e:
            print(f"[失败] {rel}: {e}")
            rows.append(
                {
                    "dataset_part": dataset_part,
                    "prompt": prompt,
                    "model": "deepseek-v3.2",
                    **{col: np.nan for col in METRIC_COLS},
                    "n_samples": np.nan,
                    "n_empty_pred": np.nan,
                    "error": str(e),
                }
            )

    result_df = pd.DataFrame(rows)
    # 与 nlu4mwps_all_models_metrics.csv 对齐的主列顺序
    ordered_cols = ["dataset_part", "prompt", "model"] + METRIC_COLS + [
        "n_samples",
        "n_empty_pred",
    ]
    if "error" in result_df.columns:
        ordered_cols.append("error")
    result_df = result_df[ordered_cols]

    result_df["_prompt_key"] = result_df["prompt"].map(
        lambda x: tuple(_natural_sort_key(str(x)))
    )
    result_df["_part_key"] = result_df["dataset_part"].map(
        lambda x: tuple(_natural_sort_key(str(x)))
    )
    result_df = result_df.sort_values(by=["_prompt_key", "_part_key"]).drop(
        columns=["_prompt_key", "_part_key"]
    ).reset_index(drop=True)

    result_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    # 额外写一张按 prompt 平均的汇总 sheet
    summary = (
        result_df.groupby(["prompt", "model"], as_index=False)[METRIC_COLS]
        .mean(numeric_only=True)
    )
    summary["_prompt_key"] = summary["prompt"].map(
        lambda x: tuple(_natural_sort_key(str(x)))
    )
    summary = summary.sort_values(by="_prompt_key").drop(columns=["_prompt_key"]).reset_index(
        drop=True
    )

    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        result_df.to_excel(writer, sheet_name="per_file", index=False)
        summary.to_excel(writer, sheet_name="prompt_mean", index=False)

    print("\n===== 按 prompt 平均 =====")
    print(summary.to_string(index=False))
    print(f"\n已保存 CSV : {OUTPUT_CSV}")
    print(f"已保存 Excel: {OUTPUT_XLSX}")


if __name__ == "__main__":
    main()
