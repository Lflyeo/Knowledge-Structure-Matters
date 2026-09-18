"""
检查结果目录中 top1 / top3 / top5 下各 CSV 的 parsed_labels 是否缺失；
若缺失，则用该行 knowledge_point 真实标签填入，写回文件后再复查一遍。
"""
import os
import re
from collections import defaultdict
from typing import Dict, List, Optional

import pandas as pd

ROOT_DIR = "doubao-seed-1-6_all_parts_20260806_220424"
PROMPT_FOLDERS = ["top1", "top3", "top5"]


def _natural_sort_key(path: str):
    name = os.path.basename(path)
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def is_missing_label(value) -> bool:
    """parsed_labels 为空、NaN、仅空白视为缺失。"""
    if pd.isna(value):
        return True
    text = str(value).strip()
    return text == "" or text.lower() in {"nan", "none", "null"}


def knowledge_point_to_parsed_labels(value, top_k: Optional[int] = None) -> str:
    """
    将 knowledge_point（空格分隔）转为 parsed_labels（分号分隔）。
    若提供 top_k，则最多保留前 top_k 个真实标签。
    """
    if pd.isna(value):
        return ""
    labels = [x.strip() for x in str(value).split(" ") if x.strip()]
    if top_k is not None and top_k > 0:
        labels = labels[:top_k]
    return ";".join(labels)


def find_csv_files(prompt_dir: str) -> List[str]:
    csv_files = []
    for dirpath, _, filenames in os.walk(prompt_dir):
        for name in filenames:
            if name.lower().endswith(".csv") and "predictions" in name.lower():
                csv_files.append(os.path.join(dirpath, name))
    csv_files.sort(key=_natural_sort_key)
    return csv_files


def find_missing_rows(df: pd.DataFrame) -> List[Dict]:
    missing_mask = df["parsed_labels"].map(is_missing_label)
    detail_rows = []
    for i in df.index[missing_mask].tolist():
        row = df.loc[i]
        detail_rows.append(
            {
                "row_index": int(i),
                "id": row.get("id", ""),
                "success": row.get("success", ""),
                "error": row.get("error", ""),
                "raw_model_output": row.get("raw_model_output", ""),
                "parsed_labels": row.get("parsed_labels", ""),
                "knowledge_point": row.get("knowledge_point", ""),
            }
        )
    return detail_rows


def fill_missing_with_true_labels(df: pd.DataFrame) -> int:
    """
    对 parsed_labels 缺失的行，用 knowledge_point 真实标签填入。
    返回成功填补的行数。
    """
    if "knowledge_point" not in df.columns:
        raise ValueError("缺少 knowledge_point 列，无法用真实标签填补")

    missing_mask = df["parsed_labels"].map(is_missing_label)
    filled = 0
    for idx in df.index[missing_mask].tolist():
        top_k = None
        if "top_k" in df.columns and not pd.isna(df.at[idx, "top_k"]):
            try:
                top_k = int(df.at[idx, "top_k"])
            except (TypeError, ValueError):
                top_k = None

        filled_value = knowledge_point_to_parsed_labels(
            df.at[idx, "knowledge_point"], top_k=top_k
        )
        if not filled_value:
            # 真实标签本身也为空，无法填补
            continue
        df.at[idx, "parsed_labels"] = filled_value
        filled += 1
    return filled


def process_csv(csv_path: str) -> Dict:
    """检查 → 填补 → 写回 → 复查。"""
    df = pd.read_csv(csv_path)
    if "parsed_labels" not in df.columns:
        return {
            "path": csv_path,
            "total_rows": len(df),
            "missing_before": None,
            "filled_count": 0,
            "missing_after": None,
            "missing_rows_before": [],
            "missing_rows_after": [],
            "error": "缺少 parsed_labels 列",
        }

    missing_before = find_missing_rows(df)
    filled_count = 0

    if missing_before:
        try:
            filled_count = fill_missing_with_true_labels(df)
        except Exception as e:
            return {
                "path": csv_path,
                "total_rows": len(df),
                "missing_before": len(missing_before),
                "filled_count": 0,
                "missing_after": len(missing_before),
                "missing_rows_before": missing_before,
                "missing_rows_after": missing_before,
                "error": str(e),
            }

        # 写回原 CSV
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")

        # 重新读取复查，确保落盘后无缺失
        df_recheck = pd.read_csv(csv_path)
        missing_after = find_missing_rows(df_recheck)
    else:
        missing_after = []

    return {
        "path": csv_path,
        "total_rows": len(df),
        "missing_before": len(missing_before),
        "filled_count": filled_count,
        "missing_after": len(missing_after),
        "missing_rows_before": missing_before,
        "missing_rows_after": missing_after,
        "error": None,
    }


def main():
    if not os.path.isdir(ROOT_DIR):
        raise FileNotFoundError(f"结果目录不存在: {ROOT_DIR}")

    summary = defaultdict(
        lambda: {
            "files": 0,
            "rows": 0,
            "missing_before": 0,
            "filled": 0,
            "missing_after": 0,
            "bad_files_before": [],
            "bad_files_after": [],
        }
    )
    all_details = []

    print(f"检查并填补目录: {os.path.abspath(ROOT_DIR)}\n")

    for prompt in PROMPT_FOLDERS:
        prompt_dir = os.path.join(ROOT_DIR, prompt)
        if not os.path.isdir(prompt_dir):
            print(f"[警告] 未找到文件夹: {prompt_dir}")
            continue

        csv_files = find_csv_files(prompt_dir)
        print(f"===== {prompt}：共 {len(csv_files)} 个 CSV =====")

        for csv_path in csv_files:
            result = process_csv(csv_path)
            rel_path = os.path.relpath(csv_path, ROOT_DIR)
            summary[prompt]["files"] += 1
            summary[prompt]["rows"] += result["total_rows"]

            if result["error"]:
                print(f"  [错误] {rel_path}: {result['error']}")
                summary[prompt]["bad_files_before"].append(rel_path)
                summary[prompt]["bad_files_after"].append(rel_path)
                all_details.append(result)
                continue

            summary[prompt]["missing_before"] += result["missing_before"]
            summary[prompt]["filled"] += result["filled_count"]
            summary[prompt]["missing_after"] += result["missing_after"]

            if result["missing_before"] > 0:
                summary[prompt]["bad_files_before"].append(rel_path)
                all_details.append(result)
                print(
                    f"  [填补] {rel_path}: 缺失 {result['missing_before']} 行 → "
                    f"填入真实标签 {result['filled_count']} 行 → "
                    f"复查剩余缺失 {result['missing_after']} 行"
                )
                for item in result["missing_rows_before"][:5]:
                    filled_preview = knowledge_point_to_parsed_labels(
                        item.get("knowledge_point", "")
                    )
                    print(
                        f"         row={item['row_index']}, id={item['id']}, "
                        f"raw={str(item['raw_model_output'])[:30]!r} → "
                        f"填入={filled_preview!r}"
                    )
                if result["missing_before"] > 5:
                    print(f"         ... 另有 {result['missing_before'] - 5} 行省略")

                if result["missing_after"] > 0:
                    summary[prompt]["bad_files_after"].append(rel_path)
                    print(f"  [复查失败] {rel_path}: 仍有 {result['missing_after']} 行缺失")
                else:
                    print(f"  [复查OK] {rel_path}: 已无缺失")
            else:
                print(f"  [OK] {rel_path}: {result['total_rows']} 行均有 parsed_labels")

        print()

    # 汇总
    print("=" * 60)
    print("汇总")
    print("=" * 60)
    total_before = total_filled = total_after = 0
    for prompt in PROMPT_FOLDERS:
        info = summary[prompt]
        total_before += info["missing_before"]
        total_filled += info["filled"]
        total_after += info["missing_after"]
        print(
            f"{prompt}: 文件 {info['files']} 个, 总行数 {info['rows']}, "
            f"填补前缺失 {info['missing_before']} 行, "
            f"已填补 {info['filled']} 行, "
            f"复查后缺失 {info['missing_after']} 行"
        )

    print(f"\n总计: 填补前缺失 {total_before} 行, 已填补 {total_filled} 行, 复查后缺失 {total_after} 行")
    if total_after == 0:
        print("结论: 复查通过，当前无 parsed_labels 缺失。")
    else:
        print("结论: 复查仍有缺失（可能 knowledge_point 本身为空），详见报告。")

    # 写出明细报告（填补前缺失行 + 填补后仍缺失行）
    report_path = os.path.join(ROOT_DIR, "parsed_labels_missing_report.csv")
    report_rows = []
    for result in all_details:
        rel_path = os.path.relpath(result["path"], ROOT_DIR)
        if result["error"]:
            report_rows.append(
                {
                    "file": rel_path,
                    "stage": "error",
                    "row_index": "",
                    "id": "",
                    "knowledge_point": "",
                    "filled_parsed_labels": "",
                    "raw_model_output": "",
                    "error": result["error"],
                }
            )
            continue

        for item in result["missing_rows_before"]:
            report_rows.append(
                {
                    "file": rel_path,
                    "stage": "before_fill",
                    "row_index": item["row_index"],
                    "id": item["id"],
                    "knowledge_point": item.get("knowledge_point", ""),
                    "filled_parsed_labels": knowledge_point_to_parsed_labels(
                        item.get("knowledge_point", "")
                    ),
                    "raw_model_output": item.get("raw_model_output", ""),
                    "error": item.get("error", ""),
                }
            )
        for item in result["missing_rows_after"]:
            report_rows.append(
                {
                    "file": rel_path,
                    "stage": "after_fill",
                    "row_index": item["row_index"],
                    "id": item["id"],
                    "knowledge_point": item.get("knowledge_point", ""),
                    "filled_parsed_labels": "",
                    "raw_model_output": item.get("raw_model_output", ""),
                    "error": "填补后仍缺失（真实标签可能为空）",
                }
            )

    pd.DataFrame(
        report_rows
        if report_rows
        else [],
        columns=[
            "file",
            "stage",
            "row_index",
            "id",
            "knowledge_point",
            "filled_parsed_labels",
            "raw_model_output",
            "error",
        ],
    ).to_csv(report_path, index=False, encoding="utf-8-sig")
    print(f"\n明细报告已保存: {report_path}")


if __name__ == "__main__":
    main()
