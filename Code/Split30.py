import pandas as pd
import os

def split_excel_equal_parts(
    input_excel_path: str,
    output_dir: str,
    n_parts: int = 30
):
    """
    将 Excel 数据严格等量拆分为 n_parts 份（不足部分直接丢弃）

    :param input_excel_path: 原始 Excel 文件路径
    :param output_dir: 拆分后文件保存目录
    :param n_parts: 拆分份数（默认 30）
    """

    # 1. 读取 Excel
    df = pd.read_excel(input_excel_path)
    total_rows = len(df)

    # 2. 计算每份数据量（严格一致）
    rows_per_part = total_rows // n_parts
    usable_rows = rows_per_part * n_parts

    if usable_rows == 0:
        raise ValueError("数据量过小，无法拆分为指定份数")

    print(f"Total rows: {total_rows}")
    print(f"Rows per part: {rows_per_part}")
    print(f"Used rows: {usable_rows}")
    print(f"Dropped rows: {total_rows - usable_rows}")

    # 3. 截断多余数据
    df = df.iloc[:usable_rows].reset_index(drop=True)

    # 4. 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 5. 等量切分并保存
    for i in range(n_parts):
        start = i * rows_per_part
        end = (i + 1) * rows_per_part
        part_df = df.iloc[start:end]

        output_path = os.path.join(
            output_dir,
            f"dataset_part_{i+1:02d}.xlsx"
        )
        part_df.to_excel(output_path, index=False)

        print(f"Saved: {output_path} ({len(part_df)} rows)")


if __name__ == "__main__":
    # ===== 示例调用 =====
    input_excel = "mwp_all_kps.xlsx"
    output_dir = "split_30_equal_parts"

    split_excel_equal_parts(
        input_excel_path=input_excel,
        output_dir=output_dir,
        n_parts=30
    )
