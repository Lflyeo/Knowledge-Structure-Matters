"""仅调用 doubao-seed-1-6-lite 模型，自动处理 split_30_equal_parts 下全部数据集。"""
from BasicConfig import ModelConfig
from ParallelLLM_common import run_all_datasets

MODEL_CONFIG = ModelConfig(
    name="doubao-seed-1-6",
    api_endpoint="https://api.uniapi.io/v1/chat/completions",
    model_name="doubao-seed-1-6-251015",
    api_key="sk-HC5Tg7UNpjyEIPY_04pYQ1SwFtQ-nssn87bo4TyCgQ2dUk7LXJ5vCwqjdSw",
    max_workers=8,
    extra_headers={"version": "2025-10-15"},
)

if __name__ == "__main__":
    run_all_datasets(MODEL_CONFIG, datasets_dir="split_30_equal_parts_doubao2", prompt_settings={"top3": 3})
