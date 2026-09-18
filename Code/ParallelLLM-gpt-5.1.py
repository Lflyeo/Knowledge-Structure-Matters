"""仅调用 gpt-5.1 模型，自动处理 split_30_equal_parts 下全部数据集。"""
from BasicConfig import ModelConfig
from ParallelLLM_common import run_all_datasets

MODEL_CONFIG = ModelConfig(
    name="gpt-5.1",
    api_endpoint="https://api.uniapi.io/v1/chat/completions",
    model_name="gpt-5.1",
    api_key="sk-HC5Tg7UNpjyEIPY_04pYQ1SwFtQ-nssn87bo4TyCgQ2dUk7LXJ5vCwqjdSw",
    max_workers=8,
)

if __name__ == "__main__":
    run_all_datasets(MODEL_CONFIG, datasets_dir="split_30_equal_parts_doubao", prompt_settings={"top1": 1})
