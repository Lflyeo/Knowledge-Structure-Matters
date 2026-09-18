"""仅调用 gemini-2.5-flash 模型，自动处理 split_30_equal_parts 下全部数据集。"""
from BasicConfig import ModelConfig
from ParallelLLM_common import run_all_datasets

MODEL_CONFIG = ModelConfig(
    name="gemini-2.5-flash",
    api_endpoint="https://api.uniapi.io/v1/chat/completions",
    model_name="gemini-2.5-flash",
    api_key="sk-HC5Tg7UNpjyEIPY_04pYQ1SwFtQ-nssn87bo4TyCgQ2dUk7LXJ5vCwqjdSw",
    max_workers=8,
)

if __name__ == "__main__":
    run_all_datasets(MODEL_CONFIG, datasets_dir="split_30_equal_parts_gemini", prompt_settings={"top1": 1})
