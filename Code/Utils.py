import re
import json
# import torch
import pandas as pd
# from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

def read_text(file_path):
    with open(file_path, mode="r",encoding="utf-8") as file:
        line_list=[text.strip() for text in file]
    return line_list

def json_parser(text):
    data_list = []
    text = text.strip()
    json_objects = re.findall(r'\{[^{}]*\}', text)
    
    for obj in json_objects:
        try:
            data = json.loads(obj)
            data_list.append(data)
        except json.JSONDecodeError:
            # 尝试修复常见的JSON格式问题
            try:
                # 修复可能缺少引号的键
                fixed_obj = re.sub(r'(\w+):', r'"\1":', obj)
                data = json.loads(fixed_obj)
                data_list.append(data)
            except:
                continue
    if data_list:
        return pd.DataFrame(data_list)
    else:
        print("无法解析任何JSON对象")
        return None

def read_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    df = json_parser(text)
    return df

def get_questions_answers(data_df, que_target, ans_target):
    return data_df[que_target].tolist(), data_df[ans_target].tolist()

# def load_pretrained_model(model_path):
#     print("正在加载预训练大模型...") 
#     # 使用量化加载以节省显存 (需要bitsandbytes)
#     tokenizer = AutoTokenizer.from_pretrained(model_path)
#     model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
#     # 创建文本生成管道
#     generator = pipeline(
#         "text-generation",
#         model=model,
#         tokenizer=tokenizer,
#         device=0 if torch.cuda.is_available() else "cpu"
#     )
    
#     print("模型加载完成!")
#     return generator