import os
import sys
from dotenv import load_dotenv

load_dotenv(override=True)

LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_BASE_URL = os.getenv("LLM_BASE_URL")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "qwen-max")

missing_configs = []

if not LLM_API_KEY:
    missing_configs.append("LLM_API_KEY")
if not LLM_BASE_URL:
    missing_configs.append("LLM_BASE_URL")

if missing_configs:
    print(f"错误: .env 文件中缺少配置: {', '.join(missing_configs)}")
    sys.exit(1)

print("配置加载成功")
