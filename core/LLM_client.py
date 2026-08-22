import sys
import os
import json
from openai import OpenAI


import config


class LLMClient:
    """
    统一的大模型服务客户端：
    1. chat: 对话生成
    2. rerank: 使用 ChatGPT API 进行文档重排序
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LLMClient, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        print(f"初始化 LLM 客户端: {config.LLM_MODEL_NAME}")

        try:
            self.client = OpenAI(
                api_key=config.LLM_API_KEY,
                base_url=config.LLM_BASE_URL
            )
            self._initialized = True
            print("LLM 客户端就绪")
        except Exception as e:
            print(f"LLM 客户端初始化失败: {e}")
            sys.exit(1)

    def chat(self, messages, temperature=1, stream=False):
        try:
            response = self.client.chat.completions.create(
                model=config.LLM_MODEL_NAME,
                messages=messages,
                temperature=temperature,
                stream=stream
            )

            if stream:
                return response

            return response.choices[0].message.content

        except Exception as e:
            print(f"LLM Chat Error: {e}")
            return "模型暂时无法响应，请检查网络、API Key 或额度。"

    def rerank(self, query: str, documents: list, top_n=5):
        if not documents:
            return []

        numbered_documents = "\n".join(
            [f"{i}. {doc}" for i, doc in enumerate(documents)]
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "你是一个文档重排序助手。"
                    "请根据用户问题，从候选文档中选出最相关的文档编号。"
                    "只返回 JSON 数组，不要返回解释文字。"
                )
            },
            {
                "role": "user",
                "content": f"""用户问题：
{query}

候选文档：
{numbered_documents}

请返回最相关的前 {top_n} 个文档编号。

返回格式必须是 JSON 数组，例如：
[2, 0, 4]
"""
            }
        ]

        try:
            response = self.chat(messages, temperature=1)

            indices = json.loads(response)

            results = []
            for rank, index in enumerate(indices[:top_n]):
                if isinstance(index, int) and 0 <= index < len(documents):
                    results.append({
                        "index": index,
                        "document": documents[index],
                        "relevance_score": None,
                        "rank": rank + 1
                    })

            return results

        except Exception as e:
            print(f"OpenAI Rerank Error: {e}")

            return [
                {
                    "index": i,
                    "document": doc,
                    "relevance_score": None,
                    "rank": i + 1
                }
                for i, doc in enumerate(documents[:top_n])
            ]


llm = LLMClient()
