import os
import sys
from pathlib import Path
from typing import List, Union

# 先设置 HF 镜像，再导入 sentence_transformers
# 如果你不想用镜像，可以在系统环境变量里提前设置 HF_ENDPOINT 覆盖它
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
from sentence_transformers import SentenceTransformer


# 导入项目配置
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

try:
    import config
except ImportError:
    config = None


class BioEmbeddingModel:
    """
    BGE-M3 本地向量模型加载器。

    特点：
    1. 自动适配 CUDA / Apple MPS / CPU
    2. 优先加载 config.EMBEDDING_MODEL_PATH 指定的本地模型
    3. 本地路径不可用时，自动从 HuggingFace / HF-Mirror 加载 BAAI/bge-m3
    4. 对 LangChain 风格的 embed_query / embed_documents 接口友好
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(BioEmbeddingModel, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.device = self._detect_device()
        print(f"[Core] Embedding 推理设备: {self.device}")

        model_name_or_path = self._resolve_model_path()
        print(f"[Core] 准备加载 Embedding 模型: {model_name_or_path}")

        try:
            self.model = SentenceTransformer(
                model_name_or_path,
                device=self.device,
                trust_remote_code=True,
            )

            # BGE-M3 支持长文本；实际 encode 时会自动截断
            self.model.max_seq_length = 8192

            self._initialized = True
            print(
                f"[Core] Embedding 模型加载完成 "
                f"(Device: {self.device}, Dimension: {self.dimension})"
            )

        except Exception as e:
            print(f"[Core] 模型加载失败: {repr(e)}")
            print("[Core] 请检查：")
            print("  1. config.EMBEDDING_MODEL_PATH 是否指向有效模型目录")
            print("  2. 网络是否能访问 HuggingFace 或 hf-mirror.com")
            print("  3. sentence-transformers / torch 是否安装正确")
            raise

    @staticmethod
    def _detect_device() -> str:
        """
        自动选择当前机器可用的最佳推理设备。
        """
        if torch.cuda.is_available():
            return "cuda"

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"

        return "cpu"

    @staticmethod
    def _resolve_model_path() -> str:
        """
        优先使用 config.EMBEDDING_MODEL_PATH。
        如果本地路径不存在，则回退到 HuggingFace 模型名。
        """
        default_model = "BAAI/bge-m3"

        if config is None:
            print("[Core] 未找到 config.py，将使用默认模型 BAAI/bge-m3")
            return default_model

        model_path = getattr(config, "EMBEDDING_MODEL_PATH", None)

        if model_path:
            model_path = os.path.expanduser(str(model_path))
            model_path = os.path.expandvars(model_path)

            if os.path.exists(model_path):
                print(f"[Core] 使用本地 Embedding 模型路径: {model_path}")
                return model_path

            print(f"[Core] 本地模型路径不存在: {model_path}")

        print("[Core] 将尝试在线加载默认模型: BAAI/bge-m3")
        return default_model

    def embed_query(self, text: str) -> List[float]:
        """
        将单个查询字符串转换为向量。
        """
        if not text:
            return []

        vector = self.model.encode(
            text,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        return vector.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        批量将文本转换为向量。
        """
        if not texts:
            return []

        cleaned_texts = [text if text is not None else "" for text in texts]

        vectors = self.model.encode(
            cleaned_texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=8,
        )

        return vectors.tolist()

    def __call__(self, texts: Union[str, List[str]]):
        """
        兼容直接调用：
        - embedder("文本")
        - embedder(["文本1", "文本2"])
        """
        if isinstance(texts, str):
            return self.embed_query(texts)

        return self.embed_documents(texts)

    @property
    def dimension(self) -> int:
        """
        返回向量维度。BGE-M3 通常是 1024。
        """
        return self.model.get_sentence_embedding_dimension()


# 创建全局单例，兼容原项目中 from embedding_model import embedder 的用法
embedder = BioEmbeddingModel()


if __name__ == "__main__":
    print("[Test] 正在测试 Embedding 模型...")

    test_text = "This is a test sentence for BGE-M3 embedding."
    vector = embedder.embed_query(test_text)

    print(f"[Test] 模型维度: {embedder.dimension}")
    print(f"[Test] 测试文本: {test_text}")
    print(f"[Test] 向量长度: {len(vector)}")
    print(f"[Test] 前 5 位: {vector[:5]}")

    if len(vector) == embedder.dimension:
        print("[Test] Embedding 模型测试通过")
    else:
        print("[Test] Embedding 模型测试异常：向量长度与模型维度不一致")
