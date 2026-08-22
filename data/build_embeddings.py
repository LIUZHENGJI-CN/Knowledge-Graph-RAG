import os
import time
from typing import List, Dict
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import torch

# --- 1. 配置区域 ---
# 数据库连接
NEO4J_URI = "neo4j://localhost:7687"
NEO4J_AUTH = ("neo4j", "your_neo4j_password_here")


# 模型配置
MODEL_NAME = "BAAI/bge-m3"
# 显存优化: 如果显存 < 8GB，建议设为 16 或 8；如果用 CPU，设为 4
BATCH_SIZE = 128

class BioKGEmbedder:
    def __init__(self):
        print(f"🔌 连接 Neo4j: {NEO4J_URI}")
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
        
        # 检查设备 (GPU/CPU/MPS)
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"🚀 加载模型: {MODEL_NAME} (Device: {device})")
        
        # 加载 BGE-M3
        # 在 BioKGEmbedder.__init__ 方法里，加载模型时增加参数：
        self.model = SentenceTransformer(
            MODEL_NAME, 
            device=device, 
            trust_remote_code=True,
            model_kwargs={"torch_dtype": torch.float16} # ⚡️ 开启半精度加速
        )
        
        # BGE-M3 支持的最大长度是 8192，这里我们设个安全值，防止内存溢出
        self.model.max_seq_length = 8192

    def close(self):
        self.driver.close()

    def fetch_papers_batch(self, limit=100):
        """
        获取没有 embedding 的论文，同时把 5 大要素全部抓出来(这里因为第一遍跑错了所以要覆盖原来结果，
        并非只抓取没有 embedding,但是会无限循环抓取同一批数据，所以后面改成一次性抓取所有数据，在python里分批处理，
        避免重复从 Neo4j 取到同一批数据）

        """
        cypher = """
        MATCH (p:Paper)
        WHERE p.title IS NOT NULL
        // 至少要有 Title，其他字段没有可以填空
        AND p.title IS NOT NULL
        
        // 1. 抓取关联的方法 (LLM提取的实体)
        
        
        // 2. 抓取关联的关键词 (LLM提取的实体)
        OPTIONAL MATCH (p)-[:HAS_KEYWORD]->(k:Keyword)
        
        WITH p, 
             p.proposed_method as proposed_method,
             collect(distinct k.name) as keywords
        
        RETURN elementId(p) as id, 
               p.title as title, 
               p.abstract as abstract, 
               p.background as background, 
               proposed_method,
               keywords
        LIMIT $limit
        """
        with self.driver.session() as session:
            result = session.run(cypher, limit=limit)
            return [dict(r) for r in result]

    def construct_semantic_text(self, data):
        """
        💎 核心：构建【五合一】高密度语义文本
        """
        # 数据清洗 (处理 None)
        title = data.get('title', 'Unknown Title')
        
        # 列表转字符串 (实体部分)
        keywords_str = ", ".join(data.get('keywords', []))
        if not keywords_str: keywords_str = "None"
        
        proposed_method = data.get('proposed_method')
        if isinstance(proposed_method, list):
            methods = [str(item).strip() for item in proposed_method if item and str(item).strip()]
        elif proposed_method and str(proposed_method).strip():
            methods = [str(proposed_method).strip()]
        else:
            methods = []

        methods_str = ", ".join(methods)
        if not methods_str: methods_str = "None"
        
        # 文本部分
        abstract = data.get('abstract', '')
        if not abstract: abstract = "No abstract available."
        
        background = data.get('background', '')
        if not background: background = "No background context."

        # 🏗️ 拼接模板 (Prompt Engineering for Embedding)
        # 这种 key: value 的结构能极大提升检索准确率
        text = f"""
Title: {title} 
Keywords: {keywords_str}
Proposed Methods: {methods_str}
Background Context: {background}
Abstract: {abstract}
"""
        return text.strip()

    def update_embeddings(self, updates: List[Dict]):
        """
        批量写回数据库
        """
        if not updates: return 
        
        cypher = """
        UNWIND $updates AS row
        MATCH (p:Paper) 
        WHERE elementId(p) = row.id
        // BGE-M3 生成的是 float 列表
        SET p.embedding = row.vector
        """
        with self.driver.session() as session:
            session.run(cypher, updates=updates)

    def run(self):
        total_processed = 0
    
        # 打印一个示例，让你看看拼接出来是啥样
        print("\n🔎 --- 语义文本拼接示例 (Preview) ---")
        all_papers = self.fetch_papers_batch(limit=1000000)
        if all_papers:
            print(self.construct_semantic_text(all_papers[0]))
        else:
            print("⚠️ 数据库中没有需要处理的论文！")
            return
        print("--------------------------------------\n")

        # 1. 在 Python 里分批处理，避免重复从 Neo4j 取到同一批数据
        for start in range(0, len(all_papers), BATCH_SIZE):
            papers = all_papers[start:start + BATCH_SIZE]
            
            # 2. 构建文本
            texts = []
            valid_papers = []
            
            for p in papers:
                txt = self.construct_semantic_text(p)
                texts.append(txt)
                valid_papers.append(p)
            
            # 3. 计算向量 (BGE-M3 核心时刻)
            # normalize_embeddings=True 对余弦相似度搜索至关重要
            print(f"⏳ Embedding {len(texts)} papers...")
            embeddings = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            
            # 4. 准备写入
            updates = []
            for i, p in enumerate(valid_papers):
                updates.append({
                    "id": p['id'],
                    "vector": embeddings[i].tolist() # Numpy 转 List
                })
            
            # 5. 写回 Neo4j
            self.update_embeddings(updates)
            total_processed += len(updates)
            print(f"✅ Saved batch. Total processed: {total_processed}")

        print("🎉 所有论文向量化完成！")


if __name__ == "__main__":
    embedder = BioKGEmbedder()
    try:
        embedder.run()
    except KeyboardInterrupt:
        print("\n🛑 用户手动停止 (进度已保存)")
    finally:
        embedder.close()

