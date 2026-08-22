from neo4j_driver import db
from embedding_model import embedder


class VectorSearchTool:
    """
    Fast RAG 的向量检索模块。

    作用：
    1. 将用户问题转换为向量。
    2. 把向量作为参数传入写好的 Cypher。
    3. 调 Neo4j 向量索引，返回语义最相关的论文。
    """

    def search(self, user_query: str, top_k: int = 10):
        if not user_query or not user_query.strip():
            return []

        try:
            query_vector = embedder.embed_query(user_query)
        except Exception as e:
            print(f"[VectorSearch] 用户问题向量化失败: {e}")
            return []

        cypher = """
        CALL db.index.vector.queryNodes('paper_embeddings', $top_k, $query_vector)
        YIELD node AS p, score

        OPTIONAL MATCH (p)-[:HAS_KEYWORD]->(k:Keyword)

        WITH p, score,
             collect(DISTINCT k.name) AS keywords

        RETURN
            elementId(p) AS id,
            p.entry_id AS entry_id,
            p.title AS title,
            p.abstract AS abstract,
            p.background AS background,
            p.proposed_method AS proposed_method,
            score,
            keywords
        ORDER BY score DESC
        """

        try:
            rows = db.query(cypher, {
                "top_k": top_k,
                "query_vector": query_vector
            })
        except Exception as e:
            print(f"[VectorSearch] Neo4j 向量查询失败: {e}")
            return []

        results = []
        for row in rows:
            proposed_method = row.get("proposed_method")

            if isinstance(proposed_method, list):
                methods = [str(m).strip() for m in proposed_method if m]
            elif proposed_method:
                methods = [str(proposed_method).strip()]
            else:
                methods = []

            keywords = row.get("keywords") or []

            results.append({
                "id": row.get("entry_id") or row.get("id"),
                "neo4j_id": row.get("id"),
                "entry_id": row.get("entry_id"),
                "score": row.get("score", 0),
                "title": row.get("title") or "Unknown Title",
                "abstract": row.get("abstract") or "",
                "background": row.get("background") or "",
                "methods": ", ".join(dict.fromkeys(methods)) if methods else "未提及",
                "keywords": ", ".join(keywords) if keywords else "无",
                "source": "Vector Search"
            })

        return results


vector_search = VectorSearchTool()

if __name__ == "__main__":
    query = "什么是分子网络定量建模？"

    results = vector_search.search(query, top_k=5)

    print(f"共返回 {len(results)} 条结果")

    for i, item in enumerate(results, 1):
        print(f"\n--- 第 {i} 条 ---")
        print("标题：", item.get("title"))
        print("相似度：", item.get("score"))
        print("方法：", item.get("methods"))
        print("关键词：", item.get("keywords"))
        print("摘要：", item.get("abstract")[:300])
