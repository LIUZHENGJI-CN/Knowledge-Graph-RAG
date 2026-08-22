from textwrap import dedent

from vector_search import vector_search
from LLM_client import llm


FAST_RAG_SYSTEM_PROMPT = """
你是一个科研文献问答助手。

你只能根据提供的论文信息回答问题，不要编造。
如果材料中没有足够证据，请直接说明“当前检索到的论文信息不足以回答”。

回答要求：
1. 用中文回答。
2. 回答要简洁、清楚。
3. 涉及具体论文观点时，用 [Paper ID] 标注来源。
4. 不要写得太长，优先给出直接结论。
"""


FAST_RAG_USER_TEMPLATE = """
用户问题：
{question}

检索到的相关论文：
{context}

请基于以上论文信息回答用户问题。
"""


class FastRAGService:
    """
    简单版 Fast RAG 回答器。

    流程：
    1. 调用 vector_search 做向量召回
    2. 调用 llm.rerank 做简单重排序
    3. 把前几篇论文拼成上下文
    4. 调用 llm.chat 生成回答
    """

    def __init__(self, recall_top_k=10, final_top_n=5):
        self.recall_top_k = recall_top_k
        self.final_top_n = final_top_n

    def _shorten(self, text, max_chars=1000):
        """
        防止单篇摘要太长，让 Fast RAG 保持“快”。
        """
        if not text:
            return ""

        text = str(text).strip()

        if len(text) <= max_chars:
            return text

        return text[:max_chars].rstrip() + "..."

    def _build_rerank_docs(self, papers):
        """
        给 rerank 模型看的候选文档。
        """
        docs = []

        for p in papers:
            doc = f"""
            Title: {p.get("title", "")}
            Background: {p.get("background", "")}
            Methods: {p.get("methods", "")}
            Keywords: {p.get("keywords", "")}
            Abstract: {self._shorten(p.get("abstract", ""), 800)}
            """
            docs.append(dedent(doc).strip())

        return docs

    def _select_final_papers(self, raw_papers, reranked_results):
        """
        根据 rerank 结果选择最终论文。
        如果 rerank 失败，就直接用向量检索前几篇兜底。
        """
        final_papers = []

        if reranked_results:
            for item in reranked_results:
                index = item.get("index")

                if isinstance(index, int) and 0 <= index < len(raw_papers):
                    paper = dict(raw_papers[index])
                    paper["rerank_rank"] = item.get("rank")
                    paper["rerank_score"] = item.get("relevance_score")
                    final_papers.append(paper)

        if not final_papers:
            final_papers = [dict(p) for p in raw_papers[:self.final_top_n]]

        return final_papers

    def _build_context(self, papers):
        """
        给最终 LLM 回答使用的上下文。
        """
        blocks = []

        for p in papers:
            block = f"""
            --- Paper ID: {p.get("id", "Unknown")} ---
            Title: {p.get("title", "Unknown Title")}
            Vector Score: {p.get("score", 0)}
            Keywords: {p.get("keywords", "无")}
            Methods: {p.get("methods", "未提及")}
            Background: {self._shorten(p.get("background", ""), 500)}
            Abstract: {self._shorten(p.get("abstract", ""), 1000)}
            ---------------------------
            """
            blocks.append(dedent(block).strip())

        return "\n\n".join(blocks)

    def answer(self, user_query):
        """
        执行 Fast RAG 问答。

        返回：
        answer: str
        sources: list[dict]
        """
        user_query = (user_query or "").strip()

        if not user_query:
            return "请输入一个具体问题。", []

        print(f"[FastRAG] 用户问题：{user_query}")

        # 1. 向量召回
        print(f"[FastRAG] Step 1: 向量检索 Top {self.recall_top_k}")
        raw_papers = vector_search.search(user_query, top_k=self.recall_top_k)

        if not raw_papers:
            return "抱歉，知识图谱中没有检索到相关论文。", []

        # 2. Rerank 精排
        print(f"[FastRAG] Step 2: Rerank 选择 Top {self.final_top_n}")
        docs_for_rerank = self._build_rerank_docs(raw_papers)
        reranked_results = llm.rerank(
            query=user_query,
            documents=docs_for_rerank,
            top_n=self.final_top_n
        )

        final_papers = self._select_final_papers(raw_papers, reranked_results)

        # 3. 构建上下文
        context = self._build_context(final_papers)

        # 4. 生成回答
        print("[FastRAG] Step 3: 生成回答")
        user_prompt = FAST_RAG_USER_TEMPLATE.format(
            question=user_query,
            context=context
        )

        answer = llm.chat([
            {"role": "system", "content": FAST_RAG_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ])

        return answer, final_papers


fast_rag = FastRAGService()


if __name__ == "__main__":
    question = "什么是分子网络定量建模？"

    answer, sources = fast_rag.answer(question)

    print("\n========== Answer ==========")
    print(answer)

    print("\n========== Sources ==========")
    for i, p in enumerate(sources, 1):
        score = p.get("rerank_score")
        if score is None:
            score = p.get("score", 0)

        print(f"{i}. [{score}] {p.get('title', 'Unknown Title')}")
