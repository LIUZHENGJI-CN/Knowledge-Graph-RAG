from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional
import json

from fast_rag import fast_rag
from text2cypher import t2c_tool
from LLM_client import llm


RouteType = Literal["fast_rag", "text2cypher", "hybrid"]


@dataclass
class RouterResult:
    question: str
    route: RouteType
    answer: str
    cypher: Optional[str] = None
    data: Optional[list] = None
    sources: Optional[list] = None
    raw: Optional[Any] = None
    reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "route": self.route,
            "answer": self.answer,
            "cypher": self.cypher,
            "data": self.data,
            "sources": self.sources,
            "raw": self.raw,
            "reason": self.reason,
        }


class QueryRouter:
    """
    轻量问答路由器：
    - 图谱结构查询：text2cypher
    - 论文语义问答：fast RAG
    - 两者都有：hybrid
    """

    def route(self, question: str) -> Dict[str, Any]:
        question = (question or "").strip()

        if not question:
            return RouterResult(
                question="",
                route="fast_rag",
                answer="请输入问题。",
                reason="empty question",
            ).to_dict()

        route_type, reason = self._classify(question)

        if route_type == "text2cypher":
            return self._run_text2cypher(question, reason).to_dict()

        if route_type == "hybrid":
            return self._run_hybrid(question, reason).to_dict()

        return self._run_fast_rag(question, reason).to_dict()

    def _classify(self, question: str) -> tuple[RouteType, str]:
        q = question.lower()

        graph_keywords = [
            "多少", "几个", "数量", "统计", "列出", "有哪些", "谁", "哪位",
            "作者", "机构", "学校", "单位", "会议", "期刊", "年份", "发表",
            "引用", "被引", "合作", "共同作者", "关系", "路径", "邻居",
            "找", "寻找", "专家",
            "paper", "papers", "author", "authors", "institution",
            "venue", "citation", "citations", "cite", "cited",
            "collaborate", "collaboration", "count", "list",
            "which", "who", "where", "when",
        ]

        rag_keywords = [
            "总结", "概括", "介绍", "解释", "分析", "趋势", "研究方向",
            "背景", "贡献", "方法", "创新", "不足", "为什么", "如何",
            "什么是",
            "summary", "summarize", "explain", "trend", "method",
            "contribution", "background", "compare", "comparison",
            "analyze", "analysis",
        ]

        graph_score = sum(keyword in q for keyword in graph_keywords)
        rag_score = sum(keyword in q for keyword in rag_keywords)

        if graph_score > 0 and rag_score > 0:
            return "hybrid", "同时包含图谱结构查询和语义总结需求"

        if graph_score > 0:
            return "text2cypher", "更适合使用 text2cypher 查询知识图谱结构"

        return "fast_rag", "更适合使用 fast RAG 做论文语义检索问答"

    def _run_fast_rag(self, question: str, reason: str) -> RouterResult:
        try:
            answer, sources = fast_rag.answer(question)

            return RouterResult(
                question=question,
                route="fast_rag",
                answer=answer,
                sources=sources,
                raw={
                    "answer": answer,
                    "sources": sources,
                },
                reason=reason,
            )
        except Exception as exc:
            return RouterResult(
                question=question,
                route="fast_rag",
                answer=f"fast RAG 调用失败：{exc}",
                reason=reason,
            )

    def _run_text2cypher(self, question: str, reason: str) -> RouterResult:
        try:
            result = t2c_tool.run(question)

            message = result.get("message", "")
            data = result.get("data", [])
            cypher = result.get("cypher", "")

            if data:
                answer = self._format_graph_answer(message, data)
            else:
                answer = message or result.get("error", "没有查询到结果。")

            return RouterResult(
                question=question,
                route="text2cypher",
                answer=answer,
                cypher=cypher,
                data=data,
                raw=result,
                reason=reason,
            )
        except Exception as exc:
            return RouterResult(
                question=question,
                route="text2cypher",
                answer=f"text2cypher 调用失败：{exc}",
                reason=reason,
            )

    def _run_hybrid(self, question: str, reason: str) -> RouterResult:
        graph_result = self._run_text2cypher(question, reason)

        if graph_result.answer.startswith("text2cypher 调用失败"):
            fallback_result = self._run_fast_rag(
                question,
                "hybrid 中 text2cypher 调用失败，回退到 fast RAG"
            )
            fallback_result.route = "hybrid"
            return fallback_result

        if not graph_result.data:
            return RouterResult(
                question=question,
                route="hybrid",
                answer=graph_result.answer,
                cypher=graph_result.cypher,
                data=graph_result.data,
                raw={
                    "text2cypher": graph_result.raw,
                },
                reason=reason,
            )

        system_prompt = """
你是一个科研知识图谱问答助手。

你需要根据提供的知识图谱查询结果回答用户问题。
不要编造查询结果之外的信息。
如果查询结果不足，请说明不足。
回答要用中文，简洁清楚，适合网页演示展示。
"""

        user_prompt = (
            f"用户问题：{question}\n\n"
            f"知识图谱查询结果：\n{self._compact_json(graph_result.data)}\n\n"
            "请基于这些查询结果，直接回答用户问题，并适当总结规律或特点。"
        )

        try:
            answer = llm.chat([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ])

            return RouterResult(
                question=question,
                route="hybrid",
                answer=answer,
                cypher=graph_result.cypher,
                data=graph_result.data,
                raw={
                    "text2cypher": graph_result.raw,
                    "summary": answer,
                },
                reason=reason,
            )

        except Exception as exc:
            return RouterResult(
                question=question,
                route="hybrid",
                answer=f"hybrid 总结失败：{exc}\n\n图谱查询结果：\n{graph_result.answer}",
                cypher=graph_result.cypher,
                data=graph_result.data,
                raw={
                    "text2cypher": graph_result.raw,
                },
                reason=reason,
            )

    def _format_graph_answer(self, message: str, data: list) -> str:
        preview = data[:5]
        preview_text = json.dumps(preview, ensure_ascii=False, indent=2, default=str)

        if len(data) > 5:
            preview_text += f"\n... 共 {len(data)} 条，仅展示前 5 条"

        return f"{message}\n\n{preview_text}"

    def _compact_json(self, data: Any, max_chars: int = 2500) -> str:
        text = json.dumps(data, ensure_ascii=False, default=str)

        if len(text) > max_chars:
            return text[:max_chars] + "\n...[内容过长，已截断]"

        return text


router = QueryRouter()


def answer_question(question: str) -> Dict[str, Any]:
    return router.route(question)


def route_question(question: str) -> Dict[str, Any]:
    return answer_question(question)


def ask(question: str) -> Dict[str, Any]:
    return answer_question(question)


if __name__ == "__main__":
    test_questions = [
        "寻找与 Ilya Nemenman 教授合作最频繁的作者",
        "什么是分子网络定量建模？",
        "复旦大学的作者主要研究什么关键词？请总结这些研究方向的特点",
    ]

    for question in test_questions:
        print("=" * 80)
        print(f"问题：{question}")

        result = answer_question(question)

        print(f"选择路线：{result.get('route')}")
        print(f"选择原因：{result.get('reason')}")
        print(f"Cypher：{result.get('cypher')}")
        print(f"回答：{result.get('answer')}")
