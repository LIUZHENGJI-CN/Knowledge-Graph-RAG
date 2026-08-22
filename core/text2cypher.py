import sys
import os
import json
import re

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from LLM_client import llm
from neo4j_driver import db


class Text2CypherTool:
    """
    Text2Cypher 工具。

    用途：
    1. 针对“找专家、找关系、找机构、找合作网络”等图谱结构问题；
    2. 基于真实 Neo4j Schema，通过 LLM 生成 Cypher；
    3. 使用 core.neo4j_driver.db 执行查询；
    4. 返回 cypher + data，供前端展示表格和网络图。
    """

    def __init__(self):
        self.schema_str = """
[Node Labels and Properties]

1. Paper
   - entry_id: String, paper id / arXiv id
   - title: String
   - abstract: String
   - background: String
   - proposed_method: String or List[String]
   - published: String
   - full_text: String, never return this property

2. Author
   - name: String

3. Institution
   - name: String

4. Keyword
   - name: String

5. Category
   - name: String

6. Journal
   - name: String

[Relationship Types and Directions]

- (:Paper)-[:AUTHORED_BY]->(:Author)
- (:Paper)-[:AFFILIATED_WITH]->(:Institution)
- (:Paper)-[:HAS_KEYWORD]->(:Keyword)
- (:Paper)-[:BELONGS_TO]->(:Category)
- (:Paper)-[:PUBLISHED_IN]->(:Journal)
"""

    def _build_prompt(self):
        return f"""
You are an expert Neo4j Cypher developer for an academic knowledge graph.

Your task:
Convert the user's natural language question into ONE read-only Cypher query.

Graph schema:
{self.schema_str}

Rules:
1. Return ONLY the Cypher query. No markdown. No explanation.
2. Only generate read-only queries: MATCH, OPTIONAL MATCH, WHERE, WITH, RETURN, ORDER BY, LIMIT.
3. Never use CREATE, MERGE, DELETE, DETACH, SET, REMOVE, DROP, LOAD CSV, CALL apoc, or database write operations.
4. Never return p.full_text.
5. Always add LIMIT 20 unless the query already has a smaller LIMIT.
6. For string matching, use:
   toLower(x.name) CONTAINS toLower('value')
   or
   toLower(p.title) CONTAINS toLower('value')
7. For expert-finding questions, identify experts through papers they authored and rank them by relevant paper count.
8. For questions about a paper's affiliated institution, use AFFILIATED_WITH between Paper and Institution.
9. If the user asks in Chinese, still generate valid Cypher.

Examples:

Q: 什么是分子网络定量建模？
A: MATCH (p:Paper)-[:HAS_KEYWORD]->(k:Keyword) WHERE toLower(p.title) CONTAINS toLower('分子网络') OR toLower(p.abstract) CONTAINS toLower('分子网络') OR toLower(k.name) CONTAINS toLower('分子网络') RETURN p.entry_id AS entry_id, p.title AS title, p.abstract AS abstract, p.background AS background, p.proposed_method AS proposed_method LIMIT 10

Q: 找深度学习方向的专家
A:  MATCH (a:Author)<-[:AUTHORED_BY]-(p:Paper)-[:HAS_KEYWORD]->(k:Keyword)
    WHERE toLower(k.name) CONTAINS toLower('deep learning')
       OR toLower(p.title) CONTAINS toLower('deep learning')
       OR toLower(p.abstract) CONTAINS toLower('deep learning')
    RETURN a.name AS author,
       count(DISTINCT p) AS paper_count,
       collect(DISTINCT p.title)[0..5] AS representative_papers
    ORDER BY paper_count DESC
    LIMIT 20

Q: 寻找与 Ilya Nemenman 教授合作最频繁的作者
A: MATCH (a:Author)<-[:AUTHORED_BY]-(p:Paper)-[:AUTHORED_BY]->(co:Author)
WHERE toLower(a.name) CONTAINS toLower('Ilya Nemenman')
  AND co <> a
RETURN co.name AS collaborator,
       count(DISTINCT p) AS collaboration_count,
       collect(DISTINCT p.title)[0..5] AS representative_papers
ORDER BY collaboration_count DESC
LIMIT 20
    
Q: 复旦大学的作者主要研究什么关键词？
A:  MATCH (i:Institution)<-[:AFFILIATED_WITH]-(p:Paper)-[:AUTHORED_BY]->(a:Author),
          (p)-[:HAS_KEYWORD]->(k:Keyword)
    WHERE toLower(i.name) CONTAINS toLower('Fudan University')
       OR toLower(i.name) CONTAINS toLower('复旦')
    RETURN k.name AS keyword,
       count(DISTINCT p) AS paper_count,
       collect(DISTINCT a.name)[0..10] AS authors
    ORDER BY paper_count DESC
    LIMIT 20
"""

    def _clean_cypher(self, response: str):
        if not response:
            return ""

        cypher = response.strip()
        cypher = cypher.replace("```cypher", "").replace("```", "").strip()

        prefixes = [
            "Cypher Query:",
            "Cypher:",
            "Query:",
            "A:",
        ]
        for prefix in prefixes:
            if cypher.startswith(prefix):
                cypher = cypher[len(prefix):].strip()

        match = re.search(r"\b(MATCH|OPTIONAL MATCH)\b", cypher, flags=re.IGNORECASE)
        if match:
            cypher = cypher[match.start():].strip()

        cypher = cypher.rstrip(";").strip()

        if not re.search(r"\bLIMIT\s+\d+\b", cypher, flags=re.IGNORECASE):
            cypher += " LIMIT 20"

        return cypher

    def _is_safe_query(self, cypher: str):
        if not cypher:
            return False

        lowered = cypher.lower()

        forbidden_keywords = [
            " create ",
            " merge ",
            " delete ",
            " detach ",
            " set ",
            " remove ",
            " drop ",
            " load csv ",
            " call apoc",
            " dbms.",
            " grant ",
            " deny ",
        ]

        normalized = f" {lowered} "
        return not any(keyword in normalized for keyword in forbidden_keywords)

    def generate_cypher(self, user_query: str):
        """
        使用 LLM 将自然语言问题转换为 Cypher。
        """
        system_prompt = self._build_prompt()

        response = llm.chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query},
            ]
        )

        cypher = self._clean_cypher(response)
        return cypher

    def run(self, user_query: str):
        """
        Text2Cypher 执行主流程：
        用户问题 -> 生成 Cypher -> 安全检查 -> Neo4j 查询 -> 返回结果。
        """
        if not user_query or not user_query.strip():
            return {
                "cypher": "",
                "data": [],
                "message": "请输入有效问题。",
            }

        cypher = self.generate_cypher(user_query)
        print(f"[Text2Cypher] 生成语句: {cypher}")

        if not self._is_safe_query(cypher):
            return {
                "cypher": cypher,
                "data": [],
                "error": "Unsafe Cypher query blocked.",
                "message": "生成的 Cypher 不符合只读查询要求，已阻止执行。",
            }

        try:
            results = db.query(cypher)

            if not results:
                return {
                    "cypher": cypher,
                    "data": [],
                    "message": "查询成功，但没有找到匹配的数据。",
                }

            return {
                "cypher": cypher,
                "data": results,
                "message": f"成功找到 {len(results)} 条结果。",
            }

        except Exception as e:
            print(f"[Text2Cypher] 执行报错: {e}")
            return {
                "cypher": cypher,
                "data": [],
                "error": str(e),
                "message": "生成的查询语句执行失败。",
            }


t2c_tool = Text2CypherTool()


if __name__ == "__main__":
    questions = [
        "寻找与 Ilya Nemenman 教授合作最频繁的作者",
        "谁和 Ilya Nemenman 合作最多？",
        "找深度学习方向的专家",
        "复旦大学的作者主要研究什么关键词？",
    ]

    for question in questions:
        print(f"\nQuestion: {question}")
        result = t2c_tool.run(question)
        print("Cypher:", result.get("cypher"))
        print("Message:", result.get("message"))
        print(json.dumps(result.get("data", [])[:3], indent=2, ensure_ascii=False))
