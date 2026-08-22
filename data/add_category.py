import json
from py2neo import Graph

# =====================================================
# 1. 配置区域
# =====================================================
 
CAT_MAP_FILE = r"C:\Users\lzj\Desktop\申请博士\复现知识图谱\data\category.json"
INPUT_JSON = r"C:\Users\lzj\Desktop\申请博士\复现知识图谱\data\full_paper_knowledge_base.json"


NEO4J_URI = "neo4j://localhost:7687" # 连接到本地 Neo4j 实例
NEO4J_AUTH = ("neo4j", "your_neo4j_password_here") # 替换为您的 Neo4j 用户名和密码
graph = Graph(NEO4J_URI, auth=NEO4J_AUTH)

# =====================================================
# 2. 初始化与加载
# =====================================================
with open(CAT_MAP_FILE, "r", encoding="utf-8") as f:
    CATEGORY_MAP = json.load(f)

with open(INPUT_JSON, "r", encoding="utf-8") as f:
    papers_data = json.load(f)


def init_constraints():
    """初始化数据库约束，保证实体唯一。"""
    graph.run(
        "CREATE CONSTRAINT category_code_unique IF NOT EXISTS "
        "FOR (c:Category) REQUIRE c.code IS UNIQUE"
    )
    graph.run(
        "CREATE CONSTRAINT paper_id_unique IF NOT EXISTS "
        "FOR (p:Paper) REQUIRE p.entry_id IS UNIQUE"
    )


def build_category_payload(categories):
    """把 JSON 中的 categories 转成可写入 Neo4j 的分类对象。"""
    payload = []
    seen_codes = set()

    for value in categories or []:
        if not value:
            continue

        if value in CATEGORY_MAP:
            code = value
            name = CATEGORY_MAP[value]
        else:
            reverse_matches = [k for k, v in CATEGORY_MAP.items() if v == value]
            if reverse_matches:
                code = reverse_matches[0]
                name = value
            else:
                code = value
                name = value

        if code in seen_codes:
            continue

        seen_codes.add(code)
        payload.append({"code": code, "name": name})

    return payload


def sync_paper_task(paper_item):
    """单篇论文的分类同步任务：只把本地 JSON 写入 Neo4j。"""
    metadata = paper_item.get("metadata", {})
    entry_id_url = metadata.get("entry_id")
    categories = metadata.get("categories", [])

    if not entry_id_url:
        return "SKIP: missing entry_id"

    cat_params = build_category_payload(categories)
    if not cat_params:
        return f"SKIP: {entry_id_url} (no categories in JSON)"

    try:
        paper_exists = graph.evaluate(
            "MATCH (p:Paper {entry_id: $entry_id}) RETURN count(p)",
            entry_id=entry_id_url,
        )
        if not paper_exists:
            return f"SKIP: {entry_id_url} (Paper not found in Neo4j)"

        cypher = """
        MATCH (p:Paper {entry_id: $entry_id})
        UNWIND $cats AS cat_info
        MERGE (c:Category {code: cat_info.code})
        SET c.name = cat_info.name
        MERGE (p)-[:BELONGS_TO]->(c)
        """
        graph.run(cypher, entry_id=entry_id_url, cats=cat_params)
        return f"SUCCESS: {entry_id_url} -> {len(cat_params)} categories"
    except Exception as e:
        return f"ERROR: {entry_id_url} ({str(e)[:120]})"


# =====================================================
# 3. 主逻辑控制
# =====================================================
def main():
    init_constraints()

    to_process = [p for p in papers_data if p.get("metadata", {}).get("categories")]
    total_to_process = len(to_process)
    print(f"检测到 JSON 中有 {total_to_process} 篇论文可同步分类到 Neo4j。")

    if total_to_process == 0:
        print("JSON 中没有可同步的 categories，脚本结束。")
        return

    completed = 0
    success = 0
    skipped = 0

    for paper_item in to_process:
        result_msg = sync_paper_task(paper_item)
        completed += 1
        print(f"[{completed}/{total_to_process}] {result_msg}")

        if result_msg.startswith("SUCCESS"):
            success += 1
        elif result_msg.startswith("SKIP"):
            skipped += 1

    print("\n同步完成。")
    print(f"成功: {success}")
    print(f"跳过: {skipped}")
    print(f"总计: {total_to_process}")


if __name__ == "__main__":
    main()

