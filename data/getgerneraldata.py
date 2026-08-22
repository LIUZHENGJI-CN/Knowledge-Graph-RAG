
import os
import re
import json
import time
import threading
import requests
import arxiv
import fitz  # PyMuPDF
from py2neo import Graph, Node, Relationship
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

# =====================================================
# 1. 配置区域
# =====================================================
NEO4J_URI = "neo4j://localhost:7687" # 连接到本地 Neo4j 实例
NEO4J_AUTH = ("neo4j", "your_neo4j_password_here") # 替换为您的 Neo4j 用户名和密码

API_KEY = "your_openai_compatible_api_key_here"  # 替换为您的 OpenAI API Key
BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1" # 替换为您的 OpenAI 兼容接口地址
MODEL = "qwen-plus"

MAP_FILE = "C:/Users/lzj/Desktop/申请博士/复现知识图谱/datainstitution_map.json" 
CAT_MAP_FILE = "C:/Users/lzj/Desktop/申请博士/复现知识图谱/category.json"
OUTPUT_JSON = "C:/Users/lzj/Desktop/申请博士/复现知识图谱/data/full_paper_knowledge_base.json"

# ----- 运行参数（测试阶段用少量论文）-----
MAX_PAPERS = 2000          # 先跑5篇，成功后可以改成 100、500
MAX_WORKERS = 5         # 并发数，避免限流

db_lock = threading.Lock()
data_lock = threading.Lock()
extracted_results = [] # 线程安全收集区

# =====================================================
# 2. 系统提示词 (核心：统合关键词语义以减少冗余)
# =====================================================
SYSTEM_PROMPT = """
Analyze the academic abstract and extract:
1. "background": 1-2 sentence summary of context.
2. "methods": THE OVERALL PROPOSED ARCHITECTURE/SOLUTION NAME (e.g., 'BioLLM Framework').
3. "keywords": 3-5 standard academic terms. 
   - REQUIREMENT: Use canonical, singular, and full forms (e.g., 'Neural Network' instead of 'NNs'). 
   - This ensures multiple papers link to the same core entities in a knowledge graph.

Return strictly as JSON:
{"background": "...", "methods": "...", "keywords": ["..."]}
"""

# =====================================================
# 3. 初始化与辅助函数
# =====================================================
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
graph = Graph(NEO4J_URI, auth=NEO4J_AUTH)

def load_json(path, default):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return default

CATEGORY_MAP = load_json(CAT_MAP_FILE, {
    "q-bio.BM": "Biomolecules",
    "q-bio.CB": "Cell Behavior",
    "q-bio.GN": "Genomics",
    "q-bio.MN": "Molecular Networks",
    "q-bio.NC": "Neurons and Cognition",
    "q-bio.OT": "Other Quantitative Biology",
    "q-bio.PE": "Populations and Evolution",
    "q-bio.QM": "Quantitative Methods",
    "q-bio.SC": "Subcellular Processes",
    "q-bio.TO": "Tissues and Organs"
})

# =====================================================
# 4. 核心执行 Worker
# =====================================================

def process_paper(paper, idx):
    print(f"[{idx}] 正在处理: {paper.title[:50]}...")
    
    # A. 语义知识提取
    schema = extract_ai_knowledge(f"Title: {paper.title}\nAbstract: {paper.summary}")
    
    # B. 构造【全量知识 JSON 对象】
    paper_full_record = {
        "metadata": {
            "title": paper.title,
            "entry_id": paper.entry_id,
            "published_date": paper.published.strftime("%Y-%m-%d"),
            "doi": paper.doi if paper.doi else "N/A",
            "pdf_url": paper.pdf_url,
            "authors": [author.name for author in paper.authors],
            "categories": [CATEGORY_MAP.get(cat, cat) for cat in paper.categories]
        },
        "semantic_extraction": {
            "background": schema["background"],
            "proposed_overall_method": schema["methods"],
            "unified_keywords": [kw.strip().title() for kw in schema.get("keywords", []) if kw]
        }
    }

    # C. 同步至图谱
    with db_lock:
        # 1. 论文节点 (将背景和整体方法存为属性)
        p_node = Node(
            "Paper", 
            entry_id=paper.entry_id, 
            title=paper.title, 
            published=paper_full_record["metadata"]["published_date"],
            background=schema["background"],
            proposed_method=schema["methods"]
        )
        graph.merge(p_node, "Paper", "entry_id")

        # 2. 作者节点
        for author_name in paper_full_record["metadata"]["authors"]:
            a_node = Node("Author", name=author_name)
            graph.merge(a_node, "Author", "name")
            graph.merge(Relationship(p_node, "AUTHORED_BY", a_node))

        # 3. 关键词实体 (跨论文聚合的核心)
        for kw in paper_full_record["semantic_extraction"]["unified_keywords"]:
            k_node = Node("Keyword", name=kw)
            graph.merge(k_node, "Keyword", "name")
            graph.merge(Relationship(p_node, "HAS_KEYWORD", k_node))

    # D. 线程安全保存全量数据用于导出
    with data_lock:
        extracted_results.append(paper_full_record)
    print(f"[{idx}] 完成全量抽取与同步。")

def extract_ai_knowledge(text):
    try:
        resp = client.chat.completions.create(
            model=MODEL, messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": text}], temperature=0
        )
        return json.loads(resp.choices[0].message.content)
    except: 
        return {"background": "", "methods": "", "keywords": []}

# =====================================================
# 5. 主流程
# =====================================================

def main():
    print("--- 正在初始化数据库 ---")
    graph.run("MATCH (n) DETACH DELETE n")
    graph.run("CREATE CONSTRAINT IF NOT EXISTS FOR (p:Paper) REQUIRE p.entry_id IS UNIQUE")
    graph.run("CREATE CONSTRAINT IF NOT EXISTS FOR (k:Keyword) REQUIRE k.name IS UNIQUE")

    query = " OR ".join([f"cat:{k}" for k in CATEGORY_MAP.keys()])
    search = arxiv.Search(query=query, max_results=MAX_PAPERS)
    papers = list(arxiv.Client().results(search))

    print(f"成功获取 {len(papers)} 篇论文。开始并行全量抽取...\n")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        pool.map(lambda p: process_paper(p[1], p[0]), enumerate(papers, 1))

    # 导出包含所有属性的精美 JSON
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(extracted_results, f, indent=4, ensure_ascii=False)

    print(f"\n🎉 构建完成！全量知识已同步至 Neo4j 并导出至 {OUTPUT_JSON}。")

if __name__ == "__main__":
    main()
