import os
import re
import time
import arxiv
from neo4j import GraphDatabase
from tqdm import tqdm


# --- 1. 配置信息 ---
# 请替换为你的真实信息
NEO4J_URI = "neo4j://localhost:7687" # 连接到本地 Neo4j 实例
NEO4J_AUTH = ("neo4j", "your_neo4j_password_here") # 替换为您的 Neo4j 用户名和密码

class ArxivAbstractFetcher:
    def __init__(self):
        print(f"🔌 连接数据库: {NEO4J_URI}")
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
        # 配置 Arxiv 客户端
        self.arxiv_client = arxiv.Client(
            page_size=10,        # 每次请求的批次大小
            delay_seconds=0.5,   # 稍微快一点，因为只抓元数据
            num_retries=5        # 失败重试次数
        )

    def close(self):
        self.driver.close()
        
    def fill_abstract_from_background(self):
        """
        🔹 新增方法：先把 background 填到 abstract
        """
        print("📝 批量填充 abstract（从 background）...")
        query = """
        MATCH (p:Paper)
        WHERE exists(p.background) AND (p.abstract IS NULL OR p.abstract = "")
        SET p.abstract = p.background
        RETURN count(p) AS updated_count
        """
        with self.driver.session() as session:
            result = session.run(query).single()
            updated_count = result["updated_count"] if result else 0
        print(f"✅ 已填充 {updated_count} 篇论文的 abstract。")
        return updated_count

    def get_target_papers(self):
        """
        查找有 entry_id (Arxiv链接) 但还没有 abstract 的论文
        """
        print("🔍 正在扫描需要补充摘要的论文...")
        query = """
        MATCH (p:Paper)
        WHERE p.entry_id CONTAINS 'arxiv.org' 
          AND (p.abstract IS NULL OR p.abstract = "")
        RETURN elementId(p) as node_id, p.entry_id as url, p.title as title
        """
        with self.driver.session() as session:
            result = session.run(query)
            return [dict(r) for r in result]

    def extract_arxiv_id(self, url):
        """
        严格提取标准 ArXiv ID (例如 1705.09708 或 1705.09708v1)
        过滤掉 adap, cond, cs 等非 ID 字符串
        """
        if not url: return None
        
        # 1. 尝试匹配新版 ID (数字.数字)
        # 例如: http://arxiv.org/abs/1705.09708v1 -> 1705.09708v1
        match_new = re.search(r'arxiv\.org/abs/(\d+\.\d+v?\d*)', url)
        if match_new:
            return match_new.group(1)
            
        # 2. 尝试匹配旧版 ID (类别/数字) - 如果你需要支持 2007年以前的老论文
        # 例如: http://arxiv.org/abs/cs/050101 -> cs/050101
        match_old = re.search(r'arxiv\.org/abs/([a-zA-Z\-\.]+\/\d+)', url)
        if match_old:
            return match_old.group(1)

        # 如果提取出来的是纯字母 (如 'cs', 'cond')，直接丢弃
        return None

    def update_abstract(self, node_id, abstract_text):
        """
        将清洗后的摘要写入 Neo4j
        """
        query = """
        MATCH (p:Paper)
        WHERE elementId(p) = $id
        SET p.abstract = $text
        """
        with self.driver.session() as session:
            session.run(query, id=node_id, text=abstract_text)

    def run(self):
        papers = self.get_target_papers()
        total = len(papers)
        print(f"📋 共发现 {total} 篇待处理论文")

        if total == 0:
            print("🎉 所有论文都已经有摘要了！无需操作。")
            return

        success_count = 0
        
        # 咱们使用 tqdm 显示进度条
        for paper in tqdm(papers, desc="正在抓取"):
            url = paper['url']
            node_id = paper['node_id']
            
            arxiv_id = self.extract_arxiv_id(url)
            if not arxiv_id:
                # print(f"⚠️ 跳过无效链接: {url}")
                continue

            try:
                # 调用 Arxiv API
                search = arxiv.Search(id_list=[arxiv_id])
                # next() 获取第一个结果
                result = next(self.arxiv_client.results(search))
                
                # 获取摘要并清洗 (把换行符换成空格，保持语义连贯)
                raw_summary = result.summary
                clean_summary = raw_summary.replace("\n", " ").strip()
                
                # 写入数据库
                self.update_abstract(node_id, clean_summary)
                success_count += 1
                
            except StopIteration:
                # 找不到结果，默默跳过即可
                pass
            except Exception as e:
                # 【新增】如果是 400 错误，说明 ID 不对，打印一下但不要崩
                if "400" in str(e):
                    print(f"⚠️ 跳过无效ID: {arxiv_id}")
                else:
                    print(f"❌ 处理出错 {arxiv_id}: {e}")
                # 出错时短暂休眠，防止连环报错卡死网络
                time.sleep(1)
        print(f"\n✅ 处理完成！成功导入: {success_count}/{total}")

if __name__ == "__main__":
    fetcher = ArxivAbstractFetcher()
    try:
        fetcher.run()
    finally:
        fetcher.close()
