import os
import re
import time
import random
import requests
import fitz  # PyMuPDF
from neo4j import GraphDatabase
from tqdm import tqdm

# ================= ⚙️ 配置区域 =================
# 1. 数据库配置
NEO4J_URI = "neo4j://localhost:7687" # 连接到本地 Neo4j 实例
NEO4J_AUTH = ("neo4j", "your_neo4j_password_here") # 替换为您的 Neo4j 用户名和密码



# 2. 镜像源选择 (二选一)
# 方案A: 中科院理论物理所 (通常速度最快，推荐)
MIRROR_DOMAIN = "arxiv.org"# 方案B: Arxiv官方中国镜像 (如果A连不上就换B)
# MIRROR_DOMAIN = "cn.arxiv.org"

# 3. 伪装头 (假装自己是浏览器，防止被拦截)
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}
# ===============================================

class TextCleaner:
    """
    清洗器：负责把 PDF 里的烂文本洗干净
    """
    def __init__(self):
        # 匹配参考文献的开头，一旦遇到这些词，后面的全砍掉
        self.ref_patterns = [
            r'(?:^|\n)\s*R\s*E\s*F\s*E\s*R\s*E\s*N\s*C\s*E\s*S', 
            r'(?:^|\n)\s*References',
            r'(?:^|\n)\s*Bibliography',
            r'(?:^|\n)\s*LITERATURE CITED',
            r'(?:^|\n)\s*Acknowledgements', # 致谢后面通常就是引用
        ]

    def clean(self, text):
        if not text: return ""
        
        # 1. 🔪 斩断尾部 (参考文献)
        for pattern in self.ref_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                text = text[:match.start()]
                break 

        # 2. 🩹 修复连字符 (Algorithm- \n based -> Algorithmbased -> Algorithm based)
        # 这一步稍微激进一点，把连字符去掉
        text = re.sub(r'(\w+)-\s*\n\s*([a-z])', r'\1\2', text)
        
        # 3. 🩹 修复强制换行 (把断开的句子拼回去)
        text = re.sub(r'(?<!\.)\n\s*([a-z])', r' \1', text)

        # 4. 🧼 去除页眉噪音 (如 arXiv:1705.xxx)
        text = re.sub(r'arXiv:[\d\.]+\w*\s*\[.*?\]', '', text)
        
        # 5. 压缩多余空格
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

class ArxivFullTextFiller:
    def __init__(self):
        print(f"🔌 连接数据库: {NEO4J_URI}")
        print(f"🇨🇳 使用纯净镜像: {MIRROR_DOMAIN} (无API查询)")
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
        self.cleaner = TextCleaner()

    def close(self):
        self.driver.close()

    def get_target_papers(self):
        print("🔍 扫描需要补充全文的论文...")
        query = """
        MATCH (p:Paper)
        WHERE p.entry_id CONTAINS 'arxiv.org' 
          AND (p.full_text IS NULL OR p.full_text = "")
        RETURN elementId(p) as id, p.entry_id as url, p.title as title
        """
        with self.driver.session() as session:
            result = session.run(query)
            return [dict(r) for r in result]

    def extract_strict_id(self, url):
        """
        只提取标准的数字ID，过滤掉 adap, cond, cs 等脏数据
        """
        if not url: return None
        # 匹配标准格式: 1705.09708 或 1705.09708v1
        match = re.search(r'arxiv\.org/abs/(\d+\.\d+v?\d*)', url)
        if match:
            return match.group(1)
        # 匹配老式格式: math/0501231
        match_old = re.search(r'arxiv\.org/abs/([a-z\-]+/\d+)', url)
        if match_old:
            return match_old.group(1)
        return None


    # 2. 在 download_pdf 函数中，禁用 verify (防止SSL报错)
    def download_pdf(self, arxiv_id):

        pdf_url = f"https://{MIRROR_DOMAIN}/pdf/{arxiv_id}.pdf"
        try:
            response = requests.get(
                pdf_url, 
                headers=HEADERS, 
                timeout=60, 
                verify=False  # 👈 加上这句，防止代理导致的证书报错
            )
            if response.status_code == 200 and len(response.content) > 10000: # 👈 加个保险：小于10KB的不要
                return response.content
            return None
        except Exception:
            return None

    def process_pdf_bytes(self, pdf_bytes):
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            text_list = []
            
            # 只读前20页，防止遇到几百页的书爆内存
            max_pages = min(len(doc), 20)
            for i in range(max_pages):
                # sort=True 专门处理双栏排版
                page_text = doc[i].get_text("text", sort=True)
                if len(page_text) > 50:
                    text_list.append(page_text)
            
            raw_text = "\n".join(text_list)
            clean_text = self.cleaner.clean(raw_text)
            
            # ⚠️ 免费版数据库保险丝：截断到 50,000 字符
            MAX_CHARS = 50000
            if len(clean_text) > MAX_CHARS:
                clean_text = clean_text[:MAX_CHARS] + "...(truncated)"
                
            return clean_text
        except Exception:
            return None

    def update_neo4j(self, node_id, text):
        query = """
        MATCH (p:Paper)
        WHERE elementId(p) = $id
        SET p.full_text = $text
        """
        with self.driver.session() as session:
            session.run(query, id=node_id, text=text)

    def run(self):
        papers = self.get_target_papers()
        total = len(papers)
        print(f"📋 任务队列: {total} 篇待处理")
        
        if total == 0:
            print("🎉 所有全文已填充完毕！无需操作。")
            return

        success = 0
        
        # 进度条循环
        pbar = tqdm(papers, desc="🚀 Downloading")
        for paper in pbar:
            url = paper['url']
            node_id = paper['id']
            
            # 1. 严格提取 ID (跳过脏数据)
            arxiv_id = self.extract_strict_id(url)
            if not arxiv_id:
                continue

            # 2. 纯净下载 (不查API，不走梯子)
            pdf_bytes = self.download_pdf(arxiv_id)
            
            if pdf_bytes:
                # 3. 解析清洗
                final_text = self.process_pdf_bytes(pdf_bytes)
                
                # 4. 存入数据库 (确保内容有效)
                if final_text and len(final_text) > 200:
                    self.update_neo4j(node_id, final_text)
                    success += 1
                    pbar.set_postfix({"Saved": success})
            
            # 礼貌休眠 (防止镜像站封IP)
            time.sleep(random.uniform(0.5, 1.5))

        print(f"\n✅ 任务完成！成功导入: {success}/{total}")

if __name__ == "__main__":
    # 依赖检查: pip install neo4j requests pymupdf tqdm
    filler = ArxivFullTextFiller()
    try:
        filler.run()
    finally:
        filler.close()
