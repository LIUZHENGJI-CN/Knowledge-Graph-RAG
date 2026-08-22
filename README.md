
## 1. 当前项目结构

```text
复现知识图谱/
├── data/                  # 知识图谱构造、数据补全、向量写入
├── core/                  # 核心问答层：Neo4j、Embedding、RAG、Text2Cypher、前端
├── 测试/                  # 零散测试脚本
└── neo4j实例和管理员凭据/  # 本地或云端 Neo4j 相关截图/说明文件
```

当前主要代码集中在 `data` 和 `core` 两个文件夹。

## 2. data：知识图谱构造层

`data` 文件夹负责把论文数据采集、增强、清洗后写入 Neo4j，并为语义检索准备向量。

### 2.1 基础数据与映射文件

- `category.json`：arXiv `q-bio` 分类代码到学科名称的映射。
- `institution_map.json`：邮箱域名到机构名称的映射。
- `full_paper_knowledge_base.json`：本地保存的论文知识库 JSON，包含论文元数据、作者、分类、机构、venue、语义抽取结果等。
  
### 2.2 图谱初始化与论文采集

- `getinstitutionmap.py`
  - 从 GitHub 开源大学域名数据中扩充机构域名映射，生成或更新 `institution_map.json`。

- `getgerneraldata.py`
  - 根据 `q-bio` 分类从 arXiv 抓取论文。
  - 调用 LLM 从标题和摘要中抽取 `background`、`methods`、`keywords`。
  - 在 Neo4j 中创建或合并 `Paper`、`Author`、`Keyword`。
  - 建立 `(Paper)-[:AUTHORED_BY]->(Author)` 和 `(Paper)-[:HAS_KEYWORD]->(Keyword)`。
  - 同步生成 `full_paper_knowledge_base.json`。

### 2.3 摘要与全文补全

- `import_abstracts.py`
  - 扫描 Neo4j 中有 arXiv `entry_id` 但缺少 `abstract` 的论文。
  - 通过 arXiv API 补充摘要。

- `arxiv_content_filler.py`
  - 根据 arXiv ID 下载 PDF。
  - 使用 PyMuPDF 提取正文文本。
  - 清洗参考文献、页眉、换行和连字符。
  - 将清洗后的全文写入 `Paper.full_text`。

### 2.4 分类、机构与发表渠道增强

- `add_category.py`
  - 从 `full_paper_knowledge_base.json` 读取论文分类。
  - 创建 `Category` 节点。
  - 建立 `(Paper)-[:BELONGS_TO]->(Category)`。

- `add_insti_and_venue.py`
  - 基于 DOI、Crossref、OpenAlex、PDF 首页邮箱域名等信息补充机构和发表渠道。
  - 创建 `Institution`、`Journal` 或 `Conference` 节点。
  - 建立 `(Paper)-[:AFFILIATED_WITH]->(Institution)` 和 `(Paper)-[:PUBLISHED_IN]->(Journal/Conference)`。

- `getinstitution.py`
  - 对仍缺少机构信息的论文，下载 PDF 首页文本。
  - 调用 OpenAI Responses API 抽取作者机构。
  - 将机构信息同步到 JSON 和 Neo4j。

### 2.5 向量化

- `build_embeddings.py`
  - 从 Neo4j 读取论文标题、摘要、背景、方法和关键词。
  - 使用 `BAAI/bge-m3` 生成语义向量。
  - 将向量写入 `Paper.embedding`。
  - 后续 `core/vector_search.py` 会通过 Neo4j 向量索引 `paper_embeddings` 检索这些向量。

## 3. core：核心问答与前端层

`core` 文件夹负责把已经构建好的 Neo4j 知识图谱包装成可交互问答系统。

### 3.1 配置与连接

- `config.py`
  - 从 `.env` 读取 `LLM_API_KEY`、`LLM_BASE_URL`、`LLM_MODEL_NAME`。
  - 当前只集中管理 LLM 配置。

- `neo4j_driver.py`
  - 管理 Neo4j 连接。
  - 提供 `query()`、`execute()` 和 `health_check()`。

- `LLM_client.py`
  - 封装 OpenAI-compatible Chat 接口。
  - 提供 `chat()`。
  - 提供基于 LLM 的简易 `rerank()`，用于 RAG 结果重排序。

- `embedding_model.py`
  - 加载 `BAAI/bge-m3`。
  - 提供 `embed_query()` 和 `embed_documents()`。
  - 自动选择 CUDA、MPS 或 CPU。

### 3.2 检索与问答

- `vector_search.py`
  - 将用户问题转成向量。
  - 调用 Neo4j 向量索引 `paper_embeddings`。
  - 返回相关论文的标题、摘要、背景、方法、关键词和相似度分数。

- `fast_rag.py`
  - 面向论文内容类问题。
  - 流程：向量召回、LLM rerank、拼接论文上下文、生成中文回答。

- `text2cypher.py`
  - 面向图谱结构类问题，例如找作者、找机构、统计论文、查合作关系。
  - 将自然语言问题转换成只读 Cypher。
  - 内置当前图谱 schema，执行查询后返回 Cypher 和查询结果。

- `router.py`
  - 根据关键词判断问题类型。
  - 路由到 `fast_rag`、`text2cypher` 或 `hybrid`。
  - `hybrid` 用于同时包含结构查询和语义总结的问题。

### 3.3 前端

- `frontend.py`
  - Streamlit 聊天式前端。
  - 调用 `router.answer_question()`。
  - 展示最终回答、查询方式、Fast RAG 参考论文、Text2Cypher 生成的 Cypher 和 Neo4j 查询结果。

启动方式：

```bash
cd core
streamlit run frontend.py
```

## 4. 当前图谱 schema

### 节点

- `Paper`
  - `entry_id`
  - `title`
  - `abstract`
  - `background`
  - `proposed_method`
  - `published`
  - `doi`
  - `full_text`
  - `embedding`

- `Author`
  - `name`

- `Keyword`
  - `name`

- `Category`
  - `code`
  - `name`

- `Institution`
  - `name`

- `Journal`
  - `name`

- `Conference`
  - `name`

### 关系

- `(Paper)-[:AUTHORED_BY]->(Author)`
- `(Paper)-[:HAS_KEYWORD]->(Keyword)`
- `(Paper)-[:BELONGS_TO]->(Category)`
- `(Paper)-[:AFFILIATED_WITH]->(Institution)`
- `(Paper)-[:PUBLISHED_IN]->(Journal)`
- `(Paper)-[:PUBLISHED_IN]->(Conference)`

## 5. 推荐运行顺序

首次复现时建议按下面顺序执行：

```bash
# 1. 扩充机构域名映射
python data/getinstitutionmap.py

# 2. 抓取论文并创建基础图谱
python data/getgerneraldata.py

# 3. 补充摘要
python data/import_abstracts.py

# 4. 补充全文
python data/arxiv_content_filler.py

# 5. 补充分学科分类
python data/add_category.py

# 6. 补充机构和发表渠道
python data/add_insti_and_venue.py

# 7. 对剩余缺机构论文使用 LLM 补全
python data/getinstitution.py

# 8. 写入论文向量
python data/build_embeddings.py

# 9. 启动核心问答前端
cd core
streamlit run frontend.py
```

实际运行时可以根据已有数据跳过部分步骤。例如 Neo4j 中已经有论文节点时，可以只补分类、机构或向量。


## 6. Neo4j 向量索引

`core/vector_search.py` 默认调用名为 `paper_embeddings` 的 Neo4j 向量索引。

如果数据库中还没有索引，需要在 Neo4j Browser 中执行类似语句：

```cypher
CREATE VECTOR INDEX paper_embeddings IF NOT EXISTS
FOR (p:Paper)
ON (p.embedding)
OPTIONS {
  indexConfig: {
    `vector.dimensions`: 1024,
    `vector.similarity_function`: 'cosine'
  }
};
```

`BAAI/bge-m3` 的向量维度通常是 1024。

