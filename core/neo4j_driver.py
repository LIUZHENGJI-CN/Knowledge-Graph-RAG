import sys
from typing import Any, Dict, List, Optional

from neo4j import GraphDatabase
from neo4j.exceptions import AuthError, Neo4jError, ServiceUnavailable


# =========================
# 直接在这里填写 Neo4j 信息
# =========================
NEO4J_URI = "neo4j://localhost:7687"
NEO4J_AUTH = ("neo4j", "your_neo4j_password_here")

# 如果你没有单独数据库名，就保持 None
NEO4J_DATABASE = None


class Neo4jDriver:
    """
    Neo4j 数据库连接管理器。

    后面其他模块统一这样用：
        from core.neo4j_driver import db
        results = db.query(cypher, params)
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Neo4jDriver, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.uri = NEO4J_URI
        self.auth = NEO4J_AUTH
        self.database = NEO4J_DATABASE
        self.driver = None

        self._connect()
        self._initialized = True

    def _connect(self):
        print(f"[Core] 正在连接 Neo4j: {self.uri} ...")

        try:
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=self.auth,
                max_connection_lifetime=3600,
                connection_timeout=30,
            )
            self.driver.verify_connectivity()
            print("[Core] Neo4j 连接成功")

        except AuthError as e:
            print(f"[Core] Neo4j 认证失败，请检查用户名或密码: {e}")
            sys.exit(1)

        except ServiceUnavailable as e:
            print(f"[Core] Neo4j 服务不可用，请检查数据库是否启动或网络是否可达: {e}")
            sys.exit(1)

        except Exception as e:
            print(f"[Core] Neo4j 连接失败: {e}")
            sys.exit(1)

    def _session_kwargs(self):
        if self.database:
            return {"database": self.database}
        return {}

    def query(
        self,
        cypher_query: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        执行查询语句，主要用于 MATCH ... RETURN。
        """
        if parameters is None:
            parameters = {}

        try:
            with self.driver.session(**self._session_kwargs()) as session:
                result = session.run(cypher_query, parameters)
                return [dict(record) for record in result]

        except Neo4jError as e:
            print(f"[Query Error] Cypher 执行出错: {e}")
            print(f"[Query] {cypher_query}")
            print(f"[Parameters] {parameters}")
            return []

        except Exception as e:
            print(f"[Query Error] 未知错误: {e}")
            print(f"[Query] {cypher_query}")
            print(f"[Parameters] {parameters}")
            return []

    def execute(
        self,
        cypher_query: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        执行写入语句，主要用于 CREATE / MERGE / SET / DELETE。
        """
        if parameters is None:
            parameters = {}

        try:
            with self.driver.session(**self._session_kwargs()) as session:
                result = session.run(cypher_query, parameters)
                summary = result.consume()
                counters = summary.counters

                return {
                    "nodes_created": counters.nodes_created,
                    "nodes_deleted": counters.nodes_deleted,
                    "relationships_created": counters.relationships_created,
                    "relationships_deleted": counters.relationships_deleted,
                    "properties_set": counters.properties_set,
                    "labels_added": counters.labels_added,
                    "labels_removed": counters.labels_removed,
                }

        except Neo4jError as e:
            print(f"[Execute Error] Cypher 执行出错: {e}")
            print(f"[Query] {cypher_query}")
            print(f"[Parameters] {parameters}")
            return {}

        except Exception as e:
            print(f"[Execute Error] 未知错误: {e}")
            print(f"[Query] {cypher_query}")
            print(f"[Parameters] {parameters}")
            return {}

    def health_check(self) -> bool:
        """
        测试 Neo4j 是否连接正常。
        """
        result = self.query("RETURN 1 AS ok")
        return bool(result and result[0].get("ok") == 1)

    def close(self):
        """
        关闭数据库连接。
        """
        if self.driver:
            self.driver.close()
            self.driver = None
            print("[Core] Neo4j 连接已关闭")


# 全局单例，后面模块直接 import 它
db = Neo4jDriver()


if __name__ == "__main__":
    if db.health_check():
        print("[Test] Neo4j 连接测试成功")
        print(db.query("RETURN 'hello neo4j' AS message"))
    else:
        print("[Test] Neo4j 连接测试失败")

    db.close()
