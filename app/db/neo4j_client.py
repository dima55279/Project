# app/db/neo4j_client.py
from neo4j import GraphDatabase
from typing import List, Dict, Any, Optional


class Neo4jClient:
    _instance = None

    def __new__(cls, uri="bolt://localhost:7687", user="neo4j", password="password"):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_driver(uri, user, password)
        return cls._instance

    def _init_driver(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(
            uri,
            auth=(user, password),
            max_connection_pool_size=30,
            max_connection_lifetime=3600,
            keep_alive=True
        )

    def close(self):
        if hasattr(self, 'driver') and self.driver:
            self.driver.close()

    def execute(self, query: str, params: Optional[dict] = None) -> List[Dict[str, Any]]:
        """Для READ-запросов (SELECT)"""
        params = params or {}
        with self.driver.session() as session:
            result = session.run(query, params)
            return [dict(record) for record in result]

    def execute_write(self, query: str, params: Optional[dict] = None) -> List[Dict[str, Any]]:
        """Для WRITE-запросов (MERGE, CREATE, DELETE, SET)"""
        params = params or {}
        with self.driver.session() as session:
            # Выполняем внутри транзакции и сразу материализуем результат
            def tx_func(tx):
                result = tx.run(query, params)
                return [dict(record) for record in result]

            return session.execute_write(tx_func)