# app/db/neo4j_client.py
from neo4j import GraphDatabase
from typing import List, Dict, Any


class Neo4jClient:
    _instance = None

    def __new__(cls, uri="bolt://localhost:7687", user="neo4j", password="password"):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_driver(uri, user, password)
        return cls._instance

    def _init_driver(self, uri, user, password):
        self.driver = GraphDatabase.driver(
            uri,
            auth=(user, password),
            max_connection_pool_size=50,
            max_connection_lifetime=3600,
            keep_alive=True
        )

    def close(self):
        if hasattr(self, 'driver') and self.driver:
            self.driver.close()

    def execute(self, query: str, params: dict = None) -> List[Dict[str, Any]]:
        """Чтение (SELECT)"""
        params = params or {}
        with self.driver.session() as session:
            result = session.run(query, params)
            return [dict(record) for record in result]   # полностью вычитываем

    def execute_write(self, query: str, params: dict = None) -> List[Dict[str, Any]]:
        """Запись (MERGE, CREATE, DELETE)"""
        params = params or {}
        with self.driver.session() as session:
            result = session.execute_write(
                lambda tx: tx.run(query, params)
            )
            # Важно: вычитываем результат внутри транзакции
            return [dict(record) for record in result]