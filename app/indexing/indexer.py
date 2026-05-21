# app/indexing/indexer.py
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tqdm import tqdm

from app.indexing.parser import MarkdownParser
from app.indexing.graph_builder import GraphBuilder
from app.indexing.term_extractor import TermExtractor
from app.indexing.bm25_index import BM25Indexer
from app.db.neo4j_client import Neo4jClient


class Indexer:
    def __init__(self):
        self.parser = MarkdownParser()
        self.graph = GraphBuilder()
        self.extractor = TermExtractor()
        self.bm25 = BM25Indexer()

    def clear_database(self):
        """Полная очистка Neo4j перед новой индексацией"""
        db = Neo4jClient()
        print("🧹 Очистка базы Neo4j...")
        db.execute_write("""
            MATCH (n)
            DETACH DELETE n
        """)
        print("✅ База Neo4j очищена.")

    def process_file(self, path):

        document = self.parser.parse(path)

        self.graph.create_document(document)

        for section in document["sections"]:
            for article in section["articles"]:

                terms = self.extractor.extract(article["text"])

                self.bm25.add_document(
                    article["text"],
                    {
                        "document": document["title"],
                        "article": article["number"],
                        "terms": terms,
                        "text": article["text"]
                    }
                )

    def build(self,
              docs_dir="data/md"):

        files = list(Path(docs_dir).glob("*.md"))

        with ThreadPoolExecutor(max_workers=8) as executor:
            list(tqdm(
                executor.map(self.process_file, files),
                total=len(files),
                desc="INDEXING"
            ))

        self.bm25.build()
        self.bm25.save()
