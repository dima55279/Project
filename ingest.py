from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
from tqdm import tqdm

from parser.markdown_parser import MarkdownLawParser
from parser.legal_reference_extractor import LegalReferenceExtractor

from indexing.chunker import LegalChunker
from indexing.bm25_index import BM25Indexer
from indexing.embedding_index import EmbeddingIndexer

from graph.neo4j_loader import Neo4jLoader


DATA_DIR = "data/laws_md"
MAX_WORKERS = 8


parser = MarkdownLawParser()
extractor = LegalReferenceExtractor()
chunker = LegalChunker()

neo4j = Neo4jLoader(
    "bolt://localhost:7687",
    "neo4j",
    "password"
)

neo4j.clear_database()


def process_law(filepath):

    law = parser.parse(filepath)

    law_name = Path(filepath).stem

    chunks = []

    for article in law["articles"]:

        neo4j.create_article(
            law_name,
            article["article_id"],
            article["text"]
        )

        refs = extractor.extract(
            article["text"]
        )

        for ref in refs:
            neo4j.create_reference(
                law_name,
                article["article_id"],
                ref
            )

        article_chunks = chunker.chunk_article(
            law_name,
            article["article_id"],
            article["text"]
        )

        chunks.extend(article_chunks)

    return chunks


law_files = list(
    Path(DATA_DIR).glob("*.md")
)

all_documents = []

with ThreadPoolExecutor(
    max_workers=MAX_WORKERS
) as executor:

    results = executor.map(
        process_law,
        law_files
    )

    for r in tqdm(
        results,
        total=len(law_files),
        desc="Processing laws"
    ):

        all_documents.extend(r)


with open(
    "indexes/documents.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        all_documents,
        f,
        ensure_ascii=False,
        indent=2
    )


bm25 = BM25Indexer(
    all_documents
)

bm25.save(
    "indexes/bm25.pkl"
)


embedding = EmbeddingIndexer()

embedding.build(
    all_documents
)

embedding.save(
    "indexes/faiss.index"
)

print("Ingestion completed")
