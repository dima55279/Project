from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import json
import argparse
from tqdm import tqdm

from indexing.bm25_index import BM25Indexer
from indexing.embedding_index import EmbeddingIndexer
from indexing.hybrid_retriever import HybridRetriever

from llm.prompts import SYSTEM_PROMPT
from llm.ollama_client import OllamaClient

from graph.neo4j_loader import Neo4jLoader

from agent.planner import LegalPlanner
from agent.tools import RetrievalTools
from agent.legal_agent import LegalAgent


MAX_WORKERS = 8
OUTPUT_FILE = "outputs/results.csv"


parser = argparse.ArgumentParser()

parser.add_argument(
    "--question",
    type=str
)

parser.add_argument(
    "--csv",
    type=str
)

args = parser.parse_args()


with open(
    "indexes/documents.json",
    encoding="utf-8"
) as f:

    documents = json.load(f)


bm25 = BM25Indexer.load(
    "indexes/bm25.pkl"
)

embedding = EmbeddingIndexer()

embedding.load(
    "indexes/faiss.index",
    documents
)


neo4j = Neo4jLoader(
    "bolt://localhost:7687",
    "neo4j",
    "password"
)


retriever = HybridRetriever(
    bm25,
    embedding,
    neo4j.driver
)

planner = LegalPlanner()
tools = RetrievalTools(retriever)
agent = LegalAgent(planner, tools)

llm = OllamaClient(
    model="mistral"
)


questions = []

if args.question:
    questions.append(args.question)

elif args.csv:

    df = pd.read_csv(args.csv)

    questions = df[
        "question"
    ].tolist()

else:
    raise ValueError(
        "Provide --question or --csv"
    )



def build_context(results):

    context = []
    docs = set()

    for item in results["primary"]:

        context.append(
            f"""
LAW: {item['law']}
ARTICLE: {item['article_id']}
TEXT:
{item['text']}
"""
        )

        docs.add(item["law"])

    return {
        "context": "\n".join(context),
        "documents": list(docs)
    }



def process_question(question):

    retrieval = agent.run(question)

    context_data = build_context(
        retrieval
    )

    if not context_data["documents"]:

        return {
            "question": question,
            "answer": "Недостаточно данных в нормативной базе",
            "document": []
        }

    prompt = f"""
КОНТЕКСТ:
{context_data['context']}

ВОПРОС:
{question}

Ответь строго по контексту.
"""

    answer = llm.generate(
        SYSTEM_PROMPT,
        prompt
    )

    return {
        "question": question,
        "answer": answer,
        "document": context_data["documents"]
    }


outputs = []

with ThreadPoolExecutor(
    max_workers=MAX_WORKERS
) as executor:

    results = executor.map(
        process_question,
        questions
    )

    for r in tqdm(
        results,
        total=len(questions),
        desc="Processing questions"
    ):

        outputs.append(r)


output_df = pd.DataFrame(outputs)

output_df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

print(output_df)
