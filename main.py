from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import ollama
import argparse
import pandas as pd

from tqdm import tqdm

from parser.markdown_parser import MarkdownLawParser
from parser.legal_reference_extractor import LegalReferenceExtractor
from graph.neo4j_loader import Neo4jLoader

from indexing.bm25_index import BM25Indexer
from indexing.embedding_index import EmbeddingIndexer
from indexing.hybrid_retriever import HybridRetriever

from agent.planner import LegalPlanner
from agent.tools import RetrievalTools
from agent.legal_agent import LegalAgent

from llm.grounded_answer import GroundedAnswerBuilder
from llm.prompts import SYSTEM_PROMPT


# =====================================================
# CONFIG
# =====================================================

DATA_DIR = "data/laws_md"
OLLAMA_MODEL = "mistral"
MAX_WORKERS = 8
OUTPUT_FILE = "outputs/results.csv"


# =====================================================
# ARGUMENTS
# =====================================================

arg_parser = argparse.ArgumentParser()

arg_parser.add_argument(
    "--question",
    type=str,
    help="Single legal question"
)

arg_parser.add_argument(
    "--csv",
    type=str,
    help="CSV file with questions"
)

args = arg_parser.parse_args()


# =====================================================
# INIT
# =====================================================

parser = MarkdownLawParser()
extractor = LegalReferenceExtractor()

neo4j = Neo4jLoader(
    "bolt://localhost:7687",
    "neo4j",
    "password"
)


# =====================================================
# PARSE SINGLE LAW
# =====================================================


def process_law(filepath):

    law = parser.parse(filepath)

    law_name = Path(filepath).stem

    articles_texts = []

    for article in law["articles"]:

        article_text = article["text"]

        neo4j.create_article(
            law_name=law_name,
            article_id=article["id"],
            text=article_text
        )

        refs = extractor.extract(article_text)

        for ref in refs:
            neo4j.create_reference(
                source_law=law_name,
                source_article=article["id"],
                target_article=ref
            )

        articles_texts.append({
            "law": law_name,
            "article_id": article["id"],
            "text": article_text
        })

    return articles_texts


# =====================================================
# MULTITHREADED DOCUMENT PROCESSING
# =====================================================

law_files = list(Path(DATA_DIR).glob("*.md"))

all_articles = []

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:

    results = executor.map(
        process_law,
        law_files
    )

    for r in tqdm(
        results,
        total=len(law_files),
        desc="Processing laws",
        colour="green"
    ):
        all_articles.extend(r)


# =====================================================
# BUILD INDEXES
# =====================================================

texts = [x["text"] for x in all_articles]

bm25 = BM25Indexer(texts)

embedding = EmbeddingIndexer()

print("Building embedding index...")
embedding.build(texts)
print("Embedding index completed")


# =====================================================
# HYBRID RETRIEVER
# =====================================================

retriever = HybridRetriever(
    bm25=bm25,
    embedding=embedding,
    neo4j_driver=neo4j.driver
)


# =====================================================
# AGENT
# =====================================================

planner = LegalPlanner()
tools = RetrievalTools(retriever)
agent = LegalAgent(planner, tools)

builder = GroundedAnswerBuilder()


# =====================================================
# OLLAMA + MISTRAL
# =====================================================


def generate_answer(question, context):

    prompt = f"""
{SYSTEM_PROMPT}

КОНТЕКСТ:
{context}

ВОПРОС:
{question}

Верни ответ строго в формате:
question,answer,document
"""

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]


# =====================================================
# SINGLE QUESTION PIPELINE
# =====================================================


def process_question(question):

    retrieval_results = agent.run(question)

    context_data = builder.build_context(retrieval_results)

    answer = generate_answer(
        question,
        context_data["context"]
    )

    final_output = {
        "question": question,
        "answer": answer,
        "document": context_data["documents"]
    }

    return final_output


# =====================================================
# LOAD QUESTIONS
# =====================================================

questions = []

# CLI mode
if args.question:
    questions.append(args.question)

# CSV mode
elif args.csv:

    df = pd.read_csv(args.csv)

    if "question" not in df.columns:
        raise ValueError(
            "CSV must contain 'question' column"
        )

    questions = df["question"].tolist()

else:
    raise ValueError(
        "Provide --question or --csv"
    )


# =====================================================
# MULTITHREADED QUESTION PROCESSING
# =====================================================

outputs = []

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:

    results = executor.map(
        process_question,
        questions
    )

    for r in tqdm(
        results,
        total=len(questions),
        desc="Processing questions",
        colour="blue"
    ):
        outputs.append(r)


# =====================================================
# SAVE CSV OUTPUT
# =====================================================

output_df = pd.DataFrame(outputs)

output_df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

print(f"Saved results to {OUTPUT_FILE}")


# =====================================================
# PRINT RESULTS
# =====================================================

for item in outputs:

    print(json.dumps(
        item,
        ensure_ascii=False,
        indent=2
    ))
