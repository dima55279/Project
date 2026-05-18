import json
import pickle

from config import *

from indexing.loaders import load_documents
from indexing.splitter import split_documents
from indexing.embeddings import load_embeddings
from indexing.vector_index import build_vectorstore
from indexing.entity_extraction import extract_graph_data
from indexing.graph_builder import build_graph
from indexing.community_detection import detect_communities
from indexing.summarization import summarize_communities


print("Loading docs...")
docs = load_documents(DOCS_DIR)

print("Splitting...")
chunks = split_documents(docs)

print("Embeddings...")
embeddings = load_embeddings()

print("Building vector store...")
vectorstore = build_vectorstore(
    chunks,
    embeddings
)

vectorstore.save_local(FAISS_DIR)

print("Extracting graph...")
extractions = extract_graph_data(chunks)

print("Building graph...")
graph = build_graph(extractions)

print("Community detection...")
communities = detect_communities(graph)

print("Summarization...")
summaries = summarize_communities(
    graph,
    communities
)

print("Saving graph...")

with open(GRAPH_PATH, "wb") as f:
    pickle.dump(graph, f)

with open(COMMUNITIES_PATH, "w", encoding="utf-8") as f:
    json.dump(communities, f, ensure_ascii=False, indent=2)

with open(SUMMARIES_PATH, "w", encoding="utf-8") as f:
    json.dump(summaries, f, ensure_ascii=False, indent=2)

print("DONE")
