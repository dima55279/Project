DOCS_DIR = "./data/docs"
QUESTIONS_FILE = "./data/questions/questions_500.csv"

FAISS_DIR = "./storage/faiss_store"
GRAPH_PATH = "./storage/graph.gpickle"
ENTITIES_PATH = "./storage/entities.json"
RELATIONS_PATH = "./storage/relationships.json"
COMMUNITIES_PATH = "./storage/communities.json"
SUMMARIES_PATH = "./storage/summaries.json"

RESULTS_DIR = "./results"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

TOP_K = 20
RERANK_TOP_K = 5
GRAPH_HOPS = 2

EMBED_MODEL = "intfloat/multilingual-e5-base"
RERANK_MODEL = "BAAI/bge-reranker-base"
OLLAMA_MODEL = "mistral"

MAX_WORKERS = 10
