from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DOCS_DIR = BASE_DIR / "documents"
OUTPUT_DIR = BASE_DIR / "outputs"
GRAPH_EXPORT_DIR = BASE_DIR / "storage"
COMMUNITIES_DIR = GRAPH_EXPORT_DIR / "communities"
RAW_EXPORT_DIR = GRAPH_EXPORT_DIR / "raw"
DEBUG_DIR = GRAPH_EXPORT_DIR / "debug_extraction"

# NEO4J
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"

# MODELS
OLLAMA_MODEL = "mistral"   # Рекомендую: llama3.1:8b или phi3:medium
TEMPERATURE = 0

# OPTIMIZATION
BATCH_SIZE = 8                    # документов на один LLM-запрос
NUM_WORKERS = 8                   # потоков для загрузки и обработки