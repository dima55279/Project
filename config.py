from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DOCS_DIR = BASE_DIR / "documents"

OUTPUT_DIR = BASE_DIR / "outputs"

GRAPH_EXPORT_DIR = BASE_DIR / "storage"

COMMUNITIES_DIR = GRAPH_EXPORT_DIR / "communities"

RAW_EXPORT_DIR = GRAPH_EXPORT_DIR / "raw"


# NEO4J

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"


# MODELS

OLLAMA_MODEL = "mistral"

TEMPERATURE = 0

MAX_DEPTH = 3

