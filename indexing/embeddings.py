from langchain_huggingface import HuggingFaceEmbeddings
from config import EMBED_MODEL

def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
