from langchain_ollama import ChatOllama

from config import (
    OLLAMA_MODEL,
    TEMPERATURE
)


llm = ChatOllama(
    model=OLLAMA_MODEL,
    temperature=0,
    timeout=120,
    max_tokens=4096
)
