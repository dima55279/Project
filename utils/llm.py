from langchain_ollama import ChatOllama

from config import (
    OLLAMA_MODEL,
    TEMPERATURE
)


llm = ChatOllama(
    model=OLLAMA_MODEL,
    temperature=TEMPERATURE,
    format="json"
)
