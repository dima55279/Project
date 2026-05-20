from langchain_ollama import ChatOllama

from config import (
    OLLAMA_MODEL,
    TEMPERATURE
)


llm = ChatOllama(
    model=OLLAMA_MODEL,
    temperature=0.1,      # немного креативности, но не сильно
    format="json",
    timeout=120,
    max_tokens=4096
)
