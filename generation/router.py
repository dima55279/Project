from langchain_ollama import ChatOllama

from generation.prompts import ROUTER_PROMPT
from config import OLLAMA_MODEL

llm = ChatOllama(
    model=OLLAMA_MODEL,
    temperature=0
)

chain = ROUTER_PROMPT | llm

def route_query(question):
    result = chain.invoke({
        "question": question
    })
    text = result.content.upper()
    if "GLOBAL" in text:
        return "global"

    return "local"
