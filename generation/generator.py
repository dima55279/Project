from langchain_ollama import ChatOllama

from generation.prompts import GENERATION_PROMPT
from config import OLLAMA_MODEL

llm = ChatOllama(
    model=OLLAMA_MODEL,
    temperature=0,
    num_predict=512
)

chain = GENERATION_PROMPT | llm

def generate_answer(question, context):
    result = chain.invoke({
        "question": question,
        "context": context
    })

    return result.content
