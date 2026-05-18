from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

from config import OLLAMA_MODEL

llm = ChatOllama(
    model=OLLAMA_MODEL,
    temperature=0
)

SUMMARY_PROMPT = ChatPromptTemplate.from_template("""
Сделай summary сообщества knowledge graph.

Entities:
{entities}

Relationships:
{relationships}
""")

chain = SUMMARY_PROMPT | llm

def summarize_communities(graph, communities):
    summaries = {}

    for cid, nodes in communities.items():
        relationships = []

        for node in nodes:
            for edge in graph.edges(node, data=True):
                relationships.append(edge)
        result = chain.invoke({
            "entities": nodes,
            "relationships": relationships[:50]
        })
        summaries[cid] = result.content

    return summaries
