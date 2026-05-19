from langchain_core.prompts import ChatPromptTemplate

from utils.llm import llm


PROMPT = ChatPromptTemplate.from_template("""
Сделай summary сообщества knowledge graph.

ENTITIES:
{entities}
""")



def summarize_community(entities):

    chain = PROMPT | llm

    result = chain.invoke({
        "entities": ", ".join(entities)
    })

    return result.content
