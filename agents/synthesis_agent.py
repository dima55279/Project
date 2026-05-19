from langchain_core.prompts import ChatPromptTemplate

from utils.llm import llm


PROMPT = ChatPromptTemplate.from_template("""
Ответь на вопрос.

QUESTION:
{question}

ENTITIES:
{entities}

GRAPH RELATIONS:
{relations}

EVIDENCE:
{evidence}
""")



def synthesize_answer(
    question,
    entities,
    relations,
    evidence
):

    chain = PROMPT | llm

    result = chain.invoke({
        "question": question,
        "entities": str(entities),
        "relations": str(relations),
        "evidence": str(evidence)
    })

    return result.content
