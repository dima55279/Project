from langchain_core.prompts import ChatPromptTemplate

from utils.llm import llm


PROMPT = ChatPromptTemplate.from_template("""
Выбери стратегию:

- local_graph_search
- global_community_search
- path_reasoning

QUESTION:
{question}
""")



def choose_strategy(question):

    chain = PROMPT | llm

    result = chain.invoke({
        "question": question
    })

    text = result.content.lower()

    if "global" in text:
        return "global_community_search"

    if "path" in text:
        return "path_reasoning"

    return "local_graph_search"
