from langchain_core.prompts import ChatPromptTemplate
from utils.llm import llm
from utils.parsing import safe_json_parse

PROMPT = ChatPromptTemplate.from_template("""
Ты система извлечения knowledge graph. Извлеки сущности, отношения, юридические концепции и ссылки.

Верни строго JSON в формате:

{{
  "entities": [
    {{"id": "Короткий уникальный id", "type": "PERSON|ORGANIZATION|CONCEPT|LAW|...", "description": "..." }}
  ],
  "relationships": [
    {{"source": "id1", "target": "id2", "relation": "RELATION_TYPE", "description": "..." }}
  ]
}}

Текст для анализа:
{text}
""")

def extract_graph_batch(texts: list) -> dict:
    """Обрабатывает несколько документов за один вызов LLM"""
    combined_text = "\n\n---\n\n".join([f"DOC {i+1}:\n{t}" for i, t in enumerate(texts)])
    
    chain = PROMPT | llm
    result = chain.invoke({"text": combined_text})
    return safe_json_parse(result.content)