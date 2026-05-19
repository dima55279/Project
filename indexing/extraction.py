from langchain_core.prompts import ChatPromptTemplate
from utils.llm import llm
from utils.parsing import safe_json_parse

PROMPT = ChatPromptTemplate.from_template("""
Ты эксперт по извлечению knowledge graph из нормативно-правовых и технических документов.

Извлеки важные сущности и отношения из предоставленного раздела.

Верни строго валидный JSON:
{{
  "entities": [{{"id": "...", "type": "...", "description": "..."}}],
  "relationships": [{{"source": "...", "target": "...", "relation": "...", "description": "..."}}]
}}

Раздел:
{text}
""")

def extract_graph_batch(texts: list) -> dict:
    """Обрабатывает несколько документов за один вызов LLM"""
    combined_text = "\n\n---\n\n".join([f"DOC {i+1}:\n{t}" for i, t in enumerate(texts)])
    
    chain = PROMPT | llm
    result = chain.invoke({"text": combined_text})
    return safe_json_parse(result.content)