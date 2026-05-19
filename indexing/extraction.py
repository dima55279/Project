from langchain_core.prompts import ChatPromptTemplate

from utils.llm import llm

from utils.parsing import safe_json_parse


PROMPT = ChatPromptTemplate.from_template("""
Ты система извлечения knowledge graph.

Извлеки:

1. entities
2. relationships
3. legal concepts
4. references

Верни JSON.

FORMAT:

{{
  "entities": [
    {{
      "id": "",
      "type": "",
      "description": ""
    }}
  ],
  "relationships": [
    {{
      "source": "",
      "target": "",
      "relation": "",
      "description": ""
    }}
  ]
}}

TEXT:
{text}
""")



def extract_graph(text):

    chain = PROMPT | llm

    result = chain.invoke({
        "text": text
    })

    return safe_json_parse(result.content)
