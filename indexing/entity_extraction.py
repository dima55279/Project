import json
from tqdm import tqdm

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

from config import OLLAMA_MODEL


EXTRACTION_PROMPT = ChatPromptTemplate.from_template("""
Ты information extraction system.

Извлеки:
- entities
- relationships

Верни JSON.

FORMAT:

{
  "entities": [
    {
      "id": "...",
      "type": "...",
      "description": "..."
    }
  ],
  "relationships": [
    {
      "source": "...",
      "target": "...",
      "relation": "...",
      "description": "..."
    }
  ]
}

TEXT:
{text}
""")

llm = ChatOllama(
    model=OLLAMA_MODEL,
    temperature=0,
    format="json"
)

chain = EXTRACTION_PROMPT | llm

def extract_from_chunk(chunk):
    try:
        result = chain.invoke({
            "text": chunk.page_content
        })
        data = json.loads(result.content)
        return {
            "chunk_id": chunk.metadata["chunk_id"],
            "source": chunk.metadata.get("source"),
            "entities": data.get("entities", []),
            "relationships": data.get("relationships", [])
        }

    except Exception as e:
        print("EXTRACTION ERROR", e)
        return {
            "chunk_id": chunk.metadata["chunk_id"],
            "entities": [],
            "relationships": []
        }


def extract_graph_data(chunks):
    results = []
    for chunk in tqdm(chunks, desc="Extracting graph"):
        results.append(extract_from_chunk(chunk))
    return results
