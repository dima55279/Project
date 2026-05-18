from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP
)


def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(docs)
    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = idx
    return chunks
