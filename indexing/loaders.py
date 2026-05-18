import os

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader
)

def get_loader(path):
    lower = path.lower()
    if lower.endswith(".pdf"):
        return PyPDFLoader(path)
    if lower.endswith(".txt"):
        return TextLoader(path, encoding="utf-8")
    if lower.endswith(".md"):
        return TextLoader(path, encoding="utf-8")
    if lower.endswith(".docx"):
        return Docx2txtLoader(path)
    return None


def load_documents(folder):
    docs = []
    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)
        loader = get_loader(path)
        if loader is None:
            continue
        try:
            loaded = loader.load()
            for d in loaded:
                d.metadata["source"] = filename
            docs.extend(loaded)
        except Exception as e:
            print(f"ERROR {filename}: {e}")
    return docs
