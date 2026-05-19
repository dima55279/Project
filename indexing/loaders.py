from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader
)



def get_loader(path):

    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return PyPDFLoader(str(path))

    if suffix == ".txt":
        return TextLoader(
            str(path),
            encoding="utf-8"
        )

    if suffix == ".docx":
        return Docx2txtLoader(str(path))

    return None



def load_documents(folder):

    folder = Path(folder)

    docs = []

    for file in folder.iterdir():

        loader = get_loader(file)

        if loader is None:
            continue

        loaded = loader.load()

        for d in loaded:

            d.metadata["source"] = file.name
            d.metadata["filepath"] = str(file)

        docs.extend(loaded)

    return docs
