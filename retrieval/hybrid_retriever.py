from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

from indexing.embeddings import load_embeddings
from config import (
    FAISS_DIR,
    TOP_K
)

embeddings = load_embeddings()

vectorstore = FAISS.load_local(
    FAISS_DIR,
    embeddings,
    allow_dangerous_deserialization=True
)

documents = list(
    vectorstore.docstore._dict.values()
)

dense_retriever = vectorstore.as_retriever(
    search_kwargs={"k": TOP_K}
)

bm25 = BM25Retriever.from_documents(documents)
bm25.k = TOP_K

retriever = EnsembleRetriever(
    retrievers=[dense_retriever, bm25],
    weights=[0.5, 0.5]
)

def retrieve_chunks(query):
    return retriever.invoke(query)
