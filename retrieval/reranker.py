from langchain_community.cross_encoders import (
    HuggingFaceCrossEncoder
)

from langchain.retrievers.document_compressors import (
    CrossEncoderReranker
)

from langchain.retrievers import (
    ContextualCompressionRetriever
)

from retrieval.hybrid_retriever import (
    ensemble_retriever
)

from config import (
    RERANK_MODEL,
    RERANK_TOP_K
)

cross_encoder = HuggingFaceCrossEncoder(
    model_name=RERANK_MODEL
)

compressor = CrossEncoderReranker(
    model=cross_encoder,
    top_n=RERANK_TOP_K
)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=ensemble_retriever
)

def rerank_chunks(query):
    return compression_retriever.invoke(query)
