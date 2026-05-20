class RetrievalTools:

    def __init__(self,
                 retriever):

        self.retriever = retriever

    def hybrid_retrieve(self,
                        question):

        return self.retriever.retrieve(
            question
        )
