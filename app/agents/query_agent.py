from app.indexing.term_extractor import TermExtractor


class QueryAgent:

    def __init__(self):
        self.extractor = TermExtractor()

    def run(self, question):

        terms = self.extractor.extract(question)

        return {
            "question": question,
            "terms": terms
        }
