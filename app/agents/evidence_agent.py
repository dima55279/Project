class EvidenceAgent:

    def rank(self,
             graph_results,
             bm25_results):

        merged = []

        for item in graph_results:
            merged.append({
                "score": 1.0,
                "data": item
            })

        for score, item in bm25_results:
            merged.append({
                "score": float(score),
                "data": item
            })

        merged = sorted(
            merged,
            key=lambda x: x["score"],
            reverse=True
        )

        return merged[:15]
