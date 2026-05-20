class LegalChunker:

    def chunk_article(self,
                      law_name,
                      article_id,
                      text,
                      chunk_size=1200):

        chunks = []

        paragraphs = text.split("\n")

        current = []
        current_len = 0
        chunk_id = 0

        for p in paragraphs:

            if current_len + len(p) > chunk_size:

                chunks.append({
                    "chunk_id": f"{law_name}_{article_id}_{chunk_id}",
                    "law": law_name,
                    "article_id": article_id,
                    "text": "\n".join(current)
                })

                current = []
                current_len = 0
                chunk_id += 1

            current.append(p)
            current_len += len(p)

        if current:
            chunks.append({
                "chunk_id": f"{law_name}_{article_id}_{chunk_id}",
                "law": law_name,
                "article_id": article_id,
                "text": "\n".join(current)
            })

        return chunks
