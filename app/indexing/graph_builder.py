from app.db.neo4j_client import Neo4jClient


class GraphBuilder:

    def __init__(self):
        self.db = Neo4jClient()

    def create_document(self, document):

        title = document["title"]

        self.db.execute(
            """
            MERGE (d:Document {title: $title})
            """,
            {"title": title}
        )

        for section in document["sections"]:

            section_title = section["title"]

            self.db.execute(
                """
                MATCH (d:Document {title: $doc_title})
                MERGE (s:Section {
                    title: $section_title
                })
                MERGE (d)-[:HAS_SECTION]->(s)
                """,
                {
                    "doc_title": title,
                    "section_title": section_title
                }
            )

            for article in section["articles"]:

                self.db.execute(
                    """
                    MATCH (s:Section {
                        title: $section_title
                    })

                    MERGE (a:Article {
                        article_number: $article_number,
                        document: $document
                    })

                    SET a.title = $title
                    SET a.text = $text

                    MERGE (s)-[:HAS_ARTICLE]->(a)
                    """,
                    {
                        "section_title": section_title,
                        "article_number": article["number"],
                        "title": article["title"],
                        "text": article["text"],
                        "document": title
                    }
                )
