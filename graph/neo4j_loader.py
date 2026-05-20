from neo4j import GraphDatabase


class Neo4jLoader:

    def __init__(self,
                 uri,
                 user,
                 password):

        self.driver = GraphDatabase.driver(
            uri,
            auth=(user, password)
        )

    def clear_database(self):

        query = """
        MATCH (n)
        DETACH DELETE n
        """

        with self.driver.session() as session:
            session.run(query)

    def create_article(self,
                       law_name,
                       article_id,
                       text):

        query = """
        MERGE (a:Article {
            law:$law,
            article_id:$article_id
        })

        SET a.text=$text
        """

        with self.driver.session() as session:
            session.run(
                query,
                law=law_name,
                article_id=article_id,
                text=text
            )

    def create_reference(self,
                         source_law,
                         source_article,
                         target_article):

        query = """
        MATCH (a:Article {
            law:$source_law,
            article_id:$source_article
        })

        MATCH (b:Article {
            article_id:$target_article
        })

        MERGE (a)-[:REFERS_TO]->(b)
        """

        with self.driver.session() as session:
            session.run(
                query,
                source_law=source_law,
                source_article=source_article,
                target_article=target_article
            )

