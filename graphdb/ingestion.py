from graphdb.neo4j_client import run_query

def create_documents_batch(docs):
    query = """
    UNWIND $docs AS doc
    MERGE (d:Document {name: doc.name})
    SET d.content = doc.content,
        d.filepath = doc.filepath
    """
    run_query(query, {"docs": docs})


def create_entities_batch(entities):
    query = """
    UNWIND $entities AS e
    MERGE (ent:Entity {id: e.id})
    SET ent.type = e.type,
        ent.description = e.description,
        ent.article_reference = e.article_reference
    """
    run_query(query, {"entities": entities})


def create_relationships_batch(relationships):
    query = """
    UNWIND $rels AS r
    MATCH (a:Entity {id: r.source})
    MATCH (b:Entity {id: r.target})
    MERGE (a)-[rel:RELATED_TO {relation: r.relation}]->(b)
    SET rel.description = r.description,
        rel.strength = r.strength
    """
    run_query(query, {"rels": relationships})


def create_facts_batch(facts):
    query = """
    UNWIND $facts AS f
    MERGE (fact:Fact {statement: f.statement})
    SET fact.article_reference = f.article_reference
    WITH fact, f
    UNWIND f.entities AS ent_id
    MATCH (e:Entity {id: ent_id})
    MERGE (fact)-[:MENTIONS]->(e)
    """
    run_query(query, {"facts": facts})


def connect_documents_batch(connections):
    query = """
    UNWIND $conn AS c
    MATCH (e:Entity {id: c.entity})
    MATCH (d:Document {name: c.document})
    MERGE (e)-[:MENTIONED_IN]->(d)
    """
    run_query(query, {"conn": connections})