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
        ent.description = e.description
    """
    run_query(query, {"entities": entities})

def create_relationships_batch(relationships):
    query = """
    UNWIND $rels AS r
    MATCH (a:Entity {id: r.source})
    MATCH (b:Entity {id: r.target})
    MERGE (a)-[rel:RELATED_TO {relation: r.relation}]->(b)
    SET rel.description = r.description
    """
    run_query(query, {"rels": relationships})

def connect_documents_batch(connections):
    query = """
    UNWIND $conn AS c
    MATCH (e:Entity {id: c.entity})
    MATCH (d:Document {name: c.document})
    MERGE (e)-[:MENTIONED_IN]->(d)
    """
    run_query(query, {"conn": connections})
