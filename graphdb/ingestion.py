from graphdb.neo4j_client import run_query



def create_document(doc):

    query = """
    MERGE (d:Document {
        name: $name
    })

    SET d.content = $content
    SET d.filepath = $filepath
    """

    run_query(query, doc)



def create_entity(entity):

    query = """
    MERGE (e:Entity {
        id: $id
    })

    SET e.type = $type
    SET e.description = $description
    """

    run_query(query, entity)



def create_relationship(rel):

    query = """
    MATCH (a:Entity {id:$source})
    MATCH (b:Entity {id:$target})

    MERGE (a)-[r:RELATED_TO {
        relation: $relation
    }]->(b)

    SET r.description = $description
    """

    run_query(query, rel)



def connect_document(entity_id, document_name):

    query = """
    MATCH (e:Entity {id:$entity})
    MATCH (d:Document {name:$document})

    MERGE (e)-[:MENTIONED_IN]->(d)
    """

    run_query(query, {
        "entity": entity_id,
        "document": document_name
    })
