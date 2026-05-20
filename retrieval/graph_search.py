from graphdb.neo4j_client import run_query

import re


def extract_keywords(text):

    words = re.findall(
        r"\w+",
        text.lower()
    )

    stopwords = {
        "какие",
        "какая",
        "какой",
        "что",
        "где",
        "когда",
        "для",
        "при",
        "или",
        "как",
        "по"
    }

    return [
        w for w in words
        if len(w) > 3
        and w not in stopwords
    ]


def retrieve_entities(question):

    keywords = extract_keywords(question)

    if not keywords:

        return []

    conditions = []

    params = {}

    for i, kw in enumerate(keywords):

        key = f"kw{i}"

        conditions.append(
            f"toLower(e.id) CONTAINS ${key}"
        )

        params[key] = kw

    query = f"""
    MATCH (e:Entity)

    WHERE {" OR ".join(conditions)}

    RETURN DISTINCT e.id as entity

    LIMIT 20
    """

    rows = run_query(
        query,
        params
    )

    return [r["entity"] for r in rows]


def expand_neighbors(entities):

    if not entities:

        return []

    query = """
    MATCH (e:Entity)-[r:RELATED_TO]-(n:Entity)

    WHERE e.id IN $entities

    RETURN
        e.id as source,
        r.relation as relation,
        n.id as target

    LIMIT 50
    """

    rows = run_query(query, {
        "entities": entities
    })

    return rows