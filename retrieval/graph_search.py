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