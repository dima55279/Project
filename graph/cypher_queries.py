GRAPH_QUERY = """
MATCH (a:Article)
WHERE toLower(a.text) CONTAINS toLower($query)
OPTIONAL MATCH (a)-[:REFERS_TO]->(b)
RETURN a, collect(b) as refs
LIMIT 5
"""
