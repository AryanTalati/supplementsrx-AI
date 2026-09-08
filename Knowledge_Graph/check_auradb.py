import os
from neo4j import GraphDatabase

# ← Put your real AuraDB credentials here
URI      = "neo4j+s://8976bb7e.databases.neo4j.io"
USER     = "8976bb7e"
PASSWORD = os.environ.get("NEO4J_PASSWORD", "")

driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))

with driver.session() as s:
    # Check community IDs
    r = list(s.run("MATCH (s:Supplement) WHERE s.louvain_community IS NOT NULL RETURN count(s) AS count"))
    print(f"Supplements with community IDs: {r[0]['count']}")

    # Check total nodes
    r2 = list(s.run("MATCH (n) RETURN labels(n)[0] AS label, count(n) AS count ORDER BY count DESC"))
    print("\nNode counts:")
    for rec in r2:
        print(f"  {rec['label']}: {rec['count']:,}")

driver.close()
print("\nDone!")
