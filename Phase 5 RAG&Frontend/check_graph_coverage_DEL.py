import os
from neo4j import GraphDatabase
import json

URI      = "neo4j+s://8976bb7e.databases.neo4j.io"
USER     = "8976bb7e"
PASSWORD = os.environ.get("NEO4J_PASSWORD", "")

driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))

with driver.session() as session:

    # 1. Total node counts by label
    print("=== NODE COUNTS BY LABEL ===")
    result = list(session.run("""
        MATCH (n)
        RETURN labels(n)[0] AS label, count(n) AS count
        ORDER BY count DESC
    """))
    total = 0
    for r in result:
        print(f"  {r['label']:20s}: {r['count']:,}")
        total += r['count']
    print(f"  {'TOTAL':20s}: {total:,}")

    # 2. Supplement nodes specifically
    print("\n=== SUPPLEMENT NODES ===")
    result = list(session.run("MATCH (s:Supplement) RETURN count(s) AS count"))
    print(f"  Total Supplement nodes: {result[0]['count']:,}")

    # 3. Check what source the supplements came from
    print("\n=== SUPPLEMENT SOURCES ===")
    result = list(session.run("""
        MATCH (s:Supplement)
        RETURN 
            CASE 
                WHEN s.overview CONTAINS 'Clinical drug from MIMIC' THEN 'MIMIC-IV Drug'
                WHEN s.overview IS NULL OR s.overview = '' THEN 'No overview (empty)'
                ELSE 'NatMed/MedlinePlus'
            END AS source,
            count(s) AS count
        ORDER BY count DESC
    """))
    for r in result:
        print(f"  {r['source']:30s}: {r['count']:,}")

    # 4. How many have actual supplement data vs empty
    print("\n=== DATA QUALITY ===")
    result = list(session.run("""
        MATCH (s:Supplement)
        RETURN
            count(s) AS total,
            sum(CASE WHEN s.overview IS NOT NULL AND s.overview <> '' THEN 1 ELSE 0 END) AS has_overview,
            sum(CASE WHEN EXISTS((s)-[:TREATS]->()) THEN 1 ELSE 0 END) AS has_conditions,
            sum(CASE WHEN EXISTS((s)-[:INTERACTS_WITH]->()) THEN 1 ELSE 0 END) AS has_interactions
    """))
    r = result[0]
    print(f"  Total supplements     : {r['total']:,}")
    print(f"  Has overview          : {r['has_overview']:,}")
    print(f"  Has conditions        : {r['has_conditions']:,}")
    print(f"  Has interactions      : {r['has_interactions']:,}")

driver.close()

# 5. Check embedding metadata
print("\n=== FAISS EMBEDDING COVERAGE ===")
with open("supplement_metadata.json", encoding="utf-8") as f:
    metadata = json.load(f)

with open("supplement_names.json", encoding="utf-8") as f:
    names = json.load(f)

print(f"  Total embeddings      : {len(names):,}")
has_overview = sum(1 for m in metadata if m.get("overview"))
has_conditions = sum(1 for m in metadata if m.get("conditions"))
has_drugs = sum(1 for m in metadata if m.get("drugs"))
empty = sum(1 for m in metadata if not m.get("overview") and not m.get("conditions"))

print(f"  Has overview          : {has_overview:,}")
print(f"  Has conditions        : {has_conditions:,}")
print(f"  Has drug interactions : {has_drugs:,}")
print(f"  Completely empty      : {empty:,}")

# Sample of empty ones
print("\n  Sample empty embeddings:")
empty_names = [names[i] for i, m in enumerate(metadata) 
               if not m.get("overview") and not m.get("conditions")][:10]
for n in empty_names:
    print(f"    - {n}")
