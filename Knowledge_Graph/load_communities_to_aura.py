"""
Loads Louvain community IDs from neo4j_communities.json
directly into AuraDB (since GDS is not available on free tier).
"""
import os

import json
from neo4j import GraphDatabase
from tqdm import tqdm

# ← Put your real AuraDB credentials here
URI      = "neo4j+s://8976bb7e.databases.neo4j.io"
USER     = "8976bb7e"
PASSWORD = os.environ.get("NEO4J_PASSWORD", "")

COMMUNITIES_FILE = "neo4j_communities.json"

driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))

# Load community assignments
print(f"Loading {COMMUNITIES_FILE}...")
with open(COMMUNITIES_FILE, encoding="utf-8") as f:
    assignments = json.load(f)

print(f"Loaded {len(assignments):,} supplement → community assignments")

# Load in batches of 500
batch_size = 500
items      = list(assignments.items())
total      = 0

with driver.session() as session:
    for i in tqdm(range(0, len(items), batch_size), desc="Uploading"):
        batch = items[i:i+batch_size]
        batch_data = [{"name": name, "community": community}
                      for name, community in batch]

        session.run("""
            UNWIND $batch AS item
            MATCH (s:Supplement {name: item.name})
            SET s.louvain_community = item.community
        """, batch=batch_data)

        total += len(batch)

# Verify
with driver.session() as session:
    result = list(session.run("""
        MATCH (s:Supplement)
        WHERE s.louvain_community IS NOT NULL
        RETURN count(s) AS count
    """))
    count = result[0]["count"]
    print(f"\n✅ Community IDs loaded: {count:,} supplements")

    # Show sample communities
    result2 = list(session.run("""
        MATCH (s:Supplement)
        WHERE s.louvain_community IS NOT NULL
        RETURN s.louvain_community AS community, count(s) AS size
        ORDER BY size DESC
        LIMIT 10
    """))
    print("\nTop 10 communities by size:")
    for r in result2:
        print(f"  Community {r['community']:4d}: {r['size']:4d} supplements")

driver.close()
print("\nDone! AuraDB is ready for Phase 4.")
