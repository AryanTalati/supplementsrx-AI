"""
Run this AFTER migrate_to_neo4j.py completes.
Verifies the migration and runs Louvain GDS clustering.
"""
import os

from neo4j import GraphDatabase
import json

NEO4J_URI      = "neo4j+s://8976bb7e.databases.neo4j.io"
NEO4J_USER     = "8976bb7e"
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "")

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

def run(query, params=None):
    with driver.session() as session:
        result = session.run(query, params or {})
        return list(result)   # ← consume immediately into list

# ── Verify node counts ────────────────────────────────────────
print("🔍 Verifying migration...\n")

records = run("MATCH (n) RETURN labels(n)[0] AS label, count(n) AS count ORDER BY count DESC")
print("Node counts:")
for r in records:
    print(f"  {r['label']:20s}: {r['count']:,}")

records = run("MATCH ()-[r]->() RETURN type(r) AS type, count(r) AS count ORDER BY count DESC")
print("\nRelationship counts:")
for r in records:
    print(f"  {r['type']:25s}: {r['count']:,}")

# ── Verify Ashwagandha ────────────────────────────────────────
records = run("""
    MATCH (s:Supplement {name: 'Ashwagandha'})
    OPTIONAL MATCH (s)-[:TREATS]->(c:Condition)
    OPTIONAL MATCH (s)-[:HAS_SIDE_EFFECT]->(se:SideEffect)
    OPTIONAL MATCH (s)-[:INTERACTS_WITH]->(d:Drug)
    RETURN s.name AS name,
           count(DISTINCT c)  AS conditions,
           count(DISTINCT se) AS side_effects,
           count(DISTINCT d)  AS drug_interactions
""")
if records:
    r = records[0]
    print(f"\nAshwagandha check:")
    print(f"  Conditions       : {r['conditions']}")
    print(f"  Side effects     : {r['side_effects']}")
    print(f"  Drug interactions: {r['drug_interactions']}")

# ── Run Louvain via GDS ───────────────────────────────────────
print("\n🔬 Running Louvain via Neo4j GDS...")

# Check GDS is available
try:
    records = run("RETURN gds.version() AS version")
    print(f"   GDS version: {records[0]['version']}")
except Exception as e:
    print(f"   ❌ GDS not available: {e}")
    print("   Install GDS: Neo4j Desktop → Your DB → Plugins → Graph Data Science")
    driver.close()
    exit()

# Drop existing projection
try:
    run("CALL gds.graph.drop('supplementGraph', false)")
    print("   Dropped old projection")
except Exception:
    pass

# Create projection using SIMILAR_TO edges
print("   Creating graph projection...")
run("""
    CALL gds.graph.project(
        'supplementGraph',
        'Supplement',
        {
            SIMILAR_TO: {
                orientation: 'UNDIRECTED',
                properties: 'weight'
            }
        }
    )
""")
print("   ✅ Projection created")

# Run Louvain
print("   Running Louvain algorithm...")
result = run("""
    CALL gds.louvain.write(
        'supplementGraph',
        {
            writeProperty: 'louvain_community',
            relationshipWeightProperty: 'weight',
            maxIterations: 10,
            maxLevels: 10
        }
    )
    YIELD communityCount, modularity, ranLevels
    RETURN communityCount, modularity, ranLevels
""")

if result:
    r = result[0]
    print(f"   ✅ Louvain done!")
    print(f"   Communities : {r['communityCount']}")
    print(f"   Modularity  : {r['modularity']:.4f}")
    print(f"   Levels ran  : {r['ranLevels']}")

# Top communities by size
print("\n   Top 15 communities:")
records = run("""
    MATCH (s:Supplement)
    WHERE s.louvain_community IS NOT NULL
    RETURN s.louvain_community AS community, count(s) AS size
    ORDER BY size DESC
    LIMIT 15
""")
for r in records:
    print(f"   Community {r['community']:4d}: {r['size']:4d} supplements")

# Drop projection
try:
    run("CALL gds.graph.drop('supplementGraph', false)")
except Exception:
    pass

# ── Generate community labels ─────────────────────────────────
print("\n🏷️  Generating community labels...")
run("""
    MATCH (s:Supplement)-[:TREATS]->(c:Condition)
    WHERE s.louvain_community IS NOT NULL
    WITH s.louvain_community AS community,
         c.name AS condition,
         count(*) AS freq
    ORDER BY community, freq DESC
    WITH community,
         collect(condition)[0..3] AS top_conditions
    MATCH (s:Supplement)
    WHERE s.louvain_community = community
    WITH community, top_conditions,
         collect(s.name)[0..5] AS sample_members
    MERGE (cl:CommunityLabel {community_id: community})
    SET cl.top_conditions = top_conditions,
        cl.sample_members = sample_members,
        cl.label          = top_conditions[0]
""")

labels = run("MATCH (cl:CommunityLabel) RETURN count(cl) AS count")
print(f"   ✅ Created {labels[0]['count']} community labels")

# ── Export community assignments to JSON ──────────────────────
print("\n💾 Exporting community assignments...")
records = run("""
    MATCH (s:Supplement)
    WHERE s.louvain_community IS NOT NULL
    RETURN s.name AS name, s.louvain_community AS community
    ORDER BY s.name
""")

assignments = {r["name"]: r["community"] for r in records}
with open("neo4j_communities.json", "w") as f:
    json.dump(assignments, f, indent=2)

print(f"   Saved {len(assignments):,} assignments → neo4j_communities.json")

# ── Sample community view ─────────────────────────────────────
print("\n📋 Sample community content:")
records = run("""
    MATCH (cl:CommunityLabel)
    RETURN cl.community_id AS id,
           cl.label AS label,
           cl.top_conditions AS conditions,
           cl.sample_members AS members
    ORDER BY cl.community_id
    LIMIT 10
""")
for r in records:
    print(f"\n  Community {r['id']} — {r['label']}")
    print(f"    Conditions : {r['conditions']}")
    print(f"    Sample     : {r['members'][:3]}")

driver.close()

print(f"""
{'='*60}
  ✅ ALL DONE!

  Neo4j now has:
  - All nodes and relationships migrated
  - Louvain community IDs on every Supplement node
  - Community labels created
  - neo4j_communities.json exported

  Open Neo4j Browser: http://localhost:7474
  Try: MATCH (s:Supplement {{name:'Ashwagandha'}})-[r]->(n)
       RETURN s, r, n LIMIT 50

  Next → python extract_triples_llm.py  (needs OpenAI key)
  Or   → python phase4_langchain.py     (if skipping LLM extraction)
{'='*60}
""")
