"""
=============================================================
  Migrate graph.json → Neo4j
  SupplementsRx AI Guidance System

  This script:
  1. Loads your existing graph.json (from Phase 2)
  2. Creates all nodes in Neo4j with correct labels
  3. Creates all relationships with properties
  4. Verifies the migration

  BEFORE RUNNING:
  - Neo4j Desktop must be running
  - GDS plugin must be installed
  - Update NEO4J_PASSWORD below if different

  HOW TO RUN:
    pip install neo4j
    python migrate_to_neo4j.py
=============================================================
"""
import os

from neo4j import GraphDatabase
import json
import time

# ── Connection settings ───────────────────────────────────────
NEO4J_URI      = "neo4j+s://8976bb7e.databases.neo4j.io"
NEO4J_USER     = "8976bb7e"
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "")

INPUT_JSON     = "graph.json"

# ── Node label mapping ────────────────────────────────────────
NODE_LABEL_MAP = {
    "supplement":  "Supplement",
    "condition":   "Condition",
    "side_effect": "SideEffect",
    "drug":        "Drug",
    "class":       "SupplementClass",
}

# ── Relationship type mapping ─────────────────────────────────
EDGE_TYPE_MAP = {
    "TREATS":          "TREATS",
    "CAUSES":          "HAS_SIDE_EFFECT",
    "INTERACTS_WITH":  "INTERACTS_WITH",
    "BELONGS_TO":      "BELONGS_TO_CLASS",
    "SIMILAR_TO":      "SIMILAR_TO",
}


class Neo4jMigrator:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        print("✅ Connected to Neo4j")

    def close(self):
        self.driver.close()

    def run(self, query, params=None):
        with self.driver.session() as session:
            return session.run(query, params or {})

    # ── STEP 1: Clear existing data ───────────────────────────
    def clear_database(self):
        print("\n🗑️  Clearing existing data...")
        self.run("MATCH (n) DETACH DELETE n")
        print("   Done")

    # ── STEP 2: Create constraints & indexes ──────────────────
    def create_constraints(self):
        print("\n📋 Creating constraints and indexes...")
        constraints = [
            "CREATE CONSTRAINT supp_name IF NOT EXISTS FOR (n:Supplement)   REQUIRE n.name IS UNIQUE",
            "CREATE CONSTRAINT cond_name IF NOT EXISTS FOR (n:Condition)    REQUIRE n.name IS UNIQUE",
            "CREATE CONSTRAINT se_name   IF NOT EXISTS FOR (n:SideEffect)   REQUIRE n.name IS UNIQUE",
            "CREATE CONSTRAINT drug_name IF NOT EXISTS FOR (n:Drug)         REQUIRE n.name IS UNIQUE",
            "CREATE CONSTRAINT cls_name  IF NOT EXISTS FOR (n:SupplementClass) REQUIRE n.name IS UNIQUE",
        ]
        for c in constraints:
            try:
                self.run(c)
            except Exception as e:
                print(f"   Note: {e}")
        print("   Done")

    # ── STEP 3: Create nodes in batches ───────────────────────
    def create_nodes(self, nodes):
        print(f"\n📦 Creating {len(nodes):,} nodes...")

        # Group by type
        by_type = {}
        for node in nodes:
            nt = node.get("node_type", "supplement")
            if nt not in by_type:
                by_type[nt] = []
            by_type[nt].append(node)

        total = 0
        for node_type, node_list in by_type.items():
            label = NODE_LABEL_MAP.get(node_type, "Node")
            print(f"   Creating {len(node_list):,} {label} nodes...")

            # Batch in groups of 500
            for i in range(0, len(node_list), 500):
                batch = node_list[i:i+500]
                props_list = []
                for node in batch:
                    props = {
                        "name":          node.get("id", ""),
                        "scientific":    node.get("scientific", ""),
                        "also_known_as": node.get("also_known_as", ""),
                        "source_url":    node.get("source_url", ""),
                        "overview":      node.get("overview", "")[:500],
                        "pregnancy":     node.get("pregnancy", ""),
                        "last_reviewed": node.get("last_reviewed", ""),
                        "node_type":     node_type,
                    }
                    props_list.append(props)

                query = f"""
                UNWIND $batch AS props
                MERGE (n:{label} {{name: props.name}})
                SET n += props
                """
                self.run(query, {"batch": props_list})
                total += len(batch)

        print(f"   ✅ {total:,} nodes created")
        return total

    # ── STEP 4: Create relationships in batches ───────────────
    def create_relationships(self, edges):
        print(f"\n🔗 Creating {len(edges):,} relationships...")

        # Group by edge type
        by_type = {}
        for edge in edges:
            et = edge.get("edge_type", "UNKNOWN")
            if et not in by_type:
                by_type[et] = []
            by_type[et].append(edge)

        total = 0
        for edge_type, edge_list in by_type.items():
            rel_type = EDGE_TYPE_MAP.get(edge_type, edge_type)
            print(f"   Creating {len(edge_list):,} {rel_type} relationships...")

            # Batch in groups of 1000
            for i in range(0, len(edge_list), 1000):
                batch = edge_list[i:i+1000]
                batch_data = []
                for edge in batch:
                    batch_data.append({
                        "source": edge.get("source", ""),
                        "target": edge.get("target", ""),
                        "weight": edge.get("weight", 1.0),
                        "description": str(edge.get("description", ""))[:200],
                        "shared_conditions": ", ".join(
                            edge.get("shared_conditions", [])
                        ) if isinstance(edge.get("shared_conditions"), list)
                        else str(edge.get("shared_conditions", "")),
                    })

                query = f"""
                UNWIND $batch AS rel
                MATCH (a {{name: rel.source}})
                MATCH (b {{name: rel.target}})
                MERGE (a)-[r:{rel_type}]->(b)
                SET r.weight      = rel.weight,
                    r.description = rel.description,
                    r.shared_conditions = rel.shared_conditions
                """
                self.run(query, {"batch": batch_data})
                total += len(batch)

        print(f"   ✅ {total:,} relationships created")
        return total

    # ── STEP 5: Verify migration ──────────────────────────────
    def verify(self):
        print("\n🔍 Verifying migration...")

        counts = self.run("""
            MATCH (n)
            RETURN labels(n)[0] AS label, count(n) AS count
            ORDER BY count DESC
        """)
        print("   Node counts:")
        for record in counts:
            print(f"     {record['label']:20s}: {record['count']:,}")

        rel_counts = self.run("""
            MATCH ()-[r]->()
            RETURN type(r) AS type, count(r) AS count
            ORDER BY count DESC
        """)
        print("   Relationship counts:")
        for record in rel_counts:
            print(f"     {record['type']:25s}: {record['count']:,}")

        # Verify a specific supplement
        test = self.run("""
            MATCH (s:Supplement {name: 'Ashwagandha'})
            OPTIONAL MATCH (s)-[:TREATS]->(c:Condition)
            OPTIONAL MATCH (s)-[:HAS_SIDE_EFFECT]->(se:SideEffect)
            OPTIONAL MATCH (s)-[:INTERACTS_WITH]->(d:Drug)
            RETURN s.name AS name,
                   count(DISTINCT c)  AS conditions,
                   count(DISTINCT se) AS side_effects,
                   count(DISTINCT d)  AS drug_interactions
        """)
        for record in test:
            print(f"\n   Ashwagandha verification:")
            print(f"     Conditions      : {record['conditions']}")
            print(f"     Side effects    : {record['side_effects']}")
            print(f"     Drug interactions: {record['drug_interactions']}")

    # ── STEP 6: Run Louvain GDS ───────────────────────────────
    def run_louvain_gds(self):
        print("\n🔬 Running Louvain clustering via Neo4j GDS...")

        # Check if GDS is available
        try:
            self.run("RETURN gds.version() AS version")
        except Exception:
            print("   ⚠️  GDS plugin not found!")
            print("   Install it: Neo4j Desktop → Your DB → Plugins → Graph Data Science")
            print("   Then restart the DB and run this script again")
            return False

        # Drop existing projection if it exists
        try:
            self.run("CALL gds.graph.drop('supplementGraph', false)")
        except Exception:
            pass

        # Create graph projection
        print("   Creating graph projection...")
        self.run("""
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

        # Run Louvain
        print("   Running Louvain algorithm...")
        self.run("""
            CALL gds.louvain.write(
                'supplementGraph',
                {
                    writeProperty: 'louvain_community',
                    relationshipWeightProperty: 'weight',
                    maxIterations: 10,
                    maxLevels: 10
                }
            )
        """)

        # Get community stats
        stats = self.run("""
            MATCH (s:Supplement)
            WHERE s.louvain_community IS NOT NULL
            RETURN s.louvain_community AS community,
                   count(s) AS size
            ORDER BY size DESC
            LIMIT 20
        """)

        communities = list(stats)
        print(f"   ✅ Louvain done! Found {len(communities)}+ communities")
        print("   Top communities by size:")
        for r in communities[:10]:
            print(f"     Community {r['community']:4d}: {r['size']:4d} supplements")

        # Clean up projection
        self.run("CALL gds.graph.drop('supplementGraph', false)")
        return True

    # ── STEP 7: Generate community summaries ─────────────────
    def generate_community_labels(self):
        print("\n🏷️  Generating community labels...")

        # For each community, find the most common conditions
        self.run("""
            MATCH (s:Supplement)-[:TREATS]->(c:Condition)
            WHERE s.louvain_community IS NOT NULL
            WITH s.louvain_community AS community,
                 c.name AS condition,
                 count(*) AS freq
            ORDER BY community, freq DESC
            WITH community,
                 collect(condition)[0..3] AS top_conditions
            MATCH (s:Supplement {louvain_community: community})
            WITH community, top_conditions,
                 collect(s.name)[0..5] AS sample_members
            MERGE (label:CommunityLabel {community_id: community})
            SET label.top_conditions  = top_conditions,
                label.sample_members  = sample_members,
                label.label           = top_conditions[0]
        """)

        count = self.run("""
            MATCH (cl:CommunityLabel)
            RETURN count(cl) AS count
        """).single()["count"]

        print(f"   ✅ Created {count} community labels")

        # Export community assignments back to JSON for Phase 4
        print("   Exporting community assignments...")
        results = self.run("""
            MATCH (s:Supplement)
            WHERE s.louvain_community IS NOT NULL
            RETURN s.name AS name,
                   s.louvain_community AS community
            ORDER BY s.name
        """)

        assignments = {r["name"]: r["community"] for r in results}
        with open("neo4j_communities.json", "w") as f:
            json.dump(assignments, f, indent=2)

        print(f"   Saved {len(assignments):,} assignments → neo4j_communities.json")
        return assignments


def main():
    print("=" * 60)
    print("  Neo4j Migration — SupplementsRx")
    print("=" * 60)

    # Load graph.json
    print(f"\n📂 Loading {INPUT_JSON}...")
    with open(INPUT_JSON, encoding="utf-8") as f:
        data = json.load(f)

    nodes = data.get("nodes", [])
    edges = data.get("edges", [])
    print(f"   Nodes: {len(nodes):,}")
    print(f"   Edges: {len(edges):,}")

    # Connect
    try:
        migrator = Neo4jMigrator(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    except Exception as e:
        print(f"\n❌ Cannot connect to Neo4j: {e}")
        print("   Make sure Neo4j Desktop is running and the DB is started")
        return

    try:
        migrator.clear_database()
        migrator.create_constraints()
        migrator.create_nodes(nodes)
        migrator.create_relationships(edges)
        migrator.verify()
        louvain_ok = migrator.run_louvain_gds()
        if louvain_ok:
            migrator.generate_community_labels()

        print(f"""
{'='*60}
  ✅ MIGRATION COMPLETE!

  Your graph is now in Neo4j with:
  - Typed nodes (Supplement, Condition, Drug etc.)
  - Typed relationships (TREATS, INTERACTS_WITH etc.)
  - Louvain community IDs on every Supplement node
  - Community labels saved to neo4j_communities.json

  Open Neo4j Browser to explore:
  http://localhost:7474

  Try this query to see Ashwagandha:
  MATCH (s:Supplement {{name:'Ashwagandha'}})-[r]->(n)
  RETURN s, r, n LIMIT 50

  Next → python phase4_langchain.py
{'='*60}
""")

    finally:
        migrator.close()


if __name__ == "__main__":
    main()
