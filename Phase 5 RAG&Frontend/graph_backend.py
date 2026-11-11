"""
graph_backend.py — knowledge-graph access for the Phase 4 LangGraph pipeline.

Two interchangeable backends expose the lookups used by phase4_langgraph.py:

  local_lookup(name)         -> 1-hop neighbourhood of up to 3 fuzzy-matched Supplements
  supplements_by_name(names) -> the same neighbourhood for exact names (FAISS seeds)
  two_hop(name, k)           -> k most similar Supplements (SIMILAR_TO by weight)
                                with their interactions / side effects / conditions
  community_summary(cid)     -> conditions/members summary for a Louvain community

Neo4jGraphBackend runs the original Cypher. InMemoryGraphBackend reproduces
those queries in Python from Knowledge_Graph/graph.json + neo4j_communities.json,
applying the same node-label and relationship-type mapping that
Knowledge_Graph/migrate_to_neo4j.py used to load Neo4j, so no database is
needed. Neo4j is used only when NEO4J_URI and NEO4J_PASSWORD are set and
verify_connectivity() succeeds.
"""

import json
import logging
import os
import re
from typing import Dict, List, Optional, Tuple

log = logging.getLogger("supplementsrx.graph")

# ── Mappings copied from Knowledge_Graph/migrate_to_neo4j.py ──
NODE_LABEL_MAP = {
    "supplement":  "Supplement",
    "condition":   "Condition",
    "side_effect": "SideEffect",
    "drug":        "Drug",
    "class":       "SupplementClass",
}
EDGE_TYPE_MAP = {
    "TREATS":          "TREATS",
    "CAUSES":          "HAS_SIDE_EFFECT",
    "INTERACTS_WITH":  "INTERACTS_WITH",
    "BELONGS_TO":      "BELONGS_TO_CLASS",
    "SIMILAR_TO":      "SIMILAR_TO",
}

# ── The two Cypher queries (moved verbatim from phase4_langgraph.py) ──
LOCAL_QUERY = """
                MATCH (s:Supplement)
                WHERE toLower(s.name) CONTAINS toLower($name)
                    OR toLower($name) CONTAINS toLower(s.name)
                    OR s.name =~ ('(?i).*' + $name + '.*')
                WITH s ORDER BY
                    CASE WHEN toLower(s.name) = toLower($name) THEN 0
                        WHEN toLower(s.name) STARTS WITH toLower($name) THEN 1
                        ELSE 2 END,
                    size(s.name) ASC
                LIMIT 3
                OPTIONAL MATCH (s)-[:TREATS]->(c:Condition)
                OPTIONAL MATCH (s)-[:INTERACTS_WITH]->(d:Drug)
                OPTIONAL MATCH (s)-[:HAS_SIDE_EFFECT]->(se:SideEffect)
                OPTIONAL MATCH (s)-[:BELONGS_TO_CLASS]->(sc:SupplementClass)
                OPTIONAL MATCH (s)-[:SIMILAR_TO]->(s2:Supplement)
                RETURN s.name                               AS name,
                       s.overview                           AS overview,
                       s.pregnancy                          AS pregnancy,
                       s.scientific                         AS scientific,
                       collect(DISTINCT c.name)[0..8]  AS conditions,
                       collect(DISTINCT d.name)[0..8]  AS drugs,
                       collect(DISTINCT se.name)[0..5] AS side_effects,
                       collect(DISTINCT sc.name)[0..3] AS classes,
                       collect(DISTINCT s2.name)[0..5] AS similar
            """

COMMUNITY_QUERY = """
                MATCH (s:Supplement {louvain_community: $cid})-[:TREATS]->(c:Condition)
                WITH s.louvain_community AS community,
                    collect(DISTINCT c.name)[0..3] AS conditions,
                    collect(DISTINCT s.name)[0..3] AS members
                RETURN community, conditions, members
                LIMIT 1
            """

LOCAL_FIELDS = ("name", "overview", "pregnancy", "scientific", "conditions",
                "drugs", "side_effects", "classes", "similar")

# Same 1-hop neighbourhood as LOCAL_QUERY, for exact supplement names
# (used to seed the traversal from FAISS hits).
BY_NAME_QUERY = """
                UNWIND $names AS n
                MATCH (s:Supplement {name: n})
                OPTIONAL MATCH (s)-[:TREATS]->(c:Condition)
                OPTIONAL MATCH (s)-[:INTERACTS_WITH]->(d:Drug)
                OPTIONAL MATCH (s)-[:HAS_SIDE_EFFECT]->(se:SideEffect)
                OPTIONAL MATCH (s)-[:BELONGS_TO_CLASS]->(sc:SupplementClass)
                OPTIONAL MATCH (s)-[:SIMILAR_TO]->(s2:Supplement)
                RETURN s.name                               AS name,
                       s.overview                           AS overview,
                       s.pregnancy                          AS pregnancy,
                       s.scientific                         AS scientific,
                       collect(DISTINCT c.name)[0..8]  AS conditions,
                       collect(DISTINCT d.name)[0..8]  AS drugs,
                       collect(DISTINCT se.name)[0..5] AS side_effects,
                       collect(DISTINCT sc.name)[0..3] AS classes,
                       collect(DISTINCT s2.name)[0..5] AS similar
            """

# Second hop: supplement -[SIMILAR_TO]-> most similar supplements (by edge
# weight) -> their drug interactions, side effects and conditions.
TWO_HOP_QUERY = """
                MATCH (s:Supplement {name: $name})-[r:SIMILAR_TO]->(s2:Supplement)
                WITH s2, r ORDER BY r.weight DESC, s2.name ASC
                LIMIT $k
                OPTIONAL MATCH (s2)-[:INTERACTS_WITH]->(d:Drug)
                OPTIONAL MATCH (s2)-[:HAS_SIDE_EFFECT]->(se:SideEffect)
                OPTIONAL MATCH (s2)-[:TREATS]->(c:Condition)
                RETURN s2.name                          AS name,
                       r.weight                         AS weight,
                       r.shared_conditions              AS shared_conditions,
                       collect(DISTINCT d.name)[0..5]   AS drugs,
                       collect(DISTINCT se.name)[0..3]  AS side_effects,
                       collect(DISTINCT c.name)[0..3]   AS conditions
                ORDER BY weight DESC, name ASC
            """
TWO_HOP_MAX_K = 5
TWO_HOP_FIELDS = ("name", "weight", "shared_conditions", "drugs",
                  "side_effects", "conditions")


#═══════════════════════════════════════════════════════════════
#  NEO4J BACKEND
#═══════════════════════════════════════════════════════════════

class Neo4jGraphBackend:
    name = "neo4j"

    def __init__(self, driver):
        self.driver = driver

    def local_lookup(self, name: str) -> List[dict]:
        with self.driver.session() as session:
            return [{k: r[k] for k in LOCAL_FIELDS}
                    for r in session.run(LOCAL_QUERY, name=name)]

    def supplements_by_name(self, names: List[str]) -> List[dict]:
        if not names:
            return []
        with self.driver.session() as session:
            rows = {r["name"]: {k: r[k] for k in LOCAL_FIELDS}
                    for r in session.run(BY_NAME_QUERY, names=list(names))}
        return [rows[n] for n in dict.fromkeys(names) if n in rows]

    def two_hop(self, name: str, k: int = 3) -> List[dict]:
        k = max(0, min(int(k), TWO_HOP_MAX_K))
        if not k:
            return []
        with self.driver.session() as session:
            return [{f: r[f] for f in TWO_HOP_FIELDS}
                    for r in session.run(TWO_HOP_QUERY, name=name, k=k)]

    def community_summary(self, cid) -> Optional[dict]:
        with self.driver.session() as session:
            rows = list(session.run(COMMUNITY_QUERY, cid=cid))
        if not rows:
            return None
        r = rows[0]
        return {"community": r["community"], "conditions": r["conditions"],
                "members": r["members"]}

    def close(self):
        self.driver.close()


#═══════════════════════════════════════════════════════════════
#  IN-MEMORY BACKEND
#═══════════════════════════════════════════════════════════════

# Neighbour lists kept per supplement, already cut to the query's slice sizes.
# (relationship type, required target label, slice size) — order matches the
# OPTIONAL MATCH clauses / RETURN columns above.
_NEIGHBOUR_SPECS = (
    ("TREATS",           "Condition",       8),   # conditions
    ("INTERACTS_WITH",   "Drug",            8),   # drugs
    ("HAS_SIDE_EFFECT",  "SideEffect",      5),   # side_effects
    ("BELONGS_TO_CLASS", "SupplementClass", 3),   # classes
    ("SIMILAR_TO",       "Supplement",      5),   # similar
)


def _cypher_size(s: str) -> int:
    """size() on a Cypher string counts UTF-16 code units (Java String.length)."""
    return len(s.encode("utf-16-le")) // 2


class InMemoryGraphBackend:
    """
    Pure-Python equivalent of the Cypher queries above.

    Semantics reproduced from the Neo4j load (migrate_to_neo4j.py +
    load_communities_to_aura.py):
      * node label from node_type (default "supplement"); name = node id
      * s.overview is stored truncated to 500 chars; missing props are ""
      * relationship types renamed via EDGE_TYPE_MAP, duplicates merged
      * each OPTIONAL MATCH only follows edges whose target has the right
        label (e.g. INTERACTS_WITH -> a Supplement node is NOT a :Drug)
      * louvain_community is set only on Supplement nodes
    Ties in ORDER BY / collect() order follow graph.json order, which is
    the order nodes and relationships were created in Neo4j.

    Only compact lookup structures are kept; the parsed graph.json is
    discarded after construction.
    """
    name = "in-memory"

    def __init__(self, graph_path: str, communities: Dict[str, int]):
        with open(graph_path, encoding="utf-8") as f:
            data = json.load(f)

        labels: Dict[str, str] = {}
        canon: Dict[str, str] = {}          # share one str object per name
        supp_order: List[str] = []
        props: Dict[str, Tuple[str, str, str]] = {}

        for node in data.get("nodes", []):
            nid = node.get("id", "")
            label = NODE_LABEL_MAP.get(node.get("node_type", "supplement"), "Node")
            # MERGE on (label, name): first occurrence wins the label slot
            if nid in labels:
                continue
            labels[nid] = label
            canon[nid] = nid
            if label == "Supplement":
                supp_order.append(nid)
                props[nid] = (
                    (node.get("overview", "") or "")[:500],
                    node.get("pregnancy", "") or "",
                    node.get("scientific", "") or "",
                )

        spec_index = {(rel, tgt): i for i, (rel, tgt, _) in enumerate(_NEIGHBOUR_SPECS)}
        full: Dict[str, Tuple[list, ...]] = {
            s: tuple([] for _ in _NEIGHBOUR_SPECS) for s in supp_order
        }
        seen = set()
        similar_w: Dict[str, List[Tuple[float, str, str]]] = {}
        for edge in data.get("edges", []):
            src, tgt = edge.get("source", ""), edge.get("target", "")
            if src not in labels or tgt not in labels:
                continue  # MATCH on a missing endpoint creates nothing
            rel = EDGE_TYPE_MAP.get(edge.get("edge_type", "UNKNOWN"),
                                    edge.get("edge_type", "UNKNOWN"))
            key = (src, rel, tgt)
            if key in seen:  # MERGE dedupes identical relationships
                continue
            seen.add(key)
            if labels[src] != "Supplement":
                continue
            i = spec_index.get((rel, labels[tgt]))
            if i is not None:
                full[src][i].append(canon[tgt])
            if rel == "SIMILAR_TO" and labels[tgt] == "Supplement":
                # relationship properties as migrate_to_neo4j.py stores them
                sc = edge.get("shared_conditions", [])
                sc = ", ".join(sc) if isinstance(sc, list) else str(sc)
                similar_w.setdefault(src, []).append(
                    (float(edge.get("weight", 1.0)), canon[tgt], sc))
        del data, seen

        # Top SIMILAR_TO neighbours by weight (ORDER BY r.weight DESC, s2.name ASC)
        self._similar_top: Dict[str, Tuple[Tuple[float, str, str], ...]] = {
            s: tuple(sorted(lst, key=lambda t: (-t[0], t[1]))[:TWO_HOP_MAX_K])
            for s, lst in similar_w.items()
        }
        del similar_w

        # Precompute community summaries before trimming neighbour lists.
        # Rows of MATCH (s {louvain_community})-[:TREATS]->(c:Condition) in
        # node order, then relationship order.
        comm_rows: Dict[int, Tuple[List[str], List[str]]] = {}
        for s in supp_order:
            cid = communities.get(s)
            if cid is None:
                continue
            treats = full[s][0]
            if not treats:
                continue
            conds, members = comm_rows.setdefault(cid, ([], []))
            members.append(s)
            for c in treats:
                if c not in conds:
                    conds.append(c)
        self._communities: Dict[int, Tuple[Tuple[str, ...], Tuple[str, ...]]] = {
            cid: (tuple(conds[:3]), tuple(members[:3]))
            for cid, (conds, members) in comm_rows.items()
        }

        # Per-supplement record with neighbour lists cut to the slice sizes.
        self._supplements: List[Tuple[str, str, int]] = [
            (s, s.lower(), _cypher_size(s)) for s in supp_order
        ]
        self._records: Dict[str, Tuple] = {}
        for s in supp_order:
            neigh = tuple(tuple(full[s][i][:n]) for i, (_, _, n) in enumerate(_NEIGHBOUR_SPECS))
            self._records[s] = props[s] + neigh
        del full, props

        log.info("In-memory graph: %d supplements, %d communities with TREATS edges",
                 len(self._supplements), len(self._communities))

    # ── local 2-hop lookup ────────────────────────────────────
    def local_lookup(self, name: str) -> List[dict]:
        name = "" if name is None else str(name)
        lname = name.lower()
        try:
            # Cypher =~ is a full, case-insensitive (ASCII) match of a Java regex
            rx = re.compile("(?i).*" + name + ".*", re.ASCII | re.DOTALL)
        except re.error:
            # Neo4j would raise on an invalid pattern and fail the whole query;
            # here only the regex clause is dropped and CONTAINS still applies.
            rx = None

        matches = []
        for pos, (sname, slower, ssize) in enumerate(self._supplements):
            if (lname in slower or slower in lname
                    or (rx is not None and rx.fullmatch(sname))):
                rank = 0 if slower == lname else 1 if slower.startswith(lname) else 2
                matches.append((rank, ssize, pos, sname))
        matches.sort()

        return [self._record(sname) for _, _, _, sname in matches[:3]]

    def _record(self, sname: str) -> dict:
        overview, pregnancy, scientific, conds, drugs, ses, classes, similar = self._records[sname]
        return {
            "name": sname, "overview": overview, "pregnancy": pregnancy,
            "scientific": scientific, "conditions": list(conds),
            "drugs": list(drugs), "side_effects": list(ses),
            "classes": list(classes), "similar": list(similar),
        }

    # ── exact-name 1-hop lookup (FAISS seeds) ─────────────────
    def supplements_by_name(self, names: List[str]) -> List[dict]:
        return [self._record(n) for n in dict.fromkeys(names or []) if n in self._records]

    # ── second hop via SIMILAR_TO ─────────────────────────────
    def two_hop(self, name: str, k: int = 3) -> List[dict]:
        k = max(0, min(int(k), TWO_HOP_MAX_K))
        out = []
        for weight, s2, shared in self._similar_top.get(name, ())[:k]:
            _, _, _, conds, drugs, ses, _, _ = self._records[s2]
            out.append({
                "name": s2, "weight": weight, "shared_conditions": shared,
                "drugs": list(drugs[:5]), "side_effects": list(ses[:3]),
                "conditions": list(conds[:3]),
            })
        return out

    # ── global community summary ──────────────────────────────
    def community_summary(self, cid) -> Optional[dict]:
        try:
            hit = self._communities.get(cid)
        except TypeError:  # unhashable cid never matches a property value
            return None
        if hit is None:
            return None
        return {"community": cid, "conditions": list(hit[0]), "members": list(hit[1])}

    def close(self):
        pass


#═══════════════════════════════════════════════════════════════
#  FACTORY
#═══════════════════════════════════════════════════════════════

def create_graph_backend(graph_path: str, communities: Dict[str, int]):
    """Neo4j if configured and reachable, else the in-memory backend."""
    uri = os.environ.get("NEO4J_URI", "").strip()
    password = os.environ.get("NEO4J_PASSWORD", "")
    user = os.environ.get("NEO4J_USER", "neo4j").strip() or "neo4j"

    if uri and password:
        driver = None
        try:
            from neo4j import GraphDatabase
            driver = GraphDatabase.driver(uri, auth=(user, password),
                                          connection_timeout=10)
            driver.verify_connectivity()
            log.info("Graph backend: Neo4j at %s", uri)
            return Neo4jGraphBackend(driver)
        except Exception as e:
            log.warning("Neo4j at %s unavailable (%s: %s) — falling back to in-memory graph",
                        uri, type(e).__name__, e)
            if driver is not None:
                try:
                    driver.close()
                except Exception:
                    pass
    else:
        log.info("NEO4J_URI/NEO4J_PASSWORD not set — using in-memory graph")

    backend = InMemoryGraphBackend(graph_path, communities)
    log.info("Graph backend: in-memory (from %s)", graph_path)
    return backend
