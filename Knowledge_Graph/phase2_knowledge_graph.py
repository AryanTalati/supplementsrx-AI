"""
=============================================================
  Phase 2 — Knowledge Graph Builder (v4 FINAL)
  SupplementsRx AI Guidance System

  Key design decisions in this version:
  
  1. SIDE EFFECTS: Strict clinical whitelist — only real 
     symptom names (nausea, headache, liver damage etc.)
     Conditions and drug mechanism terms are REJECTED.

  2. SIMILAR_TO: Excluded the 8 broadest conditions from 
     similarity calculation. "Pain & Inflammation" (1118 
     supps) causes every pair to match. We exclude these 
     super-broad categories and only use specific ones.

  3. CLASSES: Added a reject list for drug mechanism terms 
     that are NOT supplement classes.

  4. SIMILAR_TO threshold: Jaccard > 0.6 (very strict)

  INPUT : all_supplements.csv
  OUTPUT: graph.json, graph.graphml, graph_stats.txt, 
          graph_preview.png

  HOW TO RUN:
    pip install networkx pandas matplotlib
    python phase2_knowledge_graph.py
=============================================================
"""

import pandas as pd
import networkx as nx
import json, re, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

INPUT_CSV   = "merged_supplements.csv"
OUT_JSON    = "graph.json"
OUT_GRAPHML = "graph.graphml"
OUT_STATS   = "graph_stats.txt"
OUT_PLOT    = "graph_preview.png"

NODE_SUPPLEMENT  = "supplement"
NODE_CONDITION   = "condition"
NODE_SIDE_EFFECT = "side_effect"
NODE_DRUG        = "drug"
NODE_CLASS       = "class"

EDGE_TREATS         = "TREATS"
EDGE_CAUSES         = "CAUSES"
EDGE_INTERACTS_WITH = "INTERACTS_WITH"
EDGE_BELONGS_TO     = "BELONGS_TO"
EDGE_SIMILAR_TO     = "SIMILAR_TO"

# ── Conditions too broad to use for similarity ────────────────
# These apply to 500+ supplements so they're useless for 
# distinguishing between supplements
BROAD_CONDITIONS_EXCLUDE_FROM_SIMILARITY = {
    "pain & inflammation",
    "digestive health",
    "cardiovascular health",
    "immune support",
    "liver & detox",
    "weight management",
    "skin health",
    "cancer support",
    "respiratory health",
    "blood sugar management",
    "antioxidant & anti-aging",
    "sports & athletic performance",
    "kidney health",
    "anxiety & stress",
    "allergy relief",
    "cognitive health",
    "energy & fatigue",
    "thyroid & hormones",
}

# ── Real clinical side effects whitelist ──────────────────────
# ONLY strings containing one of these exact terms are accepted
SIDE_EFFECT_WHITELIST = [
    "nausea", "vomiting", "diarrhea", "diarrhoea", "constipation",
    "headache", "head ache", "migraine",
    "dizziness", "vertigo", "lightheaded",
    "drowsiness", "sedation", "somnolence",
    "fatigue", "tiredness", "weakness", "lethargy",
    "insomnia", "sleep disturbance", "sleep disorder",
    "rash", "skin rash", "hives", "urticaria", "itching", "pruritus",
    "abdominal pain", "stomach pain", "stomach upset", "stomach ache",
    "stomach cramp", "abdominal cramp",
    "bloating", "flatulence", "gas", "indigestion", "heartburn",
    "liver damage", "liver injury", "hepatotoxic", "hepatitis",
    "jaundice", "liver failure",
    "allergic reaction", "anaphylaxis", "hypersensitivity",
    "bleeding", "bruising", "hemorrhage",
    "low blood pressure", "hypotension",
    "high blood pressure", "hypertension",
    "heart palpitation", "palpitation", "tachycardia", "arrhythmia",
    "chest pain",
    "nervousness", "agitation", "irritability", "restlessness",
    "muscle pain", "muscle weakness", "myalgia", "muscle cramp",
    "joint pain",
    "dry mouth", "mouth sore", "oral irritation",
    "sweating", "excessive sweating", "flushing",
    "edema", "swelling", "water retention",
    "shortness of breath", "dyspnea",
    "cough", "throat irritation",
    "seizure", "convulsion",
    "confusion", "disorientation",
    "low blood sugar", "hypoglycemia",
    "photosensitivity", "sun sensitivity",
    "kidney damage", "renal toxicity",
    "thyroid dysfunction",
    "hormonal disruption",
    "appetite loss", "anorexia",
    "weight gain", "weight loss",
    "hair loss", "alopecia",
    "bruising", "easy bruising",
]

# ── Terms that look like side effects but are actually ────────
# conditions, drug mechanisms, or effectiveness ratings
SIDE_EFFECT_REJECT_KEYWORDS = [
    "disease", "disorder", "syndrome", "failure", "cancer",
    "deficiency", "infection", "decline", "dysfunction",
    "inhibitor", "inducer", "substrate", "agent",
    "possibly", "likely", "insufficient", "evidence",
    "effective", "ineffective",
    "diet", "exercise", "regimen", "therapy",
    "cyp", "p450",
]

# ── Terms that are NOT real supplement classes ────────────────
CLASS_REJECT_KEYWORDS = [
    "inhibitor", "inducer", "substrate", "cyp",
    "p450", "hypoglycemic", "antiplatelet",
    "hepatotoxic", "nephrotoxic", "cardiotoxic",
    "diets", "dietary", "salicylate", #new classes
]


#═══════════════════════════════════════════════════════════════
#  HELPERS
#═══════════════════════════════════════════════════════════════

def load_csv(filepath):
    print(f"Loading {filepath}...")
    df = pd.read_csv(filepath, encoding="utf-8").fillna("")
    print(f"   Rows: {len(df)}")
    return df


def parse_pipe_list(value):
    if not value or str(value).strip() == "":
        return []
    cleaned = []
    for p in str(value).split("|"):
        p = p.strip().lstrip("-*•").strip()
        p = re.sub(r'\(\s*[\d\s,]+\)', '', p).strip()
        if p and len(p) > 2:
            cleaned.append(p)
    return cleaned


def normalize(name):
    name = str(name).strip()
    name = re.sub(r'\s+', ' ', name)
    return name.title()


def is_real_side_effect(text):
    """Accept ONLY text containing a known clinical symptom"""
    tl = text.lower()
    # Reject if it contains condition/mechanism terms
    for kw in SIDE_EFFECT_REJECT_KEYWORDS:
        if kw in tl:
            return False
    # Accept only if it contains a known symptom
    for kw in SIDE_EFFECT_WHITELIST:
        if kw in tl:
            return True
    return False


def is_real_class(text):
    """Reject drug mechanism terms masquerading as classes"""
    tl = text.lower()
    for kw in CLASS_REJECT_KEYWORDS:
        if kw in tl:
            return False
    return True


def extract_side_effect_names(raw_text):
    """Extract clean side effect names using whitelist"""
    # Remove body system tag [Gastrointestinal]
    text = re.sub(r'^\[.+?\]\s*', '', str(raw_text)).strip()
    text = re.sub(r'\(\s*[\d\s,]+\)', '', text).strip()

    if not text or len(text) < 3:
        return []

    # Try to extract after "may cause", "causes", "include"
    m = re.search(
        r'(?:may cause|causes?|reported?|include[s]?|such as|including)'
        r'[:\s]+([^.]+)',
        text, re.IGNORECASE
    )
    if m:
        parts = re.split(r',\s*|\s+and\s+', m.group(1))
        results = []
        for p in parts:
            p = re.sub(r'\s+', ' ', p.strip().rstrip('.'))
            # Skip fragments starting with "and"
            if p.lower().startswith("and "):
                continue
            if 3 < len(p) < 60 and is_real_side_effect(p):
                results.append(normalize(p))
        return results[:5]

    # Use full text if short and valid
    text = text.rstrip('.')
    if 3 < len(text) < 60 and is_real_side_effect(text):
        return [normalize(text)]

    return []


def extract_drug_name(interaction_text):
    """Extract drug/interaction category name"""
    text = str(interaction_text).strip()

    if ':' in text:
        name = text.split(':')[0].strip()
    else:
        # Only accept if it clearly looks like a drug category
        tl = text.lower()
        drug_keywords = [
            "drug", "agent", "antibiotic", "antifungal", "antiviral",
            "anticoagulant", "antiplatelet", "antidiabetes", "antihypertensive",
            "antidepressant", "supplement", "herbs", "cyp", "warfarin",
            "substrate", "inhibitor", "inducer", "depressant", "stimulant",
            "cross-allerg", "perioperative", "potential",
        ]
        if not any(kw in tl for kw in drug_keywords):
            return None
        name = text

    # Clean
    name = re.sub(r'Interaction Rating.*', '', name, flags=re.IGNORECASE).strip()
    name = re.sub(r'\(\s*[\d\s,]+\)', '', name).strip()
    name = re.sub(r'\s+', ' ', name).strip()

    if len(name) < 3 or len(name) > 100:
        return None
    if name.count(' ') > 8:
        return None

    return normalize(name)


#═══════════════════════════════════════════════════════════════
#  BUILD GRAPH
#═══════════════════════════════════════════════════════════════

def build_graph(df):
    print("\nBuilding Knowledge Graph...")
    G     = nx.DiGraph()
    stats = defaultdict(int)

    for _, row in df.iterrows():
        supp = normalize(row.get("supplement_name", ""))
        if not supp:
            continue

        # ── Supplement node ───────────────────────────────────
        G.add_node(supp,
            node_type     = NODE_SUPPLEMENT,
            scientific    = str(row.get("scientific_name", "")).strip(),
            also_known_as = str(row.get("also_known_as", "")).strip(),
            source_url    = str(row.get("source_url", "")).strip(),
            overview      = str(row.get("overview", ""))[:300].strip(),
            pregnancy     = str(row.get("pregnancy_safety", "")).strip(),
            last_reviewed = str(row.get("last_reviewed", "")).strip()
        )
        stats["supplements"] += 1

        # ── TREATS ───────────────────────────────────────────
        for ind in parse_pipe_list(row.get("therapeutic_indications", "")):
            cond = normalize(re.sub(r'[^\w\s&/()-]', '', ind).strip())
            if not cond or len(cond) < 2:
                continue
            if not G.has_node(cond):
                G.add_node(cond, node_type=NODE_CONDITION)
                stats["condition_nodes"] += 1
            if not G.has_edge(supp, cond):
                G.add_edge(supp, cond, edge_type=EDGE_TREATS, weight=1.0)
                stats["treats_edges"] += 1

        # ── CAUSES (strict whitelist) ─────────────────────────
        for raw in parse_pipe_list(row.get("side_effects", "")):
            for ename in extract_side_effect_names(raw):
                if not ename:
                    continue
                if not G.has_node(ename):
                    G.add_node(ename, node_type=NODE_SIDE_EFFECT)
                    stats["side_effect_nodes"] += 1
                if not G.has_edge(supp, ename):
                    G.add_edge(supp, ename, edge_type=EDGE_CAUSES, weight=0.8)
                    stats["causes_edges"] += 1

        # ── INTERACTS_WITH ────────────────────────────────────
        for inter in parse_pipe_list(row.get("drug_interactions", "")):
            drug = extract_drug_name(inter)
            if not drug:
                continue
            if not G.has_node(drug):
                G.add_node(drug, node_type=NODE_DRUG)
                stats["drug_nodes"] += 1
            if not G.has_edge(supp, drug):
                G.add_edge(supp, drug,
                           edge_type=EDGE_INTERACTS_WITH,
                           weight=0.9,
                           description=str(inter)[:200])
                stats["interacts_edges"] += 1

        # ── BELONGS_TO (reject mechanism terms) ──────────────
        for cls in parse_pipe_list(row.get("classes", "")):
            cls_clean = normalize(cls.strip())
            if not cls_clean or len(cls_clean) < 2:
                continue
            if not is_real_class(cls_clean):
                continue  # skip drug mechanism terms
            if not G.has_node(cls_clean):
                G.add_node(cls_clean, node_type=NODE_CLASS)
                stats["class_nodes"] += 1
            if not G.has_edge(supp, cls_clean):
                G.add_edge(supp, cls_clean,
                           edge_type=EDGE_BELONGS_TO, weight=1.0)
                stats["belongs_edges"] += 1

    # ── SIMILAR_TO ────────────────────────────────────────────
    # IMPORTANT: Exclude broad conditions from similarity
    # Only use SPECIFIC conditions to find truly similar supplements
    print("   Computing similarity (excluding broad conditions, Jaccard > 0.6)...")

    supp_specific_conds = {}
    for node, attrs in G.nodes(data=True):
        if attrs.get("node_type") == NODE_SUPPLEMENT:
            specific = set(
                n for n in G.successors(node)
                if G.nodes[n].get("node_type") == NODE_CONDITION
                and n.lower() not in BROAD_CONDITIONS_EXCLUDE_FROM_SIMILARITY
            )
            if len(specific) >= 2:
                supp_specific_conds[node] = specific

    supp_list = list(supp_specific_conds.keys())
    sim_count = 0

    for i in range(len(supp_list)):
        for j in range(i + 1, len(supp_list)):
            s1, s2     = supp_list[i], supp_list[j]
            set1, set2 = supp_specific_conds[s1], supp_specific_conds[s2]
            shared     = set1 & set2
            union      = set1 | set2
            jaccard    = len(shared) / len(union) if union else 0

            # Very strict: Jaccard > 0.6 on specific conditions only
            if jaccard > 0.6 and len(shared) >= 2:
                G.add_edge(s1, s2, edge_type=EDGE_SIMILAR_TO,
                           weight=round(jaccard, 3),
                           shared_conditions=list(shared)[:5])
                G.add_edge(s2, s1, edge_type=EDGE_SIMILAR_TO,
                           weight=round(jaccard, 3),
                           shared_conditions=list(shared)[:5])
                sim_count += 1

    stats["similar_edges"] = sim_count
    print(f"   Similar pairs: {sim_count:,}")
    return G, stats


#═══════════════════════════════════════════════════════════════
#  EXPORT
#═══════════════════════════════════════════════════════════════

def export_json(G, filepath):
    print(f"\nSaving {filepath}...")
    data = {
        "nodes": [], "edges": [],
        "metadata": {
            "total_nodes": G.number_of_nodes(),
            "total_edges": G.number_of_edges(),
            "node_types": {}, "edge_types": {}
        }
    }
    ntc = defaultdict(int)
    for node, attrs in G.nodes(data=True):
        nd = {"id": node}
        nd.update(attrs)
        data["nodes"].append(nd)
        ntc[attrs.get("node_type", "unknown")] += 1
    data["metadata"]["node_types"] = dict(ntc)

    etc = defaultdict(int)
    for src, dst, attrs in G.edges(data=True):
        ed = {"source": src, "target": dst}
        ed.update(attrs)
        if "shared_conditions" in ed:
            ed["shared_conditions"] = list(ed["shared_conditions"])
        data["edges"].append(ed)
        etc[attrs.get("edge_type", "unknown")] += 1
    data["metadata"]["edge_types"] = dict(etc)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"   Done: {G.number_of_nodes():,} nodes | {G.number_of_edges():,} edges")


def export_graphml(G, filepath):
    print(f"Saving {filepath}...")
    G2 = G.copy()
    for n, attrs in G2.nodes(data=True):
        for k, v in attrs.items():
            if not isinstance(v, (str, int, float)):
                G2.nodes[n][k] = str(v)
    for s, d, attrs in G2.edges(data=True):
        for k, v in attrs.items():
            if isinstance(v, list):
                G2[s][d][k] = ", ".join(str(x) for x in v)
            elif not isinstance(v, (str, int, float)):
                G2[s][d][k] = str(v)
    nx.write_graphml(G2, filepath)
    print("   Done")


def save_stats(G, stats, filepath):
    def top_by(node_type, metric="in_degree", n=15):
        nodes = [nd for nd, d in G.nodes(data=True)
                 if d.get("node_type") == node_type]
        fn = G.in_degree if metric == "in_degree" else G.degree
        return sorted(nodes, key=lambda x: fn(x), reverse=True)[:n]

    lines = [
        "=" * 60,
        "  SupplementsRx Knowledge Graph — Statistics",
        "=" * 60, "",
        f"Total Nodes : {G.number_of_nodes():,}",
        f"Total Edges : {G.number_of_edges():,}", "",
        "NODE BREAKDOWN:",
        f"  Supplements  : {stats['supplements']:,}",
        f"  Conditions   : {stats['condition_nodes']:,}",
        f"  Side Effects : {stats['side_effect_nodes']:,}",
        f"  Drugs        : {stats['drug_nodes']:,}",
        f"  Classes      : {stats['class_nodes']:,}", "",
        "EDGE BREAKDOWN:",
        f"  TREATS         : {stats['treats_edges']:,}",
        f"  CAUSES         : {stats['causes_edges']:,}",
        f"  INTERACTS_WITH : {stats['interacts_edges']:,}",
        f"  BELONGS_TO     : {stats['belongs_edges']:,}",
        f"  SIMILAR_TO     : {stats['similar_edges']*2:,}", "",
        "TOP 20 MOST CONNECTED SUPPLEMENTS:",
    ]
    for s in top_by(NODE_SUPPLEMENT, "degree", 20):
        lines.append(f"  {s:45s} degree={G.degree(s)}")

    lines += ["", "TOP 15 CONDITIONS (most supplements treat):"]
    for c in top_by(NODE_CONDITION, n=15):
        lines.append(f"  {c:45s} <- {G.in_degree(c)} supplements")

    lines += ["", "TOP 15 SIDE EFFECTS (most common):"]
    for s in top_by(NODE_SIDE_EFFECT, n=15):
        lines.append(f"  {s:45s} <- {G.in_degree(s)} supplements")

    lines += ["", "TOP 10 DRUG INTERACTIONS:"]
    for d in top_by(NODE_DRUG, n=10):
        lines.append(f"  {d:45s} <- {G.in_degree(d)} supplements")

    lines += ["", "TOP 10 SUPPLEMENT CLASSES:"]
    for c in top_by(NODE_CLASS, n=10):
        lines.append(f"  {c:45s} <- {G.in_degree(c)} supplements")

    text = "\n".join(lines)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(text)
    print("\n" + text)


def visualize_sample(G, filepath):
    print(f"\nCreating visualization...")
    sample = ["Ashwagandha", "Turmeric", "Ginger", "Ginseng",
              "Melatonin", "Vitamin D", "Magnesium", "Echinacea",
              "Valerian", "St. John'S Wort", "Garlic", "Zinc"]
    sample = [s for s in sample if G.has_node(s)]

    if not sample:
        supps = [(n, G.degree(n)) for n, d in G.nodes(data=True)
                 if d.get("node_type") == NODE_SUPPLEMENT]
        sample = [n for n, _ in sorted(supps, key=lambda x: x[1],
                  reverse=True)[:12]]

    nodes = set(sample)
    for s in sample:
        for nb in G.successors(s):
            if G.nodes[nb].get("node_type") in [NODE_CONDITION, NODE_CLASS]:
                nodes.add(nb)

    sub = G.subgraph(nodes).copy()
    colors = {
        NODE_SUPPLEMENT: "#4A90D9", NODE_CONDITION: "#7ED321",
        NODE_SIDE_EFFECT: "#F5A623", NODE_DRUG: "#D0021B",
        NODE_CLASS: "#9B59B6",
    }
    nc = [colors.get(sub.nodes[n].get("node_type"), "#CCCCCC") for n in sub.nodes()]
    ns = [900 if sub.nodes[n].get("node_type") == NODE_SUPPLEMENT else 450
          for n in sub.nodes()]

    plt.figure(figsize=(18, 13))
    plt.title(
        "SupplementsRx — Knowledge Graph Preview\n"
        "Blue=Supplement | Green=Condition | Purple=Class",
        fontsize=13, fontweight='bold'
    )
    pos = nx.spring_layout(sub, seed=42, k=2.5)
    nx.draw_networkx_nodes(sub, pos, node_color=nc, node_size=ns, alpha=0.9)
    nx.draw_networkx_labels(sub, pos, font_size=7, font_weight='bold')
    nx.draw_networkx_edges(sub, pos, arrows=True, arrowsize=12,
                           alpha=0.4, edge_color="#999999")
    from matplotlib.patches import Patch
    legend = [
        Patch(color="#4A90D9", label="Supplement"),
        Patch(color="#7ED321", label="Condition"),
        Patch(color="#9B59B6", label="Class"),
        Patch(color="#F5A623", label="Side Effect"),
        Patch(color="#D0021B", label="Drug"),
    ]
    plt.legend(handles=legend, loc="upper left", fontsize=9)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved {filepath}")


#═══════════════════════════════════════════════════════════════
#  MAIN
#═══════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  Phase 2 — Knowledge Graph Builder (v4 Final)")
    print("=" * 60)

    df       = load_csv(INPUT_CSV)
    G, stats = build_graph(df)
    export_json(G, OUT_JSON)
    export_graphml(G, OUT_GRAPHML)
    save_stats(G, stats, OUT_STATS)
    visualize_sample(G, OUT_PLOT)

    print(f"\n{'='*60}")
    print("  PHASE 2 COMPLETE!")
    print(f"  Total Nodes : {G.number_of_nodes():,}")
    print(f"  Total Edges : {G.number_of_edges():,}")
    print("  graph.json        -> Phase 3 Clustering")
    print("  graph.graphml     -> Open in Gephi")
    print("  graph_stats.txt   -> Summary")
    print("  graph_preview.png -> Visualization")
    print("  Next -> python phase3_clustering.py")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()