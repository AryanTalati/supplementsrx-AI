"""
=============================================================
  Phase 4 — Step 1: Generate Supplement Embeddings
  SupplementsRx AI Guidance System

  Uses OpenAI text-embedding-3-small for embeddings.
  Falls back to all-MiniLM-L6-v2 if no OpenAI key.

  OUTPUT:
    supplement_embeddings.npy    — embedding vectors
    supplement_names.json        — supplement names
    supplement_metadata.json     — full metadata
    faiss_index.bin              — FAISS index
    embedding_config.json        — model config for Phase 4

  HOW TO RUN:
    pip install openai faiss-cpu numpy neo4j tqdm
    set OPENAI_API_KEY=sk-...
    python generate_embeddings.py
=============================================================
"""

import os
import json
import time
import numpy as np
from neo4j import GraphDatabase
from tqdm import tqdm

try:
    import faiss
except ImportError:
    os.system("pip install faiss-cpu -q")
    import faiss

NEO4J_URI      = "bolt://localhost:7687"
NEO4J_USER     = "neo4j"
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "")

OPENAI_API_KEY        = os.environ.get("OPENAI_API_KEY", "")
EMBEDDING_MODEL_OPENAI = "text-embedding-3-small"
EMBEDDING_MODEL_LOCAL  = "all-MiniLM-L6-v2"

OUT_EMBEDDINGS = "supplement_embeddings.npy"
OUT_NAMES      = "supplement_names.json"
OUT_METADATA   = "supplement_metadata.json"
OUT_FAISS      = "faiss_index.bin"
OUT_CONFIG     = "embedding_config.json"


def fetch_supplements():
    print("\n📊 Fetching supplements from Neo4j...")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    with driver.session() as session:
        result = list(session.run("""
            MATCH (s:Supplement)
            OPTIONAL MATCH (s)-[:TREATS]->(c:Condition)
            OPTIONAL MATCH (s)-[:BELONGS_TO_CLASS]->(sc:SupplementClass)
            OPTIONAL MATCH (s)-[:INTERACTS_WITH]->(d:Drug)
            OPTIONAL MATCH (s)-[:HAS_SIDE_EFFECT]->(se:SideEffect)
            RETURN s.name               AS name,
                   s.overview           AS overview,
                   s.pregnancy          AS pregnancy,
                   s.scientific         AS scientific,
                   s.louvain_community  AS community,
                   collect(DISTINCT c.name)[0..8]  AS conditions,
                   collect(DISTINCT sc.name)[0..3] AS classes,
                   collect(DISTINCT d.name)[0..5]  AS drugs,
                   collect(DISTINCT se.name)[0..5] AS side_effects
            ORDER BY s.name
        """))
    driver.close()
    print(f"   Fetched {len(result):,} supplements")

    supplements = []
    for record in result:
        name         = record["name"] or ""
        overview     = record["overview"] or ""
        scientific   = record["scientific"] or ""
        pregnancy    = record["pregnancy"] or ""
        community    = record["community"]
        conditions   = record["conditions"] or []
        classes      = record["classes"] or []
        drugs        = record["drugs"] or []
        side_effects = record["side_effects"] or []

        parts = [f"Supplement: {name}"]
        if scientific:
            parts.append(f"Also known as: {scientific}")
        if conditions:
            parts.append(f"Used for: {', '.join(conditions)}")
        if classes:
            parts.append(f"Type: {', '.join(classes)}")
        if drugs:
            parts.append(f"Interacts with: {', '.join(drugs)}")
        if side_effects:
            parts.append(f"Side effects: {', '.join(side_effects)}")
        if overview:
            parts.append(f"Overview: {overview[:300]}")
        if pregnancy:
            parts.append(f"Pregnancy: {pregnancy[:100]}")

        supplements.append({
            "name": name, "text": ". ".join(parts),
            "overview": overview, "scientific": scientific,
            "pregnancy": pregnancy, "community": community,
            "conditions": conditions, "classes": classes,
            "drugs": drugs, "side_effects": side_effects,
        })
    return supplements


def embed_openai(texts):
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    all_embeddings = []
    batch_size = 100

    for i in tqdm(range(0, len(texts), batch_size), desc="OpenAI embeddings"):
        batch = texts[i:i+batch_size]
        try:
            response = client.embeddings.create(
                model=EMBEDDING_MODEL_OPENAI, input=batch
            )
            all_embeddings.extend([item.embedding for item in response.data])
        except Exception as e:
            print(f"\n   Error: {e} — retrying in 5s...")
            time.sleep(5)
            response = client.embeddings.create(
                model=EMBEDDING_MODEL_OPENAI, input=batch
            )
            all_embeddings.extend([item.embedding for item in response.data])
        time.sleep(0.1)

    return np.array(all_embeddings, dtype=np.float32)


def embed_local(texts):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        os.system("pip install sentence-transformers -q")
        from sentence_transformers import SentenceTransformer
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   Device: {device}")
    model = SentenceTransformer(EMBEDDING_MODEL_LOCAL)
    return model.encode(
        texts, show_progress_bar=True, batch_size=128,
        device=device, convert_to_numpy=True,
        normalize_embeddings=True
    ).astype(np.float32)


def build_faiss_index(embeddings):
    print("\n🔍 Building FAISS index...")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normalized = embeddings / norms
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(normalized)
    print(f"   {index.ntotal:,} vectors, {embeddings.shape[1]} dims")
    return index, normalized


def test_search(supplements, index, use_openai):
    print("\n🧪 Testing semantic search...")
    queries = [
        "supplements for anxiety and stress",
        "natural sleep aid",
        "joint pain relief",
        "heart health cardiovascular",
        "blood sugar diabetes",
    ]
    names = [s["name"] for s in supplements]

    if use_openai:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        for q in queries:
            resp  = client.embeddings.create(
                model=EMBEDDING_MODEL_OPENAI, input=[q]
            )
            q_emb = np.array([resp.data[0].embedding], dtype=np.float32)
            q_emb = q_emb / np.linalg.norm(q_emb)
            D, I  = index.search(q_emb, 5)
            print(f"\n   '{q}'")
            print(f"   → {', '.join(names[i] for i in I[0] if i < len(names))}")
    else:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(EMBEDDING_MODEL_LOCAL)
        for q in queries:
            q_emb = model.encode([q], normalize_embeddings=True)
            D, I  = index.search(q_emb.astype(np.float32), 5)
            print(f"\n   '{q}'")
            print(f"   → {', '.join(names[i] for i in I[0] if i < len(names))}")


def main():
    print("=" * 60)
    print("  Generate Embeddings — Phase 4 Step 1")
    print("=" * 60)

    use_openai = bool(OPENAI_API_KEY)
    if use_openai:
        print(f"\n✅ OpenAI key detected — using {EMBEDDING_MODEL_OPENAI}")
        print("   Cost: ~3,841 supplements × ~150 tokens ≈ $0.03 total")
    else:
        print(f"\n⚠️  No key — falling back to {EMBEDDING_MODEL_LOCAL}")

    supplements = fetch_supplements()
    if not supplements:
        print("❌ No supplements found! Run migrate_to_neo4j.py first.")
        return

    texts = [s["text"] for s in supplements]
    print(f"\n📐 Embedding {len(texts):,} supplements...")

    embeddings = embed_openai(texts) if use_openai else embed_local(texts)
    print(f"   Shape: {embeddings.shape}")

    index, normalized = build_faiss_index(embeddings)

    print("\n💾 Saving...")
    np.save(OUT_EMBEDDINGS, embeddings)

    with open(OUT_NAMES, "w", encoding="utf-8") as f:
        json.dump([s["name"] for s in supplements], f, indent=2)

    with open(OUT_METADATA, "w", encoding="utf-8") as f:
        json.dump([{
            "name":        s["name"],
            "community":   s["community"],
            "conditions":  s["conditions"],
            "classes":     s["classes"],
            "drugs":       s["drugs"],
            "side_effects": s["side_effects"],
            "overview":    (s["overview"] or "")[:300],
            "pregnancy":   s["pregnancy"],
            "scientific":  s["scientific"],
        } for s in supplements], f, indent=2, ensure_ascii=False)

    faiss.write_index(index, OUT_FAISS)

    with open(OUT_CONFIG, "w") as f:
        json.dump({
            "embedding_model":   EMBEDDING_MODEL_OPENAI if use_openai else EMBEDDING_MODEL_LOCAL,
            "use_openai":        use_openai,
            "dimensions":        int(embeddings.shape[1]),
            "total_supplements": len(supplements),
        }, f, indent=2)

    for f in [OUT_EMBEDDINGS, OUT_NAMES, OUT_METADATA, OUT_FAISS, OUT_CONFIG]:
        print(f"   ✅ {f}")

    test_search(supplements, index, use_openai)

    print(f"""
{'='*60}
  ✅ DONE!

  Model      : {EMBEDDING_MODEL_OPENAI if use_openai else EMBEDDING_MODEL_LOCAL}
  Supplements: {len(supplements):,}
  Dimensions : {embeddings.shape[1]}

  Next → python phase4_langgraph.py
{'='*60}
""")


if __name__ == "__main__":
    main()
