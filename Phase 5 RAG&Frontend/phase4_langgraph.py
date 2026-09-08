"""
=============================================================
  Phase 4 — LangGraph Pipeline
  SupplementsRx AI Guidance System

  Implements the full Global-to-Local Graph Intelligence
  architecture from the proposal:

  1. Intent Classification  — Global / Local / Out-of-Scope
  2. Global Path            — Community summary retrieval
  3. Local Path             — FAISS + 2-hop Neo4j traversal
  4. Safety Intercept       — Interactions, pregnancy, dosage
  5. GPT-4o Response        — Grounded, cited, confident

  HOW TO RUN:
    pip install openai langgraph neo4j faiss-cpu numpy
    set OPENAI_API_KEY=sk-...
    python phase4_langgraph.py

  Then chat interactively or import ask() for Flask API.
=============================================================
"""

import os
import json
import re
import numpy as np
from typing import TypedDict, Literal, List, Optional
from neo4j import GraphDatabase

try:
    import faiss
except ImportError:
    os.system("pip install faiss-cpu -q")
    import faiss

try:
    from langgraph.graph import StateGraph, END
except ImportError:
    os.system("pip install langgraph -q")
    from langgraph.graph import StateGraph, END

from openai import OpenAI

# ── Config ────────────────────────────────────────────────────
NEO4J_URI      = "neo4j+s://8976bb7e.databases.neo4j.io"
NEO4J_USER     = "8976bb7e"
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

FAISS_INDEX    = "faiss_index.bin"
NAMES_FILE     = "supplement_names.json"
METADATA_FILE  = "supplement_metadata.json"
CONFIG_FILE    = "embedding_config.json"
COMMUNITIES_FILE = "neo4j_communities.json"

TOP_K_FAISS    = 10   # top K from vector search
TOP_K_RERANK   = 6    # top K after reranking
MAX_HOPS       = 2    # Neo4j graph traversal hops

client = OpenAI(api_key=OPENAI_API_KEY)


#═══════════════════════════════════════════════════════════════
#  STATE DEFINITION
#═══════════════════════════════════════════════════════════════

class AgentState(TypedDict):
    query:          str
    intent:         Optional[str]           # global / local / oos
    entities:       Optional[List[str]]     # extracted supplement/drug names
    faiss_results:  Optional[List[dict]]    # vector search results
    graph_results:  Optional[List[dict]]    # Neo4j subgraph results
    community_summaries: Optional[List[str]]
    safety_alerts:  Optional[List[str]]
    evidence:       Optional[str]           # fused evidence block
    response:       Optional[str]           # final answer
    citations:      Optional[List[str]]
    confidence:     Optional[str]           # High / Medium / Low


#═══════════════════════════════════════════════════════════════
#  LOAD RESOURCES
#═══════════════════════════════════════════════════════════════

def load_resources():
    print("Loading resources...")

    # FAISS index
    index    = faiss.read_index(FAISS_INDEX)
    with open(NAMES_FILE, encoding="utf-8") as f:
        names = json.load(f)
    with open(METADATA_FILE, encoding="utf-8") as f:
        metadata = json.load(f)
    with open(CONFIG_FILE) as f:
        config = json.load(f)

    # Community assignments
    communities = {}
    if os.path.exists(COMMUNITIES_FILE):
        with open(COMMUNITIES_FILE) as f:
            communities = json.load(f)

    # Neo4j driver
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        driver.verify_connectivity()
    except Exception:
        print("❌ Neo4j is not running!")
        print("   Open Neo4j Desktop → click Start on supplementsrx database")
        exit()

    print(f"   FAISS: {index.ntotal:,} vectors")
    print(f"   Metadata: {len(metadata):,} supplements")
    print(f"   Embedding model: {config['embedding_model']}")
    print("   Ready!\n")

    return index, names, metadata, config, communities, driver


#═══════════════════════════════════════════════════════════════
#  NODE 1: INTENT CLASSIFICATION
#═══════════════════════════════════════════════════════════════

def classify_intent(state: AgentState) -> AgentState:
    """
    Classify query as Global, Local, or Out-of-Scope.
    Uses GPT-4o-mini (fast + cheap) for classification.
    """
    query = state["query"]

    prompt = f"""You are an intent classifier for a supplement health AI.
Classify the query into exactly one of these categories:

GLOBAL  — broad questions about supplement categories, general safety,
          or population-level advice
          Examples: "What supplements help with heart health?"
                    "Are supplements safe during pregnancy?"
                    "What are adaptogens?"

LOCAL   — specific questions about a named supplement, drug,
          vitamin, mineral, herb, or ANY health-related substance
          including over-the-counter medications and pain relievers
          Examples: "Can I take Ashwagandha with Metformin?"
                    "What is the dose of Vitamin D?"
                    "Is St. John's Wort safe?"

OOS     — out of scope, not related to supplements or health
          Examples: "What is the weather today?"
                    "Write me a poem"
                    "Who is the president?"

Query: "{query}"

Respond with ONLY one word: GLOBAL, LOCAL, or OOS"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=10
    )

    intent_raw = response.choices[0].message.content.strip().upper()
    if "GLOBAL" in intent_raw:
        intent = "global"
    elif "LOCAL" in intent_raw:
        intent = "local"
    else:
        intent = "oos"

    print(f"   [Intent] {intent.upper()}")
    return {**state, "intent": intent}


#═══════════════════════════════════════════════════════════════
#  NODE 2: ENTITY EXTRACTION
#═══════════════════════════════════════════════════════════════

def extract_entities(state: AgentState) -> AgentState:
    """Extract supplement and drug names from query"""
    query = state["query"]

    prompt = f"""Extract supplement, herb, vitamin, mineral, and drug names from this query.
Common foods used as supplements (turmeric, ginger, garlic) should be included.
Do NOT include generic foods like milk, water, food, juice unless they are supplements.
Return ONLY a JSON array. Examples: ["Turmeric"] or ["Vitamin D", "Calcium"] or []

Query: "{query}"
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=100
        )
        raw     = response.choices[0].message.content.strip()
        raw     = re.sub(r'```json|```', '', raw).strip()
        entities = json.loads(raw)
        if not isinstance(entities, list):
            entities = []
    except Exception:
        entities = []

    print(f"   [Entities] {entities}")
    return {**state, "entities": entities}


#═══════════════════════════════════════════════════════════════
#  NODE 3A: GLOBAL PATH — Community Summary Retrieval
#═══════════════════════════════════════════════════════════════

def global_search(state: AgentState,
                  index, names, metadata, config, driver) -> AgentState:
    """
    For broad queries: retrieve community summaries via
    FAISS similarity search on query embedding.
    """
    query = state["query"]
    entities = state.get("entities") or []

    # Get query embedding
    # Expand query with entities for better embedding match
    
    expanded_query = query
    if entities:
        expanded_query = f"{query} {' '.join(entities)} supplement benefits"
    
    if config["use_openai"]:
        resp = client.embeddings.create(
            model=config["embedding_model"], input=[expanded_query] 
        )
        q_emb = np.array([resp.data[0].embedding], dtype=np.float32)
    else:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(config["embedding_model"])
        q_emb = model.encode([expanded_query], normalize_embeddings=True)

    q_emb = q_emb / np.linalg.norm(q_emb)
    D, I  = index.search(q_emb.astype(np.float32), TOP_K_FAISS)

    # Get community IDs for top results
    top_communities = set()
    faiss_results   = []
    for i, score in zip(I[0], D[0]):
        if i < len(metadata):
            m = metadata[i]
            faiss_results.append({
                "name":       m["name"],
                "score":      float(score),
                "conditions": m.get("conditions", []),
                "community":  m.get("community"),
            })
            if m.get("community") is not None:
                top_communities.add(m["community"])

    # Fetch community summaries from Neo4j
    # Fetch community summaries from supplement data directly
    summaries = []
    with driver.session() as session:
        for comm_id in list(top_communities)[:5]:
            result = list(session.run("""
                MATCH (s:Supplement {louvain_community: $cid})-[:TREATS]->(c:Condition)
                WITH s.louvain_community AS community,
                    collect(DISTINCT c.name)[0..3] AS conditions,
                    collect(DISTINCT s.name)[0..3] AS members
                RETURN community, conditions, members
                LIMIT 1
            """, cid=comm_id))
            if result:
                r = result[0]
                summary = (
                    f"Community {r['community']} | "
                    f"Conditions: {', '.join(r['conditions'] or [])} | "
                    f"Examples: {', '.join(r['members'] or [])}"
                )
                summaries.append(summary)

    print(f"   [Global] {len(faiss_results)} FAISS results, "
          f"{len(summaries)} community summaries")
    return {**state,
            "faiss_results": faiss_results,
            "community_summaries": summaries}


#═══════════════════════════════════════════════════════════════
#  NODE 3B: LOCAL PATH — FAISS + Neo4j 2-hop Traversal
#═══════════════════════════════════════════════════════════════

def local_search(state: AgentState,
                 index, names, metadata, config, driver) -> AgentState:
    """
    For specific queries: hybrid FAISS + Neo4j traversal.
    """
    query    = state["query"]
    entities = state.get("entities", [])

    # ── FAISS vector search ───────────────────────────────────
    if config["use_openai"]:
        resp  = client.embeddings.create(
            model=config["embedding_model"], input=[query]
        )
        q_emb = np.array([resp.data[0].embedding], dtype=np.float32)
    else:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(config["embedding_model"])
        q_emb = model.encode([query], normalize_embeddings=True)

    q_emb = q_emb / np.linalg.norm(q_emb)
    D, I  = index.search(q_emb.astype(np.float32), TOP_K_FAISS)

    faiss_results = []
    for i, score in zip(I[0], D[0]):
        if i < len(metadata):
            m = metadata[i]
            faiss_results.append({
                "name":        m["name"],
                "score":       float(score),
                "conditions":  m.get("conditions", []),
                "classes":     m.get("classes", []),
                "drugs":       m.get("drugs", []),
                "side_effects": m.get("side_effects", []),
                "overview":    m.get("overview", ""),
                "pregnancy":   m.get("pregnancy", ""),
            })

    faiss_results = [r for r in faiss_results
                     if (r.get("overview") or r.get("conditions"))]
                     #and r.get("score", 0) > 0.3]

    # ── Neo4j 2-hop subgraph traversal ────────────────────────
    graph_results = []
    search_names  = entities if entities else [r["name"] for r in faiss_results[:3]]

    with driver.session() as session:
        for entity_name in search_names[:5]:
            result = list(session.run("""
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
            """, name=entity_name))

            for r in result:
                graph_results.append({
                    "name":        r["name"],
                    "overview":    r["overview"] or "",
                    "pregnancy":   r["pregnancy"] or "",
                    "scientific":  r["scientific"] or "",
                    "conditions":  r["conditions"] or [],
                    "drugs":       r["drugs"] or [],
                    "side_effects": r["side_effects"] or [],
                    "classes":     r["classes"] or [],
                    "similar":     r["similar"] or [],
                    "source":      "neo4j_2hop"
                })

    print(f"   [Local] {len(faiss_results)} FAISS + "
          f"{len(graph_results)} Neo4j graph results")
    
    # print(f"   [FAISS top 3] {[r['name'] for r in faiss_results[:3]]}") #Debug
    # for r in faiss_results[:3]: #Debug
    #     print(f"      {r['name']}: conditions={r.get('conditions', [])[:3]}") #Debug

    return {**state,
            "faiss_results": faiss_results,
            "graph_results": graph_results}

    


#═══════════════════════════════════════════════════════════════
#  NODE 4: SAFETY INTERCEPT
#═══════════════════════════════════════════════════════════════

PREGNANCY_RISK_TERMS = [
    "avoid", "unsafe", "contraindicated", "not recommended",
    "category c", "category d", "category x", "do not use",
    "risk", "harmful", "danger"
]

MAJOR_INTERACTION_DRUGS = {
    "warfarin", "coumadin", "blood thinner", "anticoagulant",
    "ssri", "antidepressant", "maoi",
    "cyclosporine", "tacrolimus", "immunosuppressant",
    "digoxin", "lithium", "phenytoin",
    "insulin", "metformin", "antidiabetes",
    "chemotherapy", "cancer drug",
}


def safety_intercept(state: AgentState) -> AgentState:
    """
    Rule-based safety checks on retrieved evidence.
    Checks: drug interactions, pregnancy risk, high dose warnings.
    """
    alerts       = []
    graph_results = state.get("graph_results", []) or []
    faiss_results = state.get("faiss_results", []) or []
    query         = state["query"].lower()

    for result in graph_results:
        name  = result.get("name", "")
        drugs = result.get("drugs", [])

        # Check drug interactions
        for drug in drugs:
            drug_lower = drug.lower()
            if any(major in drug_lower for major in MAJOR_INTERACTION_DRUGS):
                alerts.append(
                    f"⚠️ INTERACTION ALERT: {name} may interact with "
                    f"{drug}. Consult a healthcare provider before combining."
                )

        # Check pregnancy safety
        pregnancy = result.get("pregnancy", "").lower()
        if pregnancy and any(term in pregnancy for term in PREGNANCY_RISK_TERMS):
            alerts.append(
                f"🤰 PREGNANCY WARNING: {name} — {result['pregnancy'][:150]}"
            )

    # Check if query mentions pregnancy
    if any(term in query for term in ["pregnan", "breastfeed", "nursing", "trimester"]):
        query_entities = state.get("entities") or []
        for result in graph_results + (faiss_results or []):
            name     = result.get("name", "")
            pregnancy = result.get("pregnancy", "")
            already   = any(name in a for a in alerts)

            # Only fire for supplements explicitly in the query
            name_in_query = (
                name.lower() in query.lower() or
                any(e.lower() in name.lower() or 
                    name.lower() in e.lower() 
                    for e in query_entities)
            )

            if not already and name_in_query:
                if pregnancy:
                    alerts.append(
                        f"🤰 PREGNANCY INFO for {name}: {pregnancy[:200]}"
                    )
                else:
                    alerts.append(
                        f"🤰 PREGNANCY NOTE: Always consult your doctor before "
                        f"taking {name} during pregnancy."
                    )


    if alerts:
        print(f"   [Safety] {len(alerts)} alerts triggered")
    else:
        print("   [Safety] No alerts")

    # Deduplicate — keep unique interaction types only
    seen = set()
    deduped = []
    for alert in alerts:
        key = alert.split("may interact with")[-1].strip() if "may interact with" in alert else alert[:80]
        if key not in seen:
            seen.add(key)
            deduped.append(alert)

    return {**state, "safety_alerts": deduped}
    #return {**state, "safety_alerts": alerts}


#═══════════════════════════════════════════════════════════════
#  NODE 5: EVIDENCE FUSION
#═══════════════════════════════════════════════════════════════

def fuse_evidence(state: AgentState) -> AgentState:
    """Combine FAISS + graph results into unified evidence block"""
    parts          = []
    graph_results  = state.get("graph_results", []) or []
    faiss_results  = state.get("faiss_results", []) or []
    community_sums = state.get("community_summaries", []) or []
    safety_alerts  = state.get("safety_alerts", []) or []

    graph_has_content = any(
    r.get("overview") or r.get("conditions") or r.get("drugs")
    for r in graph_results
    )

    # Add safety alerts first
    if safety_alerts:
        parts.append("=== SAFETY ALERTS ===")
        for alert in safety_alerts:
            parts.append(alert)
        parts.append("")

    # Add community summaries (for global queries)
    if community_sums:
        parts.append("=== COMMUNITY SUMMARIES ===")
        for s in community_sums[:3]:
            parts.append(f"• {s}")
        parts.append("")

    # Add Neo4j graph evidence (most detailed)
    if graph_results:
        parts.append("=== KNOWLEDGE GRAPH EVIDENCE ===")
        for r in graph_results[:TOP_K_RERANK]:
            name = r.get("name", "")
            parts.append(f"[{name}]")
            if r.get("overview"):
                parts.append(f"Overview: {r['overview'][:300]}")
            if r.get("conditions"):
                parts.append(f"Treats: {', '.join(r['conditions'][:6])}")
            if r.get("drugs"):
                parts.append(f"Interacts with: {', '.join(r['drugs'][:5])}")
            if r.get("side_effects"):
                parts.append(f"Side effects: {', '.join(r['side_effects'][:4])}")
            if r.get("classes"):
                parts.append(f"Class: {', '.join(r['classes'])}")
            if r.get("pregnancy"):
                parts.append(f"Pregnancy: {r['pregnancy'][:150]}")
            if r.get("similar"):
                parts.append(f"Similar supplements: {', '.join(r['similar'][:4])}")
            parts.append("")

    # Add FAISS evidence (semantic matches)
    if faiss_results and (not graph_results or not graph_has_content):
        parts.append("=== SEMANTIC SEARCH RESULTS ===")
        for r in faiss_results[:TOP_K_RERANK]:
            parts.append(f"[{r['name']}] similarity={r['score']:.3f}")
            if r.get("conditions"):
                parts.append(f"  Treats: {', '.join(r['conditions'][:5])}")
            if r.get("overview"):
                parts.append(f"  {r['overview'][:200]}")
            parts.append("")

    # If no graph results, promote FAISS results to main evidence
    if not graph_results and faiss_results:
        parts.append("=== KNOWLEDGE GRAPH EVIDENCE ===")
        for r in faiss_results[:TOP_K_RERANK]:
            parts.append(f"[{r['name']}]")
            if r.get("conditions"):
                parts.append(f"Treats: {', '.join(r['conditions'][:6])}")
            if r.get("drugs"):
                parts.append(f"Interacts with: {', '.join(r['drugs'][:5])}")
            if r.get("overview"):
                parts.append(f"Overview: {r['overview'][:200]}")
            parts.append("")
    
    if not parts:
        parts.append("No specific evidence found for this query.")

    evidence = "\n".join(parts)


    # Compute confidence based on evidence quality
    if graph_results and graph_has_content:
        confidence = "High"
    elif faiss_results and len(faiss_results) >= 3:
        confidence = "Medium"
    else:
        confidence = "Low"

    print(f"   [Evidence] {len(parts)} lines, confidence={confidence}")
    return {**state, "evidence": evidence, "confidence": confidence}


#═══════════════════════════════════════════════════════════════
#  NODE 6: RESPONSE GENERATION
#═══════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are SupplementsRx AI — a warm, knowledgeable health advisor specialising in dietary supplements and natural health.

GOLDEN RULE: Answer ONLY from the HEALTH CONTEXT provided. If the context contains related supplements, use them to give a helpful answer. Only say you don't know if the context has nothing relevant at all.

STRICT RULES:
1. Answer ONLY from the HEALTH CONTEXT provided below.
   IMPORTANT — check the HEALTH CONTEXT carefully first:
   - If the HEALTH CONTEXT contains relevant information about the question → use it and answer warmly
   - If the HEALTH CONTEXT is empty OR contains no relevant information → then respond warmly:
     "Hmm, I don't have solid information on that one in my knowledge base — 
     it might be outside the supplement world, or just something I haven't 
     come across yet. Worth checking with your doctor or pharmacist!"
   Vary the decline response naturally each time.
   Never use your training knowledge to fill gaps when context is missing.
   Never sound robotic. Sound like a warm, honest friend.
2. Do NOT add [Source: ...] tags
3. Do NOT write "Confidence: High/Medium/Low"
4. Do NOT end with "This is not medical advice"
5. Always include safety alerts naturally in the flow
6. If safety alerts appear in evidence — weave them in conversationally
7. If the user's query contains unclear terms, typos, symbols, or ambiguous names
   that might refer to a supplement (e.g. "omega #", "vit d", "ash", "mag"):
   Ask a warm clarifying question instead of declining.
   Examples:
   * "Did you mean Omega-3? I have lots on that one!"
   * "Are you asking about Vitamin D? Just want to make sure!"
   * "Did you mean Ashwagandha? Ask me anything about it!"
   Keep it short, soft, warm, one sentence max. Never ask multiple questions at once.
8. Answer the specific question in the FIRST sentence. Never start with background.
   Get straight to the point, then add details after.
   "What is the RDA of Vitamin D?" → Start with "The RDA for Vitamin D is 600 IU..."
   NOT → "Vitamin D is an important nutrient that plays many roles..."

Let evidence quality shape your language:
- Strong evidence → speak directly: "Melatonin is well known for supporting sleep..."
- Moderate evidence → hedge gently: "There's decent evidence that valerian may help..."
- Limited evidence → be honest: "I don't have a lot of detail on that one..."

CORE RULES:
1. Read conversation history first — track the current topic
2. Answer exactly what was asked — nothing more
3. Never repeat yourself across messages
4. Explain like a warm doctor talking to a patient — plain language, relatable comparisons
5. Sound human — never say "I appreciate your question"
6. Weave safety naturally — no separate warning sections
7. Never invent facts or dosages
8. Keep responses under 150 words

When someone asks what helps with a condition — look at the context and name the relevant supplements warmly and specifically."""


def generate_response(state: AgentState) -> AgentState:
    """Generate final grounded response using GPT-4o"""
    query      = state["query"]
    evidence   = state.get("evidence", "")
    confidence = state.get("confidence", "Low")
    intent     = state.get("intent", "local")

    if intent == "oos":
        return {**state,
                "response":  "I'm a supplement health assistant and can only answer "
                             "questions about dietary supplements, vitamins, minerals, "
                             "and related health topics. Please ask me something related "
                             "to supplements!",
                "citations": [],
                "confidence": "N/A"}

    if not evidence.strip():
        return {**state,
                "response":  "I don't have enough verified information in my knowledge "
                             "base to answer this question accurately.",
                "citations": [],
                "confidence": "Low"}

    user_message = f"""EVIDENCE CONTEXT:
{evidence}

USER QUESTION: {query}

IMPORTANT: 
- If the EVIDENCE CONTEXT above contains information relevant to the question, USE IT to answer warmly and helpfully.
- If the query contains symbols like #, *, @, or is genuinely garbled text 
  (e.g. "omega #", "vit@min", "a$hwagandha"):
  ask a warm clarifying question like "Did you mean Omega-3?"
  
  However, common abbreviations should ALWAYS be answered directly without asking:
  "vit d" → Vitamin D, "vit c" → Vitamin C, "vit b" → Vitamin B,
  "mag" → Magnesium, "omega" → Omega-3, "ash" → Ashwagandha,
  "fish oil" → Fish Oil, "cod liver" → Cod Liver Oil
  Never ask for clarification on these — just answer them directly.
- Only if the evidence is completely empty or irrelevant, decline warmly using one of these varied responses (pick randomly, never repeat exact wording):
  * "Hmm, I don't have much on that one in my knowledge base. Try asking about a specific supplement like ashwagandha, melatonin, or fish oil!"
  * "That one's not coming up in my database — it might be worth rephrasing or asking about a related supplement?"
  * "I'm not finding that one — could you try a different name? For example, 'omega 3' or 'fish oil' instead."
  * "Not something I have solid data on right now. Ask me about a specific supplement and I'll do my best!"

Answer based strictly on the evidence above. Include all safety alerts."""

    # print(f"   [Debug Evidence]\n{evidence[:500]}") #Debug
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message}
        ],
        temperature=0.1,
        max_tokens=1000
    )

    answer = response.choices[0].message.content.strip()

    # Add confidence if not already in response
    if "Confidence:" not in answer:
        answer += f"\n\nConfidence: {confidence}"

    # Extract citations using regex
    citations = re.findall(r'\[Source:[^\]]+\]', answer)

    print(f"   [Response] {len(answer)} chars, {len(citations)} citations")
    return {**state,
            "response":   answer,
            "citations":  citations,
            "confidence": confidence}


#═══════════════════════════════════════════════════════════════
#  ROUTING FUNCTIONS
#═══════════════════════════════════════════════════════════════

def route_intent(state: AgentState) -> Literal["global", "local", "oos"]:
    return state.get("intent", "local")


def route_after_intent(state: AgentState) -> str:
    intent = state.get("intent", "local")
    if intent == "oos":
        return "generate_response"
    elif intent == "global":
        return "global_search"
    else:
        return "extract_entities"


#═══════════════════════════════════════════════════════════════
#  BUILD LANGGRAPH
#═══════════════════════════════════════════════════════════════

def build_graph(index, names, metadata, config, communities, driver):
    """Build the LangGraph state machine"""

    # Create closures that capture loaded resources
    def _global_search(state):
        return global_search(state, index, names, metadata, config, driver)

    def _local_search(state):
        return local_search(state, index, names, metadata, config, driver)

    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("classify_intent",  classify_intent)
    workflow.add_node("extract_entities", extract_entities)
    workflow.add_node("global_search",    _global_search)
    workflow.add_node("local_search",     _local_search)
    workflow.add_node("safety_intercept", safety_intercept)
    workflow.add_node("fuse_evidence",    fuse_evidence)
    workflow.add_node("generate_response", generate_response)

    # Entry point
    workflow.set_entry_point("classify_intent")

    # Routing after classification
    workflow.add_conditional_edges(
        "classify_intent",
        route_after_intent,
        {
            "global_search":    "global_search",
            "extract_entities": "extract_entities",
            "generate_response": "generate_response",
        }
    )

    # Local path
    workflow.add_edge("extract_entities", "local_search")
    workflow.add_edge("local_search",     "safety_intercept")

    # Global path
    workflow.add_edge("global_search", "safety_intercept")

    # Shared path after safety check
    workflow.add_edge("safety_intercept",  "fuse_evidence")
    workflow.add_edge("fuse_evidence",     "generate_response")
    workflow.add_edge("generate_response", END)

    return workflow.compile()


#═══════════════════════════════════════════════════════════════
#  PUBLIC API
#═══════════════════════════════════════════════════════════════

# Global pipeline instance
_pipeline = None
_resources = None

def initialize():
    global _pipeline, _resources
    _resources = load_resources()
    _pipeline  = build_graph(*_resources)
    print("✅ SupplementsRx pipeline ready!\n")


def ask(query: str) -> dict:
    """Ask a question and get a grounded response"""
    if _pipeline is None:
        initialize()

    print(f"\n{'─'*50}")
    print(f"Query: {query}")
    print(f"{'─'*50}")

    initial_state: AgentState = {
        "query":              query,
        "intent":             None,
        "entities":           None,
        "faiss_results":      None,
        "graph_results":      None,
        "community_summaries": None,
        "safety_alerts":      None,
        "evidence":           None,
        "response":           None,
        "citations":          None,
        "confidence":         None,
    }

    result = _pipeline.invoke(initial_state)

    return {
        "query":        result["query"],
        "response":     result["response"],
        "intent":       result["intent"],
        "confidence":   result["confidence"],
        "citations":    result["citations"] or [],
        "safety_alerts": result["safety_alerts"] or [],
        "entities":     result["entities"] or [],
        "evidence":     result.get("evidence") or "",
    }


#═══════════════════════════════════════════════════════════════
#  INTERACTIVE CLI
#═══════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  SupplementsRx AI — Phase 4 LangGraph Pipeline")
    print("=" * 60)

    if not OPENAI_API_KEY:
        print("❌ OPENAI_API_KEY not set!")
        print("   set OPENAI_API_KEY=sk-...")
        return

    # Check required files
    for f in [FAISS_INDEX, NAMES_FILE, METADATA_FILE, CONFIG_FILE]:
        if not os.path.exists(f):
            print(f"❌ Missing: {f}")
            print("   Run generate_embeddings.py first!")
            return

    initialize()

    # Test queries
    test_queries = [
        "What supplements help with anxiety and stress?",
        "Can I take Ashwagandha with Metformin?",
        "Is Vitamin D safe during pregnancy?",
        "What is the recommended dose of Magnesium?",
        "What are the best supplements for heart health?",
    ]

    print("\n🧪 Running test queries...\n")
    for query in test_queries:
        result = ask(query)
        print(f"\n{'='*60}")
        print(f"Q: {query}")
        print(f"Intent: {result['intent'].upper()} | "
              f"Confidence: {result['confidence']}")
        if result["safety_alerts"]:
            print("\nSAFETY ALERTS:")
            for alert in result["safety_alerts"]:
                print(f"  {alert}")
        print(f"\nA: {result['response'][:500]}...")
        print(f"\nCitations: {result['citations']}")

    # Interactive mode
    print("\n" + "="*60)
    print("  Interactive Mode — type 'quit' to exit")
    print("="*60)

    while True:
        print()
        query = input("Your question: ").strip()
        if query.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        if not query:
            continue

        result = ask(query)
        print(f"\n[{result['intent'].upper()} | {result['confidence']}]")
        if result["safety_alerts"]:
            for alert in result["safety_alerts"]:
                print(f"{alert}")
        print(f"\n{result['response']}")


if __name__ == "__main__":
    main()
