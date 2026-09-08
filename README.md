# SupplementsRx AI
### Evidence-Based Supplement Guidance via Adaptive GraphRAG

**Arizona State University — CSE 573 Semantic Web Mining — Spring 2026**  
**Group 3:** Shivam Sibal · Priyanshu Gupta · Pranay Reddy Palle · Aryan Talati · Divyesh Jain · Omkar Deshpande

---

## Overview

SupplementsRx AI is a conversational health supplement advisor that delivers grounded, citation-backed guidance on supplement efficacy, drug interactions, dosage, and pregnancy safety. The system integrates a Neo4j Knowledge Graph, FAISS vector search, Louvain community clustering, and a LangGraph state machine to ensure every response is traceable to verified biomedical data.

---

## Architecture

```
User Query
    ↓
Intent Classifier (GPT-4o-mini) → Global / Local / Out-of-Scope
    ↓
LOCAL  → Entity Extraction → FAISS Search + Neo4j 2-hop Traversal
GLOBAL → Community Summary Retrieval (Louvain clusters)
OOS    → Graceful warm decline
    ↓
Safety Intercept Layer (rule-based)
→ Drug interactions via INTERACTS_WITH edges
→ Pregnancy risk from supplement safety data
→ Deduplication of alerts
    ↓
Evidence Fusion (graph + vector results)
    ↓
GPT-4o Response Generation (grounded, warm, human tone)
    ↓
FastAPI → Chat UI
```

---

## Knowledge Graph

| Metric | Value |
|--------|-------|
| Total nodes | 5,577 |
| Total relationships | 63,573 |
| Supplement nodes | 3,841 |
| Condition nodes | 200 |
| Drug nodes | 572 |
| SideEffect nodes | 921 |
| SupplementClass nodes | 43 |
| Louvain communities | 2,925 |
| Modularity score | 0.6053 |

---

## Data Sources

| Source | Records | Usage |
|--------|---------|-------|
| Natural Medicines (NatMed) | 1,437 | Knowledge Graph + FAISS |
| MedlinePlus (NIH NLM) | 2,076 | Knowledge Graph + FAISS |
| DSLD NIH Label Database | 214,780 | FAISS corpus + dosage reference |
| MIMIC-IV Clinical Demo | 582 drugs, 1,388 conditions | Graph drug-condition pairs |

---

## Evaluation Results (RAGAS)

Evaluated on 130-question SupplementRx-Bench dataset across 5 categories.

| Metric | Score |
|--------|-------|
| Faithfulness | 0.906 |
| Answer Relevancy | 0.260 |
| Context Precision | 0.430 |
| Context Recall | 0.322 |
| Answer Correctness | 0.494 |

**Safety checks category — best performance:**
- Faithfulness: 0.951
- Answer Correctness: 0.561
- Context Precision: 0.593

---

## Project Structure

```
CSE-573_26-Semantic-Web-Mininig/
├── Knowledge_Graph/
│   ├── phase2_knowledge_graph.py      # Build graph from merged CSV
│   ├── migrate_to_neo4j.py            # Load graph to Neo4j AuraDB
│   ├── verify_and_louvain.py          # Verify migration + Louvain
│   ├── load_communities_to_aura.py    # Load community IDs to AuraDB
│   └── check_auradb.py               # Verify AuraDB contents
│
├── Phase 3/
│   └── generate_embeddings.py         # Generate FAISS embeddings
│
├── Phase 5 RAG&Frontend/
│   ├── main.py                        # FastAPI server
│   ├── rag_functionality.py           # RAG + LangGraph integration
│   ├── phase4_langgraph.py            # Full LangGraph pipeline
│   ├── index.html                     # Chat UI (green/gold theme)
│   ├── run_eval.py                    # Evaluation runner
│   ├── evaluate.py                    # RAGAS scorer
│   ├── supplementsrx_qa_dataset.json  # 130-question eval dataset
│   ├── faiss_index.bin                # FAISS vector index
│   ├── supplement_embeddings.npy      # Embedding vectors
│   ├── supplement_names.json          # Supplement name index
│   ├── supplement_metadata.json       # Full supplement metadata
│   ├── embedding_config.json          # Embedding model config
│   └── .env                          # API keys (not committed)
│
├── scrapers/
│   ├── natmed_complete.py
│   ├── medlineplus_scraper.py
│   └── dsld_scraper.py
│
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## Setup & Run

### Prerequisites
- Python 3.10+
- OpenAI API key ([platform.openai.com](https://platform.openai.com))
- Neo4j AuraDB free instance ([console.neo4j.io](https://console.neo4j.io))

### 1. Clone the repository
```bash
git clone https://github.com/your-team/CSE-573_26-Semantic-Web-Mininig.git
cd CSE-573_26-Semantic-Web-Mininig
```

### 2. Create virtual environment
```bash
python -m venv venv

# Windows
.\venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Create .env file
Create a `.env` file inside `Phase 5 RAG&Frontend/`:
```
OPENAI_API_KEY=sk-your-openai-key-here
NEO4J_URI=neo4j+s://your-instance.databases.neo4j.io
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-aura-password
```

### 5. Run the application
```bash
cd "Phase 5 RAG&Frontend"
uvicorn main:app --reload --port 8000
```

### 6. Open browser
Navigate to `http://127.0.0.1:8000`

---

## Example Queries

```
✅ "Tell me about ashwagandha"
✅ "Can I take Vitamin D with Metformin?"
✅ "What supplements help with sleep?"
✅ "Is melatonin safe during pregnancy?"
✅ "What are the side effects of turmeric?"
✅ "What supplements support heart health?"
✅ "Can I take fish oil with warfarin?"
```

---

## Running Evaluation

```bash
cd "Phase 5 RAG&Frontend"

# Step 1 — Run pipeline on 130 questions (~15 mins)
python run_eval.py

# Step 2 — Score with RAGAS (~10 mins)
python evaluate.py

# Results saved to ragas_scores.csv
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Data Collection | Selenium, BeautifulSoup, Requests |
| Knowledge Graph | Neo4j AuraDB (cloud) |
| Graph Clustering | Louvain algorithm |
| Embeddings | OpenAI text-embedding-3-small |
| Vector Search | FAISS IndexFlatIP |
| Intent Classification | GPT-4o-mini |
| Pipeline Orchestration | LangGraph state machine |
| Response Generation | GPT-4o |
| Safety Layer | Rule-based graph queries |
| Backend API | FastAPI + Uvicorn |
| Frontend | HTML/CSS/JS |
| Evaluation | RAGAS 0.4.3 |

---

## Team

| Member | Role |
|--------|------|
| Shivam Sibal | Data Engineering, RAG pipeline |
| Priyanshu Gupta | KG Architecture, LangGraph pipeline, AuraDB |
| Pranay Reddy Palle | Clustering, Evaluation |
| Aryan Talati | NLU, Retrieval fusion |
| Divyesh Jain | Safety layer, Benchmarks |
| Omkar Deshpande | Frontend UI, API integration |

---

## Known Limitations

- AuraDB Free tier does not support GDS plugin — Louvain run locally, IDs loaded via `load_communities_to_aura.py`
- 582 MIMIC-IV clinical drugs are labeled as Supplement nodes due to data ingestion — correctly handled by the system
- 1,845 supplement nodes have no content data — FAISS fallback activates automatically
- DSLD 214K records used for FAISS text corpus only, not in the knowledge graph

---

## Deadline

**April 24, 2026** — GitHub submission + demo video
