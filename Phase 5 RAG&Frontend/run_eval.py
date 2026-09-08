"""
run_eval.py
SupplementsRx AI — Evaluation Runner

Runs 130 QA questions through the full LangGraph pipeline
and saves predictions to predictions.jsonl for RAGAS scoring.

HOW TO RUN:
  1. Make sure uvicorn is NOT running (we call pipeline directly)
  2. Copy this file to Phase 5 RAG&Frontend/ folder
  3. Copy supplementsrx_qa_dataset.json to same folder
  4. pip install ragas langchain-openai
  5. python run_eval.py

OUTPUT:
  predictions.jsonl  — one JSON line per question
"""

import json
import os
import time
from collections import Counter
from dotenv import load_dotenv

load_dotenv()

# ── Load pipeline ─────────────────────────────────────────────
print("Loading pipeline...")
from phase4_langgraph import ask, initialize
initialize()
print("Pipeline ready!\n")

# ── Load dataset ──────────────────────────────────────────────
DATASET_PATH = "supplementsrx_qa_dataset.json"

with open(DATASET_PATH, encoding="utf-8") as f:
    data = json.load(f)

#questions = data["questions"][:5]  # Debug: test with first 5
questions = data["questions"]
print(f"Loaded {len(questions)} questions")
print(f"Categories: {set(q['category'] for q in questions)}\n")

# ── Run evaluation ────────────────────────────────────────────
results = []
failed  = 0

for i, q in enumerate(questions, 1):
    question  = q["question"]
    reference = q["answer"]
    category  = q["category"]
    qtype     = q["type"]
    qid       = q["id"]

    print(f"[{i:3d}/{len(questions)}] [{category}] {question[:70]}...")

    try:
        result = ask(question)

        response = result.get("response", "")
        intent   = result.get("intent", "")
        evidence = result.get("evidence", "")
        safety   = result.get("safety_alerts", [])
        entities = result.get("entities", [])

        # ── Build contexts for RAGAS ──────────────────────────
        contexts = []

        # 1. Evidence chunks (best source — actual retrieved data)
        if evidence:
            chunks = [c.strip() for c in evidence.split("\n\n")
                      if c.strip() and len(c.strip()) > 20]
            contexts = chunks[:6]

        # 2. Safety alerts as additional context
        if safety:
            contexts.append("Safety alerts: " + " | ".join(safety[:3]))

        # 3. Entities found
        if entities:
            contexts.append(f"Entities identified: {', '.join(entities)}")

        # 4. Fallback — only triggered if everything above is empty
        if not contexts:
            if response and len(response) > 10:
                contexts = [response]
            else:
                contexts = [f"Query about: {question}"]

        results.append({
            "id":           qid,
            "type":         qtype,
            "category":     category,
            "question":     question,
            "reference":    reference,
            "answer":       response,
            "contexts":     contexts,
            "intent":       intent,
            "safety_alerts": safety,
        })

        print(f"         Intent={intent} | "
              f"Contexts={len(contexts)} | "
              f"Response={len(response)} chars")

    except Exception as e:
        import traceback #Debug
        print(f"         ERROR: {e}")
        print(f"         {traceback.format_exc()[:300]}") #debug
        failed += 1
        results.append({
            "id":           qid,
            "type":         qtype,
            "category":     category,
            "question":     question,
            "reference":    reference,
            "answer":       "",
            "contexts":     [f"Query about: {question}"],  # always have context
            "intent":       "error",
            "safety_alerts": [],
        })

    # Small delay to avoid rate limiting
    time.sleep(0.5)

# ── Save predictions ──────────────────────────────────────────
output_path = "predictions.jsonl"
with open(output_path, "w", encoding="utf-8") as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

# ── Summary stats ─────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  Done!")
print(f"  Total    : {len(results)}")
print(f"  Failed   : {failed}")
print(f"  Output   : {output_path}")
print(f"{'='*60}")

intents = Counter(r["intent"] for r in results)
print(f"\nIntent distribution:")
for intent, count in intents.most_common():
    print(f"  {intent:10s}: {count}")

answered = sum(1 for r in results if r["answer"] and len(r["answer"]) > 50)
declined = sum(1 for r in results if r["answer"] and len(r["answer"]) <= 50)
empty_ctx = sum(1 for r in results if not r["contexts"])

print(f"\nAnswered       : {answered}")
print(f"Declined/short : {declined}")
print(f"Errors         : {failed}")
print(f"Empty contexts : {empty_ctx}")
print(f"\nNext: python evaluate.py")