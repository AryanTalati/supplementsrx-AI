"""
evaluate.py
SupplementsRx AI — RAGAS Evaluation Scorer
Tested with RAGAS 0.4.3

HOW TO RUN:
  python evaluate.py
REQUIRES: predictions.jsonl (from run_eval.py)
"""

import json
import os
import warnings
warnings.filterwarnings("ignore")   # suppress deprecation warnings

from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# ── Load predictions ──────────────────────────────────────────
print("Loading predictions...")
rows = [json.loads(l) for l in open("predictions.jsonl", encoding="utf-8")]
valid_rows = [r for r in rows if r.get("answer") and r.get("contexts")]
print(f"  Total rows : {len(rows)}")
print(f"  Valid rows : {len(valid_rows)}")
print(f"  Skipped    : {len(rows) - len(valid_rows)}")

if not valid_rows:
    print("No valid rows!")
    exit()

# ── Build RAGAS dataset ───────────────────────────────────────
from ragas import EvaluationDataset

samples = [{
    "user_input":         r["question"],
    "response":           r["answer"],
    "retrieved_contexts": r["contexts"],
    "reference":          r["reference"],
} for r in valid_rows]

ds = EvaluationDataset.from_list(samples)

# ── LLM + Embeddings ──────────────────────────────────────────
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

judge_llm = LangchainLLMWrapper(
    ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY)
)
judge_emb = LangchainEmbeddingsWrapper(
    OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)
)

# ── Metrics — use lowercase instances (proper Metric subclasses) ──
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_correctness,
)

# ── Evaluate ──────────────────────────────────────────────────
from ragas import evaluate

print(f"\nEvaluating {len(valid_rows)} rows...")
print("This may take 5-15 minutes...\n")

result = evaluate(
    dataset=ds,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
        answer_correctness,
    ],
    llm=judge_llm,
    embeddings=judge_emb,
)

# ── Results ───────────────────────────────────────────────────
print("\n" + "="*60)
print("  RAGAS EVALUATION RESULTS")
print("="*60)
print(result)

df = result.to_pandas()
df.to_csv("ragas_scores.csv", index=False)
print(f"\nSaved to ragas_scores.csv")

# ── Per-category breakdown ────────────────────────────────────
print("\n=== PER-CATEGORY BREAKDOWN ===")
category_map = {r["question"]: r["category"] for r in valid_rows}
df["category"] = df["user_input"].map(category_map)

metric_cols = [c for c in df.columns
               if c not in ["user_input", "response",
                            "retrieved_contexts", "reference", "category"]]

for cat in sorted(df["category"].dropna().unique()):
    cat_df = df[df["category"] == cat]
    print(f"\n{cat} (n={len(cat_df)}):")
    for m in metric_cols:
        if m in cat_df.columns:
            val = cat_df[m].dropna().mean()
            print(f"  {m:35s}: {val:.3f}")

print("\n✅ Evaluation complete!")