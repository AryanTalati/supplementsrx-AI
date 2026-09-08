"""
rag_functionality.py
Graph RAG — Healthcare Supplement Guidance Agent
Uses FAISS index + Neo4j graph metadata for retrieval.

Updated imports for langchain>=1.0 (latest version compatible).
All logic is identical to Shivam's original — only the 3 broken
LangChain imports were replaced with modern equivalents:
  langchain.prompts        → langchain_core.prompts
  langchain.memory         → simple Python list (same behavior)
  langchain.chains         → langchain_core.runnables
  langchain.schema.Document→ langchain_core.documents
"""

import os
import json
import numpy as np
import re
import faiss
import time
import re as _re

from typing import List, Dict


from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.messages import HumanMessage, AIMessage
from pydantic import Field

# ─────────────────────────────────────────────
# CONFIG — update your key and paths here
# ─────────────────────────────────────────────
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
FAISS_INDEX_PATH = "faiss_index.bin"
NAMES_PATH       = "supplement_names.json"
METADATA_PATH    = "supplement_metadata.json"
CONFIG_PATH      = "embedding_config.json"
TOP_K            = 10


# ─────────────────────────────────────────────
# LOAD GRAPH EMBEDDINGS
# ─────────────────────────────────────────────
print("Loading FAISS index and supplement metadata...")

with open(CONFIG_PATH, "r") as f:
    embedding_config = json.load(f)

USE_OPENAI_EMBED = embedding_config.get("use_openai", True)
EMBED_MODEL      = embedding_config.get("embedding_model", "text-embedding-3-small")

faiss_index = faiss.read_index(FAISS_INDEX_PATH)

with open(NAMES_PATH, "r", encoding="utf-8") as f:
    supplement_names: List[str] = json.load(f)

with open(METADATA_PATH, "r", encoding="utf-8") as f:
    supplement_metadata: List[Dict] = json.load(f)

print(f"Loaded {len(supplement_names)} supplements.")

# ── Try to connect Phase 4 LangGraph pipeline ────────────────
_langgraph_available = False
_langgraph_ask       = None

try:
    import sys
    phase4_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "phase4_langgraph.py")
    if os.path.exists(phase4_path):
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from phase4_langgraph import ask as lg_ask, initialize as lg_init
        lg_init()
        _langgraph_ask       = lg_ask
        _langgraph_available = True
        print("✅ Phase 4 LangGraph pipeline connected!")
    else:
        print("⚠️  phase4_langgraph.py not found — using FAISS-only RAG")
except Exception as e:
    print(f"⚠️  LangGraph not available ({e}) — using FAISS-only RAG")

IDENTITY_PATTERNS = [
    r"\bwho are you\b",
    r"\bwhat are you\b",
    r"\btell me about yourself\b",
    r"\babout yourself\b",
    r"\bwhat do you do\b",
    r"\bwhat can you do\b",
    r"\bwhat is your name\b",
    r"\bwhat are your capabilities\b",
    r"\bhow can you help\b",
    r"\bwho made you\b",
    r"\bintroduce yourself\b",
]
 
IDENTITY_RESPONSE = (
    "I'm your SupplementRx health advisor — think of me as a knowledgeable friend "
    "who can help you understand supplements and natural health approaches. "
    "You can ask me about any supplement, what it does, whether it's safe, "
    "how it might interact with your medications, what might help with a specific "
    "health concern, or even broader topics like stress, sleep, or mental wellbeing. "
    "I have information on thousands of supplements and natural therapies, "
    "so feel free to ask me anything you have in mind!"
)
 
def is_identity_question(question: str) -> bool:
    q = question.lower().strip()
    q = q.replace('\u2019', "'").replace('\u2018', "'")
    return any(re.search(pattern, q) for pattern in IDENTITY_PATTERNS)

OUT_OF_SCOPE_PATTERNS = [
    # maths
    r"\d+\s*[\+\-\*\/]\s*\d+",
    r"\bwhat is \d+\s*[\+\-\*\/x]",
    r"\bcalculate\b",
    r"\bsolve\b.*\d",
    r"\bequals\b",                       # "2+2 equals"
    r"\bhow much is \d+",               # "how much is 2+2"
 
    # coding / tech
    r"\bwrite.*code\b",
    r"\bpython\b",
    r"\bjavascript\b",
    r"\bhtml\b",
    r"\bcss\b",
    r"\bsql\b",
    r"\bprogram\b",
    r"\balgorithm\b",
    r"\bdebug\b",
 
    # general knowledge / trivia
    r"\bwhat is the capital\b",
    r"\bwho is the president\b",
    r"\bwho won\b.*\b(match|game|election|war)\b",
    r"\bhistory of\b.*\b(country|war|world)\b",
    r"\bweather\b",
    r"\bstock price\b",
    r"\bcurrency\b",
    r"\bexchange rate\b",
 
    # food / recipes (not health)
    r"\brecipe for\b",
    r"\bhow to cook\b",
    r"\bhow to bake\b",
    r"\bingredients for\b",
 
    # entertainment
    r"\bmovie\b",
    r"\bsong\b",
    r"\blyrics\b",
    r"\bsport(s)?\b.*\bscore\b",
    r"\bwho is .*(actor|singer|celebrity)\b",
]
 
OUT_OF_SCOPE_RESPONSE = (
    "That one's a bit outside what I can help with — I'm specialised in supplements "
    "and natural health. If you have any questions about a supplement, a health concern, "
    "or a natural therapy, I'm all yours!"
)
 
FAREWELL_PATTERNS = [
    r"\bthat.?s all\b",
    r"\bthats all\b",
    r"\ball done\b",
    r"^(okay|ok|alright|bye|goodbye|thanks|thank you|cheers|great|cool|got it|perfect|noted|sure|sounds good|see you|take care|done|nothing else|no more|all good|awesome|wonderful)[\s!.]*$",
]

FAREWELL_RESPONSES = [
    "Happy to help anytime! Come back if you have more supplement questions.",
    "Take care! Feel free to return whenever you have health questions.",
    "Anytime! Stay well and don't hesitate to ask if you need anything.",
]

def is_farewell(question: str) -> bool:
    q = question.lower().strip()
    q = q.replace('\u2019', "'").replace('\u2018', "'")  # normalize smart quotes
    
    return any(re.search(p, q) for p in FAREWELL_PATTERNS)

def is_out_of_scope(question: str) -> bool:
    q = question.lower().strip()
    q = q.replace('\u2019', "'").replace('\u2018', "'")
    return any(re.search(pattern, q) for pattern in OUT_OF_SCOPE_PATTERNS)
 
OUT_OF_SCOPE_PROMPT = """You are SupplementRx, a warm and friendly health advisor who specialises in supplements and natural health. 
 
A patient just asked you something completely outside your area — they asked: "{question}"
 
Respond in a warm, friendly, energetic way like a good friend would. Acknowledge what they asked in a lighthearted way, let them know you can't help with that, and enthusiastically redirect them to what you CAN help with — supplements, natural health, sleep, stress, energy, mental wellbeing, vitamins, herbs etc.
 
Keep it short — 2-3 sentences max. Sound like a real person, not a customer service bot. Be warm, fun, and genuine. Never say "I appreciate your question." Vary your response — don't use the same opener every time."""

def generate_out_of_scope_response(question: str, llm) -> str:
    try:
        prompt = OUT_OF_SCOPE_PROMPT.format(question=question)
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content
    except Exception:
        return (
            "Ha, that one's a bit out of my lane! I live and breathe supplements "
            "and natural health — ask me anything in that space and I'm all yours!"
        )

CONTINUATION_PATTERNS = [
    r"^(yes|yeah|yep|sure|ok|okay|please|go ahead|tell me|correct|right|exactly)[\s!.]*$",
]

CONTINUATION_RESPONSE = (
    "Of course! Could you ask your question again so I can help properly? "
    "I don't have memory of what we discussed before in this context."
)

def is_continuation(question: str) -> bool:
    q = question.lower().strip()
    return any(re.search(p, q) for p in CONTINUATION_PATTERNS)

# ─────────────────────────────────────────────
# EMBED QUERY
# ─────────────────────────────────────────────
def embed_query(text: str) -> np.ndarray:
    if USE_OPENAI_EMBED:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.embeddings.create(model=EMBED_MODEL, input=[text])
        vec = np.array(response.data[0].embedding, dtype=np.float32)
    else:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(EMBED_MODEL)
        vec = model.encode([text], normalize_embeddings=True)[0].astype(np.float32)

    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec.reshape(1, -1)


# ─────────────────────────────────────────────
# FAISS GRAPH RETRIEVER
# ─────────────────────────────────────────────
class FAISSGraphRetriever(BaseRetriever):
    top_k: int = Field(default=TOP_K)

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str) -> List[Document]:
        query_vec = embed_query(query)
        distances, indices = faiss_index.search(query_vec, self.top_k)

        docs = []
        for score, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(supplement_metadata):
                continue
            meta = supplement_metadata[idx]
            name = supplement_names[idx]

            parts = [f"Supplement: {name}"]
            if meta.get("scientific"):
                parts.append(f"Scientific name / Also known as: {meta['scientific']}")
            if meta.get("conditions"):
                parts.append(f"Indicated for: {', '.join(meta['conditions'])}")
            if meta.get("classes"):
                parts.append(f"Supplement class: {', '.join(meta['classes'])}")
            if meta.get("drugs"):
                parts.append(f"Known drug interactions: {', '.join(meta['drugs'])}")
            if meta.get("side_effects"):
                parts.append(f"Possible side effects: {', '.join(meta['side_effects'])}")
            if meta.get("overview"):
                parts.append(f"Clinical overview: {meta['overview'][:500]}")
            if meta.get("pregnancy"):
                parts.append(f"Pregnancy & safety notes: {meta['pregnancy']}")

            docs.append(Document(
                page_content="\n".join(parts),
                metadata={"name": name, "score": float(score), "source": "graph_rag"}
            ))
        return docs

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        return self._get_relevant_documents(query)


# ─────────────────────────────────────────────
# HEALTHCARE SUPPLEMENT AGENT PROMPT
# ─────────────────────────────────────────────
supplement_agent_template = """You are SupplementRx — a warm, knowledgeable health advisor who specialises in dietary supplements and natural health approaches. You speak like a caring, experienced doctor who explains things simply and honestly to patients. Your knowledge covers both supplements (vitamins, minerals, herbs, botanicals) and natural health techniques and therapies (like biofeedback, naturopathy, mindfulness, etc.).
You are SupplementRx — a health advisor who ONLY speaks from the HEALTH CONTEXT provided to you below. You have no general knowledge. You do not know celebrities, maths, history, coding, sports, politics, or anything outside health supplements and natural therapies. The ONLY thing you know is what is written in the HEALTH CONTEXT section of this message,Health context that is only from data not from any net or something just from the data given to you.
GOLDEN RULE — READ THIS FIRST:
If the answer to the patient's question is not found in the HEALTH CONTEXT below — do not answer it. Do not try. Do not use anything you know from outside. Just say warmly:
"That's not something I can help with — I'm only here for supplements and natural health questions. Feel free to ask me anything in that space!"

This applies to EVERYTHING outside health — celebrities, people, maths, coding, recipes, sports, news, geography, movies, music. You simply do not know these things. You are not a general assistant.
Your ONLY source of knowledge is the HEALTH CONTEXT provided below in this message.
You do not have internet access. You do not use general knowledge. You do not use
anything from your AI training. You do not search the web. You do not know anything
that is not written in the HEALTH CONTEXT below.
If the answer is not in the HEALTH CONTEXT — you do not know it. Say warmly:
"That's not something I can help with — I'm only here for supplements and natural 
health questions. Feel free to ask me anything in that space!

If someone asks "are you a doctor?", "are you human?", "are you an AI?" — 
answer honestly and warmly. You are not a doctor, you are an AI health advisor. 
Say something like: "Not a doctor — I'm an AI health advisor! I can share 
information about supplements and natural therapies, but for medical diagnosis 
or treatment, your doctor is always the right person

This applies to EVERYTHING outside the context — celebrities, people, maths, coding,
recipes, sports, news, geography, movies, music, general health advice not in the 
context. You simply do not know these things. You are not a general assistant.
You are not connected to the internet. Your only source is the HEALTH CONTEXT below.
 
════════════════════════════════════════
WHAT YOU COVER
════════════════════════════════════════
You cover everything in the context provided — this includes:
- Dietary supplements: vitamins, minerals, herbs, amino acids, botanicals
- Natural health techniques and therapies: biofeedback, naturopathy, mindfulness, relaxation therapy, and similar approaches
 
When someone asks a broad health question like "what helps with anxiety?" or 
"what helps with stress?" — look at the context carefully and separate what is 
a SUPPLEMENT (vitamin, mineral, herb, amino acid, botanical) from what is a 
TECHNIQUE or THERAPY (naturopathy, biofeedback, mindfulness, relaxation therapy).

Present them clearly in two parts like a doctor would:
- First talk about relevant SUPPLEMENTS — herbs, vitamins, amino acids etc.
- Then mention relevant TECHNIQUES if they appear in the context — but always 
  make it clear these are practices or therapies, NOT supplements.

NEVER present a technique like naturopathy, biofeedback, or mindfulness as if 
it is a supplement. They are completely different things. A supplement is something 
you take. A technique is something you do.

Example:
"For anxiety, a few supplements worth knowing about are ashwagandha, magnesium, 
and GABA — each works a little differently. On the therapy side, something like 
biofeedback or mindfulness can also be really effective alongside supplements."
 
When someone asks to "explain more" or "tell me more" about something already discussed — go DEEPER into that specific thing. Explain how it actually works, why it helps, what to expect, in simple human terms. Do NOT repeat what you already said. A good doctor, when a patient says "can you explain that more?", explains it in a clearer, more relatable way — maybe with an analogy, maybe with a real-world example.
 
════════════════════════════════════════
CORE RULES
════════════════════════════════════════
 
1. READ CONVERSATION HISTORY FIRST — AND TRACK THE CURRENT TOPIC.
   Always know what the MOST RECENT topic of conversation is.
   When a patient asks a vague follow-up like "what techniques can I consider?", 
   "any side effects?", "what else can I take?", "tell me more" — these are ALWAYS 
   about the MOST RECENT topic discussed, not something from earlier in the conversation.
   
   Never jump back to an earlier topic unless the patient explicitly mentions it.
   The most recent question is always the active context.
 
2. ANSWER EXACTLY WHAT WAS ASKED — nothing more.
   - "Does it cause dizziness?" → yes or no, brief explanation only.
   - "Tell me about X" → warm overview, key points, most important things to know.
   - "Explain more about X" → go deeper on HOW and WHY it works. Use a simple analogy.
   - "Any side effects?" → just side effects, conversationally.
   - Short questions get short answers. Long questions get thorough ones.
 
3. NEVER REPEAT YOURSELF across messages.
 
4. EXPLAIN THINGS LIKE A WARM DOCTOR TALKING TO A PATIENT.
   Use plain language and relatable comparisons.
   Make complex things feel simple. Use "you" and "your body" naturally.
 
5. SOUND HUMAN — not robotic.
   - Never say "I appreciate your question" or "That's a great question."
   - Never end with "Your health and wellbeing are important to me."
   - Never say "the data", "the database", "in the data", "according to the data."
   - Use natural phrases: "Honestly...", "That said...", "Worth knowing..."
 
6. SAFETY — weave it in naturally.
   No separate warning sections. Just say it in the flow.
 
7. ONLY use information from the context provided. Never invent facts or dosages.
 
8. If the context doesn't have enough: "Honestly, I don't have enough on that to give you a solid answer — your doctor would be the right person for that one."
 
════════════════════════════════════════
EXAMPLES
════════════════════════════════════════
 
Patient: "what helps with mental health?"
GOOD: "There's quite a bit depending on what you're dealing with. On the supplement side, ashwagandha is well known for helping the body handle stress — it works on your cortisol levels over time rather than being an instant fix. Magnesium is another one that comes up a lot, especially for anxiety and sleep. Omega-3s have decent evidence for mood support too. On the technique side, biofeedback is really interesting — it trains you to actually see and control your stress signals in real time. Mindfulness-based approaches also show up consistently for anxiety and low mood. Want me to go deeper on any of these?"
 
Patient: "explain more about biofeedback"
GOOD: "Sure — think of it like giving your nervous system a mirror. Normally your heart rate, breathing, and muscle tension are running in the background without you paying attention. Biofeedback connects you to sensors that show you those signals on a screen in real time. Once you can see them, you can start to influence them — slow your breathing, relax a tense muscle, bring your heart rate down. Over time your brain actually learns to do this on its own without the equipment. It's used a lot for stress, anxiety, chronic pain, and blood pressure. It takes some practice but there are no side effects and it's completely non-invasive."
 
Patient already asked "what is turmeric?" and got a full answer.
Patient asks: "does it cause dizziness?"
GOOD: "Not specifically, no. Side effects with turmeric are mostly digestive — nausea, bloating, that sort of thing. Dizziness isn't one of the commonly reported ones. That said, if you're experiencing it, worth mentioning to your doctor since everyone reacts a little differently."
 
════════════════════════════════════════
 
CONVERSATION SO FAR:
{chat_history}
 
HEALTH CONTEXT:
{context}
 
PATIENT'S QUESTION:
{question}
 
YOUR RESPONSE (warm, human, a real doctor talking to a real patient):"""


# ─────────────────────────────────────────────
# BUILD LLM
# ─────────────────────────────────────────────
llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model="gpt-4o-mini", #gpt-3.5-turbo, gpt-4o-mini
    temperature=0.3
)

retriever = FAISSGraphRetriever(top_k=TOP_K)

# Simple in-memory conversation history (replaces ConversationBufferMemory)
chat_history: List = []
last_clarification: dict = {}


# ─────────────────────────────────────────────
# CORE QA FUNCTION (replaces ConversationalRetrievalChain)
# ─────────────────────────────────────────────
def run_qa(question: str) -> str:
    # 1. Retrieve relevant documents via FAISS
    docs = retriever._get_relevant_documents(question)
    context = "\n\n".join(doc.page_content for doc in docs)

    # 2. Format chat history as text
    history_text = ""
    for msg in chat_history[-6:]:  # last 6 turns
        if isinstance(msg, HumanMessage):
            history_text += f"Patient: {msg.content}\n"
        elif isinstance(msg, AIMessage):
            history_text += f"SupplementRx: {msg.content}\n"

    # 3. Build final prompt
    final_prompt = supplement_agent_template.format(
        chat_history=history_text,
        context=context,
        question=question
    )

    # 4. Call LLM directly
    response = llm.invoke([HumanMessage(content=final_prompt)])
    answer = response.content

    # 5. Save to history
    chat_history.append(HumanMessage(content=question))
    chat_history.append(AIMessage(content=answer))

    # Keep history manageable
    if len(chat_history) > 20:
        chat_history.clear()
        chat_history.extend(chat_history[-20:])

    return answer


# ─────────────────────────────────────────────
# RAG FUNCTION — called by main.py
# ─────────────────────────────────────────────
# def rag(question: str) -> str:
#     try:
#         #print(f"   [RAG] Checking farewell for: '{question}'") #Debug
#         if is_farewell(question):
#             import random
#             return random.choice(FAREWELL_RESPONSES)
#         # Identity questions caught before FAISS runs
#         if is_continuation(question):
#             return CONTINUATION_RESPONSE
#         if is_identity_question(question):
#             return IDENTITY_RESPONSE
#         if is_out_of_scope(question):
#             return generate_out_of_scope_response(question, llm)

#         # Use Phase 4 LangGraph if available
#         if _langgraph_available and _langgraph_ask:
#             result = _langgraph_ask(question)
#             return result.get("response", "")

#         answer = run_qa(question)

#         if answer:
#             return answer
#         return (
#             "Honestly, I don't have enough on that to give you a solid answer. "
#             "Your doctor or a licensed nutritionist would be the right person to ask."
#         )
#     except Exception as e:
#         print(f"RAG error: {e}")
#         return "Something went wrong on my end — please try asking again."

def rag(question: str) -> str:
    global last_clarification
    try:
        # Clear clarification if older than 60 seconds
        if last_clarification and time.time() - last_clarification.get("timestamp", 0) > 60:
            print("   [Clarification expired]")
            last_clarification.clear()

        if is_farewell(question):
            last_clarification.clear()
            import random
            return random.choice(FAREWELL_RESPONSES)

        # If user confirmed a clarification
        if is_continuation(question):
            if last_clarification:
                actual_query = f"tell me about {last_clarification['suggested']}"
                print(f"   [Clarification resolved] '{question}' → '{actual_query}'")
                last_clarification.clear()
                if _langgraph_available and _langgraph_ask:
                    result = _langgraph_ask(actual_query)
                    return result.get("response", "")
                return run_qa(actual_query)
            return CONTINUATION_RESPONSE

        if is_identity_question(question):
            last_clarification.clear()
            return IDENTITY_RESPONSE

        if is_out_of_scope(question):
            last_clarification.clear()
            return generate_out_of_scope_response(question, llm)

        # New question — clear any stale clarification
        last_clarification.clear()

        # Use Phase 4 LangGraph if available
        if _langgraph_available and _langgraph_ask:
            result = _langgraph_ask(question)
            response = result.get("response", "")

            # Detect if GPT asked a clarification question
            match = _re.search(r"[Dd]id you mean ([A-Za-z0-9\-\s]+)\?", response)
            if match:
                last_clarification["suggested"] = match.group(1).strip()
                last_clarification["original"]  = question
                last_clarification["timestamp"] = time.time()
                print(f"   [Clarification pending] → '{last_clarification['suggested']}'")

            return response

        answer = run_qa(question)
        if answer:
            return answer
        return (
            "Honestly, I don't have enough on that to give you a solid answer. "
            "Your doctor or a licensed nutritionist would be the right person to ask."
        )
    except Exception as e:
        print(f"RAG error: {e}")
        return "Something went wrong on my end — please try asking again."