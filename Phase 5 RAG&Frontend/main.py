"""
SupplementsRx AI — FastAPI server.

Configuration comes from environment variables. For local runs they are also
read from a .env file next to this module; on Hugging Face Spaces they come
from the Space's secrets/variables.

  OPENAI_API_KEY        required for answers
  NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD
                        optional; without them (or if unreachable) the
                        in-memory graph built from graph.json is used
  RATE_LIMIT_PER_MINUTE per-client limit on /chat            (default 10)
  RATE_LIMIT_PER_DAY    per-client limit on /chat, 24h       (default 60)
  GLOBAL_DAILY_LIMIT    all clients combined, 24h            (default 500)
  MAX_QUESTION_CHARS    max question length                  (default 500)
  TRUSTED_PROXY_HOPS    reverse proxies in front of the app  (default 1)
  CLIENT_IP_HEADER      header holding the client IP, set by a trusted edge
                        proxy (e.g. cf-connecting-ip on Render/Cloudflare);
                        only set this when that proxy always overwrites it
  CORS_ORIGINS          comma-separated origins allowed to call the API
                        cross-origin (default: none — the UI is same-origin)
  LOG_LEVEL             default INFO
  HOST / PORT           for `python main.py` (default 0.0.0.0:8000)
"""

import logging
import os
import re
import threading
import time
from collections import deque

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load .env before rag_functionality is imported (it reads OPENAI_API_KEY at
# import time). Real environment variables win over the file.
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(BASE_DIR, ".env"), override=False)
except ImportError:
    pass

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("supplementsrx")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        log.warning("Invalid %s=%r, using %d", name, os.environ.get(name), default)
        return default


RATE_LIMIT_PER_MINUTE = _env_int("RATE_LIMIT_PER_MINUTE", 10)
RATE_LIMIT_PER_DAY    = _env_int("RATE_LIMIT_PER_DAY", 60)
GLOBAL_DAILY_LIMIT    = _env_int("GLOBAL_DAILY_LIMIT", 500)
MAX_QUESTION_CHARS    = _env_int("MAX_QUESTION_CHARS", 500)
TRUSTED_PROXY_HOPS    = _env_int("TRUSTED_PROXY_HOPS", 1)
CLIENT_IP_HEADER      = os.environ.get("CLIENT_IP_HEADER", "").strip().lower()
CORS_ORIGINS = [o.strip() for o in os.environ.get("CORS_ORIGINS", "").split(",") if o.strip()]
INDEX_HTML = os.path.join(BASE_DIR, "index.html")

from fastapi import FastAPI, Request                     # noqa: E402
from fastapi.middleware.cors import CORSMiddleware       # noqa: E402
from fastapi.responses import FileResponse, JSONResponse  # noqa: E402
from pydantic import BaseModel                           # noqa: E402

from rag_functionality import rag_with_meta, pipeline_status  # noqa: E402

app = FastAPI(title="SupplementsRx AI", docs_url=None, redoc_url=None)

if CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )


# ─────────────────────────────────────────────
# RATE LIMITING (in-process; run a single worker)
# ─────────────────────────────────────────────
class RateLimiter:
    """Sliding-window limits per client plus one global daily cap."""

    def __init__(self, per_minute: int, per_day: int, global_per_day: int):
        self.per_minute = per_minute
        self.per_day = per_day
        self.global_per_day = global_per_day
        self._clients: dict = {}     # ip -> deque of timestamps (last 24h)
        self._global = deque()
        self._lock = threading.Lock()
        self._last_sweep = time.time()

    def check(self, client: str):
        """Record a request. Returns None if allowed, else (message, retry_after_s)."""
        now = time.time()
        day_ago, minute_ago = now - 86400, now - 60
        with self._lock:
            if now - self._last_sweep > 600:   # drop idle clients
                for k in [k for k, q in self._clients.items() if not q or q[-1] < day_ago]:
                    del self._clients[k]
                self._last_sweep = now

            while self._global and self._global[0] < day_ago:
                self._global.popleft()
            q = self._clients.setdefault(client, deque())
            while q and q[0] < day_ago:
                q.popleft()

            if self.per_minute and sum(1 for t in q if t >= minute_ago) >= self.per_minute:
                oldest = next(t for t in q if t >= minute_ago)
                return ("You're sending questions a little fast — please wait a moment.",
                        int(oldest + 60 - now) + 1)
            if self.per_day and len(q) >= self.per_day:
                return ("You've reached today's question limit for this demo. Please come back tomorrow.",
                        int(q[0] + 86400 - now) + 1)
            if self.global_per_day and len(self._global) >= self.global_per_day:
                return ("The demo has reached its daily question limit. Please try again later.",
                        int(self._global[0] + 86400 - now) + 1)

            q.append(now)
            self._global.append(now)
            return None


limiter = RateLimiter(RATE_LIMIT_PER_MINUTE, RATE_LIMIT_PER_DAY, GLOBAL_DAILY_LIMIT)


def client_ip(request: Request) -> str:
    """
    Client address. If CLIENT_IP_HEADER names a header set by a trusted edge
    proxy (e.g. cf-connecting-ip behind Cloudflare), that is used. Otherwise,
    behind N trusted proxies, the client IP is the N-th entry from the right
    of X-Forwarded-For; entries further left are set by the client and can
    be spoofed, so they are ignored.
    """
    if CLIENT_IP_HEADER:
        trusted = request.headers.get(CLIENT_IP_HEADER, "").strip()
        if trusted:
            return trusted
    xff = request.headers.get("x-forwarded-for", "")
    hops = [h.strip() for h in xff.split(",") if h.strip()]
    if TRUSTED_PROXY_HOPS > 0 and hops:
        return hops[-min(TRUSTED_PROXY_HOPS, len(hops))]
    return request.client.host if request.client else "unknown"


def _error(status: int, message: str, headers: dict = None) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": message}, headers=headers)


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────
@app.get("/")
def serve_frontend():
    return FileResponse(INDEX_HTML, media_type="text/html")


@app.get("/health")
def health():
    status = pipeline_status()
    return {
        "status": "ok",
        "openai_key_configured": bool(os.environ.get("OPENAI_API_KEY")),
        **status,
    }


class Message(BaseModel):
    question: str


_CONFIDENCE_SUFFIX = re.compile(r"\s*Confidence:\s*(High|Medium|Low|N/A)\s*$", re.IGNORECASE)


# Sync handler: FastAPI runs it in a worker thread, so the blocking OpenAI
# calls inside rag() don't stall the event loop for other visitors.
@app.post("/chat")
def chat(message: Message, request: Request):
    question = (message.question or "").strip()
    if not question:
        return _error(400, "Please type a question.")
    if len(question) > MAX_QUESTION_CHARS:
        return _error(400, f"Please keep questions under {MAX_QUESTION_CHARS} characters.")
    if not os.environ.get("OPENAI_API_KEY"):
        return _error(503, "This demo isn't configured yet (missing OpenAI API key).")

    ip = client_ip(request)
    limited = limiter.check(ip)
    if limited:
        msg, retry_after = limited
        log.info("Rate limited a client (retry in %ss)", retry_after)
        return _error(429, msg, {"Retry-After": str(max(retry_after, 1))})

    result = rag_with_meta(question, session_key=ip)
    answer, route = result["answer"], result["route"]
    payload = {"answer": answer}
    if route:
        # confidence is shown as a tag, so drop the "Confidence: X" footer
        payload["answer"] = _CONFIDENCE_SUFFIX.sub("", answer)
        payload["route"] = route
    return payload


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=os.environ.get("HOST", "0.0.0.0"),
                port=_env_int("PORT", 8000))
