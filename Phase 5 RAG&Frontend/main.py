from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag_functionality import rag
from fastapi.responses import FileResponse
import os
import asyncio  # This is the function you've written above

app = FastAPI()

@app.get("/")
async def serve_frontend():
    return FileResponse(os.path.join(os.path.dirname(__file__), "index.html"))
# Allow frontend (browser) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Message(BaseModel):
    question: str

@app.post("/chat")
async def chat(message: Message):
    response = rag(message.question)
    await asyncio.sleep(0.8)
    return {"answer": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)