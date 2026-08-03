import ollama
import asyncio
import chromadb
import json 
import time
import pathlib
from datetime import datetime
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse,StreamingResponse
from pydantic import BaseModel
from fastapi import FastAPI
from storing import embed
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage

BASE_DIR = pathlib.Path(__file__).parent          # gets main.py's folder
CHROMA_PATH = str(BASE_DIR / "chroma_db1") 
COLLECTION_NAME = "mattrlabs_doc1"
LOG_FILE="rag.log"
OLLAMA_HOST = "http://localhost:11435"


app=FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def serve_ui():   
    with open("static/index.html") as f:
        return HTMLResponse(f.read())

client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection(COLLECTION_NAME)

model = init_chat_model(
    # model="gemma4:e4b",
    model="gemma4:12b-mlx",
    # model="gemma2:9b-instruct-q2_K",
    # model="phi3:latest",
    model_provider="ollama",
    base_url=OLLAMA_HOST,
    temperature=0,
)

def log_entry(data:dict):
    """Append one json line to the log file."""
    data["time"]=datetime.now().isoformat()
    with open(LOG_FILE,"a") as f:
        f.write(json.dumps(data)+"\n")




class AskRequest(BaseModel):
    question : str
    
class AskResponse(BaseModel):
    answer: str
    
@app.get("/health")
async def health(key : str):
    if key != "1230":
        return {"error": "unauthorized"}
    health_status = {"status": "ok", "checks": {}}

    # check ollama
    try:
        test_embed = ollama.Client(host=OLLAMA_HOST).embeddings(
            model="nomic-embed-text:latest", prompt="ping"
        )
        health_status["checks"]["ollama"] = "ok"
    except Exception as e:
        health_status["status"] = "degraded"
        health_status["checks"]["ollama"] = f"fail: {e}"

    # check chromadb
    try:
        collection.count()
        health_status["checks"]["chromadb"] = "ok"
    except Exception as e:
        health_status["status"] = "degraded"
        health_status["checks"]["chromadb"] = f"fail: {e}"

    return health_status

@app.post("/ask")
async def ask(body: AskRequest):
    start = time.time()
    query = body.question

    # step 1: embed
    try:
        query_embedding = await asyncio.to_thread(embed, query)
    except Exception as e:
        print(f"[ERROR] embed failed: {e}")
        log_entry({"query": query, "error": f"embed failed: {e}", "latency_s": round(time.time() - start, 2)})
        return AskResponse(answer="I'm having trouble connecting to the knowledge base.")

    # step 2: search chroma
    try:
        results = await asyncio.to_thread(collection.query, query_embeddings=[query_embedding], n_results=3)
    except Exception as e:
        print(f"[ERROR] chroma query failed: {e}")
        log_entry({"query": query, "error": f"chroma query failed: {e}", "latency_s": round(time.time() - start, 2)})
        return AskResponse(answer="I'm having trouble searching the knowledge base.")

    context = "\n\n---\n\n".join(
        f"[Source: {meta['section']}]\n{doc}"
        for doc, meta in zip(results["documents"][0], results["metadatas"][0])
    )

    prompt = f""""You are an assistant for mattrlabs that answers questions based on the provided context. Important rules:
            Be concise: 2-4 sentences by default, elaborate only if necessary, Base your answers on the provided context, do not make up information,If the context does not contain the answer, say so clearly,Stay factual and professional"
    Context:
    {context}
    
    Question: {query}"""

    # step 3: stream the answer
    async def token_generator():
        full_answer = ""
        try:
            async for chunk in model.astream([HumanMessage(content=prompt)]):
                full_answer += chunk.content
                yield chunk.content
        except asyncio.TimeoutError:
            yield "\n\nThe request took too long. Please try again."
            full_answer = "The request took too long. Please try again."
        except Exception as e:
            print(f"[ERROR] stream failed: {e}")
            yield "\n\nI'm having trouble generating an answer right now."
            full_answer = "I'm having trouble generating an answer right now."

        log_entry({
            "query": query,
            "answer": full_answer,
            "retrieved": [f"{m['section']}-{m['chunk_index']}" for m in results["metadatas"][0]],
            "latency_s": round(time.time() - start, 2),
            "error": None,
        })

    return StreamingResponse(token_generator(), media_type="text/plain")
        