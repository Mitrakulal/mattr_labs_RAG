# A minimal FastAPI server that talks to a local Ollama model
# 
# Run it:   uvicorn fastapi_test:app --reload
# Test it:  curl -X POST http://localhost:8000/chat \
#             -H "Content-Type: application/json" \
#             -d '{"message": "hello, what is 2+2?"}'

from fastapi import FastAPI
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage

# ---------------- SETUP ----------------

app = FastAPI()

model = init_chat_model(
    model="phi3:latest",
    model_provider="ollama",
    base_url="http://localhost:11435",  # your Mac Mini
    temperature=0,
)


# ---------------- ENDPOINT ----------------

@app.post("/chat")
async def chat(body: dict):
    """
    Expects:  {"message": "your question here"}
    Returns:  {"response": "the model's answer"}
    """
    user_msg = body.get("message", "")
    response = await model.ainvoke([HumanMessage(content=user_msg)])
    return {"response": response.content}
