from langchain.chat_models import init_chat_model

llm = init_chat_model(
    "gemma4:e4b",
    model_provider="ollama",
    base_url="http://localhost:11435",
)

response = llm.invoke("Hello!")
print(response.content)