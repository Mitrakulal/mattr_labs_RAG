import ollama
import asyncio
import chromadb
from fastapi import FastAPI
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "mattrlabs_doc"
ollama_client = ollama.Client(host="http://localhost:11435")

client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection(COLLECTION_NAME)

model = init_chat_model(
    model="gemma4:e4b",
    # model="gemma4:12b-mlx",
    # model="phi3:latest",
    model_provider="ollama",
    base_url="http://localhost:11435",
    temperature=0,
)


def embed(text: str) -> list[float]:
    return ollama.embeddings(model="nomic-embed-text:latest", prompt=text)["embedding"]


async def answer_question(query: str) -> str:
    query_embedding = embed(query)                                    #  embed the question
    results = collection.query(query_embeddings=[query_embedding], n_results=3)  # search Chroma

    # build readable context from the raw results dict
    context = "\n\n---\n\n".join(
        f"[Source: {meta['section']}]\n{doc}"
        for doc, meta in zip(results["documents"][0], results["metadatas"][0])
    )

    prompt = f"""Answer the user's question using ONLY the context below.
If the context doesn't contain the answer, say so honestly.

Context:
{context}

Question: {query}
"""
    response = await model.ainvoke([HumanMessage(content=prompt)])     # ask the model
    return response.content


async def main():
    while True:
        question = input("You: ")
        if question.lower() in ("exit", "quit"):
            break
        answer = await answer_question(question)
        print(f"Bot: {answer}\n")


if __name__ == "__main__":
    asyncio.run(main())