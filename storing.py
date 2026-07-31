import ollama
import chromadb
from spliting import token_length

EMBED_MODEL = "nomic-embed-text"
COLLECTION_NAME="mattrlabs_doc"
CHORMA_PATH = "./chroma_db"

client=chromadb.PersistentClient(path=CHORMA_PATH)
collection=client.get_or_create_collection(COLLECTION_NAME)
ollama_client = ollama.Client(host="http://localhost:11434")

def embed(text: str) -> list[float]:
    return ollama_client.embeddings(model="nomic-embed-text:latest", prompt=text)["embedding"]

def store_chunks(chunks:list[dict])->None:
    
    """Embed each chunk and store it in chromaDB wiht metada"""
    
    
    for i , chunk in enumerate(chunks):
        chunk_id = f"{chunk['section'].replace(' ', '_')}-{chunk['chunk_index']}"
        collection.add(
            ids=[chunk_id],
            embeddings=[embed(chunk["text"])],
            documents=[chunk["text"]],
            metadatas=[{
                "section":chunk["section"],
                "chunk_index":chunk["chunk_index"],
                "token_count":token_length(chunk["text"])
            }]
        )
        print(f"[{i + 1}/{len(chunks)}] stored chunk_id={chunk_id}"
              f"(section='{chunk['section']}', tokens={token_length(chunk['text'])})")
    print(f"\nDone. Collection '{COLLECTION_NAME}' now has {collection.count()} chunks.")