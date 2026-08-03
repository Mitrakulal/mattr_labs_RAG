from spliting import split_into_sections,chunk_sections,load_source_text,token_length
from storing import embed ,store_chunks

def main():
    
    SOURCE_FILE="mattrlabs-website-content-enrichment-combined.txt"
    
    full_text=load_source_text(SOURCE_FILE)
    
    print("Splitting into labeled sections...")
    sections = split_into_sections(full_text)
    print(f"  Found {len(sections)} sections: {[s['section'] for s in sections]}")
    
    print("\nChunking each section (recursive, token-based)...")
    chunks = chunk_sections(sections)
    print(f"  Produced {len(chunks)} total chunks.")
    
    print("\nEmbedding and storing chunks in ChromaDB...")
    store_chunks(chunks)
 
 
if __name__ == "__main__":
    main()
 
    