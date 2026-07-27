"""
Phase 1 — Verification script.

Run this AFTER phase1_ingest.py to sanity-check the chunking quality
before moving on to retrieval (Phase 2). This does NOT test retrieval
relevance yet — it just confirms the chunks themselves look clean.
"""

import chromadb

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "mattrlabs_doc"

CHUNK_SIZE_TOKENS = 400
CHUNK_OVERLAP_TOKENS = 50
# Allow some slack around the target — the recursive splitter won't always
# hit the exact number, since it prefers clean boundaries over a hard cutoff.
TOKEN_TOLERANCE = 100


def main():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(COLLECTION_NAME)

    total = collection.count()
    print(f"Total chunks stored: {total}\n")

    all_data = collection.get(include=["documents", "metadatas"])

    sections_seen = {}
    flagged = []

    for doc, meta in zip(all_data["documents"], all_data["metadatas"]):
        section = meta.get("section", "UNKNOWN")
        token_count = meta.get("token_count", 0)
        sections_seen[section] = sections_seen.get(section, 0) + 1

        if token_count > CHUNK_SIZE_TOKENS + TOKEN_TOLERANCE:
            flagged.append((section, meta.get("chunk_index"), token_count, "TOO LARGE"))

        # crude mid-sentence-cut heuristic: flag chunks that don't end in
        # sentence-ending punctuation AND aren't the last chunk of their section
        if doc.strip() and doc.strip()[-1] not in ".!?:\"')":
            flagged.append((section, meta.get("chunk_index"), token_count, "POSSIBLE MID-SENTENCE CUT"))

    print("Chunks per section:")
    for section, count in sections_seen.items():
        print(f"  {section}: {count} chunks")

    print(f"\nFlagged chunks ({len(flagged)}):")
    for section, idx, tokens, reason in flagged:
        print(f"  [{reason}] section='{section}' chunk_index={idx} tokens={tokens}")

    print("\n--- Sample chunks for manual eyeballing ---")
    # Print a few chunks in full so you can manually confirm boundaries look right
    for doc, meta in list(zip(all_data["documents"], all_data["metadatas"]))[:5]:
        print(f"\n[section: {meta.get('section')} | chunk_index: {meta.get('chunk_index')} "
              f"| tokens: {meta.get('token_count')}]")
        print(doc)
        print("-" * 60)


if __name__ == "__main__":
    main()