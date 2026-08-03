"""
Phase 2 — Retrieval correctness testing.

Run this AFTER phase1_ingest.py (with the corrected section detection)
has populated the ChromaDB collection. This script embeds a set of
test queries with KNOWN expected answers, and checks whether the
correct chunk/section actually comes back near the top of the results.

This is NOT about whether querying "works" mechanically (that was
proven in earlier debugging) -- it's about whether retrieval is
actually RELEVANT, which is the real thing Phase 2 needs to confirm
before building the agent on top of it.
"""

import ollama
import chromadb

CHROMA_PATH = "./chroma_db1"
COLLECTION_NAME = "mattrlabs_doc1"
EMBED_MODEL = "nomic-embed-text"

N_RESULTS = 3  # how many top chunks to retrieve per query


def embed(text: str) -> list[float]:
    return ollama.embeddings(model=EMBED_MODEL, prompt=text)["embedding"]


# ---------------------------------------------------------------------
# Test cases: each has a query, and the section we EXPECT to see in the
# top result. Add more of your own as you think of edge cases.
# ---------------------------------------------------------------------
TEST_CASES = [
    {
        "query": "What does mattrlabs actually build?",
        "expected_section": "HOMEPAGE",
    },
    {
        "query": "What's the pilot program for the camera?",
        "expected_section": "AI PTZ CAMERA (Product) Page",
    },
    {
        "query": "How much does a small project cost?",
        "expected_section": "CONTACT Page",
    },
    {
        "query": "What file formats do you accept for 3D printing?",
        "expected_section": "3D PRINTING Page",
    },
    {
        "query": "What are the four engineering disciplines at the studio?",
        "expected_section": "STUDIO (About) Page",
    },
    {
        "query": "What's the target video resolution for the PTZ camera?",
        "expected_section": "AI PTZ CAMERA (Product) Page",
    },
]


def run_test_case(collection, test_case: dict) -> dict:
    query = test_case["query"]
    expected_section = test_case["expected_section"]

    query_embedding = embed(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=N_RESULTS,
    )

    top_sections = [meta.get("section") for meta in results["metadatas"][0]]
    top_match_correct = (len(top_sections) > 0 and top_sections[0] == expected_section)
    expected_in_top_n = expected_section in top_sections

    return {
        "query": query,
        "expected_section": expected_section,
        "top_sections_returned": top_sections,
        "top_match_correct": top_match_correct,
        "expected_in_top_n": expected_in_top_n,
        "top_chunk_preview": results["documents"][0][0][:150] if results["documents"][0] else "(none)",
    }


def main():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(COLLECTION_NAME)

    print(f"Running {len(TEST_CASES)} retrieval test cases against '{COLLECTION_NAME}'...\n")

    passed = 0
    for test_case in TEST_CASES:
        result = run_test_case(collection, test_case)

        status = "PASS" if result["top_match_correct"] else (
            "PARTIAL" if result["expected_in_top_n"] else "FAIL"
        )
        if result["top_match_correct"]:
            passed += 1

        print(f"[{status}] Query: {result['query']}")
        print(f"  Expected section : {result['expected_section']}")
        print(f"  Top sections got : {result['top_sections_returned']}")
        print(f"  Top chunk preview: {result['top_chunk_preview']}...")
        print()

    print(f"--- Summary: {passed}/{len(TEST_CASES)} queries had the correct section as the TOP result ---")
    print("PARTIAL = correct section was retrieved, but not ranked first (may still be usable).")
    print("FAIL = correct section did not appear in the top results at all -- worth investigating.")


if __name__ == "__main__":
    main()