from fastapi.responses import StreamingResponse

@app.post("/ask")
async def ask(body: AskRequest):
    start = time.time()
    query = body.question

    # step 1: embed
    try:
        query_embedding = embed(query)
    except Exception as e:
        print(f"[ERROR] embed failed: {e}")
        log_entry({"query": query, "error": f"embed failed: {e}", "latency_s": round(time.time() - start, 2)})
        return AskResponse(answer="I'm having trouble connecting to the knowledge base.")

    # step 2: search chroma
    try:
        results = collection.query(query_embeddings=[query_embedding], n_results=3)
    except Exception as e:
        print(f"[ERROR] chroma query failed: {e}")
        log_entry({"query": query, "error": f"chroma query failed: {e}", "latency_s": round(time.time() - start, 2)})
        return AskResponse(answer="I'm having trouble searching the knowledge base.")

    context = "\n\n---\n\n".join(
        f"[Source: {meta['section']}]\n{doc}"
        for doc, meta in zip(results["documents"][0], results["metadatas"][0])
    )

    prompt = f"""You are a mattrlabs assistant. Answer using ONLY the context. If not in context, say "I don't know."

    Context:
    {context}

    Question: {query}"""

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