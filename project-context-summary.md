# Project Context — Full Summary for LLM Handoff

This document summarizes two related but separate AI agent projects, the infrastructure connecting them, and everything learned/debugged along the way. Give this to any LLM to get full context instantly.

---

## Background / Environment

- **Hardware**: Mac Mini M4, 16GB unified memory, located at an office. Development/testing also happens from a Windows laptop at home.
- **Local LLM runtime**: Ollama, running on the Mac. Primary model: `qwen3:14b` (Q4_K_M quantization, 14.8B params, supports `tools` + `thinking` capabilities, native context length 40960 — but capped via `num_ctx` in code).
- **Stable `num_ctx` setting**: `8192` (tested working; `16384` also works but pushes the Mac into partial CPU/GPU split instead of 100% GPU; `4096` caused silent context truncation bugs earlier).
- **Remote access problem**: The office network's firewall blocks Tailscale at the control-plane level (confirmed via `tailscale netcheck`/`tailscale status` showing the coordination server unreachable, even though UDP itself tested as open). Solved using **Cloudflare Tunnel** instead, which uses outbound-only HTTPS and gets through the same firewall Tailscale couldn't.
- **Remote access solution (final, working)**: Cloudflare Tunnel + a domain (`mattrlabs.online`, added to Cloudflare) + SSH port-forwarding.
  - Daily workflow: on the Windows laptop, run:
    ```
    ssh -N -L 11434:localhost:11434 -o ProxyCommand="cloudflared access ssh --hostname ssh.mattrlabs.online" inunity@ssh.mattrlabs.online
    ```
    (foreground mode, NOT `-f` background — `-f` was unreliable on this Windows OpenSSH client, foreground works reliably)
  - Then verify with `curl http://localhost:11434/api/tags` — must show `qwen3:14b`, not a local Windows Ollama instance (a recurring gotcha: Windows has its own local Ollama running `gemma4:e4b` that squats on port 11434 if not killed first via `netstat -ano | findstr :11434` + `taskkill`).
  - The Mac's own Ollama binds to `localhost:11434` only (not `0.0.0.0`) — this is correct and intentional for the SSH-tunnel approach, since forwarded traffic arrives as a local connection on the Mac's side.

---

## Project 1 — Multi-Agent Trip Planner (COMPLETE, working end-to-end)

### Architecture
A LangGraph/LangChain-based Main Agent that orchestrates four specialist sub-agents, each exposed as a `@tool`-wrapped function. No LangGraph supervisor graph — just `create_agent` with a tool list.

```
Main Agent (qwen3:14b)
   |-- call_travel_agent   -> TransportResult (mode, cost, duration, attractions, hotels) -- uses Tavily web search
   |-- call_weather_agent  -> WeatherResult (summary, temperature, rain_probability, recommendation) -- uses an MCP weather server
   |-- call_food_agent     -> FoodResult (dishes, restaurants, average_cost) -- uses Tavily web search
   `-- call_budget_agent   -> BudgetResult (estimated_cost, remaining_budget, budget_status, suggestions) -- pure calculation, no search
Main Agent -> update_trip_state(updates) -> writes into shared TripState (extends AgentState)
```

### Key files
- `main.py` — Main Agent definition, system prompt, invocation
- `subagent.py` — the four sub-agent definitions + their tool wrappers
- `state.py` — `TripState` schema + `update_trip_state` tool
- `schema.py` — Pydantic models: `TransportResult`, `WeatherResult`, `FoodResult`, `BudgetResult`
- `tools.py` — `web_search` (Tavily)
- `whether_mcp.py` — MCP client wrapper for the weather server (stdio transport, `python -m mcp_weather_server`)

### Bugs found and fixed along the way (all resolved)
1. **JSON parsing failures** — Qwen3 inconsistently returns either a proper `structured_response` (tool-call route) or raw JSON as plain text, sometimes wrapped in markdown code fences. Fixed with a `clean_json_content()` helper (strips fences via regex) applied before `json.loads()`, plus checking `response.get("structured_response")` first before falling back to text parsing, in all four sub-agent tool wrappers.
2. **`TripState` schema bugs** — duplicate `estimated_cost` field (declared twice, second silently won); flat `food`/`weather` string fields couldn't hold the actual multi-field `FoodResult`/`WeatherResult` data. Fixed by adding explicit fields per sub-agent result (`transport_cost`, `weather_summary`, `temperature`, `rain_probability`, `weather_recommendation`, `dishes`, `restaurants`, `average_food_cost`, `estimated_cost`, `remaining_budget`, `budget_status`, `budget_suggestions`).
3. **`update_trip_state` missing `ToolMessage`** — LangGraph requires every tool call to have a matching `ToolMessage` reply in `Command.update`; the original implementation only updated state fields without adding a `ToolMessage`, causing a hard error. Fixed by including a `ToolMessage(content=..., tool_call_id=runtime.tool_call_id)` in the `Command(update={...})` return.
4. **Context window truncation at `num_ctx=4096`** — likely caused several earlier bugs (e.g., the travel agent "forgetting" its JSON-only instruction and returning a full prose itinerary instead) — fixed by raising to `8192`/`16384`.
5. **Incomplete sub-agent queries** — the Main Agent sometimes called `call_travel_agent` without including source/days, causing the sub-agent to ask a clarifying question in prose (which then crashed the JSON parser). Fixed by tightening the Main Agent's system prompt to explicitly require complete query strings at each step.
6. **Copy-paste bugs** — mislabeled debug print statements and error messages across the four sub-agent functions (all said "Budget agent" regardless of which function threw); `call_budget_agent` briefly returned `FoodResult(**data)` instead of `BudgetResult(**data)`.
7. **Occasional redundant looping** — the Main Agent sometimes re-called `call_travel_agent`/`call_budget_agent` a second time before producing the final answer, likely due to an incomplete first `call_budget_agent` query (missing hotel cost) triggering a self-correction. Not fully eliminated, but harmless — pipeline still completes correctly.

### Status: fully working
Confirmed producing a complete, correct final itinerary end-to-end, including successfully running over the Cloudflare Tunnel + SSH remote setup.

### Known remaining improvement ideas (not yet done)
- Consider a sequential tool-gating wrap-middleware to enforce travel-weather-food-budget ordering at the code level instead of relying purely on prompt instructions, closing the redundant-looping issue properly.
- Consider `SummarizationMiddleware` to bound message history growth in longer sessions.
- Consider swapping to a smaller model (`qwen3.5:9b` or `qwen3:8b`) for faster responses — analysis suggested `qwen3:14b` was oversized for the 16GB Mac Mini's practical ~11GB usable memory ceiling.
- `/no_think` prompt suffix suggested to skip Qwen3's reasoning trace for faster responses — not yet confirmed applied.

---

## Project 2 — Gmail Assistant Agent (IN PROGRESS, debugging auth flow)

### Architecture
A separate LangChain/LangGraph project (own folder: `gmail_AI/`) using MCP for Gmail access, with an authentication gate before email tools unlock, and Human-in-the-Loop (HITL) approval before sending emails.

```
User <-> Main Gmail Agent (qwen3:14b)
         tools: authenticate, ask_user, search_emails, read_email, send_email (Gmail via MCP)
         middleware: dynamic_tool_call, dynamic_prompt_call (wrap-style),
                     HumanInTheLoopMiddleware (interrupt_on={"send_email": True})
         state_schema: agent_state (user, password, auth: bool)
```

### Key files (as currently structured)
- `main.py` — agent setup, chat loop, interrupt-handling logic
- `tools.py` — `get_gmail_tool()` (MCP client), `ask_user`, `authenticate`
- `MiddleWare.py` — `dynamic_tool_call`, `dynamic_prompt_call` (both need to be `async def`)
- `state.py` — `agent_state` schema
- `praser.py` — `classify_with_phi3` (small-model classifier for translating natural language HITL replies into structured decisions)
- `gmail_prompt.py` — the Gmail system prompt
- `server/gmail_server.py` — self-written MCP server exposing `search_emails`, `read_email`, `send_email` via the Gmail API (Google Cloud OAuth: `credentials.json` + `token.json`, both gitignored)

### Core design decisions
1. **`authenticate` is a tool, not middleware** — the model calls it with user-supplied `username`/`password` as arguments; the tool itself does REAL verification (`username == stored_user and password == stored_password`, reading from `runtime.state`) and returns `Command(update={"auth": is_valid, "messages": [...]})`. The model must never be trusted to decide the auth boolean itself.
2. **`dynamic_tool_call`** (wrap-style middleware) — checks `request.state.get("auth")` on every model call; if `False`, restricts `request.tools` to only `[authenticate]`; if `True`, allows everything except `authenticate`.
3. **`dynamic_prompt_call`** (wrap-style middleware) — similarly adjusts the system prompt text based on auth state.
4. **`ask_user`** tool — uses LangGraph's `interrupt()` primitive directly (not `HumanInTheLoopMiddleware`) to pause and ask the user a question, tagging its payload with `{"type": "ask_user", "question": ...}` for later routing. Whatever is passed to `Command(resume=...)` becomes the tool's return value directly (plain string, not a structured dict).
5. **`HumanInTheLoopMiddleware`** gates `send_email` specifically (`interrupt_on={"send_email": True}`) — this is a different interrupt mechanism than `ask_user`'s manual `interrupt()`, with its own fixed resume schema: `Command(resume={"decisions": [{"type": "approve"|"reject"|"edit", ...}]})`.
6. **Dual-interrupt routing** — since both `ask_user` and `HumanInTheLoopMiddleware` produce a `"__interrupt__"` key in the response, the chat loop must distinguish them: check `value.get("type") == "ask_user"` (resume with a raw string + `config=config`) vs. `"action_requests" in value` (resume with the structured decisions format, `config=config`).
7. **`classify_with_phi3`** — a small separate model (Phi3, chosen for speed + reliable narrow-task instruction-following) used to translate the user's free-text reply (e.g. "yeah send it") into a structured decision dict for the HITL resume. Industry-practice research suggested keeping the actual approve/reject moment simple/unambiguous rather than fully trusting a parsed natural-language classification for something irreversible — so a confirmation step ("I understood this as: APPROVE — correct?") was added before actually resuming, as a safety net against misclassification.

### Bugs found and fixed
1. **`ToolMessage(conent=...)` typo** (missing "t" in "content") in both `update_state` and `authenticate` tools — caused silent/crashing behavior, contributing to the "wrong credentials still let me through" bug.
2. **`authenticate` originally took a generic `updates: dict`** instead of typed `username`/`password` parameters, meaning the model could set `auth: True` itself with no real verification — the core security bug. Fixed by requiring explicit typed parameters and doing the real comparison inside the tool.
3. **`agent_state`'s class-body defaults never applied** — `AgentState` is a `TypedDict` under the hood, and `TypedDict` does not support runtime default values (they're silently ignored — no error, no warning). Confirmed via `agent.get_state(config)` showing an empty state despite defaults being written in the class. Fix applied: explicitly pass `user`, `password`, `auth` as part of the state dict in the first `ainvoke()` call for a new thread; the checkpointer persists them across subsequent turns. LangChain's own docs confirm: use a `dataclass` instead of `TypedDict`-based `AgentState` if real defaults are wanted — not yet tested whether `@dataclass` cleanly subclasses `AgentState` (likely a metaclass conflict) — safer to build a standalone dataclass not inheriting from `AgentState`, if pursuing this cleanup later. Current working fix (explicit initial state) is valid and sufficient for now.
4. **`wrap_model_call` sync/async mismatch** — `NotImplementedError: Asynchronous implementation of awrap_model_call is not available` — occurred because `dynamic_tool_call`/`dynamic_prompt_call` were written as plain `def` while the agent is invoked via `await agent.ainvoke(...)`. Fixed by making both middleware functions `async def`, with `return await handler(request)` instead of `return handler(request)`. General rule established: MCP tools force async at the top level; every middleware in the chain must then also be async — no automatic sync-to-async bridging exists for middleware (unlike plain tools, which LangChain does auto-bridge via a background thread executor).
5. **MCP server launch config bug** — passing `-m` together with a file path is invalid; `-m` expects a dotted module name, not a file path. Fixed by dropping `-m` and just passing the file path directly as the arg.
6. **`collection.query()` passed raw text instead of an embedding** — caused a ValueError about expecting a list of floats. Caused by forgetting to call `embed(query)` before passing to `query_embeddings=[...]`. Fixed by always embedding the query string first.

### Status: in progress
Auth flow logic is now believed correct after the fixes above, but not yet fully re-confirmed end-to-end in a clean test run (wrong-credentials-rejected + correct-credentials-accepted + tools unlock correctly + send_email HITL flow all working together in one continuous conversation).

### Not yet done / next steps for this project
- Re-run a full clean test: wrong credentials -> rejected, stays gated; correct credentials -> accepted, tools unlock; attempt send_email -> HITL interrupt fires -> test approve/reject/edit paths.
- Confirm `classify_with_phi3`'s fence-stripping + try/except safety net (falls back to a reject decision on parse failure) is actually in place.
- Confirm the final print of the bot's response is present after the interrupt-handling loop in `main.py` (was flagged as missing in one review pass).

---

## Project 3 — RAG System (IN PROGRESS)

### Goal
A mid-level (not enterprise-grade) RAG system, initially planned around Gmail data but pivoted to use a fake company's plain-text dataset instead (not email content). Designed as "Agentic RAG" — retrieval exposed as an agent tool, not a hardcoded always-on retrieval step.

### Stack
- **Embeddings**: `nomic-embed-text` via Ollama (local, no cloud API)
- **Vector store**: ChromaDB, `PersistentClient(path="./chroma_db")` — a local folder on disk, no external service/account
- **Retrieval integration pattern**: exposed as a `@tool` (`search_company_knowledge`) that the Main Agent calls when relevant, rather than always-on retrieval-then-generate — chosen specifically because the agent has other capabilities too and shouldn't force retrieval on every turn.

### Planned/actual pipeline phases
1. Load plain text company data from a file
2. Chunk it (simple sliding-window chunker: chunk_size=500, overlap=50)
3. Embed each chunk via Ollama's embeddings endpoint with the nomic-embed-text model
4. Store in Chroma (ids, embeddings, documents, metadatas)
5. On a query: embed the query, then query Chroma for the top-k most similar chunks
6. Feed retrieved documents back to the LLM as context to generate a grounded answer

### Bugs found and fixed
- `collection.query()` given raw text instead of an embedding (same underlying mistake as in Project 2's bug list — fixed the same way: always call embed(query) first).

### Status: core embedding/retrieval pipeline confirmed working (embedding bug fixed)
Integration into a full agent with a retrieval tool + system prompt is understood conceptually and partially planned but not yet fully built/tested end-to-end.

### New idea under consideration (not yet started)
Self-hosting this RAG system publicly, using the already-existing Cloudflare Tunnel infrastructure from the remote-access setup:
- Wrap the RAG agent in a minimal FastAPI app (a POST endpoint that calls agent.ainvoke(...) and returns the answer) — user is a FastAPI/web-serving beginner, needs this explained from scratch.
- Add a second ingress rule to the same existing cloudflared config file (already has one rule for SSH) — e.g. a new subdomain pointing to the local FastAPI port — reusing the same tunnel, same authenticated Cloudflare account (no new login needed, tunnel login was already done once).
- Route DNS for the new subdomain, restart the tunnel service to pick up the config change.
- Key clarified concept: this is not like Vercel/GitHub-based hosting (where code is pushed and runs on someone else's infrastructure) — nothing moves off the Mac. The tunnel just makes an already-locally-running service (Ollama + ChromaDB + the Python RAG agent, all still on the Mac) reachable from the internet. The Mac must stay on/connected for it to work; if the Mac is off, the hosted service is down.
- Response time impact of this approach: negligible added latency (roughly 50-200ms network/tunnel overhead) compared to the model's own inference time (20-40+ seconds observed for qwen3:14b locally) — network hop is not the bottleneck; local inference speed remains the dominant factor either way.
- Security note flagged but not yet implemented: unlike the SSH tunnel (which requires SSH auth), a public FastAPI endpoint would have no built-in auth — would need an API key check or Cloudflare Access in front of it before genuinely exposing this publicly, to avoid abuse/unauthorized use of local compute.

---

## Cross-cutting technical concepts learned (apply to all projects)

- **Sync vs. async**: async functions can be paused (await) while other work happens elsewhere in the event loop; sync functions block fully until done. Once anything in a call chain needs await, everything above it in that chain must also be async def. MCP client libraries are async-only, which is why both projects ended up needing async throughout their agent/middleware layers. Async does not, by itself, speed up a purely sequential chain of dependent steps — its real benefit is enabling independent tasks to run concurrently (e.g. asyncio.gather() for the trip planner's non-dependent sub-agent calls — not yet implemented, but identified as a real potential optimization).
- **TypedDict vs dataclass vs Pydantic BaseModel for agent state**: TypedDict (what AgentState uses) has zero runtime behavior for defaults — it's purely a static type-checking annotation. dataclass supports real defaults via a generated __init__. Pydantic BaseModel (with extra="forbid") offers the strictest validation, useful for catching malformed/unexpected state fields in production, at higher complexity/perf cost than the other two.
- **Command + ToolMessage mechanics**: any tool that returns Command(update={...}) to modify state directly (bypassing the automatic sync-tool-to-ToolMessage wrapping LangGraph normally provides) must manually include a ToolMessage(content=..., tool_call_id=runtime.tool_call_id) in that update, or LangGraph raises a validation error about a missing matching reply to the tool call.
- **Wrap-style middleware** (the wrap_model_call decorator, receiving a ModelRequest/handler pair) runs unconditionally before every model call; any conditional behavior comes from your own if-logic inside it, not from an external trigger. request.override(tools=..., system_prompt=...) is the mechanism for dynamically changing what's available to the model. Multiple wrap middlewares nest in list order (first one wraps the rest).
- **Interrupt/resume mechanics**: two distinct mechanisms exist — (1) manual interrupt() calls inside your own tool code (you define the resume payload shape yourself, since you also define what the tool does with it), and (2) HumanInTheLoopMiddleware's built-in gating (fixed resume schema with a list of decisions, each having a type of approve/reject/edit). Both surface identically as a "__interrupt__" key in the response dict, requiring your own routing logic to tell them apart if both are used in the same agent.
- **Local model JSON reliability**: local models (tested: Qwen3:14B, Phi3) are not perfectly consistent in emitting clean, fence-free JSON even with explicit prompt instructions not to use markdown — always defensively strip fences before parsing and wrap parsing in try/except with a safe fallback, rather than assuming compliance.
- **MCP server naming**: tool names exposed by an MCP server are defined by that server's author and must be discovered by actually printing the tool names — never guessed — since middleware/gating configs rely on exact string matches that fail silently (no error) if wrong.

---

## Open questions / things not yet resolved
- Whether @dataclass can cleanly subclass AgentState (untested) — current workaround (explicit initial state in first ainvoke() call) works and is sufficient for now.
- Full end-to-end clean re-test of the Gmail auth + HITL flow after all fixes applied.
- Whether to actually proceed with publicly hosting the RAG system via a second Cloudflare Tunnel ingress rule — conceptually understood and planned, not yet built.
- Trip planner's occasional redundant tool-call looping — functional but not fully root-caused/eliminated.
