# Multi-Article Agentic QA Pipeline

Generates open-ended, multi-article research questions with grounded reference answers
and QUEST-style rubric trees. Adapts the QUEST paper's proposer pattern to a local
Wikipedia corpus using DeepSeek V4 Flash with tool use.

Script: `generate_deepresearch_qa_multi.py`

---

## Pipeline Overview

```
Phase 1: Load articles -> chunk -> hybrid search index (TF-IDF + vec)
Phase 2: Entity/keyword extraction per article (parallel LLM calls)
Phase 3: Agentic exploration with search/read tools (15-100 tool calls)
Phase 4: Synthesis -> structured question + grounded reference answer
Phase 5: Rubric tree generation (4 branches, QUEST-style)
```

Each phase feeds into the next. Phases 2-5 use DeepSeek V4 Flash via the OpenAI
API. Phase 3 uses tool calling (not `json_object` mode); phases 2, 4, 5 use
`response_format={"type": "json_object"}`.

---

## Phase 1: Chunking and Indexing

Articles are loaded from the pre-downloaded `wikipedia_en.jsonl` and split into
chunks using recursive splitting: first by double newline (sections), then by
single newline (paragraphs/headers), with a hard limit of ~1000 words per chunk.

```python
def chunk_article(text: str, max_words: int = 1000) -> list[str]:
    sections = text.split("\n\n")
    chunks: list[str] = []
    for section in sections:
        words = section.split()
        if len(words) <= max_words:
            chunks.append(section)
        else:
            paragraphs = section.split("\n")
            # accumulate paragraphs until hitting word limit, then split
            ...
    return chunks
```

Chunks are indexed in a `HybridSearchIndex` that combines TF-IDF (scikit-learn)
with dense vector embeddings (sentence-transformers `all-mpnet-base-v2`). At query
time, scores are blended with a 0.4/0.6 TF-IDF/vec weight:

```python
hybrid_scores = 0.4 * tfidf_scores + 0.6 * vec_scores
top_indices = np.argsort(hybrid_scores)[::-1][:top_k]
```

For 30 articles this produces ~2400 chunks. Embedding takes ~30s on CPU.

---

## Phase 2: Entity Extraction

Each article gets a parallel LLM call to extract named entities, domain keywords,
and high-level themes. These become seed keywords for the exploration agent.

```python
async def extract_all_entities(client, articles):
    tasks = [extract_entities(client, a) for a in articles]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    ...
```

Output per article: `EntityExtractionResponse(entities, keywords, themes)`.
These are passed to Phase 3 as the agent's starting context.

---

## Phase 3: Agentic Exploration

This is the core of the pipeline. A DeepSeek V4 Flash agent explores the corpus
using two tools:

- `search(query)` — hybrid search over all chunks, returns top-5 results
- `read_article(article_id, offset, limit)` — read raw article text at any offset

The agent's system prompt encodes a 3-stage exploration protocol adapted from
the deepresearch skill:

**Stage 1 — Breadth-First Survey (~5 tool calls):** Broad thematic searches,
skim multiple articles, identify the richest content.

**Stage 2 — Targeted Deep Dives (~5 tool calls):** Read promising articles at
multiple offsets (not just the intro), search for specific cross-references.

**Stage 3 — Cross-Referencing & Refinement (~5 tool calls):** Intersection
queries combining concepts from different articles, unexpected juxtapositions,
verification of specific claims.

### Tool call boundaries

The agent loop enforces minimum and maximum tool call counts:

```python
MIN_TOOL_CALLS = 15   # agent is pushed back if it tries to stop early
MAX_TOOL_CALLS = 100  # tools removed from API call to force summary
MAX_AGENT_ITERATIONS = 125
```

Three regimes:
- **Below MIN:** tools always provided. If the agent tries to stop, it gets
  nudged to keep exploring with varied queries and unexplored articles.
- **Between MIN and MAX:** tools provided, agent decides when to stop naturally.
- **Above MAX:** tools removed from the API call, agent is told to write its
  final summary.

```python
if msg.tool_calls:
    total_tool_calls += n_calls
    # execute tool calls, append results
    if total_tool_calls >= MAX_TOOL_CALLS:
        messages.append({"role": "user", "content": "Write your summary now."})
elif total_tool_calls < MIN_TOOL_CALLS:
    # push back — agent tried to stop too early
    messages.append({"role": "user", "content": "Keep exploring..."})
else:
    # agent chose to stop, accept the summary
    return exploration_notes, retrieved_passages
```

### Output

The agent produces two things:
1. **Exploration notes** — a structured summary of cross-article connections,
   evidence inventory, and proposed research question (~15K chars).
2. **Retrieved passages** — all chunks returned by search calls during
   exploration (~170 passages for 30 articles), deduplicated in Phase 4.

---

## Phase 4: Synthesis

A separate `json_object` LLM call receives the exploration notes and retrieved
passages, and produces a structured output:

```python
class SynthesisResponse(BaseModel):
    question: str            # open-ended multi-article research task
    reference_answer: str    # 400-700 word grounded report
    articles_used: list[int] # article IDs used in the answer
```

The synthesis prompt requires:
- Questions that need 2-3+ articles to answer
- Reference answers grounded in specific facts, dates, names from source articles
- Cross-article connections and analytical depth

---

## Phase 5: Rubric Tree

A final `json_object` call generates a QUEST-style rubric tree with 4 fixed
branches and adaptive leaves:

| Branch | Weight | Purpose |
|--------|--------|---------|
| Instruction Following | ~0.20 | Does the report address the task? |
| Comprehensiveness | ~0.30 | Does it cover all key subtopics? |
| Readability | ~0.25 | Is it well-structured and clear? |
| Insight | ~0.25 | Does it offer non-obvious analysis? |

Each branch has 2-4 leaf criteria with verification descriptions. Leaf weights
within each branch sum to 1.0.

---

## Output Format

Each run appends one JSONL line to `deepresearch_qa_multi.jsonl`:

```json
{
  "question": "open-ended multi-article research task",
  "rubric_tree": {"children": [...]},
  "reference_answer": "multi-paragraph grounded report",
  "articles_used": [{"article_id": 0, "title": "Anarchism"}, ...],
  "seed_keywords": {"0": {"entities": [...], "keywords": [...], "themes": [...]}},
  "exploration_log": "agent's full exploration summary",
  "generation_method": "multi_article_agentic"
}
```

---

## DeepSeek V4 Flash: Tool Use Quirks

- Tool calling works with `tools` parameter (OpenAI-compatible)
- `tool_choice="required"` is NOT supported in thinking mode — use default `auto`
- Cannot combine `tools` with `response_format={"type": "json_object"}` — these
  are used in separate calls (tool use for exploration, json_object for synthesis)
- Do NOT pass `temperature` (unsupported for this model)

---

## Example Run (30 articles)

```
Phase 1: 30 articles -> 2442 chunks, index built in ~35s
Phase 2: 30 parallel entity extractions in ~10s
Phase 3: 25 iterations, 70 tool calls, 170 passages in ~2.5min
Phase 4: Synthesis in ~15s
Phase 5: Rubric tree in ~12s
Total: ~3.5 minutes
```

**Sample question:**

> How do different philosophical frameworks -- Aristotelian virtue ethics, Ayn
> Rand's Objectivism, the altruism-egoism debate, and anarchist political
> philosophy -- define the relationship between the individual, ethics, and
> legitimate authority, and where do these frameworks converge, diverge, and
> critique one another?

**Articles used:** Anarchism, Aristotle, Altruism, Ayn Rand

**Reference answer:** 604 words, with direct quotes from source articles,
specific dates (384-322 BC, 1798-1857, 1905-1982), named works (The Virtue
of Selfishness, Mutual Aid: A Factor of Evolution), and a traced chain of
influence (Aristotle -> Rand -> altruism -> anarchism via Kropotkin).

---

## Usage

```bash
# Default: 10 articles
uv run python generate_deepresearch_qa_multi.py

# Custom article count
uv run python generate_deepresearch_qa_multi.py 30
```

Requires `DEEPSEEK_API_KEY` env var. Logs to `logs/generate_deepresearch_qa_multi.log`.
