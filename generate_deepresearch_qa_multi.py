"""Generate multi-article deep research QA pairs using an agentic pipeline.

Adapts the QUEST paper's proposer pattern to local Wikipedia articles:
  Phase 1: Load N articles -> recursive chunking -> hybrid search index (TF-IDF + vec)
  Phase 2: Entity/keyword extraction per article (parallel LLM calls)
  Phase 3: Agentic exploration — LLM with search/read tools explores corpus
  Phase 4: Synthesis — structured question + grounded reference answer
  Phase 5: Rubric tree generation (QUEST-style, after question exists)

Usage:
  uv run python generate_deepresearch_qa_multi.py [N]  # N articles, default 10
"""

import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger
from openai import AsyncOpenAI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from utils.wikipedia_loader import load_articles

# --- Config ---
DEEPSEEK_MODEL = "deepseek-v4-flash"
BASE_URL = "https://api.deepseek.com"
OUTPUT_FILE = Path("deepresearch_qa_multi.jsonl")
LOG_DIR = Path("logs")
N_ARTICLES = 10
MAX_CHUNK_WORDS = 1000
EMBEDDING_MODEL = "all-mpnet-base-v2"
SEARCH_TOP_K = 5
MAX_AGENT_ITERATIONS = 15
MAX_TOOL_CALLS = 8
TFIDF_WEIGHT = 0.4
VEC_WEIGHT = 0.6

# --- Logging ---
LOG_DIR.mkdir(exist_ok=True)
logger.remove()
log_level = os.environ.get("LOGGING_LEVEL", "DEBUG").upper()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level:<7} | {message}")
logger.add(
    LOG_DIR / "generate_deepresearch_qa_multi.log",
    level=log_level,
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<7} | {message}",
    rotation="10 MB",
)


# --- Pydantic models ---


class RubricLeaf(BaseModel):
    criterion: str
    weight: float
    verification: str


class RubricBranch(BaseModel):
    criterion: str
    weight: float
    children: list[RubricLeaf]


class RubricTree(BaseModel):
    children: list[RubricBranch]


class EntityExtractionResponse(BaseModel):
    entities: list[str]
    keywords: list[str]
    themes: list[str]


class SynthesisResponse(BaseModel):
    question: str
    reference_answer: str
    articles_used: list[int]


class RubricTreeResponse(BaseModel):
    rubric_tree: RubricTree


# --- Chunker ---


def chunk_article(text: str, max_words: int = MAX_CHUNK_WORDS) -> list[str]:
    """Recursive splitting: sections -> headers/paragraphs -> word limit."""
    sections = text.split("\n\n")
    chunks: list[str] = []
    for section in sections:
        section = section.strip()
        if not section:
            continue
        words = section.split()
        if len(words) <= max_words:
            chunks.append(section)
        else:
            # Split by single newline (paragraphs/headers within section)
            paragraphs = section.split("\n")
            current: list[str] = []
            current_words = 0
            for para in paragraphs:
                para_words = len(para.split())
                if current_words + para_words > max_words and current:
                    chunks.append("\n".join(current).strip())
                    current = []
                    current_words = 0
                if para_words > max_words:
                    # Hard split long paragraph by word count
                    w = para.split()
                    for i in range(0, len(w), max_words):
                        chunk = " ".join(w[i : i + max_words])
                        if chunk.strip():
                            chunks.append(chunk.strip())
                else:
                    current.append(para)
                    current_words += para_words
            if current:
                joined = "\n".join(current).strip()
                if joined:
                    chunks.append(joined)
    return chunks


# --- Search Index ---


@dataclass
class Chunk:
    article_id: int
    title: str
    chunk_idx: int
    text: str


@dataclass
class HybridSearchIndex:
    """TF-IDF + sentence-transformer hybrid search over article chunks."""

    chunks: list[Chunk]
    tfidf: TfidfVectorizer = field(init=False)
    tfidf_matrix: Any = field(init=False)
    vec_embeddings: np.ndarray = field(init=False)
    model: SentenceTransformer = field(init=False)

    def __post_init__(self) -> None:
        texts = [c.text for c in self.chunks]
        logger.info(f"Building TF-IDF index over {len(texts)} chunks")
        self.tfidf = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = self.tfidf.fit_transform(texts)

        logger.info(f"Computing embeddings with {EMBEDDING_MODEL}")
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.vec_embeddings = self.model.encode(
            texts, show_progress_bar=True, convert_to_numpy=True
        )
        logger.info("Search index ready")

    def search(self, query: str, top_k: int = SEARCH_TOP_K) -> list[dict[str, Any]]:
        q_tfidf = self.tfidf.transform([query])
        tfidf_scores = cosine_similarity(q_tfidf, self.tfidf_matrix)[0]

        q_vec = self.model.encode([query], convert_to_numpy=True)
        vec_scores = cosine_similarity(q_vec, self.vec_embeddings)[0]

        hybrid_scores = TFIDF_WEIGHT * tfidf_scores + VEC_WEIGHT * vec_scores

        top_indices = np.argsort(hybrid_scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            c = self.chunks[idx]
            results.append({
                "article_id": c.article_id,
                "title": c.title,
                "chunk_idx": c.chunk_idx,
                "text": c.text[:800],
                "score": float(hybrid_scores[idx]),
            })
        return results


# --- LLM ---


def make_client() -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url=BASE_URL,
    )


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential_jitter(initial=0.5, max=3, jitter=1),
    retry=retry_if_exception_type((json.JSONDecodeError, AssertionError)),
    before_sleep=lambda rs: logger.warning(
        f"LLM call retry {rs.attempt_number}/5: {rs.outcome.exception()!r}"
    ),
    reraise=True,
)
async def call_llm_json(
    client: AsyncOpenAI, system: str, user: str
) -> dict[str, Any]:
    """Call DeepSeek V4 Flash with json_object response format."""
    t0 = time.monotonic()
    resp = await client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_format={"type": "json_object"},
    )
    elapsed = time.monotonic() - t0
    content = resp.choices[0].message.content
    assert content, "Empty LLM response"
    logger.debug(f"LLM json call took {elapsed:.1f}s, {len(content)} chars")
    return json.loads(content)


# --- Tool schemas ---

TOOL_SEARCH: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "search",
        "description": (
            "Search the article corpus for passages relevant to a query. "
            "Returns top-5 matching chunks with article titles and text previews."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query to find relevant passages",
                }
            },
            "required": ["query"],
        },
    },
}

TOOL_READ_ARTICLE: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "read_article",
        "description": (
            "Read a section of a specific article by its article_id. "
            "Returns text from the given character offset."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "article_id": {
                    "type": "integer",
                    "description": "The article ID to read",
                },
                "offset": {
                    "type": "integer",
                    "description": "Character offset to start from (default 0)",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max characters to return (default 3000)",
                },
            },
            "required": ["article_id"],
        },
    },
}

TOOLS = [TOOL_SEARCH, TOOL_READ_ARTICLE]


# --- Prompts ---

ENTITY_SYSTEM = """\
You are an expert at extracting key entities, topics, and keywords from text.

Given a Wikipedia article, extract:
1. Named entities (people, places, organizations, events)
2. Key domain keywords (technical terms, concepts)
3. High-level themes that connect this article to broader topics

Respond with this json structure:
{
  "entities": ["entity1", "entity2", ...],
  "keywords": ["keyword1", "keyword2", ...],
  "themes": ["theme1", "theme2", ...]
}"""

ENTITY_USER = """\
Article Title: {title}

Article Text (first 4000 chars):
{text}

Extract entities, keywords, and themes. Respond in json."""


EXPLORER_SYSTEM = """\
You are a research task proposer exploring a corpus of Wikipedia articles to design \
a challenging, multi-article open-ended research question.

Your goal: find connections across multiple articles and propose a research question \
that requires synthesizing information from at least 2-3 different articles.

You have access to two tools:
- search(query): Search the corpus for relevant passages (returns top-5 chunks)
- read_article(article_id, offset, limit): Read a section of a specific article

Strategy:
1. Start by searching for themes that might connect multiple articles
2. Read relevant sections to understand the content deeply
3. Look for cross-cutting themes, contrasts, causal connections, or comparative angles
4. When you have found a good multi-article angle, stop exploring

After exploring, write a summary of:
- What connections you found across articles
- Which articles are most relevant (by ID and title)
- What research question you would propose
- Key facts and details that should appear in a reference answer

Keep exploration focused — aim for 4-8 tool calls total."""


EXPLORER_USER = """\
Corpus contains {n_articles} articles. Seed keywords and entities per article:

{seed_info}

Available articles:
{article_list}

Explore the corpus to find cross-article connections for a research question."""


SYNTHESIS_SYSTEM = """\
You are an expert at crafting open-ended deep research tasks that require \
synthesizing information from multiple sources.

Given exploration notes from a corpus analysis, generate:
1. An open-ended research question requiring synthesis across multiple articles
2. A reference report (4-6 paragraphs, 400-700 words) grounded in the source material

The question must:
- Require information from at least 2-3 different articles
- Ask for a comprehensive analytical report, not a simple factual answer
- Be broad enough that multiple valid responses exist
- Sound like an assignment a professor or research lead would give

The reference answer must:
- Synthesize specific facts, names, dates, and details from the source articles
- Be well-structured with clear paragraphs and logical flow
- Demonstrate cross-article connections and analytical depth
- Draw from multiple articles explicitly

Respond with this json structure:
{
  "question": "open-ended multi-article research task",
  "reference_answer": "multi-paragraph reference report (400-700 words)",
  "articles_used": [article_id1, article_id2, ...]
}"""

SYNTHESIS_USER = """\
Exploration Notes:
{exploration_notes}

Retrieved Passages:
{passages}

Available Articles:
{article_list}

Generate a multi-article research question and grounded reference answer. Respond in json."""


RUBRIC_SYSTEM = """\
You are an expert at designing structured evaluation rubrics \
for open-ended deep research tasks.

Given a research question, design a rubric tree for evaluation. \
The rubric tree has exactly 4 top-level branches. \
Each branch has 2-4 leaf children. \
Top-level weights must sum to 1.0. Leaf weights within each branch must sum to 1.0.

The 4 branches:
1. "Instruction Following" (weight ~0.20) — does the report address the task? \
Leaves capture specific aspects the task demands.
2. "Comprehensiveness" (weight ~0.30) — does the report cover all important subtopics? \
Leaves name key subtopics a thorough report must include.
3. "Readability" (weight ~0.25) — is the report well-structured and clear? \
Leaves describe structural qualities.
4. "Insight" (weight ~0.25) — does the report offer non-obvious analysis? \
Leaves describe analytical depth expected.

Respond with this json structure:
{
  "rubric_tree": {
    "children": [
      {
        "criterion": "Instruction Following",
        "weight": 0.20,
        "children": [
          {"criterion": "...", "weight": 0.50, "verification": "Assess whether ..."},
          ...
        ]
      },
      ...
    ]
  }
}"""

RUBRIC_USER = """\
Research Question:
{question}

This question requires synthesis across these articles: {articles_used}

Design a rubric tree for evaluating responses. Respond in json."""


# --- Phase 2: Entity extraction ---


async def extract_entities(
    client: AsyncOpenAI, article: dict,
) -> EntityExtractionResponse:
    text = article["text"][:4000]
    user = ENTITY_USER.format(title=article["title"], text=text)
    raw = await call_llm_json(client, ENTITY_SYSTEM, user)
    return EntityExtractionResponse.model_validate(raw)


async def extract_all_entities(
    client: AsyncOpenAI, articles: list[dict],
) -> dict[int, EntityExtractionResponse]:
    tasks = [extract_entities(client, a) for a in articles]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    entities: dict[int, EntityExtractionResponse] = {}
    for article, result in zip(articles, results):
        if isinstance(result, Exception):
            logger.warning(
                f"Entity extraction failed for article {article['article_id']}: {result!r}"
            )
        else:
            entities[article["article_id"]] = result
    return entities


# --- Phase 3: Agent exploration ---


def _msg_to_dict(msg: Any) -> dict[str, Any]:
    """Convert ChatCompletionMessage to a dict safe for re-sending."""
    d: dict[str, Any] = {"role": msg.role}
    if msg.content is not None:
        d["content"] = msg.content
    if msg.tool_calls:
        d["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in msg.tool_calls
        ]
    return d


async def agent_explore(
    client: AsyncOpenAI,
    articles: list[dict],
    entities: dict[int, EntityExtractionResponse],
    index: HybridSearchIndex,
) -> tuple[str, list[dict[str, Any]]]:
    """Run agentic exploration with search/read tools.

    Returns (exploration_notes, retrieved_passages).
    """
    # Build seed info
    seed_lines = []
    for a in articles:
        aid = a["article_id"]
        if aid in entities:
            e = entities[aid]
            seed_lines.append(
                f"Article {aid} '{a['title']}': "
                f"entities={e.entities[:5]}, keywords={e.keywords[:5]}, "
                f"themes={e.themes}"
            )
    seed_info = "\n".join(seed_lines)

    article_list = "\n".join(
        f"  - Article {a['article_id']}: '{a['title']}' ({len(a['text'])} chars)"
        for a in articles
    )

    article_map = {a["article_id"]: a for a in articles}
    retrieved_passages: list[dict[str, Any]] = []

    def handle_search(args: dict[str, Any]) -> str:
        query = args.get("query", "")
        results = index.search(query, top_k=SEARCH_TOP_K)
        retrieved_passages.extend(results)
        logger.debug(f"search('{query}') -> {len(results)} results")
        return json.dumps(results, indent=2)

    def handle_read_article(args: dict[str, Any]) -> str:
        aid = args.get("article_id", 0)
        offset = args.get("offset", 0)
        limit = args.get("limit", 3000)
        if aid not in article_map:
            return json.dumps({"error": f"Article {aid} not found"})
        text = article_map[aid]["text"]
        snippet = text[offset : offset + limit]
        logger.debug(
            f"read_article({aid}, offset={offset}, limit={limit}) -> {len(snippet)} chars"
        )
        return json.dumps({
            "article_id": aid,
            "title": article_map[aid]["title"],
            "offset": offset,
            "text": snippet,
            "total_chars": len(text),
        })

    tool_handlers = {
        "search": handle_search,
        "read_article": handle_read_article,
    }

    user_prompt = EXPLORER_USER.format(
        n_articles=len(articles),
        seed_info=seed_info,
        article_list=article_list,
    )
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": EXPLORER_SYSTEM},
        {"role": "user", "content": user_prompt},
    ]

    total_tool_calls = 0

    for iteration in range(MAX_AGENT_ITERATIONS):
        # After enough tool calls, drop tools to force a summary response
        use_tools = total_tool_calls < MAX_TOOL_CALLS

        t0 = time.monotonic()
        kwargs: dict[str, Any] = {
            "model": DEEPSEEK_MODEL,
            "messages": messages,
        }
        if use_tools:
            kwargs["tools"] = TOOLS
        resp = await client.chat.completions.create(**kwargs)
        elapsed = time.monotonic() - t0
        msg = resp.choices[0].message

        n_calls = len(msg.tool_calls or [])
        logger.info(
            f"Agent iter {iteration + 1}: "
            f"finish={resp.choices[0].finish_reason}, "
            f"tool_calls={n_calls}, total={total_tool_calls + n_calls}, "
            f"{elapsed:.1f}s"
        )

        messages.append(_msg_to_dict(msg))

        if msg.tool_calls:
            total_tool_calls += n_calls
            for tc in msg.tool_calls:
                handler = tool_handlers.get(tc.function.name)
                if handler:
                    args = json.loads(tc.function.arguments)
                    result = handler(args)
                else:
                    result = json.dumps({"error": f"Unknown tool: {tc.function.name}"})
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })
            # If we just crossed the threshold, nudge the agent to wrap up
            if total_tool_calls >= MAX_TOOL_CALLS:
                messages.append({
                    "role": "user",
                    "content": (
                        "You have gathered enough information. "
                        "Now write your exploration summary with the connections "
                        "you found, relevant articles, and proposed research question."
                    ),
                })
        else:
            exploration_notes = msg.content or ""
            logger.info(
                f"Agent done after {iteration + 1} iterations, "
                f"{total_tool_calls} tool calls, "
                f"{len(retrieved_passages)} passages retrieved"
            )
            return exploration_notes, retrieved_passages

    logger.warning(f"Agent hit max iterations ({MAX_AGENT_ITERATIONS})")
    last_content = ""
    for m in reversed(messages):
        if m.get("role") == "assistant" and m.get("content"):
            last_content = m["content"]
            break
    return last_content, retrieved_passages


# --- Phase 4: Synthesis ---


async def synthesize_qa(
    client: AsyncOpenAI,
    exploration_notes: str,
    retrieved_passages: list[dict[str, Any]],
    articles: list[dict],
) -> SynthesisResponse:
    # Deduplicate passages by (article_id, chunk_idx)
    seen: set[tuple[int, int]] = set()
    unique: list[dict[str, Any]] = []
    for p in retrieved_passages:
        key = (p["article_id"], p["chunk_idx"])
        if key not in seen:
            seen.add(key)
            unique.append(p)

    passages_text = "\n\n---\n\n".join(
        f"[Article {p['article_id']} '{p['title']}' chunk {p['chunk_idx']}]\n{p['text']}"
        for p in unique[:15]
    )

    article_list = "\n".join(
        f"  - Article {a['article_id']}: '{a['title']}'"
        for a in articles
    )

    user = SYNTHESIS_USER.format(
        exploration_notes=exploration_notes,
        passages=passages_text,
        article_list=article_list,
    )
    raw = await call_llm_json(client, SYNTHESIS_SYSTEM, user)
    return SynthesisResponse.model_validate(raw)


# --- Phase 5: Rubric tree ---


async def generate_rubric(
    client: AsyncOpenAI,
    question: str,
    articles_used: list[int],
    articles: list[dict],
) -> RubricTreeResponse:
    used_titles = [
        f"'{a['title']}' (id={a['article_id']})"
        for a in articles
        if a["article_id"] in articles_used
    ]
    user = RUBRIC_USER.format(
        question=question,
        articles_used=", ".join(used_titles),
    )
    raw = await call_llm_json(client, RUBRIC_SYSTEM, user)
    return RubricTreeResponse.model_validate(raw)


# --- Main ---


async def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else N_ARTICLES

    # Phase 1: Load, chunk, index
    logger.info(f"Phase 1: Loading {n} articles and building search index")
    articles = load_articles(n)
    logger.info(f"Loaded {len(articles)} articles")

    all_chunks: list[Chunk] = []
    for a in articles:
        texts = chunk_article(a["text"])
        for i, text in enumerate(texts):
            all_chunks.append(
                Chunk(article_id=a["article_id"], title=a["title"], chunk_idx=i, text=text)
            )
        logger.info(f"  Article {a['article_id']}: '{a['title']}' -> {len(texts)} chunks")
    logger.info(f"Total chunks: {len(all_chunks)}")

    index = HybridSearchIndex(chunks=all_chunks)

    # Phase 2: Entity extraction
    logger.info("Phase 2: Extracting entities from all articles")
    client = make_client()
    entities = await extract_all_entities(client, articles)
    logger.info(f"Extracted entities from {len(entities)}/{len(articles)} articles")
    for aid, e in entities.items():
        logger.info(
            f"  Article {aid}: {len(e.entities)} entities, "
            f"{len(e.keywords)} keywords, {len(e.themes)} themes"
        )

    # Phase 3: Agent exploration
    logger.info("Phase 3: Agent exploration with tools")
    exploration_notes, retrieved_passages = await agent_explore(
        client, articles, entities, index
    )
    logger.info(f"Exploration: {len(exploration_notes)} chars, {len(retrieved_passages)} passages")

    # Phase 4: Synthesis
    logger.info("Phase 4: Synthesizing question + reference answer")
    synthesis = await synthesize_qa(
        client, exploration_notes, retrieved_passages, articles
    )
    logger.info(f"Question: {synthesis.question[:120]}...")
    logger.info(f"Articles used: {synthesis.articles_used}")
    logger.info(f"Answer: {len(synthesis.reference_answer)} chars")

    # Phase 5: Rubric tree
    logger.info("Phase 5: Generating rubric tree")
    rubric = await generate_rubric(
        client, synthesis.question, synthesis.articles_used, articles
    )
    branches = len(rubric.rubric_tree.children)
    leaves = sum(len(b.children) for b in rubric.rubric_tree.children)
    logger.info(f"Rubric: {branches} branches, {leaves} leaves")

    # Output
    result = {
        "question": synthesis.question,
        "rubric_tree": rubric.rubric_tree.model_dump(),
        "reference_answer": synthesis.reference_answer,
        "articles_used": [
            {"article_id": a["article_id"], "title": a["title"]}
            for a in articles
            if a["article_id"] in synthesis.articles_used
        ],
        "seed_keywords": {
            str(aid): {
                "entities": e.entities[:5],
                "keywords": e.keywords[:5],
                "themes": e.themes,
            }
            for aid, e in entities.items()
        },
        "exploration_log": exploration_notes,
        "generation_method": "multi_article_agentic",
    }

    with open(OUTPUT_FILE, "a") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")
    logger.info(f"Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
