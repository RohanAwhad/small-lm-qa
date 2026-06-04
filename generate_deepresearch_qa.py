"""Generate deep research QA pairs with rubric trees from Wikipedia articles.

Adapts the QUEST paper's rubric-tree-based data synthesis pipeline to generate
complex, multi-criteria QA pairs from local Wikipedia articles.

Three-pass generation (primary):
  Pass 1: Article text (truncated) -> rubric tree (structured evaluation criteria)
  Pass 2: Rubric tree -> open-ended research question
  Pass 3: Question + full article text -> reference report grounded in source

Single-pass generation (experiment):
  One call: Article text -> question + rubric tree + reference answer

Usage:
  uv run python generate_deepresearch_qa.py [N]  # N articles, default 5
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from loguru import logger
from openai import AsyncOpenAI
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential_jitter, retry_if_exception_type

from utils.wikipedia_loader import load_articles

# --- Config ---
DEEPSEEK_MODEL = "deepseek-v4-flash"
BASE_URL = "https://api.deepseek.com"
OUTPUT_FILE = Path("deepresearch_qa.jsonl")
EXPERIMENT_FILE = Path("deepresearch_qa_singlepass.jsonl")
LOG_DIR = Path("logs")
N_ARTICLES = 5
MAX_ARTICLE_CHARS = 8000


# --- Logging ---
LOG_DIR.mkdir(exist_ok=True)
logger.remove()
log_level = os.environ.get("LOGGING_LEVEL", "DEBUG").upper()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level:<7} | {message}")
logger.add(
    LOG_DIR / "generate_deepresearch_qa.log",
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


class RubricTreeResponse(BaseModel):
    topic_summary: str
    key_themes: list[str]
    rubric_tree: RubricTree


class QuestionOnlyResponse(BaseModel):
    question: str


class AnswerResponse(BaseModel):
    reference_answer: str


class SinglePassResponse(BaseModel):
    question: str
    rubric_tree: RubricTree
    reference_answer: str


# --- Prompts ---

RUBRIC_SYSTEM = """\
You are an expert at analyzing text and designing structured evaluation rubrics \
for open-ended deep research tasks.

Given a Wikipedia article, your job is to:
1. Identify the key themes, subtopics, and analytical angles the article covers
2. Design a rubric tree for evaluating an open-ended research report on this topic

The rubric tree has exactly 4 top-level branches (from DeepResearch Bench). \
Each branch has 2-4 leaf children that are *adaptive to the article content*.
Top-level weights must sum to 1.0. Leaf weights within each branch must sum to 1.0.

The 4 branches:
1. "Instruction Following" (weight ~0.20) — does the report address the task requirements? \
Leaves should capture specific aspects the task demands (e.g., "addresses historical context," \
"covers both pros and cons").
2. "Comprehensiveness" (weight ~0.30) — does the report cover all important subtopics? \
Leaves should name the key subtopics from the article that a thorough report must include.
3. "Readability" (weight ~0.25) — is the report well-structured, clear, and logically organized? \
Leaves should describe structural qualities (e.g., "uses clear topic sentences," \
"logical flow from background to analysis to implications").
4. "Insight" (weight ~0.25) — does the report offer non-obvious analysis, connections, or \
deeper understanding? Leaves should describe the analytical depth expected \
(e.g., "identifies causal mechanisms," "connects to broader context").

Leaf criteria should be quality descriptors, not binary fact checks. \
They describe *what good looks like* for a research report on this topic.

Respond with this json structure:
{
  "topic_summary": "1-2 sentence summary of the article",
  "key_themes": ["theme1", "theme2", ...],
  "rubric_tree": {
    "children": [
      {
        "criterion": "Instruction Following",
        "weight": 0.20,
        "children": [
          {"criterion": "Addresses ...", "weight": 0.50, "verification": "Assess whether the report ..."},
          ...
        ]
      },
      ...
    ]
  }
}"""

RUBRIC_USER = """\
Article Title: {title}

Article Text:
{text}

Analyze this article and generate the rubric tree. Respond in json."""


QUESTION_SYSTEM = """\
You are an expert at crafting open-ended deep research tasks — tasks that ask \
for a well-structured, multi-paragraph research report on a topic.

Given a rubric tree (evaluation criteria) derived from a Wikipedia article, generate \
an open-ended research task that requires a comprehensive written report.

The task must:
- Ask the writer to produce a research report or analytical essay
- Be broad enough that multiple valid responses exist (no single correct answer)
- Require coverage of multiple subtopics, analysis, and synthesis
- Sound like an assignment a professor or research lead would give

Respond with this json structure:
{
  "question": "your open-ended research task here"
}"""

QUESTION_USER = """\
Article Title: {title}
Topic Summary: {topic_summary}

Rubric Tree (evaluation criteria):
{rubric_tree}

Generate a deep research question that covers all criteria above. Respond in json."""


ANSWER_SYSTEM = """\
You are an expert researcher writing a high-quality reference report.

Given a research question and the full source article, write a reference report \
(4-6 paragraphs, 400-700 words) that:
- Is grounded in the provided article content (use specific facts, names, dates)
- Is well-structured with clear paragraphs and logical flow
- Covers all major subtopics relevant to the question
- Demonstrates analytical depth and non-obvious insights
- Serves as a quality benchmark (not the only valid answer)

Respond with this json structure:
{
  "reference_answer": "your multi-paragraph reference report here"
}"""

ANSWER_USER = """\
Research Question:
{question}

Source Article — {title}:
{text}

Write a reference report answering the question above, grounded in the article. \
Respond in json."""


SINGLEPASS_SYSTEM = """\
You are an expert at crafting open-ended deep research tasks with evaluation frameworks.

Given a Wikipedia article, generate all three components in a single response:
1. A rubric tree — hierarchical evaluation criteria for a research report
2. An open-ended research task requiring a multi-paragraph report
3. A reference report (4-6 paragraphs, 400-700 words) exemplifying high quality

Rubric tree structure: exactly 4 branches, each with 2-4 leaves.
Branches: Instruction Following (~0.20), Comprehensiveness (~0.30), \
Readability (~0.25), Insight (~0.25). Weights sum to 1.0 at each level.
Leaves are quality descriptors, not binary fact checks.

Respond with this json structure:
{
  "question": "open-ended research task",
  "rubric_tree": {
    "children": [
      {
        "criterion": "Instruction Following",
        "weight": 0.20,
        "children": [
          {"criterion": "quality descriptor", "weight": 0.50, "verification": "how to assess"},
          ...
        ]
      },
      ...
    ]
  },
  "reference_answer": "multi-paragraph reference report (400-700 words)"
}"""

SINGLEPASS_USER = """\
Article Title: {title}

Article Text:
{text}

Generate an open-ended research task with rubric tree and reference report. Respond in json."""


# --- LLM client ---

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
async def call_llm(client: AsyncOpenAI, system: str, user: str) -> dict[str, Any]:
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
    logger.debug(f"LLM call took {elapsed:.1f}s, response {len(content)} chars")
    return json.loads(content)


def truncate_text(text: str, max_chars: int = MAX_ARTICLE_CHARS) -> str:
    """Truncate article text, preserving word boundaries."""
    if len(text) <= max_chars:
        return text
    cut = text.rfind(" ", 0, max_chars)
    if cut == -1:
        cut = max_chars
    return text[:cut] + "\n\n[Article truncated for processing]"


# --- Three-pass generation ---

async def generate_rubric_tree(
    client: AsyncOpenAI, title: str, text: str
) -> RubricTreeResponse:
    """Pass 1: Article (truncated) -> rubric tree."""
    user = RUBRIC_USER.format(title=title, text=truncate_text(text))
    raw = await call_llm(client, RUBRIC_SYSTEM, user)
    return RubricTreeResponse.model_validate(raw)


async def generate_question(
    client: AsyncOpenAI, title: str, rubric: RubricTreeResponse
) -> QuestionOnlyResponse:
    """Pass 2: Rubric tree -> question."""
    rubric_json = json.dumps(rubric.rubric_tree.model_dump(), indent=2)
    user = QUESTION_USER.format(
        title=title,
        rubric_tree=rubric_json,
        topic_summary=rubric.topic_summary,
    )
    raw = await call_llm(client, QUESTION_SYSTEM, user)
    return QuestionOnlyResponse.model_validate(raw)


async def generate_answer(
    client: AsyncOpenAI, title: str, text: str, question: str
) -> AnswerResponse:
    """Pass 3: Question + full article -> reference report."""
    user = ANSWER_USER.format(title=title, text=text, question=question)
    raw = await call_llm(client, ANSWER_SYSTEM, user)
    return AnswerResponse.model_validate(raw)


async def three_pass(client: AsyncOpenAI, article: dict) -> dict[str, Any]:
    """Three-pass pipeline: rubric tree, question, then grounded answer."""
    aid, title, text = article["article_id"], article["title"], article["text"]

    logger.info(f"[3-pass] Article {aid}: '{title}' — pass 1: rubric tree")
    rubric = await generate_rubric_tree(client, title, text)
    branches = len(rubric.rubric_tree.children)
    leaves = sum(len(b.children) for b in rubric.rubric_tree.children)
    logger.info(
        f"[3-pass] Article {aid}: rubric done "
        f"({branches} branches, {leaves} leaves)"
    )

    logger.info(f"[3-pass] Article {aid}: pass 2: question")
    qa = await generate_question(client, title, rubric)
    logger.info(f"[3-pass] Article {aid}: question done")

    logger.info(f"[3-pass] Article {aid}: pass 3: reference answer (full article)")
    answer = await generate_answer(client, title, text, qa.question)
    logger.info(f"[3-pass] Article {aid}: done")

    return {
        "article_id": aid,
        "title": title,
        "question": qa.question,
        "rubric_tree": rubric.rubric_tree.model_dump(),
        "topic_summary": rubric.topic_summary,
        "key_themes": rubric.key_themes,
        "reference_answer": answer.reference_answer,
        "generation_method": "three_pass",
    }


# --- Single-pass generation (experiment) ---

async def single_pass(client: AsyncOpenAI, article: dict) -> dict[str, Any]:
    """Single-pass: article -> question + rubric + answer in one call."""
    aid, title, text = article["article_id"], article["title"], article["text"]

    logger.info(f"[single-pass] Article {aid}: '{title}'")
    user = SINGLEPASS_USER.format(title=title, text=truncate_text(text))
    raw = await call_llm(client, SINGLEPASS_SYSTEM, user)
    result = SinglePassResponse.model_validate(raw)
    logger.info(f"[single-pass] Article {aid}: done")

    return {
        "article_id": aid,
        "title": title,
        "question": result.question,
        "rubric_tree": result.rubric_tree.model_dump(),
        "topic_summary": "",
        "key_themes": [],
        "reference_answer": result.reference_answer,
        "generation_method": "single_pass",
    }


async def _safe_three_pass(client: AsyncOpenAI, article: dict) -> dict[str, Any] | None:
    """three_pass with skip on final failure."""
    try:
        return await three_pass(client, article)
    except (json.JSONDecodeError, AssertionError, Exception) as e:
        logger.warning(f"Article {article['article_id']} '{article['title']}' failed after retries: {e!r}")
        return None


async def _safe_single_pass(client: AsyncOpenAI, article: dict) -> dict[str, Any] | None:
    """single_pass with skip on final failure."""
    try:
        return await single_pass(client, article)
    except (json.JSONDecodeError, AssertionError, Exception) as e:
        logger.warning(f"Article {article['article_id']} '{article['title']}' single-pass failed after retries: {e!r}")
        return None


# --- Main ---

async def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else N_ARTICLES
    logger.info(f"Loading {n} articles from wikipedia_en.jsonl")
    articles = load_articles(n)
    logger.info(f"Loaded {len(articles)} articles")
    for a in articles:
        logger.info(
            f"  Article {a['article_id']}: '{a['title']}' "
            f"({len(a['text'])} chars, truncated to {min(len(a['text']), MAX_ARTICLE_CHARS)})"
        )

    client = make_client()

    # --- Experiment: single-pass on article[0] for comparison ---
    logger.info("=" * 60)
    logger.info("EXPERIMENT: single-pass on article 0")
    logger.info("=" * 60)
    sp_result = await _safe_single_pass(client, articles[0])
    if sp_result is not None:
        EXPERIMENT_FILE.write_text(
            json.dumps(sp_result, ensure_ascii=False, indent=2) + "\n"
        )
        logger.info(f"Single-pass result saved to {EXPERIMENT_FILE}")
    else:
        logger.warning("Single-pass experiment failed after retries, skipping")

    # --- Main: three-pass on all articles ---
    logger.info("=" * 60)
    logger.info(f"MAIN RUN: three-pass on {len(articles)} articles")
    logger.info("=" * 60)

    OUTPUT_FILE.write_text("")  # fresh start for prototype

    saved = 0
    for article in articles:
        result = await _safe_three_pass(client, article)
        if result is None:
            continue
        with open(OUTPUT_FILE, "a") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
        saved += 1
        logger.info(f"Saved article {article['article_id']} to {OUTPUT_FILE}")

    logger.info(f"Done. {saved}/{n} deep research QA pairs in {OUTPUT_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
