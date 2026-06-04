# Deep Research QA: Design Decisions

Adapting the QUEST paper's rubric-tree-based data synthesis to Wikipedia articles.
Prototype: `generate_deepresearch_qa.py` (5 articles, DeepSeek V4 Flash).

---

## 1. Generation Architecture

**Chosen: Two-pass** (rubric tree first, then question + answer)

| Option | Description | Tradeoff |
|--------|-------------|----------|
| **(a) Single-pass** | One LLM call generates rubric + question + answer together | Simpler, fewer API calls. But the model juggles three tasks at once; rubric quality may suffer because it's not the primary focus. |
| **(b) Two-pass** [chosen] | Pass 1: article -> rubric tree. Pass 2: rubric tree -> question + answer | Rubric tree drives question design (QUEST's key insight). Model focuses on one task per call. Slightly more expensive (2x calls). |
| **(c) Three-pass** | Rubric tree -> question -> answer (separate calls) | Most faithful to QUEST. But the answer generation doesn't need a separate call; it benefits from seeing the question in the same context. |
| **(d) Rubric-free** | Generate complex questions directly without a rubric tree | Defeats the purpose. No structured evaluation criteria. Questions lack guaranteed multi-criteria coverage. |

### Experiment: Single-pass vs Two-pass on Article 0 (Anarchism)

Ran both on the same article. Observations:

- **Single-pass question**: "Critically analyze the development of anarchist thought from its ancient precursors to its modern forms, addressing both the core principles and the internal debates over strategy and ideology."
- **Two-pass question**: "Trace the historical development of anarchism from its pre-modern precursors to its contemporary resurgence, analyzing its core principles, key figures, relationship to socialism, strategic debates (revolutionary vs. evolutionary), and the controversy surrounding the term 'libertarian' and anarcho-capitalism."

**Finding**: Two-pass produces more specific, targeted questions. The two-pass question explicitly names "libertarian," "anarcho-capitalism," and "revolutionary vs. evolutionary" — these came from the rubric tree analysis in pass 1. The single-pass question is more generic ("internal debates over strategy and ideology"). The rubric trees are similar in size (12 vs 13 leaves), but two-pass leaf criteria are more precisely tied to article content because the model had full focus on extraction.

**Verdict**: Two-pass is worth the extra API call. The rubric-first approach forces the model to deeply analyze the article before formulating a question, producing better-targeted criteria and more specific questions.

---

## 2. Rubric Tree Structure

**Chosen: Fixed 4-branch + adaptive leaves**

| Option | Description | Tradeoff |
|--------|-------------|----------|
| **(a) Fully adaptive tree** | LLM decides both branch categories and leaves freely | Maximum flexibility, but inconsistent across articles. Hard to compare or aggregate scores. |
| **(b) Fixed branches + adaptive leaves** [chosen] | 4 fixed categories (Factual Accuracy, Completeness, Analytical Depth, Synthesis Quality) with article-specific leaf criteria | Consistent top-level structure enables cross-article comparison. Leaves are specific and verifiable. Sweet spot. |
| **(c) Flat checklist** | No hierarchy — just a weighted list of criteria | Loses the QUEST rubric tree insight. No distinction between what kind of criterion (fact vs. analysis). Harder to assign meaningful weights. |
| **(d) Full QUEST 3-level tree** | Root -> intermediate -> sub-intermediate -> leaf (3+ depth) | Risk of hallucination at deeper levels. DeepSeek's json_object mode struggles with deeply nested structures. Overkill for single-article questions. |

**Rationale**: The 4 fixed branches map to the capabilities QUEST identifies: factual accuracy (fact seeking), completeness (coverage), analytical depth (synthesis), and synthesis quality (cross-section connections). This gives every question a consistent evaluation framework while allowing article-specific verification criteria at the leaf level.

---

## 3. Article Text Handling

**Chosen: Fixed truncation at 8000 characters**

| Option | Description | Tradeoff |
|--------|-------------|----------|
| **(a) Full article text** | Send entire article (up to 78K chars for Alabama) | Expensive, slow. Thinking tokens eat into output budget. Risk of the model getting lost in noise. |
| **(b) Fixed truncation** [chosen] | First 8000 chars, word-boundary aware | Predictable cost and latency. Captures intro + early sections, which usually contain the most important facts. Loses tail content on long articles. |
| **(c) Adaptive truncation** | Parse article structure, keep intro + key section headers + first paragraph of each section | Better coverage of long articles. Requires section parsing logic. More complex for a prototype. |
| **(d) Summarize-then-generate** | First call: summarize article. Second call: generate rubric from summary | Loses specific details needed for verifiable leaf criteria. Summaries are lossy. Adds another API call. |

**Rationale**: 8000 chars is roughly 2000 tokens of input. For the 5 test articles (10K-78K chars), this captures the article's definition, key facts, and early history — enough for a rubric tree with 11-14 verifiable leaf criteria. Alabama (78K) and Anarchism (46K) lose their later sections, but the generated questions still have good quality. For production, option (c) would be better.

---

## 4. Question Complexity Target

**Chosen: Single-article analytical synthesis**

| Option | Description | Tradeoff |
|--------|-------------|----------|
| **(a) Single-article factoid synthesis** | Questions connecting 2-3 facts from one article | Too close to existing `generate_qa.py` hard questions. Not "deep research." |
| **(b) Single-article analytical synthesis** [chosen] | Questions requiring multi-aspect analysis within one article | Good complexity for training. Covers factual recall + analysis + synthesis. No multi-article infrastructure needed. |
| **(c) Multi-article cross-reference** | Questions spanning 2-3 related articles (e.g., Anarchism + Achilles on themes of heroism vs. collective action) | True deep research territory. Requires article selection strategy, cross-article rubric trees. Major infrastructure. Future work. |
| **(d) Report-style open-ended** | "Write a comprehensive report on X" — following QUEST's open-ended task format | Most ambitious. Requires reference report generation + pairwise scoring (QUEST Section 2.2). Evaluation is complex. Better for later iteration. |

**Rationale**: Option (b) produces questions like "Evaluate the significance of albedo in Earth's climate system by analyzing its measurement, feedback mechanisms, and potential for climate mitigation" — clearly more complex than factoid QA, but scoped to one article's content. This is the right starting point before scaling to multi-article or report-style tasks.

---

## 5. Evaluation Protocol Format

**Chosen: Verification hints embedded in rubric tree leaves**

| Option | Description | Tradeoff |
|--------|-------------|----------|
| **(a) Executable Python scripts** (QUEST objective style) | Generate a Python script that programmatically checks each rubric node against an answer | Most rigorous. But DeepSeek can't reliably generate correct evaluation scripts via json_object. QUEST uses GPT-5 for this. High failure rate expected. |
| **(b) LLM-as-judge with rubric** | Send rubric tree + answer to a judge LLM, get per-node scores | Flexible, handles nuance. But adds judge LLM cost per evaluation. Judge quality depends on model. |
| **(c) Claim checklist in rubric leaves** [chosen] | Each leaf has a `verification` field describing what to check (e.g., "Check that the answer mentions X") | Simple, human-readable, can be automated later with string matching or LLM judge. No extra generation step. Can be verified manually. |
| **(d) Hybrid: rubric + RAGAS claims** | Use rubric tree for structure, RAGAS claim decomposition for verification | Leverages existing `evaluate_ragas.py` infrastructure. But RAGAS operates on flat claims, not hierarchical rubrics. Mismatch in abstractions. |

**Rationale**: The `verification` field in each leaf is a natural-language instruction for how to check that criterion. Examples: "Check that the answer includes the scale 0 to 1", "Check that the answer mentions Paris Commune and Spanish Civil War." These can be used for manual review now, and automated with an LLM judge or string matching later. This avoids the fragility of generated Python scripts while preserving the structured evaluation framework.

---

## Summary

| Decision | Chosen | Key Alternative Considered | Experimented |
|----------|--------|---------------------------|-------------|
| Architecture | Two-pass | Single-pass | Yes (Article 0) |
| Rubric structure | Fixed 4-branch + adaptive leaves | Fully adaptive tree | No |
| Text handling | 8000 char truncation | Full text / summarize-first | No |
| Question style | Analytical synthesis | Multi-article cross-ref | No |
| Eval protocol | Verification hints in leaves | Python eval scripts | No |

## Output Files

- `deepresearch_qa.jsonl` — 5 two-pass records (primary output)
- `deepresearch_qa_singlepass.jsonl` — 1 single-pass record (experiment)
- `logs/generate_deepresearch_qa.log` — debug logs

## Next Steps

1. **Multi-article questions**: Combine 2-3 related Wikipedia articles for true cross-document deep research questions
2. **Automated evaluation**: Build an LLM-judge that scores answers against rubric tree leaves
3. **Scale**: Add resume support, concurrency, and run on 100+ articles
4. **RL integration**: Use rubric tree partial scores as fine-grained reward signals (following QUEST Section 4.4)
