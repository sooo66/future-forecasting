"""Central prompt templates and rendering helpers for forecasting methods.

This file is the single source of truth for prompt text and prompt injection.
If you want to manually edit forecasting prompts, start here.
"""

from __future__ import annotations

from typing import Any

from forecasting.memory import FlexExperience


FORECAST_SYSTEM_PROMPT = (
    "You are a forecasting assistant. "
    "Return JSON only with keys predicted_prob and reasoning_summary. "
    "predicted_prob must be a number in [0,1]. "
    "reasoning_summary must be concise and factual."
)

AGENT_SYSTEM_BASE_SECTIONS = [
    """## Role & Goal
You are a forecasting research agent.
Your task is to estimate the probability that the market resolves YES.""",
    """## Input Information
You will receive a forecasting question containing:
- question: the core prediction question
- description: background context
- resolution_criteria: how the market will be resolved as YES or NO
- cutoff_time: the information cutoff date

You may use available search and market-data tools to gather evidence.""",
    """## Tool Usage Strategy
**When to use each tool**:
- For structured historical price, index, FX, or crypto data, openbb is the most reliable and structured option
- For textual background, policy, people, events, or narrative evidence, search is the best starting point
- If the question mentions a ticker, price level, percentage move, index, FX pair, or crypto symbol, prioritize openbb

**Search rules**:
- Search is capped to 2 calls total, each returning at most 3 truncated content snippets
- Prefer named entities and concrete event descriptions over vague terms
- If results are clearly off-topic, use source filters or change your query
- Do not repeat the same tool with near-duplicate queries""",
    """## Constraints
- Never use information later than the cutoff {cutoff_time}
- Search snippets may be noisy, incomplete, stale, or off-topic even when they rank highly. Treat each hit as provisional until the entity, timeframe, source, and quote actually match the market
- Do not infer precise historical statistics, event mechanics, or hidden facts from weak search hits
- If retrieval quality is poor, state that confidence is limited and avoid overclaiming
- Keep tool interactions concise; do not use tools to restate qualitative reasoning you could express directly
- Stay within the available reasoning-step budget and keep the tool path short""",
    """## Output Format
Return the final answer as JSON only with these two keys:
- predicted_prob: a number in [0,1] representing your probability estimate for YES
- reasoning_summary: a concise, factual summary of your reasoning

**When to finalize**: If you have exhausted your evidence-gathering tools, or reached maximum reasoning steps, you MUST output JSON immediately. Do not say you need more time, more information, or more searches. Even with limited evidence, provide your best estimate.

If evidence remains weak, lower confidence instead of inventing facts.""",
]

REASONINGBANK_MEMORY_PREAMBLE = (
    "Below are memory items retrieved from past resolved forecasting questions that may be relevant to this task. "
    "These are not direct evidence but serve as guidance for what to check or consider. "
    "Review these items before deciding your evidence-gathering approach."
)

GENERIC_MEMORY_PREAMBLE = (
    "Below are memory items from earlier resolved questions that may provide useful guidance. "
    "Treat them as strategic hints, not as direct evidence."
)

FLEX_PRELOADED_PREAMBLE = (
    "The FLEX library has already surfaced a few relevant experiences. "
    "Warning items are for avoiding prior mistakes, not for direct support."
)

FLEX_MEMORY_TOOL_PREAMBLE = (
    "A memory tool is available. After checking the preloaded experiences, call it whenever you still need "
    "a more specific strategy, pattern, warning, or prior case."
)

DIRECT_USER_SUFFIX = "Use only the market text above. Estimate the probability that the market resolves YES."
AGENT_USER_SUFFIX = "Use tools as needed. When you finish, output JSON only with keys predicted_prob and reasoning_summary."
AGENT_MEMORY_USER_SUFFIX = "The injected memories above are hints for what to investigate, not evidence. Cross-check any memory guidance against actual retrieved evidence before relying on it."

REASONINGBANK_SUCCESS_EXTRACTION_PROMPT = """You are an expert in event forecasting. You will be given a user query, the corresponding trajectory
that represents how an agent successfully accomplished the task.
## Guidelines
You need to extract and summarize useful insights in the format of memory items based on the
agent's successful trajectory.
The goal of summarized memory items is to be helpful and generalizable for future similar tasks.
## Important notes
- You must first think why the trajectory is successful, and then summarize the insights.
- You can extract at most 3 memory items from the trajectory.
- You must not repeat similar or overlapping items.
- Do not mention specific websites, queries, or string contents, but rather focus on the
  generalizable insights.
## Output Format
Your output must strictly follow the Markdown format shown below:
```
# Memory Item i
## Title <the title of the memory item>
## Description <one sentence summary of the memory item>
## Content <1-3 sentences describing the insights learned to successfully accomplishing the
task>
```"""

REASONINGBANK_FAILURE_EXTRACTION_PROMPT = """You are an expert in event forecasting. You will be given a user query, the corresponding trajectory
that represents how an agent attempted to resolve the task but failed.
## Guidelines
You need to extract and summarize useful insights in the format of memory items based on the
agent's failed trajectory.
The goal of summarized memory items is to be helpful and generalizable for future similar tasks.
## Important notes
- You must first reflect and think why the trajectory failed, and then summarize what lessons
  you have learned or strategies to prevent the failure in the future.
- You can extract at most 3 memory items from the trajectory.
- You must not repeat similar or overlapping items.
- Do not mention specific websites, queries, or string contents, but rather focus on the
  generalizable insights.
## Output Format
Your output must strictly follow the Markdown format shown below:
```
# Memory Item i
## Title <the title of the memory item>
## Description <one sentence summary of the memory item>
## Content <1-3 sentences describing the lessons learned from the failure>
```"""

FLEX_SUCCESS_DISTILL_PROMPT = """You are building a FLEX experience library from a successful forecasting run.
Extract reusable experience blocks at three levels:
1. strategy: a high-level rule that transfers across similar forecasting questions.
2. pattern: a reusable evidence-gathering or reasoning workflow.
3. case: a short task-shaped cue that helps retrieve this experience for similar situations.

Requirements:
- Focus on reusable guidance from the trajectory, not the final market answer itself.
- Do not copy raw JSON, predicted probabilities, market ids, exact dates, exact search queries, or long snippets from evidence.
- Do not use generic titles like "golden strategy" or "case note".
- strategy should be broadly reusable and action-oriented.
- pattern should describe what evidence to gather and in what order.
- case should capture the shape of the task or the distinctive cue, but still avoid ids and exact dates.
- Keep summary to one sentence and content to 2-4 short sentences.

## Output Format
Your output must strictly follow the Markdown format shown below:
# Strategy
## Title <strategy title>
## Summary <one sentence summary>
## Content <2-4 short sentences describing the strategy>

# Pattern
## Title <pattern title>
## Summary <one sentence summary>
## Content <2-4 short sentences describing the pattern>

# Case
## Title <case title>
## Summary <one sentence summary>
## Content <2-4 short sentences describing the case>"""

FLEX_FAILURE_DISTILL_PROMPT = """You are building a FLEX experience library from a failed forecasting run.
Extract reusable warning blocks at three levels:
1. strategy: a high-level warning or corrective rule for similar questions.
2. pattern: a reusable failure mode or corrective workflow.
3. case: a short task-shaped cue that helps retrieve this warning for similar situations.

Requirements:
- Focus on reusable lessons from the trajectory, not the final market answer itself.
- Do not copy raw JSON, predicted probabilities, market ids, exact dates, exact search queries, or long snippets from evidence.
- Do not use generic titles like "warning strategy" or "case note".
- strategy should state the corrective principle.
- pattern should explain which evidence path failed or should have been prioritized instead.
- case should capture the shape of the failure so it can be recognized later.
- Keep summary to one sentence and content to 2-4 short sentences.

## Output Format
Your output must strictly follow the Markdown format shown below:
# Strategy
## Title <strategy title>
## Summary <one sentence summary>
## Content <2-4 short sentences describing the corrective strategy>

# Pattern
## Title <pattern title>
## Summary <one sentence summary>
## Content <2-4 short sentences describing the failure pattern>

# Case
## Title <case title>
## Summary <one sentence summary>
## Content <2-4 short sentences describing the case>"""


def forecast_system_prompt() -> str:
    return FORECAST_SYSTEM_PROMPT


def build_direct_user_prompt(question: dict[str, Any]) -> str:
    return f"{question_block(question)}\n\n{DIRECT_USER_SUFFIX}"


def build_rag_user_prompt(question: dict[str, Any], query: str, context: str) -> str:
    return (
        f"{question_block(question)}\n\n"
        f"Retrieval query: {query}\n"
        "Retrieved context is already filtered to the allowed cutoff.\n"
        f"{context}\n\n"
        "Estimate the probability that the market resolves YES."
    )


def build_agent_system_prompt(
    question: dict[str, Any],
    *,
    method_name: str,
    injected_memories: list[Any],
    flex_preloaded: list[FlexExperience],
) -> str:
    cutoff_time = question.get("sample_time") or question["open_time"]
    parts = [section.format(cutoff_time=cutoff_time) for section in AGENT_SYSTEM_BASE_SECTIONS]
    if injected_memories:
        parts.append(REASONINGBANK_MEMORY_PREAMBLE if method_name == "reasoningbank" else GENERIC_MEMORY_PREAMBLE)
        parts.append(format_memories_for_prompt(injected_memories))
    if flex_preloaded:
        parts.append(FLEX_PRELOADED_PREAMBLE)
        parts.append(format_flex_experiences_for_prompt(flex_preloaded))
    if method_name == "flex":
        parts.append(FLEX_MEMORY_TOOL_PREAMBLE)
    return "\n\n".join(parts)


def build_agent_user_prompt(question: dict[str, Any], *, injected_memories: list[Any]) -> str:
    prompt = f"{question_block(question)}\n\n{AGENT_USER_SUFFIX}"
    if injected_memories:
        prompt += f"\n\n{AGENT_MEMORY_USER_SUFFIX}"
    return prompt


def build_reasoningbank_extraction_messages(
    *,
    query: str,
    trajectory_text: str,
    success_or_failure: str,
) -> list[dict[str, str]]:
    system_prompt = (
        REASONINGBANK_SUCCESS_EXTRACTION_PROMPT
        if success_or_failure == "success"
        else REASONINGBANK_FAILURE_EXTRACTION_PROMPT
    )
    user_prompt = f"Query: {query}\n\nTrajectory: {trajectory_text}"
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def build_flex_distill_messages(
    question: dict[str, Any],
    *,
    correctness: bool,
    reasoning_summary: str,
    trajectory_highlights: str,
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                FLEX_SUCCESS_DISTILL_PROMPT
                if correctness
                else FLEX_FAILURE_DISTILL_PROMPT
            ),
        },
        {
            "role": "user",
            "content": build_flex_distill_context(
                question,
                correctness=correctness,
                reasoning_summary=reasoning_summary,
                trajectory_highlights=trajectory_highlights,
            ),
        },
    ]


def build_flex_distill_context(
    question: dict[str, Any],
    *,
    correctness: bool,
    reasoning_summary: str,
    trajectory_highlights: str,
) -> str:
    return (
        f"Question: {question['question']}\n"
        f"Domain: {question['domain']}\n"
        f"Description: {_compact_text(' '.join(str(question['description']).split()), 800)}\n"
        f"Resolution Criteria: {_compact_text(' '.join(str(question['resolution_criteria']).split()), 800)}\n"
        f"Outcome Label: {question['label']}\n"
        f"Prediction Correctness: {'correct' if correctness else 'incorrect'}\n"
        f"Reasoning Summary: {reasoning_summary}\n"
        f"Trajectory Highlights:\n{trajectory_highlights}\n"
    )


def format_reasoningbank_trajectory(result: dict[str, Any]) -> str:
    trajectory = list(result.get("trajectory") or [])
    if not trajectory:
        return str(result.get("reasoning_summary") or "")
    lines: list[str] = []
    for index, step in enumerate(trajectory, start=1):
        lines.append(f"[{index}] {step}")
    return "\n".join(lines)


def question_block(question: dict[str, Any]) -> str:
    description = _compact_text(str(question["description"]), 900)
    resolution_criteria = _compact_text(str(question["resolution_criteria"]), 900)
    if description == resolution_criteria:
        resolution_criteria = ""
    return (
        f"Question: {question['question']}\n"
        f"Description: {description}\n"
        f"Resolution criteria: {resolution_criteria or '(same as description)'}"
    )


def format_docs_for_prompt(hits: list[dict[str, Any]], *, content_chars: int) -> str:
    if not hits:
        return "Retrieved context: none."
    lines = ["Retrieved context:"]
    for index, hit in enumerate(hits, start=1):
        lines.append(
            f"[{index}] doc_id={hit['doc_id']} source={hit.get('source') or 'unknown'} "
            f"timestamp={hit['timestamp']} title={hit['title']}"
        )
        lines.append(f"    content={_compact_text(str(hit.get('content') or ''), content_chars)}")
    return "\n".join(lines)


def format_memories_for_prompt(memories: list[Any]) -> str:
    """Format memory items for the agent prompt — per the paper, title + content only."""
    lines: list[str] = []
    for index, item in enumerate(memories, start=1):
        lines.append(f"# Memory Item {index}")
        lines.append(f"## Title {getattr(item, 'title', '')}")
        lines.append(f"## Content {_compact_text(str(getattr(item, 'content', '')), 360)}")
    return "\n".join(lines)


def format_flex_experiences_for_prompt(items: list[FlexExperience]) -> str:
    lines: list[str] = []
    for index, item in enumerate(items, start=1):
        lines.append(f"[{index}] zone={item.zone} level={item.level}")
        lines.append(f"    title={item.title}")
        lines.append(f"    summary={item.summary}")
        lines.append(f"    content={_compact_text(item.content, 280)}")
    return "\n".join(lines)


def _compact_text(text: str, limit: int) -> str:
    collapsed = " ".join((text or "").split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 3].rstrip() + "..."
