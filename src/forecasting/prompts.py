"""Central prompt templates and rendering helpers for forecasting methods.

This file is the single source of truth for prompt text and prompt injection.
If you want to manually edit forecasting prompts, start here.
"""

from __future__ import annotations

import json
from typing import Any

from forecasting.memory import FlexExperience


FORECAST_SYSTEM_PROMPT = (
    "You are a forecasting assistant. "
    "Return JSON only with keys predicted_prob and reasoning_summary. "
    "predicted_prob must be a number in [0,1]. "
    "reasoning_summary must be concise and factual."
)

AGENT_SYSTEM_BASE_SECTIONS = [
    "You are a forecasting research agent.",
    "Your task is to estimate the probability that the market resolves YES.",
    "Never use information later than the cutoff {cutoff_time}.",
    "The available search and market-data tools are already constrained to the question cutoff.",
    "Use tools when they materially improve the forecast instead of relying on unsupported intuition.",
    "For structured historical price, index, FX, or crypto data, openbb is usually the most reliable and structured option.",
    "For textual background, policy, people, events, or narrative evidence, search is usually the best starting point.",
    "When using search, prefer named entities and concrete event descriptions over vague terms, and use source filters when a result set is clearly off-topic.",
    "Search is capped to 2 calls total, each returning at most 3 truncated content snippets. Repeated or low-yield searches will be blocked.",
    "Do not keep calling the same tool with near-duplicate queries if the returned evidence is off-topic; switch tools, narrow the source, or finalize with lower confidence.",
    "If the question mentions a ticker, price level, percentage move, index, FX pair, or crypto symbol, seriously consider openbb before additional broad search.",
    "Aim to stay within at most 5 LLM reasoning steps and keep the tool path short.",
    "Keep tool interactions concise. Do not use tools to restate qualitative reasoning you could express directly.",
    "Before each tool call, keep your discussion to one short sentence rather than a long plan.",
    "Return the final answer as JSON only with keys predicted_prob and reasoning_summary.",
    "If evidence remains weak, lower confidence instead of inventing facts.",
]

AGENT_CODE_INTERPRETER_ENABLED_SECTION = (
    "code_interpreter is available only because this question has an explicit numeric or interval-comparison component. "
    "Use it only for a concrete calculation you cannot do reliably in plain text, and call it at most once."
)

AGENT_CODE_INTERPRETER_DISABLED_SECTION = (
    "Do not use code_interpreter for this question. "
    "This is not an explicit numeric-calculation task; reason directly from search/openbb evidence."
)

REASONINGBANK_MEMORY_PREAMBLE = (
    "Below are some memory items that I accumulated from past interaction from the environment "
    "that may be helpful to solve the task. You can use it when you feel it's relevant. "
    "In each step, please first explicitly discuss if you want to use each memory item or not, and then take action."
)

GENERIC_MEMORY_PREAMBLE = (
    "You also have reusable historical memories from earlier resolved questions. "
    "Treat them as guidance, not as direct evidence."
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
AGENT_MEMORY_USER_SUFFIX = "The injected memories above are not evidence by themselves; use them to guide what to check."

FORCED_FINALIZER_SYSTEM_PROMPT = (
    "You are a forecasting assistant. A previous tool-using run collected evidence but did not return "
    "the required final JSON. Using only the question and collected evidence below, return JSON only "
    "with keys predicted_prob and reasoning_summary."
)

REASONINGBANK_SUCCESS_EXTRACTION_PROMPT = """You are an expert in web navigation. You will be given a user query, the corresponding trajectory
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

REASONINGBANK_FAILURE_EXTRACTION_PROMPT = """You are an expert in web navigation. You will be given a user query, the corresponding trajectory
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
## Content <1-3 sentences describing the insights learned to successfully accomplishing the
task>
```"""

FLEX_SUCCESS_DISTILL_PROMPT = """You are building a FLEX experience library from a successful forecasting run.
Extract reusable experience blocks at three levels:
1. strategy: a high-level rule that transfers across similar forecasting questions.
2. pattern: a reusable evidence-gathering or reasoning workflow.
3. case: a short task-shaped cue that helps retrieve this experience for similar situations.

Requirements:
- Return JSON only with keys strategy, pattern, case.
- Each value must be an object with keys title, summary, content.
- Focus on reusable guidance from the trajectory, not the final market answer itself.
- Do not copy raw JSON, predicted probabilities, market ids, exact dates, exact search queries, or long snippets from evidence.
- Do not use generic titles like "golden strategy" or "case note".
- strategy should be broadly reusable and action-oriented.
- pattern should describe what evidence to gather and in what order.
- case should capture the shape of the task or the distinctive cue, but still avoid ids and exact dates.
- Keep summary to one sentence and content to 2-4 short sentences."""

FLEX_FAILURE_DISTILL_PROMPT = """You are building a FLEX experience library from a failed forecasting run.
Extract reusable warning blocks at three levels:
1. strategy: a high-level warning or corrective rule for similar questions.
2. pattern: a reusable failure mode or corrective workflow.
3. case: a short task-shaped cue that helps retrieve this warning for similar situations.

Requirements:
- Return JSON only with keys strategy, pattern, case.
- Each value must be an object with keys title, summary, content.
- Focus on reusable lessons from the trajectory, not the final market answer itself.
- Do not copy raw JSON, predicted probabilities, market ids, exact dates, exact search queries, or long snippets from evidence.
- Do not use generic titles like "warning strategy" or "case note".
- strategy should state the corrective principle.
- pattern should explain which evidence path failed or should have been prioritized instead.
- case should capture the shape of the failure so it can be recognized later.
- Keep summary to one sentence and content to 2-4 short sentences."""


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
    code_interpreter_enabled: bool,
) -> str:
    cutoff_time = question.get("sample_time") or question["open_time"]
    parts = [section.format(cutoff_time=cutoff_time) for section in AGENT_SYSTEM_BASE_SECTIONS]
    parts.append(
        AGENT_CODE_INTERPRETER_ENABLED_SECTION
        if code_interpreter_enabled
        else AGENT_CODE_INTERPRETER_DISABLED_SECTION
    )
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


def build_forced_finalizer_messages(
    question: dict[str, Any],
    *,
    trajectory: list[dict[str, Any]],
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": FORCED_FINALIZER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"{question_block(question)}\n\n"
                "Collected evidence:\n"
                f"{format_trajectory_for_finalizer(trajectory)}\n\n"
                "Return JSON only with keys predicted_prob and reasoning_summary."
            ),
        },
    ]


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
    lines: list[str] = []
    for index, item in enumerate(memories, start=1):
        lines.append(f"# Memory Item {index}")
        lines.append(f"## Title {getattr(item, 'title', '')}")
        lines.append(f"## Description {getattr(item, 'description', '')}")
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


def format_trajectory_for_finalizer(trajectory: list[dict[str, Any]]) -> str:
    if not trajectory:
        return "No tool evidence was collected."
    lines: list[str] = []
    for step in trajectory[-12:]:
        name = str(step.get("step") or "step")
        payload = {key: value for key, value in step.items() if key != "step"}
        lines.append(f"[{name}] {json.dumps(payload, ensure_ascii=False)}")
    return "\n".join(lines)


def _compact_text(text: str, limit: int) -> str:
    collapsed = " ".join((text or "").split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 3].rstrip() + "..."


# Backward-compatible aliases while callers are migrated.
direct_user_prompt = build_direct_user_prompt
rag_user_prompt = build_rag_user_prompt
agent_system_prompt = build_agent_system_prompt
agent_user_prompt = build_agent_user_prompt
forced_finalizer_messages = build_forced_finalizer_messages
