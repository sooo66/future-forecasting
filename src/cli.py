"""Unified CLI for info/kb pipeline and qwen_agent runner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import warnings

warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API.*",
    category=UserWarning,
    module=r"jieba\._compat",
)

from loguru import logger
from qwen_agent.llm.base import ModelServiceError
import uvicorn

from agent import Agent
from agent.tools import build_default_tools
from core.runner import build_run_config, run_modules
from experiments import get_experiment_spec, list_experiments
from forecasting.runner import run_experiment
from modules import DEFAULT_MODULES
from tools.corpus import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_TOKENS, DEFAULT_TOKENIZER_NAME, build_corpus
from tools.rerankers import DEFAULT_RERANKER_MODEL
from tools.search import (
    build_bm25_index,
    build_dense_index,
    create_app,
    default_search_log_dir,
    default_search_root,
    find_latest_snapshot_root,
    resolve_search_api_base,
    resolve_search_retrieval_mode,
)
from tools.search_clients import resolve_search_backend
from utils.config import Config
from utils.env import get_first_env, load_dotenv
from utils.logger import setup_logger


def _parse_modules(value: str) -> list[str]:
    if not value or value.strip().lower() == "all":
        return ["all"]
    return [x.strip() for x in value.split(",") if x.strip()]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _build_llm_config(args: argparse.Namespace) -> dict[str, str]:
    model = (
        (args.model or "").strip()
        or get_first_env("MODEL_NAME", "QWEN_MODEL", "LLM_MODEL", "OPENAI_MODEL")
    )
    if not model:
        raise ValueError("LLM model is required. Pass --model or set MODEL_NAME / QWEN_MODEL / LLM_MODEL.")

    cfg = {"model": model}
    model_server = (
        (args.model_server or "").strip()
        or get_first_env(
            "V_API_BASE_URL",
            "QWEN_MODEL_SERVER",
            "OPENAI_BASE_URL",
            "BASE_URL",
            "LLM_BASE_URL",
            "MODEL_SERVER",
        )
    )
    api_key = (
        (args.api_key or "").strip()
        or get_first_env(
            "V_API_KEY",
            "QWEN_API_KEY",
            "DASHSCOPE_API_KEY",
            "OPENAI_API_KEY",
            "API_KEY",
            "LLM_API_KEY",
        )
    )
    if model_server:
        cfg["model_server"] = model_server
    if api_key:
        cfg["api_key"] = api_key
    return cfg


def _resolve_agent_prompt(args: argparse.Namespace) -> str:
    prompt = (args.prompt or "").strip()
    if prompt:
        return prompt
    return " ".join(args.prompt_text or []).strip()


def _resolve_search_api_base(args: argparse.Namespace) -> str:
    return resolve_search_api_base((getattr(args, "search_api_base", "") or "").strip())


def _resolve_search_retrieval_mode_arg(args: argparse.Namespace) -> str | None:
    value = (getattr(args, "search_retrieval_mode", "") or "").strip()
    return resolve_search_retrieval_mode(value or None)


def _resolve_search_backend_arg(args: argparse.Namespace) -> str:
    value = (getattr(args, "search_backend", "") or "").strip()
    return resolve_search_backend(value or None)


def _resolve_snapshot_root(args: argparse.Namespace) -> str:
    return (args.snapshot_root or "").strip() or get_first_env("AGENT_SNAPSHOT_ROOT", "SNAPSHOT_ROOT")


def _parse_methods_override(value: str) -> list[str]:
    if not value or value.strip().lower() == "all":
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _list_experiment_specs() -> list[str]:
    return list_experiments()


def _run_experiment_command(args: argparse.Namespace) -> int:
    spec = get_experiment_spec(args.id)
    setup_logger(
        Config({"paths": {"log_dir": str((_project_root() / "logs" / "experiments" / spec.experiment_id).resolve())}}),
        verbose=bool(args.verbose),
    )
    result = run_experiment(
        spec,
        project_root=_project_root(),
        methods_override=_parse_methods_override(args.methods),
        output_dir_override=(args.output_dir or "").strip() or None,
        search_api_base=_resolve_search_api_base(args),
        search_retrieval_mode=_resolve_search_retrieval_mode_arg(args),
        search_backend=_resolve_search_backend_arg(args),
        force=bool(args.force),
        max_parallel_methods=(args.max_parallel_methods if int(args.max_parallel_methods) > 0 else None),
    )
    logger.info(f"experiment completed: id={spec.experiment_id}")
    logger.info(f"experiment output dir: {result['output_dir']}")
    return 0


def _run_experiment_pretty_command(args: argparse.Namespace) -> int:
    input_path = Path((args.file or "").strip()).expanduser().resolve()
    rows = _read_jsonl_rows(input_path)
    rendered = json.dumps(rows, ensure_ascii=False, indent=2)
    output_path = (args.output or "").strip()
    if output_path:
        target = Path(output_path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(rendered + "\n", encoding="utf-8")
        print(str(target))
        return 0
    print(rendered)
    return 0


def _run_agent(args: argparse.Namespace) -> int:
    llm_cfg = _build_llm_config(args)
    project_root = _project_root()
    setup_logger(
        Config({"paths": {"log_dir": str(project_root / "logs" / "agent")}}),
        verbose=bool(args.verbose),
    )
    resolved_cuttime = (args.cutoff_time or "").strip() or None
    tools = build_default_tools(
        project_root=project_root,
        search_api_base=_resolve_search_api_base(args),
        search_retrieval_mode=_resolve_search_retrieval_mode_arg(args),
        search_backend=_resolve_search_backend_arg(args),
        cutoff_time=resolved_cuttime,
        search_limit=int(args.search_limit),
        enable_code_interpreter=not bool(args.no_code_interpreter),
    )

    agent = Agent(
        llm=llm_cfg,
        tools=tools,
        system_prompt=(args.system_prompt or "").strip() or None,
        max_steps=int(args.max_steps),
        raise_on_tool_error=bool(args.raise_on_tool_error),
    )

    prompt = _resolve_agent_prompt(args)
    if prompt:
        try:
            print(agent.run(prompt, cuttime=resolved_cuttime))
            return 0
        except Exception as exc:
            if isinstance(exc, ModelServiceError):
                logger.error(f"agent run failed: {exc}")
            else:
                logger.exception("agent run failed")
            return 1

    history: list[dict[str, str]] = []
    while True:
        try:
            user_input = input("User: ").strip()
        except EOFError:
            break
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            break
        try:
            response = agent.run(user_input, messages=history, cuttime=resolved_cuttime)
        except Exception as exc:
            if isinstance(exc, ModelServiceError):
                logger.error(f"agent run failed: {exc}")
            else:
                logger.exception("agent run failed")
            continue
        print(f"Agent: {response}")
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": response})
    return 0


def _run_search_serve_command(args: argparse.Namespace) -> int:
    project_root = _project_root()
    snapshot_root = (
        Path((args.snapshot_root or "").strip()).expanduser().resolve()
        if (args.snapshot_root or "").strip()
        else find_latest_snapshot_root(project_root)
    )
    search_root = (
        Path((args.search_root or "").strip()).expanduser().resolve()
        if (args.search_root or "").strip()
        else default_search_root(project_root, snapshot_root)
    )
    log_dir = Path((args.log_dir or "").strip()).expanduser().resolve() if (args.log_dir or "").strip() else default_search_log_dir(project_root)
    setup_logger(
        Config({"paths": {"log_dir": str(log_dir)}}),
        verbose=bool(args.verbose),
    )
    logger.info(f"starting search service for snapshot: {snapshot_root}")
    logger.info(f"search root: {search_root}")
    app = create_app(
        search_root,
        log_dir=log_dir,
        retrieval_mode=_resolve_search_retrieval_mode_arg(args),
        reranker_model_name=(args.reranker_model_name or "").strip() or None,
        reranker_device=(args.reranker_device or "").strip() or None,
        reranker_batch_size=int(args.reranker_batch_size),
        rerank_candidate_limit=int(args.rerank_candidates),
    )
    uvicorn.run(app, host=args.host, port=int(args.port), log_level="info")
    return 0


def _read_jsonl_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected each JSONL row to be an object in {path}")
            rows.append(payload)
    return rows


def _run_search_build_corpus_command(args: argparse.Namespace) -> int:
    project_root = _project_root()
    snapshot_root = (
        Path((args.snapshot_root or "").strip()).expanduser().resolve()
        if (args.snapshot_root or "").strip()
        else find_latest_snapshot_root(project_root)
    )
    search_root = (
        Path((args.search_root or "").strip()).expanduser().resolve()
        if (args.search_root or "").strip()
        else default_search_root(project_root, snapshot_root)
    )
    setup_logger(
        Config({"paths": {"log_dir": str((search_root / "logs").resolve())}}),
        verbose=bool(args.verbose),
    )
    corpus_path = search_root / "corpus.jsonl"
    stats = build_corpus(
        snapshot_root,
        corpus_path,
        chunk_tokens=int(args.chunk_tokens),
        chunk_overlap=int(args.chunk_overlap),
        tokenizer_name=(args.tokenizer or DEFAULT_TOKENIZER_NAME).strip() or DEFAULT_TOKENIZER_NAME,
    )
    logger.info(f"search corpus written: {corpus_path}")
    logger.info(f"passage_count={stats['passage_count']} doc_count={stats['doc_count']}")
    return 0


def _run_search_build_index_command(args: argparse.Namespace) -> int:
    project_root = _project_root()
    snapshot_root = (
        Path((args.snapshot_root or "").strip()).expanduser().resolve()
        if (args.snapshot_root or "").strip()
        else find_latest_snapshot_root(project_root)
    )
    search_root = (
        Path((args.search_root or "").strip()).expanduser().resolve()
        if (args.search_root or "").strip()
        else default_search_root(project_root, snapshot_root)
    )
    setup_logger(
        Config({"paths": {"log_dir": str((search_root / "logs").resolve())}}),
        verbose=bool(args.verbose),
    )
    corpus_path = search_root / "corpus.jsonl"
    retrieval_mode = _resolve_search_retrieval_mode_arg(args) or "bm25"
    if retrieval_mode in {"bm25", "hybrid"}:
        bm25_dir = search_root / "bm25"
        build_bm25_index(
            corpus_path,
            bm25_dir,
            overwrite=bool(args.force),
            show_progress=not bool(args.no_progress),
        )
        logger.info(f"bm25 index written: {bm25_dir}")
    if retrieval_mode in {"dense", "hybrid"}:
        dense_dir = search_root / "dense"
        build_dense_index(
            corpus_path,
            dense_dir,
            model_name=(args.embedding_model_name or "").strip() or "sentence-transformers/all-MiniLM-L6-v2",
            device=(args.embedding_device or "").strip() or None,
            batch_size=int(args.embedding_batch_size),
            workers=int(args.embedding_workers),
            overwrite=bool(args.force),
            show_progress=not bool(args.no_progress),
        )
        logger.info(f"dense index written: {dense_dir}")
    return 0


def main() -> int:
    load_dotenv(_project_root() / ".env", override=False)
    parser = argparse.ArgumentParser(description="Unified info/kb pipeline runner")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run modules into one benchmark snapshot")
    run.add_argument("--snapshot", required=True, help="Snapshot id, e.g. s2026_03_static")
    run.add_argument("--from", dest="date_from", required=True, help="Start date (YYYY-MM-DD)")
    run.add_argument("--to", dest="date_to", required=True, help="End date (YYYY-MM-DD)")
    run.add_argument("--kb-from", dest="kb_date_from", default="", help="Optional KB start date (YYYY-MM-DD)")
    run.add_argument("--kb-to", dest="kb_date_to", default="", help="Optional KB end date (YYYY-MM-DD)")
    run.add_argument(
        "--modules",
        default="all",
        help=f"Comma-separated module list or 'all'. default=all ({', '.join(DEFAULT_MODULES)})",
    )
    run.add_argument("--resume", action="store_true", help="Resume from module states")
    run.add_argument("--verbose", action="store_true", help="Enable DEBUG logs on console")
    run.add_argument(
        "--snapshot-base-dir",
        default="data/benchmark",
        help="Base dir for snapshots (default: data/benchmark)",
    )
    run.add_argument(
        "--module-workers",
        type=int,
        default=0,
        help="Parallel workers for modules. 0 means auto (all selected modules).",
    )

    agent_cmd = sub.add_parser(
        "agent",
        aliases=["chat", "ask"],
        help="Run a qwen_agent with local search/OpenBB tools",
    )
    agent_cmd.add_argument(
        "prompt_text",
        nargs="*",
        help="Optional single-turn prompt as positional text. If omitted, starts an interactive session.",
    )
    agent_cmd.add_argument(
        "--model",
        default="",
        help="LLM model name. Defaults to MODEL_NAME / QWEN_MODEL / LLM_MODEL from .env or env.",
    )
    agent_cmd.add_argument(
        "--model-server",
        default="",
        help="Optional OpenAI-compatible base URL. Defaults to V_API_BASE_URL / OPENAI_BASE_URL / BASE_URL.",
    )
    agent_cmd.add_argument(
        "--api-key",
        default="",
        help="Optional LLM API key. Defaults to V_API_KEY / OPENAI_API_KEY / QWEN_API_KEY.",
    )
    agent_cmd.add_argument("--prompt", default="", help="Single-turn prompt. Higher priority than positional prompt.")
    agent_cmd.add_argument("--system-prompt", default="", help="Override the default agent system prompt")
    agent_cmd.add_argument(
        "--search-api-base",
        default="",
        help="Optional search API base URL. Defaults to SEARCH_API_BASE or http://127.0.0.1:8000.",
    )
    agent_cmd.add_argument(
        "--search-backend",
        default="",
        help="Search backend override: local or exa. Defaults to SEARCH_BACKEND or local.",
    )
    agent_cmd.add_argument(
        "--search-retrieval-mode",
        default="",
        help="Optional retrieval mode override for search requests: bm25, dense, or hybrid.",
    )
    agent_cmd.add_argument(
        "--cutoff-time",
        default="",
        help="Optional external cutoff time for search/openbb. Defaults to current UTC time.",
    )
    agent_cmd.add_argument(
        "--search-limit",
        type=int,
        default=3,
        help="Externally fixed max number of search hits exposed to the agent.",
    )
    agent_cmd.add_argument("--max-steps", type=int, default=8, help="Maximum qwen_agent tool/LLM steps per run")
    agent_cmd.add_argument("--no-code-interpreter", action="store_true", help="Disable built-in code_interpreter")
    agent_cmd.add_argument("--raise-on-tool-error", action="store_true", help="Raise instead of returning tool errors")
    agent_cmd.add_argument("--verbose", action="store_true", help="Enable DEBUG logs on console")

    experiment_cmd = sub.add_parser("experiment", help="Run forecasting experiment configs")
    experiment_sub = experiment_cmd.add_subparsers(dest="experiment_command", required=True)

    experiment_list = experiment_sub.add_parser("list", help="List available experiment ids")
    experiment_list.add_argument("--verbose", action="store_true", help="Enable DEBUG logs on console")

    experiment_run = experiment_sub.add_parser("run", help="Run a forecasting experiment spec")
    experiment_run.add_argument("--id", required=True, help="Experiment id, e.g. pre_experiment_smoke_3")
    experiment_run.add_argument("--output-dir", default="", help="Optional override for output dir")
    experiment_run.add_argument("--methods", default="", help="Optional comma-separated method override")
    experiment_run.add_argument("--search-api-base", default="", help="Optional search API base URL override")
    experiment_run.add_argument(
        "--search-backend",
        default="",
        help="Search backend override: local or exa. Defaults to SEARCH_BACKEND or local.",
    )
    experiment_run.add_argument(
        "--search-retrieval-mode",
        default="",
        help="Optional retrieval mode override for search requests: bm25, dense, or hybrid.",
    )
    experiment_run.add_argument("--force", action="store_true", help="Force rerun even if results already exist")
    experiment_run.add_argument(
        "--max-parallel-methods",
        type=int,
        default=0,
        help="Parallel workers for experiment methods. 0 means spec default.",
    )
    experiment_run.add_argument("--verbose", action="store_true", help="Enable DEBUG logs on console")

    experiment_pretty = experiment_sub.add_parser("pretty", help="Render a JSONL result file as pretty JSON")
    experiment_pretty.add_argument("--file", required=True, help="Path to a JSONL file, e.g. artifacts/.../results_flex.jsonl")
    experiment_pretty.add_argument("--output", default="", help="Optional output path for the pretty JSON")

    search_cmd = sub.add_parser("search", help="Build and serve the local search API")
    search_sub = search_cmd.add_subparsers(dest="search_command", required=True)
    search_build_corpus = search_sub.add_parser("build-corpus", help="Build an offline passage corpus from one snapshot")
    search_build_corpus.add_argument("--snapshot-root", default="", help="Snapshot root to read. Defaults to the latest local snapshot.")
    search_build_corpus.add_argument("--search-root", default="", help="Optional output root for corpus/index artifacts.")
    search_build_corpus.add_argument("--chunk-tokens", type=int, default=DEFAULT_CHUNK_TOKENS, help="Passage chunk size in tokens.")
    search_build_corpus.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP, help="Passage overlap in tokens.")
    search_build_corpus.add_argument("--tokenizer", default=DEFAULT_TOKENIZER_NAME, help="Tokenizer name used for chunking.")
    search_build_corpus.add_argument("--verbose", action="store_true", help="Enable DEBUG logs on console")

    search_build_index = search_sub.add_parser("build-index", help="Build retrieval indexes from an offline search corpus")
    search_build_index.add_argument("--snapshot-root", default="", help="Snapshot root used to derive the default search root.")
    search_build_index.add_argument("--search-root", default="", help="Root containing corpus.jsonl.")
    search_build_index.add_argument(
        "--search-retrieval-mode",
        default="bm25",
        help="Index mode to build: bm25, dense, or hybrid (builds both).",
    )
    search_build_index.add_argument(
        "--embedding-model-name",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Hugging Face embedding model for dense retrieval.",
    )
    search_build_index.add_argument(
        "--embedding-device",
        default="",
        help="Optional torch device override for dense embedding generation, e.g. cpu or cuda.",
    )
    search_build_index.add_argument(
        "--embedding-batch-size",
        type=int,
        default=64,
        help="Dense embedding batch size. Larger is faster but uses more memory.",
    )
    search_build_index.add_argument(
        "--embedding-workers",
        type=int,
        default=0,
        help="Dense embedding worker processes. 0 means auto; non-CPU devices are forced to 1.",
    )
    search_build_index.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable dense indexing progress output.",
    )
    search_build_index.add_argument("--force", action="store_true", help="Overwrite an existing BM25 index directory.")
    search_build_index.add_argument("--verbose", action="store_true", help="Enable DEBUG logs on console")

    search_serve = search_sub.add_parser("serve", help="Serve the local search API for one snapshot")
    search_serve.add_argument("--snapshot-root", default="", help="Snapshot root to bind. Defaults to the latest local snapshot.")
    search_serve.add_argument("--search-root", default="", help="Root containing corpus.jsonl and retrieval index artifacts.")
    search_serve.add_argument(
        "--search-retrieval-mode",
        default="",
        help="Default retrieval mode for requests: bm25, dense, or hybrid.",
    )
    search_serve.add_argument(
        "--reranker-model-name",
        default="",
        help=(
            "Optional local reranker model for hybrid mode, e.g. "
            f"{DEFAULT_RERANKER_MODEL}. Disabled when omitted."
        ),
    )
    search_serve.add_argument(
        "--reranker-device",
        default="",
        help="Optional torch device override for reranking, e.g. cuda or cpu.",
    )
    search_serve.add_argument(
        "--reranker-batch-size",
        type=int,
        default=32,
        help="Batch size for reranker inference.",
    )
    search_serve.add_argument(
        "--rerank-candidates",
        type=int,
        default=40,
        help="How many hybrid first-stage candidates to rerank before truncating to the requested limit.",
    )
    search_serve.add_argument("--host", default="127.0.0.1", help="Host to bind the search API server")
    search_serve.add_argument("--port", type=int, default=8000, help="Port to bind the search API server")
    search_serve.add_argument("--log-dir", default="", help="Optional log directory for search request logs")
    search_serve.add_argument("--verbose", action="store_true", help="Enable DEBUG logs on console")

    args = parser.parse_args()
    if args.command == "agent":
        return _run_agent(args)
    if args.command == "experiment":
        if args.experiment_command == "list":
            for experiment_id in _list_experiment_specs():
                print(experiment_id)
            return 0
        if args.experiment_command == "pretty":
            return _run_experiment_pretty_command(args)
        return _run_experiment_command(args)
    if args.command == "search":
        if args.search_command == "build-corpus":
            return _run_search_build_corpus_command(args)
        if args.search_command == "build-index":
            return _run_search_build_index_command(args)
        return _run_search_serve_command(args)

    modules = _parse_modules(args.modules)
    cfg = build_run_config(
        snapshot_id=args.snapshot,
        date_from=args.date_from,
        date_to=args.date_to,
        kb_date_from=args.kb_date_from,
        kb_date_to=args.kb_date_to,
        modules=modules,
        resume=bool(args.resume),
        snapshot_base_dir=args.snapshot_base_dir,
        module_workers=args.module_workers,
        project_root=_project_root(),
    )
    setup_logger(
        Config({"paths": {"log_dir": str(cfg.snapshot_root / "logs")}}),
        verbose=bool(args.verbose),
    )
    result = run_modules(cfg)
    logger.info(f"run completed: {result['stats']}")
    logger.info(f"snapshot root: {result['snapshot_root']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
