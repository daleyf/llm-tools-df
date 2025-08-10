"""Minimal LLM runner with logging and optional grep context."""
from __future__ import annotations

import argparse
import os

from tools.llm_client import LLM
from tools.context_tools import grep_context


def parse_args():
    p = argparse.ArgumentParser(description="Minimal LLM runner with logging + optional grep context")
    # model & sampling
    p.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-4o"),
                   help="Model id (or set OPENAI_MODEL)")
    p.add_argument("--temperature", type=float, default=float(os.getenv("OPENAI_TEMPERATURE", 0.2)))
    p.add_argument("--top-p", dest="top_p", type=float, default=float(os.getenv("OPENAI_TOP_P", 1.0)))
    p.add_argument("--max-output-tokens", dest="max_output_tokens", type=int,
                   default=int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS", "1024")))
    p.add_argument("--seed", type=int, default=os.getenv("OPENAI_SEED") and int(os.getenv("OPENAI_SEED")),
                   help="Set for deterministic-ish outputs")

    # output shaping
    p.add_argument("--json", action="store_true",
                   help="Force JSON responses with response_format=json_object")

    # logging/meta
    p.add_argument("--session", default=os.getenv("LLM_SESSION", "default"))
    p.add_argument("--tags", default=os.getenv("LLM_TAGS", ""),
                   help="Comma-separated tags for the log record")

    # optional grep context
    p.add_argument("--grep", nargs="+", metavar="PATH_OR_GLOB",
                   help="Paths or globs to search for context")
    p.add_argument("--regex", help="Regex pattern to match")
    p.add_argument("--keywords", nargs="+", help="Simple substring keywords (OR logic)")
    p.add_argument("--include", action="append", default=[],
                   help="Include glob (repeatable). Example: --include '**/*.py'")
    p.add_argument("--exclude", action="append", default=[],
                   help="Exclude glob (repeatable). Example: --exclude 'node_modules/**'")
    p.add_argument("--context-lines", type=int, default=2, help="Lines of context before/after each match")
    p.add_argument("--max-context-tokens", type=int, default=1500,
                   help="Budget for context snippets (approx)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    response_format = {"type": "json_object"} if args.json else None

    llm = LLM(
        model=args.model,
        instructions=os.getenv("OPENAI_INSTRUCTIONS", "You are an open-source AI that respects privacy."),
        temperature=args.temperature,
        top_p=args.top_p,
        max_output_tokens=args.max_output_tokens,
        seed=args.seed,
        response_format=response_format,
    )

    ctx = None
    if args.grep and (args.regex or args.keywords):
        ctx_text, stats = grep_context(
            paths=args.grep,
            regex=args.regex,
            keywords=args.keywords,
            include=args.include,
            exclude=args.exclude,
            before=args.context_lines,
            after=args.context_lines,
            max_tokens=args.max_context_tokens,
        )
        if ctx_text.strip():
            print(f"[context] using {stats['snippets']} snippets from {stats['files']} files (~{stats['tokens']} toks est.)")
            ctx = ctx_text

    prompt = input("llm input:\n===\n")
    print("===\n\nllm output:\n===\n")

    resp = llm.pred(
        prompt,
        session=args.session,
        tags=[t.strip() for t in args.tags.split(",") if t.strip()],
        context=ctx,
        context_title="Grep Context",
    )
    try:
        print(resp.output_text)
    except Exception:
        print(resp)


if __name__ == "__main__":
    main()
