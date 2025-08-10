love it. let’s make this ergonomic.

Below I (1) show a main.py that exposes the most commonly tweaked params (model, temperature, top_p, max_output_tokens, seed, JSON mode, session/tags), and (2) add a tiny “grep-as-context” tool you can point at folders/globs to pull matched snippets into the prompt automatically.

Quick notes (so you know the “why”):

    output_text is a convenience property in the OpenAI SDK that aggregates all text output — perfect for logging and printing.
    OpenAI Platform

    instructions acts like a system prompt in the Responses API.
    OpenAI Platform

    The Responses API supports temperature, top_p, max_output_tokens, and seed for control & reproducibility.
    OpenAI Platform+2OpenAI Platform+2

    JSON mode is provided via response_format={"type": "json_object"} (and the broader structured-outputs guide shows the pattern).
    OpenAI Platform

    To swap to OpenRouter later, you can set OPENAI_BASE_URL=https://openrouter.ai/api/v1 and (optionally) add two headers (HTTP-Referer, X-Title). This works with the OpenAI SDK.
    OpenRouter

Updated files
main.py (exposes changeable params + optional grep context)

import os
import argparse
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
                   help="One or more paths/globs to search for context (e.g., src/ **/*.py README.md)")
    p.add_argument("--regex", help="Regex to match (e.g., '(LLM|ChatGPT)')")
    p.add_argument("--keywords", nargs="+", help="Simple substring keywords (OR logic)")
    p.add_argument("--include", action="append", default=[],
                   help="Include glob (repeatable). Example: --include '**/*.py'")
    p.add_argument("--exclude", action="append", default=[],
                   help="Exclude glob (repeatable). Example: --exclude 'node_modules/**'")
    p.add_argument("--context-lines", type=int, default=2, help="Lines of context before/after each match")
    p.add_argument("--max-context-tokens", type=int, default=1500,
                   help="Budget for context snippets (approx)")

    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # Build response_format only if requested
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

    # Optional: gather grep context
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

    # Prompt
    prompt = input('llm input:\n===\n')
    print('===\n\nllm output:\n===\n')

    resp = llm.pred(
        prompt,
        session=args.session,
        tags=[t.strip() for t in args.tags.split(",") if t.strip()],
        context=ctx,
        context_title="Grep Context",
    )
    print(resp.output_text)

tools/llm_client.py (accepts more params; easy future OpenRouter swap)

from __future__ import annotations
import os, time
from typing import Optional
from dotenv import load_dotenv
from openai import OpenAI
from .log_utils import JsonlLogger, make_record

def _mk_client() -> OpenAI:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")  # set to https://openrouter.ai/api/v1 later if you want
    # Optional headers used by OpenRouter for attribution
    referer = os.getenv("HTTP_REFERER") or os.getenv("HTTP-Referer") or os.getenv("OPENAI_HTTP_REFERER")
    x_title = os.getenv("X_TITLE") or os.getenv("OPENAI_X_TITLE")
    default_headers = {}
    if referer:
        default_headers["HTTP-Referer"] = referer
    if x_title:
        default_headers["X-Title"] = x_title
    if not default_headers:
        default_headers = None

    return OpenAI(api_key=api_key, base_url=base_url, default_headers=default_headers)

class LLM:
    def __init__(
        self,
        model: str,
        *,
        instructions: str = "You are an open-source AI that respects privacy.",
        logger: JsonlLogger | None = None,
        **params
    ):
        self.model = model
        self.instructions = instructions
        self.client = _mk_client()
        self.logger = logger or JsonlLogger()
        # only keep non-None params (e.g., seed=None shouldn't be sent)
        self.params = {k: v for k, v in params.items() if v is not None}

    def pred(
        self,
        user_text: str,
        session: str = "default",
        tags: Optional[list[str]] = None,
        context: Optional[str] = None,
        context_title: str = "Context",
    ):
        started = time.perf_counter()
        if context:
            full_input = f"=== {context_title} ===\n{context.strip()}\n\n=== USER INPUT ===\n{user_text}"
        else:
            full_input = user_text

        resp = self.client.responses.create(
            model=self.model,
            instructions=self.instructions,
            input=full_input,
            **self.params
        )

        rec = make_record(
            provider="openai",
            model=self.model,
            instructions=self.instructions,
            input_text=full_input,
            response=resp,
            started_at=started,
            params=self.params,
            session=session,
            tags=tags,
        )
        self.logger.log(rec)
        return resp

tools/context_tools.py (grep-like file search → trimmed prompt context)

from __future__ import annotations
import os, re, fnmatch
from pathlib import Path
from typing import Iterable, List, Tuple, Dict, Optional

DEFAULT_IGNORES = [
    "node_modules/**", ".git/**", ".hg/**", ".svn/**",
    ".venv/**", "venv/**", "__pycache__/**", "dist/**", "build/**",
    ".ipynb_checkpoints/**",
]
DEFAULT_TEXT_EXTS = {
    ".py",".md",".txt",".json",".yaml",".yml",".toml",".ini",".cfg",".env",
    ".js",".ts",".tsx",".jsx",".java",".go",".rs",".c",".cpp",".h",".hpp",
    ".sh",".bash",".zsh",".fish",".sql",".csv",".tsv",".html",".css",
}

def _is_probably_text(p: Path) -> bool:
    if p.suffix.lower() in DEFAULT_TEXT_EXTS:
        return True
    try:
        with p.open("rb") as f:
            chunk = f.read(2048)
        # heuristic: reject NUL-heavy content
        if b"\x00" in chunk:
            return False
        return True
    except Exception:
        return False

def _estimate_tokens(text: str) -> int:
    # prefer tiktoken if present; else ~4 chars/token heuristic
    try:
        import tiktoken  # type: ignore
        enc = tiktoken.get_encoding("o200k_base")  # works well for GPT-4o family
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text) // 4)

def _match_line(line: str, rx: Optional[re.Pattern], keywords: Optional[List[str]], case_sensitive: bool) -> bool:
    if rx and rx.search(line):
        return True
    if keywords:
        if case_sensitive:
            return any(k in line for k in keywords)
        else:
            low = line.lower()
            return any(k.lower() in low for k in keywords)
    return False

def _iter_files(paths: Iterable[str], include: List[str], exclude: List[str], max_bytes: int) -> Iterable[Path]:
    include_patterns = include or []
    exclude_patterns = (exclude or []) + DEFAULT_IGNORES
    for raw in paths:
        for p in Path().glob(raw) if any(ch in raw for ch in "*?[]") else [Path(raw)]:
            if p.is_dir():
                for f in p.rglob("*"):
                    if f.is_file():
                        yield from _filter_file(f, include_patterns, exclude_patterns, max_bytes)
            elif p.is_file():
                yield from _filter_file(p, include_patterns, exclude_patterns, max_bytes)

def _filter_file(p: Path, includes: List[str], excludes: List[str], max_bytes: int) -> Iterable[Path]:
    rel = str(p)
    if any(fnmatch.fnmatch(rel, pat) for pat in excludes):
        return
    if includes and not any(fnmatch.fnmatch(rel, pat) for pat in includes):
        return
    try:
        if p.stat().st_size > max_bytes:
            return
    except Exception:
        return
    if _is_probably_text(p):
        yield p

def grep_context(
    *,
    paths: List[str],
    regex: Optional[str] = None,
    keywords: Optional[List[str]] = None,
    include: List[str] = [],
    exclude: List[str] = [],
    before: int = 2,
    after: int = 2,
    case_sensitive: bool = False,
    max_tokens: int = 1500,
    max_bytes_per_file: int = 2_000_000,
) -> Tuple[str, Dict[str, int]]:
    """
    Search files and build a compact, prompt-ready context string.
    Returns (context_text, stats)
    """
    if not regex and not keywords:
        raise ValueError("provide either regex or keywords")

    flags = 0 if case_sensitive else re.IGNORECASE
    rx = re.compile(regex, flags) if regex else None

    snippets: List[str] = []
    seen_spans = set()
    files_seen = set()
    budget = max_tokens
    token_count = 0

    for fpath in _iter_files(paths, include, exclude, max_bytes_per_file):
        try:
            with open(fpath, "r", encoding="utf-8", errors="ignore") as fh:
                lines = fh.readlines()
        except Exception:
            continue

        matches_here = 0
        for idx, line in enumerate(lines):
            if _match_line(line, rx, keywords, case_sensitive):
                start = max(0, idx - before)
                end = min(len(lines), idx + after + 1)
                key = (str(fpath), start, end)
                if key in seen_spans:
                    continue
                seen_spans.add(key)
                snippet = "".join(lines[start:end]).rstrip()
                header = f"=== {fpath}:{idx+1} ==="
                block = f"{header}\n{snippet}\n"
                cost = _estimate_tokens(block)

                if token_count + cost > budget:
                    # stop adding more from this file if we're out of budget
                    continue

                snippets.append(block)
                token_count += cost
                matches_here += 1
                files_seen.add(str(fpath))

        # modest cap to avoid over-representing a single file
        if matches_here > 0 and token_count >= budget:
            break

    if not snippets:
        return ("", {"files": 0, "snippets": 0, "tokens": 0})

    context_text = "\n".join(snippets)
    return (context_text, {"files": len(files_seen), "snippets": len(snippets), "tokens": token_count})

if __name__ == "__main__":
    # CLI usage: python -m tools.context_tools --paths src "**/*.py" --regex "(LLM|ChatGPT)"
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", nargs="+", required=True)
    ap.add_argument("--regex")
    ap.add_argument("--keywords", nargs="+")
    ap.add_argument("--include", action="append", default=[])
    ap.add_argument("--exclude", action="append", default=[])
    ap.add_argument("--before", type=int, default=2)
    ap.add_argument("--after", type=int, default=2)
    ap.add_argument("--case-sensitive", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=1500)
    args = ap.parse_args()
    ctx, stats = grep_context(
        paths=args.paths,
        regex=args.regex,
        keywords=args.keywords,
        include=args.include,
        exclude=args.exclude,
        before=args.before,
        after=args.after,
        case_sensitive=args.case_sensitive,
        max_tokens=args.max_tokens,
    )
    print(ctx)
    print("\n---")
    print(stats)

tools/__init__.py

# makes tools a package so "from tools.llm_client import LLM" works

    (No changes to your existing tools/log_utils.py.)

How to use

    Show tweakable params right from CLI:

python main.py \
  --model gpt-4o \
  --temperature 0.2 \
  --top-p 1 \
  --max-output-tokens 1024 \
  --seed 42 \
  --json

    Add grep context on the fly (regex OR keywords):

python main.py \
  --grep src/ README.md "**/*.py" \
  --regex "(LLM|ChatGPT)" \
  --include "**/*.py" \
  --exclude "venv/**" \
  --max-context-tokens 1200

    Keep logging to a specific path (your earlier ask):

export LLM_LOG_FILE=~/ai/interactions.jsonl

    Prep for OpenRouter later (no code change needed):
    Set envs (only when you want to flip):

export OPENAI_BASE_URL="https://openrouter.ai/api/v1"
export OPENAI_API_KEY="or-xxxxxxxx"           # their key
export HTTP_REFERER="https://your.site"       # optional
export X_TITLE="Your App Name"                # optional

(Those header/env details are straight from their docs and work with the OpenAI SDK.
OpenRouter
)
Design choices (why this shape)

    Keep context in plain text snippets with file:line headers so it’s readable and diffable. Token budget prevents prompt bloat.

    Regex/keyword search happens before the LLM call, so you don’t need vector DBs or embeddings (fast feedback while hacking).

    If you want structured outputs, flip --json and parse the model output as JSON; this follows the official structured output pattern.
    OpenAI Platform

    Determinism knobs: seed + same temperature/top_p make runs reproducible-ish.
    OpenAI Platform

If you want, I can add:

    A --stream flag (event-driven streaming with the Responses API).
    OpenAI Platform

    A faster external path using ripgrep (use rg if installed; fallback to Python).

    A simple “context report” to show which files contributed the most matches.

Want me to wire in streaming next or add ripgrep acceleration?
