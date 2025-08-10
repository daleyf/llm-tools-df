from __future__ import annotations

import os
import re
import fnmatch
from pathlib import Path
from typing import Iterable, List, Tuple, Dict, Optional

DEFAULT_IGNORES = [
    "node_modules/**", ".git/**", ".hg/**", ".svn/**",
    ".venv/**", "venv/**", "__pycache__/**", "dist/**", "build/**",
    ".ipynb_checkpoints/**",
]
DEFAULT_TEXT_EXTS = {
    ".py", ".md", ".txt", ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".env",
    ".js", ".ts", ".tsx", ".jsx", ".java", ".go", ".rs", ".c", ".cpp", ".h", ".hpp",
    ".sh", ".bash", ".zsh", ".fish", ".sql", ".csv", ".tsv", ".html", ".css",
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
    """Search files and build a compact, prompt-ready context string.
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
