from __future__ import annotations

import fnmatch
import glob
import os
import re
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Set


def _estimate_tokens(text: str) -> int:
    """Roughly estimate token count (4 chars ~= 1 token)."""
    return max(1, len(text) // 4)


def _iter_files(paths: Sequence[str], include: List[str], exclude: List[str]) -> Iterable[Path]:
    for pattern in paths:
        for match in glob.glob(pattern, recursive=True):
            p = Path(match)
            if p.is_dir():
                candidates = p.rglob("*")
            else:
                candidates = [p]
            for file in candidates:
                if not file.is_file():
                    continue
                posix = file.as_posix()
                if include and not any(fnmatch.fnmatch(posix, inc) for inc in include):
                    continue
                if any(fnmatch.fnmatch(posix, exc) for exc in exclude):
                    continue
                yield file


def grep_context(
    paths: Sequence[str],
    regex: Optional[str] = None,
    keywords: Optional[Sequence[str]] = None,
    include: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None,
    before: int = 2,
    after: int = 2,
    case_sensitive: bool = False,
    max_tokens: int = 1500,
) -> Tuple[str, dict]:
    """Search files for regex/keywords and build context snippets.

    Returns a tuple of (context_text, stats).
    """

    include = include or []
    exclude = exclude or []
    keywords = list(keywords or [])
    if not case_sensitive:
        keywords = [k.lower() for k in keywords]
    regex_obj = re.compile(regex, 0 if case_sensitive else re.IGNORECASE) if regex else None

    snippets: List[str] = []
    token_count = 0
    files_seen: Set[str] = set()
    seen_spans: Set[Tuple[str, int, int]] = set()

    for fpath in _iter_files(paths, include, exclude):
        try:
            lines = fpath.read_text(encoding="utf-8", errors="ignore").splitlines(True)
        except Exception:
            continue
        matches_here = 0
        for idx, line in enumerate(lines):
            has_match = False
            if regex_obj and regex_obj.search(line):
                has_match = True
            if keywords:
                target = line if case_sensitive else line.lower()
                if any(k in target for k in keywords):
                    has_match = True
            if not has_match:
                continue

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
            if token_count + cost > max_tokens:
                continue
            snippets.append(block)
            token_count += cost
            matches_here += 1
            files_seen.add(str(fpath))
        if matches_here > 0 and token_count >= max_tokens:
            break

    if not snippets:
        return "", {"files": 0, "snippets": 0, "tokens": 0}
    context_text = "\n".join(snippets)
    return context_text, {"files": len(files_seen), "snippets": len(snippets), "tokens": token_count}


if __name__ == "__main__":
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
