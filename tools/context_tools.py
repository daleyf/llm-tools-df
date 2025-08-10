"""Utilities for gathering textual context via simple grep-like searches."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set, Tuple

import fnmatch


def _iter_files(paths: Sequence[str], include: Sequence[str], exclude: Sequence[str]) -> Iterable[Path]:
    seen: Set[Path] = set()
    for p in paths:
        for path in Path().glob(p):
            if path.is_dir():
                for file in path.rglob("*"):
                    if file.is_file():
                        if _match_globs(file, include, exclude):
                            if file not in seen:
                                seen.add(file)
                                yield file
            elif path.is_file():
                if _match_globs(path, include, exclude):
                    if path not in seen:
                        seen.add(path)
                        yield path


def _match_globs(path: Path, include: Sequence[str], exclude: Sequence[str]) -> bool:
    as_posix = path.as_posix()
    if include:
        if not any(fnmatch.fnmatch(as_posix, pat) for pat in include):
            return False
    if any(fnmatch.fnmatch(as_posix, pat) for pat in exclude):
        return False
    return True


def _estimate_tokens(text: str) -> int:
    # naive token estimate
    return max(1, len(text.split()))


def grep_context(*, paths: Sequence[str], regex: Optional[str] = None, keywords: Optional[Sequence[str]] = None,
                 include: Sequence[str] = (), exclude: Sequence[str] = (), before: int = 2, after: int = 2,
                 case_sensitive: bool = False, max_tokens: int = 1500) -> Tuple[str, dict]:
    """Search paths for regex/keywords and return snippets within token budget."""
    flags = 0 if case_sensitive else re.IGNORECASE
    pattern = re.compile(regex, flags) if regex else None
    kw_lower = [k if case_sensitive else k.lower() for k in (keywords or [])]

    snippets: List[str] = []
    files_seen: Set[str] = set()
    token_count = 0

    for file in _iter_files(paths, include, exclude):
        try:
            text = file.read_text(encoding="utf-8")
        except Exception:
            continue
        lines = text.splitlines()
        matches_here = 0
        for idx, line in enumerate(lines):
            hay = line if case_sensitive else line.lower()
            matched = False
            if pattern and pattern.search(line):
                matched = True
            if kw_lower and any(k in hay for k in kw_lower):
                matched = True
            if not matched:
                continue
            start = max(0, idx - before)
            end = min(len(lines), idx + after + 1)
            snippet = "\n".join(lines[start:end])
            header = f"=== {file}:{idx+1} ==="
            block = f"{header}\n{snippet}\n"
            cost = _estimate_tokens(block)
            if token_count + cost > max_tokens:
                continue
            snippets.append(block)
            token_count += cost
            matches_here += 1
            files_seen.add(str(file))
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
    ctx, stats = grep_context(paths=args.paths, regex=args.regex, keywords=args.keywords,
                               include=args.include, exclude=args.exclude, before=args.before,
                               after=args.after, case_sensitive=args.case_sensitive,
                               max_tokens=args.max_tokens)
    print(ctx)
    print("\n---")
    print(stats)
