"""Minimal JSONL logging helpers."""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional


def _ensure_parent(path: Path) -> None:
    if path.parent and not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)


@dataclass
class LogRecord:
    """Structure for a single LLM interaction."""

    timestamp: float
    session: str
    model: str
    prompt: str
    response: str
    tags: List[str]
    context: Optional[str] = None


def make_record(prompt: str, response: str, *, model: str, session: str,
                tags: Optional[List[str]] = None, context: Optional[str] = None) -> LogRecord:
    """Create a :class:`LogRecord` from interaction pieces."""
    return LogRecord(
        timestamp=time.time(),
        session=session,
        model=model,
        prompt=prompt,
        response=response,
        tags=tags or [],
        context=context,
    )


class JsonlLogger:
    """Append log records to a JSONL file if configured."""

    def __init__(self, path: Optional[str] = None) -> None:
        self.path = Path(path or os.getenv("LLM_LOG_FILE", ""))

    def log(self, record: LogRecord) -> None:
        if not self.path:
            return
        _ensure_parent(self.path)
        with self.path.open("a", encoding="utf-8") as fh:
            json.dump(asdict(record), fh)
            fh.write("\n")
