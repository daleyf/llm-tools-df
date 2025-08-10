"""Utility logging helpers for LLM interactions."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


class JsonlLogger:
    """Append records to a JSON Lines file.

    The default log file path can be set via the ``LLM_LOG_FILE`` environment
    variable. If not provided, ``interactions.jsonl`` in the current working
    directory is used.
    """

    def __init__(self, path: str | None = None):
        self.path = Path(os.path.expanduser(path or os.getenv("LLM_LOG_FILE", "interactions.jsonl")))
        if self.path.parent and not self.path.parent.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: Dict[str, Any]) -> None:
        """Append *record* as a JSON object on a single line."""
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def _dump_response(resp: Any) -> Any:
    """Try to serialize an OpenAI response object."""
    dump = getattr(resp, "model_dump", None)
    if callable(dump):
        try:
            return dump()
        except Exception:
            pass
    # Fallback to repr to avoid crashes if the response isn't serialisable.
    try:
        return json.loads(json.dumps(resp, default=str))
    except Exception:
        return str(resp)


def make_record(
    *,
    provider: str,
    model: str,
    instructions: str,
    input_text: str,
    response: Any,
    started_at: float,
    params: Dict[str, Any],
    session: str,
    tags: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Create a log record describing a single LLM call."""
    completed = time.perf_counter()
    return {
        "provider": provider,
        "model": model,
        "instructions": instructions,
        "input": input_text,
        "output": getattr(response, "output_text", None),
        "response": _dump_response(response),
        "started_at": started_at,
        "completed_at": completed,
        "duration": completed - started_at,
        "params": params,
        "session": session,
        "tags": list(tags) if tags else [],
    }
