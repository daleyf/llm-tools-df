from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


class JsonlLogger:
    """Append interaction records to a JSON Lines file.

    The destination file can be controlled via the ``LLM_LOG_FILE``
    environment variable.  Directories are created automatically.
    """

    def __init__(self, path: Optional[str] = None) -> None:
        log_path = os.getenv("LLM_LOG_FILE", path or "llm_log.jsonl")
        self.path = Path(log_path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: Dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


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
    """Create a dict representing one LLM interaction."""

    duration = time.perf_counter() - started_at
    output_text = getattr(response, "output_text", None)
    return {
        "ts": time.time(),
        "provider": provider,
        "model": model,
        "instructions": instructions,
        "input": input_text,
        "output": output_text,
        "params": params,
        "session": session,
        "tags": list(tags) if tags else [],
        "duration": duration,
    }
