import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


class JsonlLogger:
    """Append JSON records to a newline-delimited log file."""

    def __init__(self, path: Optional[str] = None) -> None:
        self.path = Path(path or os.getenv("LLM_LOG_FILE", "llm_interactions.jsonl")).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: Dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def make_record(prompt: str, response: str, session: str, tags: List[str]) -> Dict[str, Any]:
    """Create a simple log record."""
    return {
        "ts": time.time(),
        "session": session,
        "tags": tags,
        "prompt": prompt,
        "response": response,
    }
