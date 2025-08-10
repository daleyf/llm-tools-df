from __future__ import annotations

import os
import time
from typing import List, Optional

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - fallback if package missing
    def load_dotenv() -> None:  # type: ignore
        return None

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - allow running without package
    OpenAI = None

from .log_utils import JsonlLogger, make_record


class LLM:
    """Thin wrapper around the OpenAI Responses API with logging."""

    def __init__(
        self,
        model: str,
        instructions: str,
        temperature: float = 0.2,
        top_p: float = 1.0,
        max_output_tokens: int = 1024,
        seed: Optional[int] = None,
        response_format: Optional[dict] = None,
    ) -> None:
        load_dotenv()
        self.client = OpenAI() if OpenAI else None
        self.model = model
        self.instructions = instructions
        self.temperature = temperature
        self.top_p = top_p
        self.max_output_tokens = max_output_tokens
        self.seed = seed
        self.response_format = response_format
        self.logger = JsonlLogger()

    def pred(
        self,
        prompt: str,
        session: str,
        tags: List[str],
        context: Optional[str] = None,
        context_title: str = "Context",
    ):
        """Send a prompt to the model and log the interaction."""
        if self.client is None:
            raise RuntimeError("OpenAI client not available; install the openai package.")

        input_text = ""
        if context:
            input_text += f"{context_title}\n{context}\n\n"
        input_text += prompt

        start = time.time()
        resp = self.client.responses.create(
            model=self.model,
            input=input_text,
            instructions=self.instructions,
            temperature=self.temperature,
            top_p=self.top_p,
            max_output_tokens=self.max_output_tokens,
            seed=self.seed,
            response_format=self.response_format,
        )
        duration = time.time() - start
        output_text = getattr(resp, "output_text", "")

        record = make_record(prompt, output_text, session, tags)
        record.update({
            "model": self.model,
            "duration": duration,
        })
        self.logger.log(record)
        return resp
