"""Wrapper around the OpenAI Responses API with simple logging."""
from __future__ import annotations

import os
from typing import List, Optional

from dotenv import load_dotenv
from openai import OpenAI

from .log_utils import JsonlLogger, make_record


class LLM:
    """Lightweight convenience wrapper for OpenAI's client."""

    def __init__(self, *, model: str, instructions: str, temperature: float = 0.0,
                 top_p: float = 1.0, max_output_tokens: int = 1024,
                 seed: Optional[int] = None, response_format: Optional[dict] = None) -> None:
        load_dotenv()
        self.client = OpenAI()
        self.model = model
        self.instructions = instructions
        self.temperature = temperature
        self.top_p = top_p
        self.max_output_tokens = max_output_tokens
        self.seed = seed
        self.response_format = response_format
        self.logger = JsonlLogger()

    def pred(self, prompt: str, *, session: str = "default", tags: Optional[List[str]] = None,
             context: Optional[str] = None, context_title: str = "Context"):
        """Send prompt (optionally with context) and log the response."""
        if context:
            user_content = f"{context_title}:\n{context}\n\n{prompt}"
        else:
            user_content = prompt

        messages = [
            {"role": "system", "content": self.instructions},
            {"role": "user", "content": user_content},
        ]

        resp = self.client.responses.create(
            model=self.model,
            input=messages,
            temperature=self.temperature,
            top_p=self.top_p,
            max_output_tokens=self.max_output_tokens,
            seed=self.seed,
            response_format=self.response_format,
        )

        try:
            output_text = resp.output_text  # type: ignore[attr-defined]
        except Exception:
            output_text = str(resp)

        record = make_record(prompt, output_text, model=self.model, session=session,
                              tags=tags, context=context)
        self.logger.log(record)
        return resp
