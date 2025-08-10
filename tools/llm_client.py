from __future__ import annotations

import os
import time
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

from .log_utils import JsonlLogger, make_record


def _mk_client() -> OpenAI:
    """Create an OpenAI client honoring environment overrides."""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")  # can be OpenRouter later
    # Optional headers for attribution (used by OpenRouter)
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
        **params,
    ) -> None:
        self.model = model
        self.instructions = instructions
        self.client = _mk_client()
        self.logger = logger or JsonlLogger()
        # only keep non-None params (e.g. seed=None shouldn't be sent)
        self.params = {k: v for k, v in params.items() if v is not None}

    def pred(
        self,
        user_text: str,
        *,
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
            **self.params,
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
