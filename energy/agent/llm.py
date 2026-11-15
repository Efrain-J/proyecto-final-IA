"""The chat model behind a small interface, so the loop never sees Groq objects (D16).

``make_groq_chat`` returns a ``ChatFn``: ``(messages, tool_schemas | None) -> ChatReply``.
Models are tried in order; a malformed tool call (``tool_use_failed``) is retried once on
the same model, a rate limit that asks to wait at most ``MAX_RATE_WAIT_S`` is waited out
and retried once on the same model, and any other failure moves to the next one. When every model fails an
``LLMError`` is raised, which the loop turns into a retrieval-only answer.

Errors are classified by ``status_code`` and message text, not by SDK classes, so tests
can use plain fake exceptions. ``groq`` is imported only when a client is built.
Needs ``GROQ_API_KEY`` (environment, or ``.env`` via ``config.load_env`` in the entry point).
"""

from __future__ import annotations

import logging
import os
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from energy import config

logger = logging.getLogger(__name__)

REASONING_EFFORT = "low"  # gpt-oss only; keeps latency and token use small
MAX_RATE_WAIT_S = 15.0  # wait out a short rate limit on the same model; longer → next model


class LLMError(Exception):
    """The model could not produce a reply (no key, network, quota, every model failed)."""


class MissingKeyError(LLMError):
    """``GROQ_API_KEY`` is not set. Same for every model, so it is never retried (D16)."""


@dataclass(frozen=True)
class ToolCall:
    """One tool call requested by the model; ``arguments`` is the raw JSON string."""

    id: str
    name: str
    arguments: str


@dataclass(frozen=True)
class ChatReply:
    """A model reply: text, tool calls, or both; ``model`` is the id that answered (or None)."""

    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    model: str | None = None  # lets the evaluation record fallbacks to the 20b model (D16)


ChatFn = Callable[[list[dict], list[dict] | None], ChatReply]


def _is_tool_use_failed(exc: Exception) -> bool:
    return getattr(exc, "status_code", None) == 400 and "tool_use_failed" in str(exc)


_TRY_AGAIN = re.compile(r"try again in\s+(?:(\d+)m)?\s*(\d+(?:\.\d+)?)(ms|s)\b")


def _short_rate_limit_wait(exc: Exception) -> float | None:
    """Seconds to wait for a 429 that says "try again in <t>", if short enough; else None.

    The free tier allows only 8,000 tokens per minute, so back-to-back questions hit it.
    Waiting a moment keeps the stronger model instead of dropping to the fallback, which
    follows the citation format less reliably (seen in the live check, D16).
    """
    if getattr(exc, "status_code", None) != 429:
        return None
    match = _TRY_AGAIN.search(str(exc))
    if not match:
        return None
    minutes, value, unit = match.groups()
    seconds = float(value) / (1000 if unit == "ms" else 1) + 60 * int(minutes or 0)
    return seconds if seconds <= MAX_RATE_WAIT_S else None


def _to_reply(response: Any, model: str | None = None) -> ChatReply:
    message = response.choices[0].message
    calls = [
        ToolCall(id=c.id, name=c.function.name, arguments=c.function.arguments or "{}")
        for c in (message.tool_calls or [])
    ]
    return ChatReply(content=message.content, tool_calls=calls, model=model)


def make_groq_chat(
    api_key: str | None = None,
    models: tuple[str, ...] = config.GROQ_MODELS,
    client: Any | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> ChatFn:
    """Build the chat function.

    Args:
        api_key: Groq key; defaults to ``GROQ_API_KEY`` read when the function is called.
        models: Model ids, tried in order.
        client: Object with ``chat.completions.create``; tests inject a fake.
        sleep: Used to wait out a short rate limit; tests inject a recorder.

    Returns:
        A ``ChatFn`` that raises ``LLMError`` on failure. A missing key raises only when
        called, so the app can start without it.
    """
    holder: dict[str, Any] = {"client": client}

    def get_client() -> Any:
        key = api_key or os.environ.get("GROQ_API_KEY")
        if not key:  # checked even with an injected client: a missing key is never silent
            raise MissingKeyError("GROQ_API_KEY no está configurada")
        if holder["client"] is None:
            from groq import Groq

            holder["client"] = Groq(api_key=key, timeout=config.LLM_TIMEOUT_S, max_retries=1)
        return holder["client"]

    def chat(messages: list[dict], tools: list[dict] | None) -> ChatReply:
        groq = get_client()
        last: Exception | None = None
        for model in models:
            kwargs: dict[str, Any] = {"model": model, "messages": messages}
            if tools:
                kwargs.update(tools=tools, tool_choice="auto")
            if "gpt-oss" in model:
                kwargs["reasoning_effort"] = REASONING_EFFORT
            for attempt in range(2):
                try:
                    return _to_reply(groq.chat.completions.create(**kwargs), model)
                except Exception as exc:  # classified below; the SDK has many error types
                    last = exc
                    logger.warning("Groq %s failed (attempt %d): %s", model, attempt + 1, exc)
                    if attempt > 0:
                        break
                    wait = _short_rate_limit_wait(exc)
                    if wait is not None:
                        sleep(wait + 0.5)
                    elif not _is_tool_use_failed(exc):
                        break
        raise LLMError(f"Falló el modelo de lenguaje: {last}") from last

    return chat
