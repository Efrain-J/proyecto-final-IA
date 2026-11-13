"""Tests for `energy.agent.llm`: the Groq adapter behind `ChatFn` (step 5, D16).

Offline: a fake client exposing ``chat.completions.create(**kwargs)`` replaces the
Groq SDK, and errors are plain exceptions carrying ``status_code``, because the
adapter classifies by status and message rather than by SDK classes. Also pins the
agent settings in `energy.config`. Run with ``uv run pytest tests/test_agent_llm.py``.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from energy import config
from energy.agent.llm import ChatReply, LLMError, ToolCall, make_groq_chat

MODELS = ("model-a", "model-b")
SCHEMAS = [{"type": "function", "function": {"name": "t", "description": "d", "parameters": {"type": "object"}}}]
MESSAGES = [{"role": "user", "content": "hola"}]


class FakeAPIError(Exception):
    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code


def response(content: str | None = "ok", tool_calls: list | None = None):
    msg = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


def tool_call(id_: str, name: str, arguments: str):
    return SimpleNamespace(id=id_, function=SimpleNamespace(name=name, arguments=arguments))


class FakeClient:
    """Plays back scripted responses/exceptions, recording every ``create`` kwargs."""

    def __init__(self, script: list) -> None:
        self.script = list(script)
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    @property
    def models(self) -> list[str]:
        return [c["model"] for c in self.calls]


def make_chat(script: list):
    client = FakeClient(script)
    return make_groq_chat(api_key="test-key", models=MODELS, client=client), client


# --- config ------------------------------------------------------------------


def test_agent_settings_in_config():
    assert config.GROQ_MODELS == ("openai/gpt-oss-120b", "openai/gpt-oss-20b")
    assert config.T_MAX == 5
    assert config.HISTORY_MESSAGES == 6
    assert config.LLM_TIMEOUT_S == 30


# --- response conversion -----------------------------------------------------


def test_plain_text_response_becomes_chat_reply():
    chat, client = make_chat([response("Hola")])
    reply = chat(MESSAGES, None)
    assert isinstance(reply, ChatReply)
    assert reply.content == "Hola"
    assert reply.tool_calls == []
    assert client.calls[0]["messages"] == MESSAGES


def test_tool_calls_are_converted_to_our_type():
    raw = [tool_call("c1", "search_regulation", '{"query": "x"}'), tool_call("c2", "simulate_battery", "{}")]
    chat, _ = make_chat([response(None, raw)])
    reply = chat(MESSAGES, SCHEMAS)
    assert reply.content is None
    assert reply.tool_calls == [
        ToolCall(id="c1", name="search_regulation", arguments='{"query": "x"}'),
        ToolCall(id="c2", name="simulate_battery", arguments="{}"),
    ]


# --- tools / tool_choice -----------------------------------------------------


def test_with_tools_sends_them_with_auto_choice():
    chat, client = make_chat([response()])
    chat(MESSAGES, SCHEMAS)
    kwargs = client.calls[0]
    assert kwargs["tools"] == SCHEMAS
    assert kwargs["tool_choice"] == "auto"


def test_without_tools_does_not_allow_tool_calls():
    chat, client = make_chat([response()])
    chat(MESSAGES, None)
    kwargs = client.calls[0]
    assert not kwargs.get("tools") or kwargs.get("tool_choice") == "none"


# --- retries and fallback ----------------------------------------------------


def test_first_model_is_used_first():
    chat, client = make_chat([response()])
    chat(MESSAGES, None)
    assert client.models == ["model-a"]


def test_tool_use_failed_is_retried_once_on_the_same_model():
    err = FakeAPIError(400, "Failed to call a function: tool_use_failed")
    chat, client = make_chat([err, response("recuperado")])
    assert chat(MESSAGES, SCHEMAS).content == "recuperado"
    assert client.models == ["model-a", "model-a"]


def test_tool_use_failed_twice_moves_to_next_model():
    err = lambda: FakeAPIError(400, "tool_use_failed")  # noqa: E731
    chat, client = make_chat([err(), err(), response("desde b")])
    assert chat(MESSAGES, SCHEMAS).content == "desde b"
    assert client.models == ["model-a", "model-a", "model-b"]


@pytest.mark.parametrize(
    "error",
    [
        FakeAPIError(429, "rate limit exceeded"),
        FakeAPIError(500, "internal error"),
        FakeAPIError(404, "model not found"),
        FakeAPIError(400, "bad request"),
    ],
    ids=["rate-limit", "server-error", "not-found", "other-400"],
)
def test_other_errors_move_to_next_model_without_retry(error):
    chat, client = make_chat([error, response("desde b")])
    assert chat(MESSAGES, None).content == "desde b"
    assert client.models == ["model-a", "model-b"]


def _chat_with_sleep(script: list):
    client, waits = FakeClient(script), []
    chat = make_groq_chat(api_key="test-key", models=MODELS, client=client, sleep=waits.append)
    return chat, client, waits


@pytest.mark.parametrize(
    ("message", "seconds"),
    [("Please try again in 630ms. Need more tokens?", 0.63), ("Please try again in 1.875s.", 1.875)],
    ids=["milliseconds", "seconds"],
)
def test_short_rate_limit_is_waited_out_on_the_same_model(message, seconds):
    chat, client, waits = _chat_with_sleep([FakeAPIError(429, message), response("desde a")])
    assert chat(MESSAGES, None).content == "desde a"
    assert client.models == ["model-a", "model-a"]
    assert len(waits) == 1 and waits[0] == pytest.approx(seconds + 0.5)


def test_long_rate_limit_moves_to_next_model_without_waiting():
    chat, client, waits = _chat_with_sleep([FakeAPIError(429, "try again in 2m30s"), response("desde b")])
    assert chat(MESSAGES, None).content == "desde b"
    assert client.models == ["model-a", "model-b"]
    assert waits == []


def test_rate_limit_twice_on_the_same_model_moves_on():
    err = lambda: FakeAPIError(429, "try again in 1s")  # noqa: E731
    chat, client, waits = _chat_with_sleep([err(), err(), response("desde b")])
    assert chat(MESSAGES, None).content == "desde b"
    assert client.models == ["model-a", "model-a", "model-b"]
    assert len(waits) == 1


def test_all_models_failing_raises_llm_error_naming_last_error():
    chat, client = make_chat([FakeAPIError(429, "primero"), FakeAPIError(503, "segundo-fallo")])
    with pytest.raises(LLMError, match="segundo-fallo"):
        chat(MESSAGES, None)
    assert client.models == ["model-a", "model-b"]


def test_non_api_exceptions_also_become_llm_error():
    chat, _ = make_chat([TimeoutError("timed out"), ConnectionError("down")])
    with pytest.raises(LLMError):
        chat(MESSAGES, None)


# --- missing key ---------------------------------------------------------------


def test_missing_api_key_raises_llm_error_without_calling_the_client(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    client = FakeClient([response()])
    chat = make_groq_chat(api_key=None, models=MODELS, client=client)  # must not raise yet
    with pytest.raises(LLMError, match="GROQ_API_KEY"):
        chat(MESSAGES, None)
    assert client.calls == []
