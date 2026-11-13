"""Tests for `energy.agent.tools`: the agent's two tools (step 5, D16).

Offline. ``search_regulation`` runs against a fake retriever; ``simulate_battery``
runs on a real two-week slice of 2025 using the committed price CSV and the
committed Q-tables in ``artifacts/policies/``. Expensive objects are built once
per module. Run with ``uv run pytest tests/test_agent_tools.py``.
"""
from __future__ import annotations

import json
from datetime import date

import pandas as pd
import pytest

from energy.agent.tools import Tool, build_registry, run_tool, tool_schemas
from energy.data.prices import load_prices
from energy.rag.retrieve import Hit
from energy.rl.factory import TEST_SHIFT_SEEDS
from energy.rl.simulate import simulate

START, END = "2025-10-06", "2025-10-19"  # test period (Oct-Dec), two weeks
RULES = {"night_day", "shift_schedule", "price_threshold"}


def make_hit(**overrides) -> Hit:
    fields = dict(
        doc_id="creg101113",
        title="Resolución CREG 101 113 de 2025",
        publisher="CREG",
        article="5",
        in_force=True,
        superseded_by="",
        chunk_index=0,
        text="Texto del artículo.",
        score=0.9,
    )
    fields.update(overrides)
    return Hit(**fields)


class FakeRetriever:
    name = "fake"

    def __init__(self, hits: list[Hit]) -> None:
        self.hits = hits
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, k: int = 5) -> list[Hit]:
        self.calls.append((query, k))
        return self.hits[:k]


class ExplodingRetriever:
    name = "boom"

    def search(self, query: str, k: int = 5) -> list[Hit]:
        raise RuntimeError("index exploded")


@pytest.fixture(scope="module")
def prices():
    return load_prices()


@pytest.fixture(scope="module")
def registry(prices):
    return build_registry(FakeRetriever([make_hit()]), prices=prices)


@pytest.fixture(scope="module")
def sim(registry):
    """Parsed output of a default-policy two-week simulation."""
    raw, parsed = run_tool(
        registry,
        "simulate_battery",
        json.dumps({"policy": "q_learning", "preset": "8h", "start_date": START, "end_date": END}),
    )
    assert "error" not in parsed, raw
    return parsed


def call(registry, name: str, **args) -> dict:
    raw, parsed = run_tool(registry, name, json.dumps(args))
    assert json.loads(raw) == parsed
    return parsed


# --- schemas -----------------------------------------------------------------


def test_registry_has_exactly_the_two_tools(registry):
    assert set(registry) == {"search_regulation", "simulate_battery"}
    assert all(isinstance(t, Tool) for t in registry.values())


def test_tool_schemas_follow_openai_format(registry):
    schemas = tool_schemas(registry)
    assert len(schemas) == 2
    for s in schemas:
        assert s["type"] == "function"
        fn = s["function"]
        assert fn["name"] in registry
        assert fn["description"]
        params = fn["parameters"]
        assert params["type"] == "object"
        assert set(params.get("required", [])) <= set(params["properties"])
        json.dumps(s)  # must be serializable


def test_schema_required_arguments(registry):
    by_name = {s["function"]["name"]: s["function"]["parameters"] for s in tool_schemas(registry)}
    assert "query" in by_name["search_regulation"]["required"]
    assert {"start_date", "end_date"} <= set(by_name["simulate_battery"]["required"])
    assert "policy" not in by_name["simulate_battery"]["required"]


# --- run_tool never raises ---------------------------------------------------


@pytest.mark.parametrize(
    ("name", "raw_args"),
    [
        ("simulate_battery", "{not json"),
        ("simulate_battery", "[]"),
        ("no_such_tool", "{}"),
        ("search_regulation", "{}"),
        ("simulate_battery", json.dumps({"start_date": START})),
        ("simulate_battery", json.dumps({"preset": "5h", "start_date": START, "end_date": END})),
        ("simulate_battery", json.dumps({"policy": "magic", "start_date": START, "end_date": END})),
        ("simulate_battery", json.dumps({"start_date": "2025/10/06", "end_date": END})),
        ("simulate_battery", json.dumps({"start_date": "2023-01-01", "end_date": "2023-01-10"})),
        ("simulate_battery", json.dumps({"start_date": "2026-01-01", "end_date": "2026-01-10"})),
        ("simulate_battery", json.dumps({"start_date": END, "end_date": START})),
        ("simulate_battery", json.dumps({"start_date": "2024-12-20", "end_date": "2025-01-05"})),
        ("simulate_battery", json.dumps({"start_date": "2024-01-01", "end_date": "2025-01-05"})),
    ],
    ids=[
        "bad-json", "args-not-object", "unknown-tool", "search-missing-query", "missing-end-date",
        "bad-preset-enum", "bad-policy-enum", "bad-date-format", "before-data", "after-data",
        "start-after-end", "crosses-two-years", "more-than-366-days",
    ],
)
def test_run_tool_returns_error_json_instead_of_raising(registry, name, raw_args):
    raw, parsed = run_tool(registry, name, raw_args)
    assert isinstance(raw, str)
    assert json.loads(raw) == parsed
    assert isinstance(parsed.get("error"), str) and parsed["error"]


def test_run_tool_turns_tool_exceptions_into_errors(prices):
    reg = build_registry(ExplodingRetriever(), prices=prices)
    _, parsed = run_tool(reg, "search_regulation", json.dumps({"query": "x"}))
    assert "error" in parsed


# --- search_regulation -------------------------------------------------------


def test_search_regulation_maps_hit_fields(prices):
    hit = make_hit(article="7", in_force=False, superseded_by="CREG 101 113", doc_id="creg098")
    out = call(build_registry(FakeRetriever([hit]), prices=prices), "search_regulation", query="eficiencia")
    assert out["retriever"] == "fake"
    (row,) = out["hits"]
    assert row["cita"] == hit.citation
    assert row["doc_id"] == "creg098"
    assert row["in_force"] is False
    assert row["superseded_by"] == "CREG 101 113"
    assert row["text"] == hit.text


def test_search_regulation_truncates_long_text(prices):
    hit = make_hit(text="a" * 5000)
    out = call(build_registry(FakeRetriever([hit]), prices=prices), "search_regulation", query="q")
    assert 0 < len(out["hits"][0]["text"]) <= 1600


def test_search_regulation_passes_query_and_k_to_retriever(prices):
    retr = FakeRetriever([make_hit(chunk_index=i) for i in range(6)])
    out = call(build_registry(retr, prices=prices), "search_regulation", query="almacenamiento", k=3)
    assert retr.calls == [("almacenamiento", 3)]
    assert len(out["hits"]) == 3


def test_search_regulation_default_k_is_five(prices):
    retr = FakeRetriever([make_hit(chunk_index=i) for i in range(8)])
    call(build_registry(retr, prices=prices), "search_regulation", query="q")
    assert retr.calls[0][1] == 5


def test_search_regulation_without_hits_returns_note(prices):
    out = call(build_registry(FakeRetriever([]), prices=prices), "search_regulation", query="arepas")
    assert out["hits"] == []
    assert out["nota"]
    assert out["retriever"] == "fake"


# --- simulate_battery --------------------------------------------------------


def test_simulate_battery_reports_expected_keys(sim):
    expected = {
        "policy", "policy_used", "preset", "period", "shift_patterns", "mix", "savings_cop",
        "baselines", "best_rule", "vs_best_rule", "vs_perfect_foresight", "energy",
        "by_hour_block", "by_price_level", "in_sample", "notes",
    }
    assert expected <= set(sim)
    assert sim["policy"] == "q_learning"
    assert sim["policy_used"] == "q_learning"
    assert sim["preset"] == "8h"
    assert sim["shift_patterns"] == 5
    assert sim["mix"] == "default"
    assert START in json.dumps(sim["period"]) and END in json.dumps(sim["period"])
    assert sim["notes"]


def test_simulate_battery_has_no_prewritten_comparison(sim):
    """The renderer writes the comparison; the model gets no ready-made figures (D16)."""
    assert "comparison" not in sim


def test_simulate_battery_defaults_policy_and_preset(registry):
    out = call(registry, "simulate_battery", start_date=START, end_date=END)
    assert out["policy"] == "q_learning"
    assert out["preset"] == "8h"


def test_mean_savings_equals_mean_of_direct_simulations(sim, prices):
    direct = [
        simulate(
            "q_learning", date.fromisoformat(START), date.fromisoformat(END), "8h",
            prices=prices, mix="default", shift_seed=s,
        ).summary.savings_cop
        for s in TEST_SHIFT_SEEDS
    ]
    assert sim["savings_cop"]["mean"] == pytest.approx(sum(direct) / 5, rel=1e-6)
    assert sim["savings_cop"]["min"] == pytest.approx(min(direct), rel=1e-6)
    assert sim["savings_cop"]["max"] == pytest.approx(max(direct), rel=1e-6)


def test_savings_min_mean_max_are_ordered(sim):
    s = sim["savings_cop"]
    assert s["min"] <= s["mean"] <= s["max"]


def test_baselines_include_all_references(sim):
    b = sim["baselines"]
    assert {"no_battery", "shift_schedule", "night_day", "price_threshold", "perfect_foresight"} <= set(b)
    assert b["no_battery"] == 0
    assert b["perfect_foresight"] >= b["no_battery"]
    assert any(r in str(sim["best_rule"]) for r in RULES)
    assert isinstance(sim["vs_best_rule"], (int, float))
    assert isinstance(sim["vs_perfect_foresight"], (int, float))


def test_energy_totals_are_present_and_non_negative(sim):
    assert {"bought_kwh", "delivered_kwh", "cycles"} <= set(sim["energy"])
    assert all(v >= 0 for v in sim["energy"].values())


def test_by_hour_block_has_twelve_two_hour_rows(sim):
    rows = sim["by_hour_block"]
    assert len(rows) == 12
    assert rows[0]["hours"] == "00-01"
    assert rows[-1]["hours"] == "22-23"
    for r in rows:
        assert {"hours", "charge_pct", "discharge_pct", "mean_price"} <= set(r)
        assert 0 <= r["charge_pct"] <= 100
        assert 0 <= r["discharge_pct"] <= 100
        assert r["mean_price"] > 0


def test_by_price_level_has_four_quartile_rows(sim):
    rows = sim["by_price_level"]
    assert len(rows) == 4
    for r in rows:
        assert {"charge_pct", "discharge_pct"} <= set(r)
        assert 0 <= r["charge_pct"] <= 100
        assert 0 <= r["discharge_pct"] <= 100


def test_in_sample_true_for_training_months(registry):
    out = call(
        registry, "simulate_battery",
        policy="q_learning", start_date="2025-09-15", end_date="2025-09-28",
    )
    assert out["in_sample"] is True


def test_in_sample_false_for_test_months(registry):
    out = call(
        registry, "simulate_battery",
        policy="q_learning", start_date="2025-11-03", end_date="2025-11-16",
    )
    assert out["in_sample"] is False


def test_missing_q_table_falls_back_and_says_so(prices, tmp_path):
    reg = build_registry(FakeRetriever([]), prices=prices, policies_dir=tmp_path)
    out = call(reg, "simulate_battery", policy="q_learning", start_date=START, end_date=END)
    assert "error" not in out
    assert out["policy"] == "q_learning"
    assert out["policy_used"] != "q_learning"
