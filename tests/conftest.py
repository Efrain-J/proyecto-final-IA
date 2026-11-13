"""Shared fixtures for the data-download tests (no network)."""
from __future__ import annotations

import json
from datetime import date, timedelta

import pytest

_DROP = object()


def expected_price(day: date, hour_number: int) -> float:
    """Deterministic price used by the payload builder: day*1000 + hour number."""
    return float(day.day * 1000 + hour_number)


def _day_values(day: date, overrides: dict | None = None) -> dict:
    values: dict = {"code": "Sistema"}
    for h in range(1, 25):
        values[f"Hour{h:02d}"] = str(expected_price(day, h))
    for h, v in (overrides or {}).items():
        if v is _DROP:
            values.pop(f"Hour{h:02d}", None)
        else:
            values[f"Hour{h:02d}"] = v
    return values


def build_payload(dates, overrides: dict | None = None) -> dict:
    """Build an XM-like hourly response for the given dates (in the given order)."""
    overrides = overrides or {}
    items = [
        {
            "Date": d.isoformat(),
            "HourlyEntities": [{"Id": "Sistema", "Values": _day_values(d, overrides.get(d))}],
        }
        for d in dates
    ]
    return {"Metric": {"MetricId": "PrecBolsNaci"}, "Items": items}


def days(start: date, end: date) -> list[date]:
    return [start + timedelta(days=i) for i in range((end - start).days + 1)]


@pytest.fixture
def make_payload():
    """Factory: make_payload(dates, overrides={date: {hour_number: value}})."""
    return build_payload


@pytest.fixture
def drop_hour():
    """Sentinel: use as an override value to remove an Hour key entirely."""
    return _DROP


@pytest.fixture
def date_range():
    return days


@pytest.fixture
def price_of():
    return expected_price


@pytest.fixture
def write_month_file():
    """Factory: write_month_file(raw_dir, year, month, payload) at the contract path."""
    from energy.data.xm_prices import raw_month_path

    def _write(raw_dir, year, month, payload):
        path = raw_month_path(raw_dir, year, month)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    return _write


@pytest.fixture
def xm_handler_factory(make_payload):
    """Factory returning (handler, calls) for httpx.MockTransport.

    The handler answers from the request JSON body's StartDate/EndDate. `fail` maps a
    "YYYY-MM" key to a callable(call_number_for_that_month) -> httpx.Response | None;
    returning None falls through to the normal payload.
    """
    import httpx

    def factory(fail=None):
        calls: list[dict] = []
        per_month: dict[str, int] = {}
        fail = fail or {}

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            calls.append(body)
            key = body["StartDate"][:7]
            per_month[key] = per_month.get(key, 0) + 1
            if key in fail:
                resp = fail[key](per_month[key])
                if resp is not None:
                    return resp
            s = date.fromisoformat(body["StartDate"])
            e = date.fromisoformat(body["EndDate"])
            return httpx.Response(200, json=make_payload(days(s, e)))

        return handler, calls

    return factory


# --- RAG fixtures (chunking, index, retrieval; offline) ---

FAKE_DIM = 32


def fake_vector(text: str, dim: int = FAKE_DIM) -> list[float]:
    """Deterministic hashed bag-of-words vector, unit length."""
    import hashlib
    import math
    import re

    vec = [0.0] * dim
    for word in re.findall(r"\w+", text.lower()):
        bucket = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16) % dim
        vec[bucket] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


@pytest.fixture
def fake_dim():
    return FAKE_DIM


@pytest.fixture
def fake_embed_documents():
    """embed_documents(list[str]) -> list[list[float]] with FAKE_DIM dimensions."""
    return lambda texts: [fake_vector(t) for t in texts]


@pytest.fixture
def fake_embed_query():
    """embed_query(str) -> list[float] with FAKE_DIM dimensions."""
    return fake_vector


@pytest.fixture
def make_chunk():
    """Factory for a Chunk with sensible defaults; override any field by keyword."""
    from energy.rag.chunk import Chunk

    def _make(**overrides):
        fields = dict(
            doc_id="doc",
            title="Resolución X",
            publisher="CREG",
            article=None,
            in_force=True,
            superseded_by="",
            chunk_index=0,
            text="Texto de ejemplo suficientemente largo para ser un fragmento válido.",
        )
        fields.update(overrides)
        return Chunk(**fields)

    return _make


@pytest.fixture
def sample_chunks(make_chunk):
    """Three chunks with distinctive vocabulary, from three documents."""
    return [
        make_chunk(
            doc_id="creg131", title="Resolución CREG 131", article="2", chunk_index=0,
            text="Usuario no regulado: potencia máxima de 0.1 MW o energía mensual mínima.",
        ),
        make_chunk(
            doc_id="xmglosario", title="Glosario XM", chunk_index=0,
            text="Despacho ideal: programación del sistema sin restricciones de red.",
        ),
        make_chunk(
            doc_id="xmbolsa", title="Preguntas XM", chunk_index=0,
            text="El precio de bolsa se determina cada hora según la oferta de generadores.",
        ),
    ]
