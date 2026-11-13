import json
import math
from datetime import date

import httpx
import pandas as pd
import pytest

from energy.data import xm_prices
from energy.data.xm_prices import (
    XMDownloadError,
    XMResponseError,
    build_prices_csv,
    build_request_body,
    download_months,
    month_ranges,
    parse_hourly_items,
    raw_month_path,
    validate_prices,
)


# ---------------------------------------------------------------- month_ranges
def test_month_ranges_full_year_gives_twelve_chunks():
    chunks = month_ranges(date(2024, 1, 1), date(2024, 12, 31))
    assert len(chunks) == 12
    assert chunks[0] == (date(2024, 1, 1), date(2024, 1, 31))
    assert chunks[-1] == (date(2024, 12, 1), date(2024, 12, 31))


def test_month_ranges_leap_february_ends_on_29th():
    assert month_ranges(date(2024, 2, 1), date(2024, 2, 29)) == [
        (date(2024, 2, 1), date(2024, 2, 29))
    ]


def test_month_ranges_non_leap_february_ends_on_28th():
    assert month_ranges(date(2025, 2, 1), date(2025, 2, 28)) == [
        (date(2025, 2, 1), date(2025, 2, 28))
    ]


def test_month_ranges_clips_partial_start_and_end():
    assert month_ranges(date(2024, 1, 15), date(2024, 3, 10)) == [
        (date(2024, 1, 15), date(2024, 1, 31)),
        (date(2024, 2, 1), date(2024, 2, 29)),
        (date(2024, 3, 1), date(2024, 3, 10)),
    ]


def test_month_ranges_single_day_gives_one_chunk():
    assert month_ranges(date(2024, 12, 1), date(2024, 12, 1)) == [
        (date(2024, 12, 1), date(2024, 12, 1))
    ]


def test_month_ranges_crosses_year_boundary():
    assert month_ranges(date(2024, 12, 20), date(2025, 1, 5)) == [
        (date(2024, 12, 20), date(2024, 12, 31)),
        (date(2025, 1, 1), date(2025, 1, 5)),
    ]


def test_month_ranges_start_after_end_raises():
    with pytest.raises(ValueError):
        month_ranges(date(2024, 5, 2), date(2024, 5, 1))


# ----------------------------------------------------------- build_request_body
def test_build_request_body_exact_dict():
    assert build_request_body(date(2024, 1, 1), date(2024, 1, 31)) == {
        "MetricId": "PrecBolsNaci",
        "StartDate": "2024-01-01",
        "EndDate": "2024-01-31",
        "Entity": "Sistema",
        "Filter": [],
    }


# ------------------------------------------------------------- raw_month_path
def test_raw_month_path_file_name_format(tmp_path):
    assert raw_month_path(tmp_path, 2024, 3) == tmp_path / "PrecBolsNaci_2024-03.json"


def test_raw_month_path_two_digit_month(tmp_path):
    assert raw_month_path(tmp_path, 2025, 12).name == "PrecBolsNaci_2025-12.json"


# --------------------------------------------------------- parse_hourly_items
def test_parse_returns_24_rows_per_day_with_expected_columns(make_payload):
    df = parse_hourly_items(make_payload([date(2024, 12, 1), date(2024, 12, 2)]))
    assert len(df) == 48
    assert list(df.columns) == ["timestamp", "price_cop_kwh"]


def test_parse_timestamps_run_00_to_23(make_payload):
    df = parse_hourly_items(make_payload([date(2024, 12, 1)]))
    expected = pd.date_range("2024-12-01 00:00", periods=24, freq="h")
    assert list(df["timestamp"]) == list(expected)


def test_parse_hour01_is_midnight_and_hour24_is_2300(make_payload, price_of):
    d = date(2024, 12, 1)
    df = parse_hourly_items(make_payload([d]))
    assert df["timestamp"].iloc[0] == pd.Timestamp("2024-12-01 00:00")
    assert df["price_cop_kwh"].iloc[0] == price_of(d, 1)
    assert df["timestamp"].iloc[-1] == pd.Timestamp("2024-12-01 23:00")
    assert df["price_cop_kwh"].iloc[-1] == price_of(d, 24)


def test_parse_real_values_from_xm_example():
    values = {"code": "Sistema", **{f"Hour{h:02d}": "642.54859" for h in range(1, 25)}}
    values["Hour07"] = "152.54859"
    payload = {
        "Items": [
            {"Date": "2024-12-01", "HourlyEntities": [{"Id": "Sistema", "Values": values}]}
        ]
    }
    df = parse_hourly_items(payload)
    assert df["price_cop_kwh"].iloc[0] == pytest.approx(642.54859)
    assert df["price_cop_kwh"].iloc[6] == pytest.approx(152.54859)


def test_parse_prices_are_floats_and_timestamps_datetimes(make_payload):
    df = parse_hourly_items(make_payload([date(2024, 12, 1)]))
    assert pd.api.types.is_float_dtype(df["price_cop_kwh"])
    assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])


def test_parse_sorts_when_days_are_out_of_order(make_payload):
    df = parse_hourly_items(
        make_payload([date(2024, 12, 3), date(2024, 12, 1), date(2024, 12, 2)])
    )
    assert df["timestamp"].is_monotonic_increasing
    assert df["timestamp"].iloc[0] == pd.Timestamp("2024-12-01 00:00")
    assert df["timestamp"].iloc[-1] == pd.Timestamp("2024-12-03 23:00")


@pytest.mark.parametrize("bad_value", ["", None])
def test_parse_empty_hour_value_gives_nan(make_payload, bad_value):
    d = date(2024, 12, 1)
    df = parse_hourly_items(make_payload([d], overrides={d: {5: bad_value}}))
    assert math.isnan(df["price_cop_kwh"].iloc[4])
    assert df["price_cop_kwh"].notna().sum() == 23


def test_parse_missing_hour_key_gives_nan(make_payload, drop_hour):
    d = date(2024, 12, 1)
    df = parse_hourly_items(make_payload([d], overrides={d: {10: drop_hour}}))
    assert len(df) == 24
    assert math.isnan(df["price_cop_kwh"].iloc[9])


def test_parse_missing_items_raises_xm_response_error():
    with pytest.raises(XMResponseError):
        parse_hourly_items({"Metric": {}})


def test_xm_response_error_is_value_error():
    assert issubclass(XMResponseError, ValueError)


def test_xm_download_error_is_runtime_error():
    assert issubclass(XMDownloadError, RuntimeError)


# ------------------------------------------------------------- download_months
def _run(start, end, raw_dir, handler, **kw):
    return download_months(
        start, end, raw_dir, backoff_base=0, transport=httpx.MockTransport(handler), **kw
    )


def test_download_writes_one_file_per_month(tmp_path, xm_handler_factory):
    handler, _ = xm_handler_factory()
    paths = _run(date(2024, 1, 1), date(2024, 3, 31), tmp_path, handler)
    assert sorted(p.name for p in paths) == [
        "PrecBolsNaci_2024-01.json",
        "PrecBolsNaci_2024-02.json",
        "PrecBolsNaci_2024-03.json",
    ]
    assert all(p.exists() for p in paths)


def test_download_payload_round_trips(tmp_path, xm_handler_factory, make_payload, date_range):
    handler, _ = xm_handler_factory()
    _run(date(2024, 2, 1), date(2024, 2, 29), tmp_path, handler)
    saved = json.loads(raw_month_path(tmp_path, 2024, 2).read_text(encoding="utf-8"))
    assert saved == make_payload(date_range(date(2024, 2, 1), date(2024, 2, 29)))


def test_download_sends_bodies_matching_build_request_body(tmp_path, xm_handler_factory):
    handler, calls = xm_handler_factory()
    start, end = date(2024, 1, 20), date(2024, 2, 10)
    _run(start, end, tmp_path, handler)
    expected = [build_request_body(s, e) for s, e in month_ranges(start, end)]
    assert sorted(calls, key=lambda b: b["StartDate"]) == expected


def test_download_posts_to_api_url(tmp_path, make_payload):
    seen = []

    def handler(request):
        seen.append((request.method, str(request.url)))
        return httpx.Response(200, json=make_payload([date(2024, 1, 1)]))

    _run(date(2024, 1, 1), date(2024, 1, 1), tmp_path, handler)
    assert seen == [("POST", xm_prices.API_URL)]


def test_download_skips_existing_files(tmp_path, xm_handler_factory):
    handler, calls = xm_handler_factory()
    _run(date(2024, 1, 1), date(2024, 2, 29), tmp_path, handler)
    assert len(calls) == 2
    paths = _run(date(2024, 1, 1), date(2024, 2, 29), tmp_path, handler)
    assert len(calls) == 2  # no new requests on the second run
    assert len(paths) == 2


def test_download_force_redownloads(tmp_path, xm_handler_factory):
    handler, calls = xm_handler_factory()
    _run(date(2024, 1, 1), date(2024, 1, 31), tmp_path, handler)
    _run(date(2024, 1, 1), date(2024, 1, 31), tmp_path, handler, force=True)
    assert len(calls) == 2


def test_download_retries_after_http_500_then_succeeds(tmp_path, xm_handler_factory):
    handler, calls = xm_handler_factory(
        fail={"2024-01": lambda n: httpx.Response(500, text="boom") if n == 1 else None}
    )
    paths = _run(date(2024, 1, 1), date(2024, 1, 31), tmp_path, handler)
    assert len(calls) == 2
    assert paths[0].exists()


def test_download_retries_after_invalid_json_then_succeeds(tmp_path, xm_handler_factory):
    handler, calls = xm_handler_factory(
        fail={
            "2024-01": lambda n: httpx.Response(200, text="<html>not json</html>")
            if n == 1
            else None
        }
    )
    paths = _run(date(2024, 1, 1), date(2024, 1, 31), tmp_path, handler)
    assert len(calls) == 2
    json.loads(paths[0].read_text(encoding="utf-8"))  # valid JSON was saved


def test_download_persistent_failure_raises_with_failed_month(tmp_path, xm_handler_factory):
    handler, _ = xm_handler_factory(fail={"2024-02": lambda n: httpx.Response(500)})
    with pytest.raises(XMDownloadError) as excinfo:
        _run(date(2024, 1, 1), date(2024, 3, 31), tmp_path, handler)
    assert list(excinfo.value.failed) == ["2024-02"]


def test_download_persistent_failure_still_writes_other_months(tmp_path, xm_handler_factory):
    handler, _ = xm_handler_factory(fail={"2024-02": lambda n: httpx.Response(500)})
    with pytest.raises(XMDownloadError):
        _run(date(2024, 1, 1), date(2024, 3, 31), tmp_path, handler)
    assert raw_month_path(tmp_path, 2024, 1).exists()
    assert raw_month_path(tmp_path, 2024, 3).exists()
    assert not raw_month_path(tmp_path, 2024, 2).exists()


def test_download_failure_leaves_no_tmp_files(tmp_path, xm_handler_factory):
    handler, _ = xm_handler_factory(fail={"2024-02": lambda n: httpx.Response(500)})
    with pytest.raises(XMDownloadError):
        _run(date(2024, 1, 1), date(2024, 2, 29), tmp_path, handler)
    assert not list(tmp_path.glob("*.tmp"))


# ------------------------------------------------- build_prices_csv / validate
START, END = date(2024, 1, 30), date(2024, 2, 2)


@pytest.fixture
def raw_dir(tmp_path, make_payload, date_range, write_month_file):
    d = tmp_path / "raw"
    write_month_file(d, 2024, 1, make_payload(date_range(date(2024, 1, 28), date(2024, 1, 31))))
    write_month_file(d, 2024, 2, make_payload(date_range(date(2024, 2, 1), date(2024, 2, 3))))
    return d


def test_build_csv_has_expected_columns_and_rows(raw_dir, tmp_path):
    out = tmp_path / "out" / "prices.csv"
    df = build_prices_csv(raw_dir, out, START, END)
    assert len(df) == 24 * 4
    csv = pd.read_csv(out)
    assert list(csv.columns) == ["timestamp", "price_cop_kwh"]
    assert len(csv) == 24 * 4


def test_build_csv_is_clipped_to_start_and_end(raw_dir, tmp_path):
    df = build_prices_csv(raw_dir, tmp_path / "p.csv", START, END)
    assert df["timestamp"].min() == pd.Timestamp("2024-01-30 00:00")
    assert df["timestamp"].max() == pd.Timestamp("2024-02-02 23:00")


def test_build_csv_is_sorted(raw_dir, tmp_path):
    df = build_prices_csv(raw_dir, tmp_path / "p.csv", START, END)
    assert df["timestamp"].is_monotonic_increasing


def test_build_csv_drops_duplicate_timestamps(tmp_path, make_payload, write_month_file):
    raw = tmp_path / "raw"
    d = date(2024, 1, 1)
    write_month_file(raw, 2024, 1, make_payload([d, d]))
    df = build_prices_csv(raw, tmp_path / "p.csv", d, d)
    assert len(df) == 24
    assert df["timestamp"].is_unique


def test_validate_complete_data_reports_no_problems(raw_dir, tmp_path):
    df = build_prices_csv(raw_dir, tmp_path / "p.csv", START, END)
    report = validate_prices(df, START, END)
    assert report.expected_rows == 24 * 4
    assert report.rows == 24 * 4
    assert report.missing_timestamps == []
    assert report.duplicate_count == 0
    assert report.nan_count == 0


def test_validate_reports_missing_day(tmp_path, make_payload, date_range, write_month_file):
    raw = tmp_path / "raw"
    # Jan 31 is absent from the data
    write_month_file(raw, 2024, 1, make_payload([date(2024, 1, 30)]))
    write_month_file(raw, 2024, 2, make_payload(date_range(date(2024, 2, 1), date(2024, 2, 2))))
    df = build_prices_csv(raw, tmp_path / "p.csv", START, END)
    report = validate_prices(df, START, END)
    assert report.expected_rows == 24 * 4
    assert report.rows == 24 * 3
    assert len(report.missing_timestamps) == 24
    assert report.missing_timestamps[0] == pd.Timestamp("2024-01-31 00:00")
    assert report.missing_timestamps[-1] == pd.Timestamp("2024-01-31 23:00")


def test_validate_counts_nan_prices(raw_dir, tmp_path):
    df = build_prices_csv(raw_dir, tmp_path / "p.csv", START, END).copy()
    df.loc[[0, 5, 10], "price_cop_kwh"] = float("nan")
    assert validate_prices(df, START, END).nan_count == 3


def test_validate_counts_duplicate_timestamps(raw_dir, tmp_path):
    df = build_prices_csv(raw_dir, tmp_path / "p.csv", START, END)
    dup = pd.concat([df, df.iloc[[0, 1]]], ignore_index=True)
    assert validate_prices(dup, START, END).duplicate_count == 2


# ------------------------------------------------------------------------ main
def test_main_returns_zero_and_writes_csv(
    tmp_path, monkeypatch, make_payload, date_range, write_month_file
):
    src = tmp_path / "fixtures_src"
    write_month_file(src, 2024, 1, make_payload(date_range(date(2024, 1, 1), date(2024, 1, 3))))

    def fake_download_months(start, end, raw_dir, **kwargs):
        raw_dir.mkdir(parents=True, exist_ok=True)
        out = []
        for s, _ in month_ranges(start, end):
            target = raw_month_path(raw_dir, s.year, s.month)
            target.write_bytes(raw_month_path(src, s.year, s.month).read_bytes())
            out.append(target)
        return out

    monkeypatch.setattr(xm_prices, "download_months", fake_download_months)
    out_csv = tmp_path / "processed" / "prices.csv"
    code = xm_prices.main(
        [
            "--start", "2024-01-01",
            "--end", "2024-01-03",
            "--raw-dir", str(tmp_path / "raw"),
            "--out", str(out_csv),
        ]
    )
    assert code == 0
    df = pd.read_csv(out_csv)
    assert list(df.columns) == ["timestamp", "price_cop_kwh"]
    assert len(df) == 72


# --------------------------------------------------------------------- network
@pytest.mark.network
def test_real_xm_api_one_day(tmp_path):
    d = date(2024, 12, 1)
    paths = download_months(d, d, tmp_path)
    df = parse_hourly_items(json.loads(paths[0].read_text(encoding="utf-8")))
    assert len(df) == 24
    assert df["timestamp"].iloc[0] == pd.Timestamp("2024-12-01 00:00")
    assert df["price_cop_kwh"].iloc[0] == pytest.approx(642.54859, rel=1e-6)
