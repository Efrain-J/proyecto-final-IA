"""Download hourly Colombian bolsa prices (PrecBolsNaci) from the XM API."""

from __future__ import annotations

import argparse
import asyncio
import calendar
import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

from energy.data.io import write_bytes_atomic, write_json_atomic

API_URL = "https://servapibi.xm.com.co/hourly"
METRIC_ID = "PrecBolsNaci"
USER_AGENT = "trabajo-final-ia/0.1 (academic project)"


class XMResponseError(ValueError):
    """The XM payload is missing or has a malformed ``Items`` list."""


class XMDownloadError(RuntimeError):
    """One or more months could not be downloaded."""

    def __init__(self, failed: dict[str, str]) -> None:
        self.failed = failed
        super().__init__(f"Failed to download months: {failed}")


def month_ranges(start: date, end: date) -> list[tuple[date, date]]:
    """Split ``[start, end]`` into inclusive calendar-month chunks."""
    if start > end:
        raise ValueError(f"start {start} is after end {end}")
    ranges: list[tuple[date, date]] = []
    current = start
    while current <= end:
        last_day = calendar.monthrange(current.year, current.month)[1]
        chunk_end = min(date(current.year, current.month, last_day), end)
        ranges.append((current, chunk_end))
        current = chunk_end + timedelta(days=1)
    return ranges


def build_request_body(start: date, end: date) -> dict[str, Any]:
    """Build the JSON body for the XM hourly endpoint."""
    return {
        "MetricId": METRIC_ID,
        "StartDate": start.isoformat(),
        "EndDate": end.isoformat(),
        "Entity": "Sistema",
        "Filter": [],
    }


def _to_float(value: Any) -> float:
    if value is None or value == "":
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def parse_hourly_items(payload: dict[str, Any]) -> pd.DataFrame:
    """Convert an XM hourly payload into a ``timestamp``/``price_cop_kwh`` frame."""
    items = payload.get("Items") if isinstance(payload, dict) else None
    if not isinstance(items, list):
        raise XMResponseError("Payload has no 'Items' list")
    rows: list[tuple[pd.Timestamp, float]] = []
    for item in items:
        try:
            day = pd.Timestamp(item["Date"])
            values = item["HourlyEntities"][0]["Values"]
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise XMResponseError(f"Malformed item: {item!r}") from exc
        if not isinstance(values, dict):
            raise XMResponseError(f"Malformed item values: {item!r}")
        for hour in range(1, 25):
            rows.append(
                (day + pd.Timedelta(hours=hour - 1), _to_float(values.get(f"Hour{hour:02d}")))
            )
    df = pd.DataFrame(rows, columns=["timestamp", "price_cop_kwh"])
    df["timestamp"] = pd.to_datetime(df["timestamp"]).astype("datetime64[ns]")
    df["price_cop_kwh"] = df["price_cop_kwh"].astype(float)
    return df.sort_values("timestamp").reset_index(drop=True)


def raw_month_path(raw_dir: Path, year: int, month: int) -> Path:
    """Path of the raw JSON file for a given month."""
    return Path(raw_dir) / f"{METRIC_ID}_{year:04d}-{month:02d}.json"


async def _fetch_month(
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    start: date,
    end: date,
    path: Path,
    retries: int,
    backoff_base: float,
) -> None:
    """Fetch one month with retries; raise ``XMResponseError`` if all attempts fail."""
    body = build_request_body(start, end)
    last_error = "no attempts made"
    async with sem:
        for attempt in range(retries):
            try:
                response = await client.post(API_URL, json=body)
                if response.status_code != 200:
                    raise ValueError(f"HTTP {response.status_code}")
                payload = response.json()  # JSONDecodeError is a ValueError
                if not isinstance(payload, dict) or not isinstance(payload.get("Items"), list):
                    raise ValueError("payload without 'Items' list")
                write_json_atomic(path, payload)
                return
            except (httpx.HTTPError, ValueError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt < retries - 1 and backoff_base > 0:
                    await asyncio.sleep(backoff_base * 2**attempt)
    raise XMResponseError(last_error)


async def _download_all(
    start: date,
    end: date,
    raw_dir: Path,
    force: bool,
    concurrency: int,
    retries: int,
    timeout: float,
    backoff_base: float,
    transport: httpx.AsyncBaseTransport | None,
) -> list[Path]:
    sem = asyncio.Semaphore(concurrency)
    todo: list[tuple[date, date, Path]] = []
    paths: list[Path] = []
    for chunk_start, chunk_end in month_ranges(start, end):
        path = raw_month_path(raw_dir, chunk_start.year, chunk_start.month)
        if path.exists() and not force:
            print(f"skip {path.name} (exists)")
            paths.append(path)
        else:
            todo.append((chunk_start, chunk_end, path))
    failed: dict[str, str] = {}

    async with httpx.AsyncClient(
        transport=transport, timeout=timeout, headers={"User-Agent": USER_AGENT}
    ) as client:

        async def run(chunk_start: date, chunk_end: date, path: Path) -> None:
            try:
                await _fetch_month(
                    client, sem, chunk_start, chunk_end, path, retries, backoff_base
                )
            except XMResponseError as exc:
                failed[f"{chunk_start:%Y-%m}"] = str(exc)
            else:
                print(f"downloaded {path.name}")
                paths.append(path)

        await asyncio.gather(*(run(*item) for item in todo))
    if failed:
        raise XMDownloadError(dict(sorted(failed.items())))
    return sorted(paths)


def download_months(
    start: date,
    end: date,
    raw_dir: Path,
    *,
    force: bool = False,
    concurrency: int = 4,
    retries: int = 3,
    timeout: float = 60.0,
    backoff_base: float = 1.0,
    transport: httpx.AsyncBaseTransport | None = None,
) -> list[Path]:
    """Download one raw JSON file per calendar month in ``[start, end]``.

    Existing files are skipped unless ``force``. Every month is attempted;
    ``XMDownloadError`` is raised afterwards if any failed.
    """
    return asyncio.run(
        _download_all(
            start, end, Path(raw_dir), force, concurrency, retries,
            timeout, backoff_base, transport,
        )
    )


def build_prices_csv(raw_dir: Path, out_csv: Path, start: date, end: date) -> pd.DataFrame:
    """Merge the monthly raw files into one hourly CSV clipped to ``[start, end]``."""
    frames = [
        parse_hourly_items(json.loads(path.read_text(encoding="utf-8")))
        for path in sorted(Path(raw_dir).glob(f"{METRIC_ID}_*.json"))
    ]
    columns = ["timestamp", "price_cop_kwh"]
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)
    if not df.empty:
        lo = pd.Timestamp(start)
        hi = pd.Timestamp(end) + pd.Timedelta(hours=23)
        df = df[(df["timestamp"] >= lo) & (df["timestamp"] <= hi)]
    df = (
        df.drop_duplicates(subset="timestamp", keep="first")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    csv_text = df.to_csv(index=False, date_format="%Y-%m-%d %H:%M:%S")
    write_bytes_atomic(Path(out_csv), csv_text.encode("utf-8"))
    return df


@dataclass
class PriceReport:
    """Quality summary of an hourly price series."""

    expected_rows: int
    rows: int
    missing_timestamps: list[pd.Timestamp]
    duplicate_count: int
    nan_count: int


def validate_prices(df: pd.DataFrame, start: date, end: date) -> PriceReport:
    """Compare ``df`` against the full hourly grid of ``[start, end]``."""
    grid = pd.date_range(
        pd.Timestamp(start), pd.Timestamp(end) + pd.Timedelta(hours=23), freq="h"
    )
    present = pd.DatetimeIndex(df["timestamp"])
    missing = grid.difference(present)
    return PriceReport(
        expected_rows=len(grid),
        rows=len(df),
        missing_timestamps=list(missing),
        duplicate_count=int(present.duplicated().sum()),
        nan_count=int(df["price_cop_kwh"].isna().sum()),
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: download months, build the CSV and print a report."""
    parser = argparse.ArgumentParser(description="Download XM hourly bolsa prices.")
    parser.add_argument("--start", type=date.fromisoformat, default=date(2024, 1, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2025, 12, 31))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/xm"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/prices_hourly.csv"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    try:
        download_months(args.start, args.end, args.raw_dir, force=args.force)
    except XMDownloadError as exc:
        print(f"ERROR: failed months: {exc.failed}")
        return 1

    df = build_prices_csv(args.raw_dir, args.out, args.start, args.end)
    report = validate_prices(df, args.start, args.end)
    print(f"rows: {report.rows} (expected {report.expected_rows})")
    print(f"missing timestamps: {len(report.missing_timestamps)}")
    print(f"duplicates: {report.duplicate_count}")
    print(f"NaN prices: {report.nan_count}")
    if report.missing_timestamps:
        print(f"WARNING: first missing: {report.missing_timestamps[:5]}")
    stats = df.groupby(df["timestamp"].dt.year)["price_cop_kwh"].agg(["min", "mean", "max"])
    print("price COP/kWh per year:")
    print(stats.round(2).to_string())
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
