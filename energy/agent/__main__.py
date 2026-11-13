"""Manual check of the agent from the terminal (D16).

Run from the repo root::

    uv run python -m energy.agent "pregunta" [--bm25] [--t-max N]
        [--preset 8h] [--start 2025-10-01] [--end 2025-12-31]

``--preset``, ``--start`` and ``--end`` are the user's battery and period (D16); the
agent uses them when the question names none. Prints the answer, each tool call, the citations (valid and unverified) and why the
loop stopped. Needs ``GROQ_API_KEY`` (read from ``.env``); without it the answer is the
retrieval-only fallback.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date

from energy import config
from energy.agent.loop import open_agent, run_agent
from energy.agent.settings import UserSettings
from energy.rag.chunk import read_chunks
from energy.rag.retrieve import BM25Retriever


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    config.load_env()
    if hasattr(sys.stdout, "reconfigure"):  # Windows consoles garble Spanish accents
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Ask the battery/regulation agent.")
    parser.add_argument("question")
    parser.add_argument("--bm25", action="store_true", help="force the BM25 retriever")
    parser.add_argument("--t-max", type=int, default=config.T_MAX)
    parser.add_argument("--preset", default=config.DEFAULT_PRESET, help="battery: 4h, 8h or 12h")
    parser.add_argument("--start", type=date.fromisoformat, default=config.DEFAULT_START, help="YYYY-MM-DD")
    parser.add_argument("--end", type=date.fromisoformat, default=config.DEFAULT_END, help="YYYY-MM-DD")
    args = parser.parse_args(argv)
    try:
        settings = UserSettings(args.preset, args.start, args.end)
    except ValueError as exc:
        parser.error(str(exc))
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    retriever = BM25Retriever(read_chunks(config.CHUNKS_PATH)) if args.bm25 else None
    registry, chat = open_agent(retriever)
    result = run_agent(args.question, [], registry, chat, t_max=args.t_max, settings=settings)
    print(f"settings: preset={settings.preset} start={settings.start} end={settings.end}")
    print(result.answer)
    print("\n--- tool calls ---")
    for call in result.tool_calls:
        print(
            f"{call.name} {call.arguments} ok={call.ok}" + (f" error={call.error}" if call.error else "")
            + f" from_settings={call.from_settings}"
        )
    print(f"simulations in the results block: {len(result.simulations)}")
    print("--- citations ---")
    print("valid:", result.citations)
    print("unverified:", result.unverified_citations)
    print(f"--- stopped: {result.stopped} (retriever: {result.retriever or '-'}) ---")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
