"""Manual check of the agent from the terminal (D16).

Run from the repo root::

    uv run python -m energy.agent "pregunta" [--bm25] [--t-max N]

Prints the answer, each tool call, the citations (valid and unverified) and why the
loop stopped. Needs ``GROQ_API_KEY`` (read from ``.env``); without it the answer is the
retrieval-only fallback.
"""

from __future__ import annotations

import argparse
import logging
import sys

from energy import config
from energy.agent.loop import open_agent, run_agent
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
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    retriever = BM25Retriever(read_chunks(config.CHUNKS_PATH)) if args.bm25 else None
    registry, chat = open_agent(retriever)
    result = run_agent(args.question, [], registry, chat, t_max=args.t_max)

    print(result.answer)
    print("\n--- tool calls ---")
    for call in result.tool_calls:
        print(f"{call.name} {call.arguments} ok={call.ok}" + (f" error={call.error}" if call.error else ""))
    print(f"simulations in the results block: {len(result.simulations)}")
    print("--- citations ---")
    print("valid:", result.citations)
    print("unverified:", result.unverified_citations)
    print(f"--- stopped: {result.stopped} (retriever: {result.retriever or '-'}) ---")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
