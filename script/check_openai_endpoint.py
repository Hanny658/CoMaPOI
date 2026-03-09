#!/usr/bin/env python3
"""
Check whether an OpenAI-compatible endpoint is ready and list model IDs.
"""

import argparse
import json
from pathlib import Path

from openai import OpenAI


def main() -> int:
    parser = argparse.ArgumentParser(description="Check OpenAI-compatible endpoint.")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7863)
    parser.add_argument("--base-url", type=str, default="")
    parser.add_argument("--api-key", type=str, default="EMPTY")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    base_url = args.base_url or f"http://{args.host}:{args.port}/v1"
    client = OpenAI(api_key=args.api_key, base_url=base_url)
    try:
        models = [m.id for m in client.models.list().data]
    except Exception as e:
        if not args.quiet:
            print(f"[ERROR] Endpoint not ready: {e}")
        return 1

    payload = {"base_url": base_url, "models": models}
    if not args.quiet:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

