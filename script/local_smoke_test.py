"""
Local CPU-friendly smoke test for CoMaPOI single-model inference.

Goal:
- Validate dataset loading, prompt generation and POI parsing end-to-end.
- Run a tiny number of samples (default: 2) with one local served model.
- No fine-tuning required.
"""

import argparse
import json
import random
import time
from pathlib import Path
from types import SimpleNamespace

from openai import OpenAI

from config import DATASET_ROOT, DATASET_MAX_ITEM
from parser_tool import extract_predicted_pois
from utils import clean_predicted_pois, create_prompt_ori


class LocalSmokeTester:
    """Run a tiny local smoke test with one served model."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.client = None

    def _set_seed(self) -> None:
        random.seed(self.args.seed)

    def _resolve_data_path(self) -> Path:
        return DATASET_ROOT / self.args.dataset / self.args.mode / f"{self.args.dataset}_{self.args.mode}.jsonl"

    def _load_samples(self):
        data_path = self._resolve_data_path()
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {data_path}")

        samples = []
        with data_path.open("r", encoding="utf-8") as f:
            for line in f:
                samples.append(json.loads(line))
                if len(samples) >= self.args.num_samples:
                    break

        if not samples:
            raise ValueError(f"No samples loaded from: {data_path}")
        return samples

    def _create_client(self) -> None:
        if self.args.base_url:
            base_url = self.args.base_url
        else:
            base_url = f"http://{self.args.host}:{self.args.port}/v1"
        self.client = OpenAI(api_key=self.args.api_key, base_url=base_url)

    def _predict_one(self, prompt: str):
        retry_interval = 1
        last_error = None
        for _ in range(self.args.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.args.model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=self.args.temperature,
                    top_p=self.args.top_p,
                    n=1,
                    max_tokens=self.args.max_new_tokens,
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                last_error = e
                time.sleep(retry_interval)
                retry_interval *= 2

        raise RuntimeError(f"Inference failed after {self.args.max_retries} retries: {last_error}")

    def run(self):
        self._set_seed()
        samples = self._load_samples()
        self._create_client()

        mini_args = SimpleNamespace(top_k=self.args.top_k)
        max_item = DATASET_MAX_ITEM.get(self.args.dataset, 5091)

        results = []
        hit = 0
        for idx, sample in enumerate(samples, start=1):
            user_id, prompt, label = create_prompt_ori(mini_args, sample)
            output_text = self._predict_one(prompt)
            raw_pois = extract_predicted_pois(output_text, self.args.top_k)
            pred_pois = clean_predicted_pois(raw_pois, max_item)

            label_str = str(label)
            is_hit = label_str in [str(x) for x in pred_pois[: self.args.top_k]]
            hit += int(is_hit)

            results.append(
                {
                    "idx": idx,
                    "user_id": user_id,
                    "label": label,
                    "predicted_poi_ids": pred_pois[: self.args.top_k],
                    "hit@k": is_hit,
                    "raw_output": output_text,
                }
            )
            print(f"[{idx}/{len(samples)}] user={user_id} label={label} pred={pred_pois[:self.args.top_k]}")

        hr = hit / len(samples)
        print(f"\nSmoke test done. Samples={len(samples)}, HR@{self.args.top_k}={hr:.4f}")

        if self.args.output_json:
            output_path = Path(self.args.output_json)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"Saved outputs to: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Local service-based smoke test for CoMaPOI")
    parser.add_argument("--dataset", type=str, default="nyc", choices=["nyc", "tky", "ca"])
    parser.add_argument("--mode", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--num_samples", type=int, default=2, help="Tiny sample count for smoke test")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--model", type=str, default="local-model", help="Served model name")
    parser.add_argument("--host", type=str, default="localhost", help="Local service host")
    parser.add_argument("--port", type=int, default=1025, help="Local service port")
    parser.add_argument("--base_url", type=str, default="", help="Optional full base URL, e.g. http://localhost:1025/v1")
    parser.add_argument("--api_key", type=str, default="EMPTY", help="API key for local OpenAI-compatible server")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_json", type=str, default="results/smoke/local_smoke_outputs.json")
    return parser.parse_args()


def main():
    args = parse_args()
    tester = LocalSmokeTester(args)
    tester.run()


if __name__ == "__main__":
    main()
