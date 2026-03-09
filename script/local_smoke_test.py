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
import re
import time
from pathlib import Path
import sys

from openai import OpenAI

# Ensure project root is importable when running as a script.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import DATASET_ROOT, DATASET_MAX_ITEM
from parser_tool import extract_predicted_pois


def _clean_predicted_pois(predicted_pois, max_item):
    cleaned_pois = []
    seen = set()
    for poi in predicted_pois:
        try:
            poi_id = int(poi)
            if 1 <= poi_id <= max_item and poi_id not in seen:
                cleaned_pois.append(poi_id)
                seen.add(poi_id)
        except (ValueError, TypeError):
            continue
    return cleaned_pois


def _create_prompt_ori(top_k, sample):
    messages = sample.get("messages", [])
    user_id = None
    label = None
    current_trajectory = None

    for msg in messages:
        if msg.get("role") == "user":
            content = msg.get("content", "")
            user_match = re.search(r'"user_id":\s*"?(\d+)"?', content)
            if user_match:
                user_id = user_match.group(1)
            current_trajectory = content
        elif msg.get("role") == "assistant":
            content = msg.get("content", "")
            try:
                data = json.loads(content)
                if isinstance(data, dict) and "next_poi_id" in data:
                    label = data["next_poi_id"]
                elif isinstance(data, int):
                    label = data
            except json.JSONDecodeError:
                label_match = re.search(r'"next_poi_id":\s*(\d+)', content)
                if label_match:
                    label = label_match.group(1)

    prompt = f"""You are an expert POI Predictor specialized in predicting the next Point of Interest (POI) a user will visit based on their trajectory.

User Trajectory:
{current_trajectory}

Task: Predict the next POI ID for user_{user_id} based on their trajectory.

Respond with a JSON dictionary in a markdown's fenced code block as follows:
```json
{{"next_poi_id": ["value1", "value2", ..., "value{top_k}"]}}
```"""

    return user_id, prompt, label


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

        max_item = DATASET_MAX_ITEM.get(self.args.dataset, 5091)

        results = []
        hit = 0
        for idx, sample in enumerate(samples, start=1):
            user_id, prompt, label = _create_prompt_ori(self.args.top_k, sample)
            output_text = self._predict_one(prompt)
            raw_pois = extract_predicted_pois(output_text, self.args.top_k)
            pred_pois = _clean_predicted_pois(raw_pois, max_item)

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
