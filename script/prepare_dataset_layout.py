#!/usr/bin/env python3
"""
Prepare and validate dataset_all layout for CoMaPOI scripts.

This helper keeps original flat files and creates nested paths expected by
inference/training scripts:
  dataset_all/{dataset}/{split}/{dataset}_{split}.jsonl
"""

import argparse
import json
import shutil
from pathlib import Path


DATASETS = ("nyc", "tky", "ca")
SPLITS = ("train", "test")


def ensure_copy(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        shutil.copy2(src, dst)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare dataset_all layout for CoMaPOI.")
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset_all"))
    parser.add_argument("--datasets", nargs="*", default=list(DATASETS))
    parser.add_argument("--result-json", type=Path, default=Path("result/00_dataset_manifest.json"))
    parser.add_argument("--require-poi-info", action="store_true")
    parser.add_argument("--require-candidates", action="store_true")
    args = parser.parse_args()

    root = args.dataset_root
    manifest = {"root": str(root), "datasets": {}, "missing": []}

    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    for dataset in args.datasets:
        d = root / dataset
        d.mkdir(parents=True, exist_ok=True)
        info = {
            "train_jsonl": False,
            "test_jsonl": False,
            "poi_info": False,
            "train_candidates": False,
            "test_candidates": False,
        }

        for split in SPLITS:
            flat = root / f"{dataset}_{split}.jsonl"
            nested = d / split / f"{dataset}_{split}.jsonl"
            alt = d / f"{dataset}_{split}.jsonl"
            copied = ensure_copy(flat, nested)
            if copied:
                ensure_copy(flat, alt)
            info[f"{split}_jsonl"] = nested.exists() or alt.exists()
            if not info[f"{split}_jsonl"]:
                manifest["missing"].append(f"{dataset}:{split}_jsonl")

        poi_flat = root / f"{dataset}_poi_info.csv"
        poi_nested = d / f"{dataset}_poi_info.csv"
        ensure_copy(poi_flat, poi_nested)
        info["poi_info"] = poi_nested.exists()
        if args.require_poi_info and not info["poi_info"]:
            manifest["missing"].append(f"{dataset}:poi_info.csv")

        tr_cand_flat = root / f"{dataset}_train_candidates.jsonl"
        te_cand_flat = root / f"{dataset}_test_candidates.jsonl"
        tr_cand_nested = d / "train" / f"{dataset}_train_candidates.jsonl"
        te_cand_nested = d / "test" / f"{dataset}_test_candidates.jsonl"
        ensure_copy(tr_cand_flat, tr_cand_nested)
        ensure_copy(te_cand_flat, te_cand_nested)
        info["train_candidates"] = tr_cand_nested.exists() or (d / f"{dataset}_train_candidates.jsonl").exists()
        info["test_candidates"] = te_cand_nested.exists() or (d / f"{dataset}_test_candidates.jsonl").exists()
        if args.require_candidates:
            if not info["train_candidates"]:
                manifest["missing"].append(f"{dataset}:train_candidates")
            if not info["test_candidates"]:
                manifest["missing"].append(f"{dataset}:test_candidates")

        manifest["datasets"][dataset] = info

    args.result_json.parent.mkdir(parents=True, exist_ok=True)
    args.result_json.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 1 if manifest["missing"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

