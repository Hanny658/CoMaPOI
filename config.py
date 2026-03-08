"""
Minimal runtime configuration for CoMaPOI.
各位复现时请注意, 以下配置项可根据实际环境进行调整:
"""

from pathlib import Path

# Project-relative roots
DATASET_ROOT = Path("dataset_all")
LEGACY_DATASET_ROOT = Path("dataset")
FINETUNE_DATA_ROOT = Path("finetune") / "data"
FINETUNE_RESULTS_ROOT = Path("finetune") / "results"

# API defaults
DEFAULT_VLLM_HOST = "localhost"
DEFAULT_PORT_INVERSE = 7863

# Dataset defaults
DATASET_MAX_ITEM = {"nyc": 5091, "tky": 7851, "ca": 13630}
DATASET_TRAIN_SAMPLES = {"nyc": 3870, "tky": 11850, "ca": 6616}


def vllm_base_url(host: str, port: int) -> str:
    return f"http://{host}:{port}/v1"
