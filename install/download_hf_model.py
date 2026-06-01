import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_DIR = WORKSPACE_ROOT / ".hf"

# Released TimeOmni-1 checkpoints and their base models.
#   - TimeOmni-1-7B : Qwen2.5-7B-Instruct (text-only)
#   - TimeOmni-1-4B : Qwen3.5-4B          (VLM-backed, text used for inference)
#   - TimeOmni-1-9B : Qwen3.5-9B          (VLM-backed, text used for inference)
MODEL_REGISTRY = {
    "7B": "anton-hugging/TimeOmni-1-7B",
    "4B": "TimeOmni-1/TimeOmni-1-4B",
    "9B": "TimeOmni-1/TimeOmni-1-9B",
}


def resolve_repo_id(model: str) -> str:
    """Accept either a short tag (7B/4B/9B) or a full repo id."""
    return MODEL_REGISTRY.get(model.upper(), model)


def download(repo_id: str, cache_dir: Path, local_dir: Path | None) -> str:
    cache_dir.mkdir(parents=True, exist_ok=True)
    if local_dir is not None:
        local_dir.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path = snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        cache_dir=str(cache_dir),
        local_dir=str(local_dir) if local_dir is not None else None,
    )
    print(f"saved {repo_id} -> {snapshot_path}")
    return snapshot_path


def main():
    parser = argparse.ArgumentParser(description="Download TimeOmni-1 model(s)")
    parser.add_argument(
        "--model",
        type=str,
        default="7B",
        help="Short tag (7B/4B/9B) or a full Hugging Face repo id.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all released checkpoints (7B, 4B, 9B).",
    )
    parser.add_argument(
        "--local_dir",
        type=str,
        default=None,
        help="Optional flat directory for the snapshot. By default the model is "
             "stored in the shared Hugging Face cache (cache_dir) only.",
    )
    parser.add_argument("--cache_dir", type=str, default=str(DEFAULT_CACHE_DIR), help="Hugging Face cache directory.")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    local_dir = Path(args.local_dir) if args.local_dir else None

    if args.all:
        for repo_id in MODEL_REGISTRY.values():
            download(repo_id, cache_dir, None)
    else:
        download(resolve_repo_id(args.model), cache_dir, local_dir)


if __name__ == "__main__":
    main()
