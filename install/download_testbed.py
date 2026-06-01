import argparse
import shutil
from pathlib import Path
from huggingface_hub import hf_hub_download


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_DIR = WORKSPACE_ROOT / ".hf"


FILES = {
    "id_test.json": "timeomni1_id_test.json",
    "ood_test.json": "timeomni1_ood_test.json",
}


def main():
    parser = argparse.ArgumentParser(description="Download TimeOmni-1 testbed")
    parser.add_argument("--repo", type=str, default="anton-hugging/timeomni-1-testbed")
    parser.add_argument("--out_dir", type=str, default="data")
    parser.add_argument("--cache_dir", type=str, default=str(DEFAULT_CACHE_DIR))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    cache_dir = Path(args.cache_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    for src, dst in FILES.items():
        target = out_dir / dst
        if target.exists():
            print(f"skip {target}")
            continue
        cache_path = hf_hub_download(
            repo_id=args.repo,
            repo_type="dataset",
            filename=src,
            cache_dir=str(cache_dir),
        )
        shutil.copyfile(cache_path, target)
        print(f"saved {target}")


if __name__ == "__main__":
    main()
