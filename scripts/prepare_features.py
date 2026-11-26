import argparse
import time

from src.features import build_feature_cache
from src.paths import FEATURES_DIR


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", default=None, help="UrbanSound8K folder (found automatically if omitted)")
    parser.add_argument("--out-dir", default=FEATURES_DIR, help="where to write the cache")
    parser.add_argument("--workers", type=int, default=None, help="processes to use (default: CPU count - 1)")
    parser.add_argument("--overwrite", action="store_true", help="rebuild an existing cache")
    args = parser.parse_args()

    start = time.perf_counter()
    out_dir = build_feature_cache(args.dataset_path, args.out_dir, args.workers, args.overwrite)
    print(f"Feature cache written to {out_dir} in {time.perf_counter() - start:.0f} s")


if __name__ == "__main__":
    main()