import json
from pathlib import Path
import numpy as np
from src.data_loader import get_dataset_path, load_metadata, iter_samples, INPUT_SHAPE

def main():
    root = get_dataset_path()
    df = load_metadata(root)
    assert len(df) == 8732
    assert not df.duplicated(["fold", "slice_file_name"]).any()
    results = []
    for fold in range(1, 11):
        count = 0
        labels = set()
        for x, y in iter_samples((fold,), root):
            assert x.shape == INPUT_SHAPE and x.dtype == np.float32
            assert np.isfinite(x).all() and 0 <= x.min() <= x.max() <= 1
            assert 0 <= y < 10
            labels.add(y)
            count += 1
        assert count == int((df.fold == fold).sum())
        results.append({"fold": fold, "recordings": count, "classes": sorted(labels)})
        print(f"Fold {fold}: {count} recordings validated", flush=True)
    assert sum(r["recordings"] for r in results) == 8732
    Path("metadata/validation.json").write_text(json.dumps(results, indent=2) + "\n")

if __name__ == "__main__":
    main()
