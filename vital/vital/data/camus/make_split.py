import sys
from pathlib import Path

def make_camussplits(data_root: Path, n_folds: int = 10):
    data_root = Path(data_root)
    patients = sorted([d.name for d in data_root.iterdir() if d.is_dir() and d.name.startswith("patient")])
    
    if len(patients) != 500:
        raise ValueError(f"Expected 500 patients but found {len(patients)}")

    fold_size = len(patients) // n_folds
    fold_dir = data_root / "listSubGroups"
    fold_dir.mkdir(exist_ok=True)

    # Split patients into 10 folds (folds is a list of lists)
    folds = [patients[i*fold_size:(i+1)*fold_size] for i in range(n_folds)]

    for i in range(n_folds):
        test = folds[i]
        val = folds[(i + 1) % n_folds]
        train = [p for fold in folds if fold not in (test, val) for p in fold]

        for subset, ids in zip(["training", "validation", "testing"], [train, val, test]):
            fn = fold_dir / f"subGroup{i+1}_{subset}.txt"
            fn.write_text("\n".join(ids))

        print(f"Fold {i+1}: train={len(train)} val={len(val)} test={len(test)}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python make_splits.py <CAMUS_ROOT_PATH>")
        sys.exit(1)
    data_root = sys.argv[1]
    make_camussplits(data_root)
