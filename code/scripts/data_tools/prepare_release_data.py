"""Write local metadata CSVs from the portable release manifests.

Only dataset paths are changed. Captions, window indices, row order, and
benchmark membership are preserved. No third-party packages are required.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path, PurePosixPath


CODE_ROOT = Path(__file__).resolve().parents[2]
PATH_COLUMNS = {"path", "path_target", "context_path", "target_path"}
SPLITS = {"train", "extreme_test_split", "seasonal_test_split"}


def resolve_path(value: str, root: Path) -> str:
    if not value:
        return value
    parts = PurePosixPath(value).parts
    if not parts or parts[0] not in SPLITS or ".." in parts:
        raise ValueError(f"Expected a relative EarthNet2021 path, got {value!r}")
    return str(root.joinpath(*parts))


def prepare_file(source: Path, target: Path, root: Path, check: bool) -> int:
    with source.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        if not fields:
            raise ValueError(f"CSV has no header: {source}")
        rows = list(reader)
    for row in rows:
        for key in PATH_COLUMNS & row.keys():
            row[key] = resolve_path(row[key], root)
        if row.get("benchmark_meta_json"):
            meta = json.loads(row["benchmark_meta_json"])
            for key in PATH_COLUMNS & meta.keys():
                meta[key] = resolve_path(meta[key], root)
            row["benchmark_meta_json"] = json.dumps(meta, ensure_ascii=False, separators=(",", ":"))
        if check:
            for key in ("path", "path_target"):
                if row.get(key) and not Path(row[key]).is_file():
                    raise FileNotFoundError(row[key])
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--earthnet-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=CODE_ROOT / "data/local")
    parser.add_argument("--split", choices=["all", "train", "extreme", "seasonal"], default="all")
    parser.add_argument("--check", action="store_true", help="Check every referenced NPZ file before writing each CSV")
    args = parser.parse_args()
    root = args.earthnet_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    source_root = CODE_ROOT / "data"
    manifests = {
        "train": "metadata/earthnet2021_train_in10out20_refine-caption_multi-p_corrected-temp50.csv",
        "extreme": "benchmarks/extreme_summer/extreme_summer_benchmark_stage2_infer.csv",
        "seasonal": "benchmarks/seasonal/seasonal_benchmark_infer_subset.csv",
    }
    selected = manifests if args.split == "all" else {args.split: manifests[args.split]}
    for relative in selected.values():
        source, target = source_root / relative, output / relative
        if target == source:
            parser.error("output-dir must not overwrite the release manifests")
        count = prepare_file(source, target, root, args.check)
        print(f"{count:>6} rows -> {target}")
    if "seasonal" in selected:
        relative = "benchmarks/seasonal/final_benchmark_pairs.csv"
        source, target = source_root / relative, output / relative
        if target == source:
            parser.error("output-dir must not overwrite the release manifests")
        count = prepare_file(source, target, root, args.check)
        print(f"{count:>6} pairs -> {target}")


if __name__ == "__main__":
    main()
