"""
Build a deterministic, balanced fixed eval set from the exported test split.
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open() as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Create fixed stratified eval set")
    parser.add_argument("--input", default="./datasets/test.jsonl")
    parser.add_argument("--output", default="./eval/fixed_eval_set.jsonl")
    parser.add_argument("--per-category", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260621)
    args = parser.parse_args()

    records = load_jsonl(Path(args.input))
    grouped = defaultdict(list)
    for record in records:
        grouped[str(record.get("category", "OTHER")).upper()].append(record)

    rng = random.Random(args.seed)
    selected = []
    summary = {}
    for category in sorted(grouped):
        items = list(grouped[category])
        rng.shuffle(items)
        chosen = items[: args.per_category]
        selected.extend(chosen)
        summary[category] = {"available": len(items), "selected": len(chosen)}

    rng.shuffle(selected)
    write_jsonl(selected, Path(args.output))

    print(f"Fixed eval set: {args.output}")
    print(f"Rows: {len(selected)}")
    for category, counts in summary.items():
        print(f"  {category}: {counts['selected']}/{counts['available']}")


if __name__ == "__main__":
    main()
