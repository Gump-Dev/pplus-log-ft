"""
Dataset quality gate for PPLUS FortiGate fine-tuning JSONL files.

Checks are intentionally lightweight and stream-friendly so they can run on
Gump or DGX before training, without loading a model.
"""
import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean


VALID_CATEGORIES = {"THREAT", "BLOCKED", "DENIED", "SUSPICIOUS", "NORMAL", "OTHER"}
REQUIRED_FIELDS = {"instruction", "response", "category"}
LOG_SPLIT = "\n\nLog: "
LABEL_CHECK_FIELDS = ("action", "subtype", "threat_name")


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def response_category(response: str) -> str:
    return response.split(":", 1)[0].strip().upper()


def find_field_value(log_text: str, key: str) -> str:
    marker = f"{key}="
    start = log_text.find(marker)
    if start < 0:
        return ""
    start += len(marker)
    if start >= len(log_text):
        return ""

    if log_text[start] == '"':
        end = log_text.find('"', start + 1)
        if end < 0:
            return log_text[start + 1 :].strip()
        return log_text[start + 1 : end].strip()

    end = log_text.find(" ", start)
    if end < 0:
        return log_text[start:].strip()
    return log_text[start:end].strip()


def parse_log_fields(instruction: str) -> dict:
    if LOG_SPLIT not in instruction:
        return {}
    log_text = instruction.split(LOG_SPLIT, 1)[1]
    return {key: find_field_value(log_text, key) for key in LABEL_CHECK_FIELDS}


def expected_category_from_fields(fields: dict) -> str | None:
    subtype = fields.get("subtype", "")
    action = fields.get("action", "")

    if subtype in {"ips", "virus", "waf", "anomaly", "file-filter"}:
        return "THREAT"
    if action in {"block", "blocked", "dropped"}:
        return "BLOCKED"
    if action == "deny":
        return "DENIED"
    if action in {"client-rst", "server-rst", "timeout"}:
        return "SUSPICIOUS"
    if action in {"accept", "pass", "close", "passthrough"}:
        return "NORMAL"
    return None


def percentile(values: list[int], pct: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round((pct / 100) * (len(ordered) - 1))))
    return ordered[idx]


def new_split_report(path: str) -> dict:
    return {
        "path": path,
        "rows": 0,
        "json_errors": 0,
        "missing_required": 0,
        "empty_required": 0,
        "invalid_categories": 0,
        "response_category_mismatch": 0,
        "missing_log_marker": 0,
        "label_rule_mismatch": 0,
        "duplicate_instructions": 0,
        "conflicting_duplicate_labels": 0,
        "category_counts": Counter(),
        "response_counts": Counter(),
        "char_lengths": [],
        "word_lengths": [],
        "examples": defaultdict(list),
    }


def add_example(report: dict, key: str, line_no: int, detail: str, limit: int) -> None:
    if len(report["examples"][key]) < limit:
        report["examples"][key].append({"line": line_no, "detail": detail})


def scan_jsonl(path: Path, split_name: str, example_limit: int) -> tuple[dict, dict[str, str]]:
    report = new_split_report(str(path))
    seen: dict[str, str] = {}

    with path.open() as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue

            report["rows"] += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                report["json_errors"] += 1
                add_example(report, "json_errors", line_no, str(exc), example_limit)
                continue

            missing = REQUIRED_FIELDS - record.keys()
            if missing:
                report["missing_required"] += 1
                add_example(report, "missing_required", line_no, ",".join(sorted(missing)), example_limit)
                continue

            instruction = str(record.get("instruction", ""))
            response = str(record.get("response", "")).strip()
            category = str(record.get("category", "")).strip().upper()

            if not instruction.strip() or not response or not category:
                report["empty_required"] += 1
                add_example(report, "empty_required", line_no, f"category={category!r}", example_limit)

            if category not in VALID_CATEGORIES:
                report["invalid_categories"] += 1
                add_example(report, "invalid_categories", line_no, category, example_limit)

            resp_cat = response_category(response)
            if resp_cat != category:
                report["response_category_mismatch"] += 1
                add_example(
                    report,
                    "response_category_mismatch",
                    line_no,
                    f"response={response!r} category={category!r}",
                    example_limit,
                )

            if LOG_SPLIT not in instruction:
                report["missing_log_marker"] += 1
                add_example(report, "missing_log_marker", line_no, instruction[:160], example_limit)

            fields = parse_log_fields(instruction)
            expected = expected_category_from_fields(fields)
            if expected and expected != category:
                report["label_rule_mismatch"] += 1
                detail = f"expected={expected} category={category} action={fields.get('action','')} subtype={fields.get('subtype','')}"
                add_example(report, "label_rule_mismatch", line_no, detail, example_limit)

            key = stable_hash(instruction)
            label_key = f"{category}|{response}"
            if key in seen:
                report["duplicate_instructions"] += 1
                if seen[key] != label_key:
                    report["conflicting_duplicate_labels"] += 1
                    add_example(
                        report,
                        "conflicting_duplicate_labels",
                        line_no,
                        f"first={seen[key]} now={label_key}",
                        example_limit,
                    )
            else:
                seen[key] = label_key

            report["category_counts"][category] += 1
            report["response_counts"][response] += 1
            report["char_lengths"].append(len(instruction))
            report["word_lengths"].append(len(instruction.split()))

    return report, {f"{split_name}:{k}": v for k, v in seen.items()}


def serialize_counter(counter: Counter) -> dict:
    return dict(counter.most_common())


def summarize_lengths(values: list[int]) -> dict:
    if not values:
        return {"min": 0, "mean": 0, "p50": 0, "p95": 0, "p99": 0, "max": 0}
    return {
        "min": min(values),
        "mean": round(mean(values), 2),
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "max": max(values),
    }


def finalize_report(report: dict) -> dict:
    return {
        "path": report["path"],
        "rows": report["rows"],
        "json_errors": report["json_errors"],
        "missing_required": report["missing_required"],
        "empty_required": report["empty_required"],
        "invalid_categories": report["invalid_categories"],
        "response_category_mismatch": report["response_category_mismatch"],
        "missing_log_marker": report["missing_log_marker"],
        "label_rule_mismatch": report["label_rule_mismatch"],
        "duplicate_instructions": report["duplicate_instructions"],
        "conflicting_duplicate_labels": report["conflicting_duplicate_labels"],
        "category_counts": serialize_counter(report["category_counts"]),
        "response_counts_top20": dict(report["response_counts"].most_common(20)),
        "instruction_chars": summarize_lengths(report["char_lengths"]),
        "instruction_words": summarize_lengths(report["word_lengths"]),
        "examples": dict(report["examples"]),
    }


def write_markdown(result: dict, path: Path) -> None:
    lines = [
        "# Dataset Quality Report",
        "",
        f"- Status: **{result['status']}**",
        f"- Total rows: `{result['total_rows']}`",
        f"- Total errors: `{result['total_errors']}`",
        f"- Cross-split duplicate instructions: `{result['cross_split_duplicate_instructions']}`",
        f"- Cross-split conflicting labels: `{result['cross_split_conflicting_labels']}`",
        "",
    ]
    for name, split in result["splits"].items():
        lines.extend(
            [
                f"## {name}",
                f"- Rows: `{split['rows']}`",
                f"- Errors: `{sum(split[k] for k in result['error_fields'])}`",
                f"- Duplicates within split: `{split['duplicate_instructions']}`",
                f"- Label rule mismatches: `{split['label_rule_mismatch']}`",
                f"- Instruction words p50/p95/p99/max: `{split['instruction_words']['p50']}` / `{split['instruction_words']['p95']}` / `{split['instruction_words']['p99']}` / `{split['instruction_words']['max']}`",
                "- Category counts:",
            ]
        )
        for category, count in split["category_counts"].items():
            lines.append(f"  - `{category}`: `{count}`")
        lines.append("")
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Validate PPLUS fine-tune JSONL datasets")
    parser.add_argument("--data-dir", default="./datasets")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--output-json", default="./eval/reports/dataset_quality.json")
    parser.add_argument("--output-md", default="./eval/reports/dataset_quality.md")
    parser.add_argument("--example-limit", type=int, default=5)
    parser.add_argument("--fail-on-warnings", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    split_reports = {}
    global_seen: dict[str, tuple[str, str]] = {}
    cross_split_duplicates = 0
    cross_split_conflicts = 0

    for split in args.splits:
        path = data_dir / f"{split}.jsonl"
        if not path.exists():
            raise SystemExit(f"Missing split file: {path}")

        raw_report, split_seen = scan_jsonl(path, split, args.example_limit)
        split_reports[split] = finalize_report(raw_report)

        for scoped_key, label in split_seen.items():
            split_name, key = scoped_key.split(":", 1)
            if key in global_seen:
                cross_split_duplicates += 1
                if global_seen[key][1] != label:
                    cross_split_conflicts += 1
            else:
                global_seen[key] = (split_name, label)

    error_fields = [
        "json_errors",
        "missing_required",
        "empty_required",
        "invalid_categories",
        "response_category_mismatch",
        "missing_log_marker",
        "conflicting_duplicate_labels",
    ]
    warning_fields = ["label_rule_mismatch", "duplicate_instructions"]
    total_errors = sum(sum(split[k] for k in error_fields) for split in split_reports.values())
    total_warnings = sum(sum(split[k] for k in warning_fields) for split in split_reports.values())
    total_rows = sum(split["rows"] for split in split_reports.values())

    status = "pass"
    if total_errors or cross_split_conflicts:
        status = "fail"
    elif args.fail_on_warnings and (total_warnings or cross_split_duplicates):
        status = "fail"

    result = {
        "status": status,
        "total_rows": total_rows,
        "total_errors": total_errors,
        "total_warnings": total_warnings,
        "cross_split_duplicate_instructions": cross_split_duplicates,
        "cross_split_conflicting_labels": cross_split_conflicts,
        "error_fields": error_fields,
        "warning_fields": warning_fields,
        "splits": split_reports,
    }

    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    write_markdown(result, output_md)

    print(f"Dataset quality status: {status.upper()}")
    print(f"Rows: {total_rows}")
    print(f"Errors: {total_errors}")
    print(f"Warnings: {total_warnings}")
    print(f"Cross-split duplicates: {cross_split_duplicates}")
    print(f"Report JSON: {output_json}")
    print(f"Report MD: {output_md}")

    if status != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
