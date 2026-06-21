"""
Data extraction from v3x ClickHouse.
Samples balanced dataset from fortigate_traffic and exports as JSONL.
"""
import os
import json
import argparse
from pathlib import Path
from datetime import datetime, timedelta

import clickhouse_connect
import yaml
import pandas as pd


def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    # Expand env vars in password
    cfg["clickhouse"]["password"] = os.path.expandvars(cfg["clickhouse"]["password"])
    return cfg


def get_client(cfg: dict):
    ch = cfg["clickhouse"]
    return clickhouse_connect.get_client(
        host=ch["host"],
        port=ch["port"],
        username=ch["user"],
        password=ch["password"],
        database=ch["database"],
    )


COLUMNS = [
    "timestamp", "tenant_name", "device", "action", "src_ip", "dst_ip",
    "src_port", "dst_port", "protocol_name", "service", "policy_name",
    "bytes_sent", "bytes_recv", "duration", "app_name", "app_cat",
    "src_country", "dst_country", "threat_type", "threat_name",
    "threat_level", "threat_matched", "url_category", "url",
    "message",
]


def format_log_row(row) -> str:
    """Format a ClickHouse row as a human-readable FortiGate log line."""
    parts = []
    for col in COLUMNS:
        val = getattr(row, col, None) if hasattr(row, col) else row.get(col)
        if val is not None and str(val) != "":
            parts.append(f"{col}={val}")
    return " ".join(parts)


def classify_label(row) -> str:
    """Determine the label for a log row."""
    threat_matched = row.get("threat_matched", 0)
    action = row.get("action", "")
    threat_type = row.get("threat_type", "")
    threat_level = row.get("threat_level", "")

    if threat_matched == 1 and threat_type:
        severity = threat_level if threat_level else "unknown"
        return f"THREAT:{threat_type}:{severity}"

    # Suspicious patterns: deny to many ports
    if action in ("deny", "timeout"):
        return "SUSPICIOUS:denied"

    if action in ("client-rst", "server-rst"):
        return "SUSPICIOUS:reset"

    return "NORMAL"


def build_instruction(log_text: str, label: str) -> dict:
    """Build instruction-response pair for fine-tuning."""
    return {
        "instruction": (
            "You are a FortiGate firewall log analyst. "
            "Classify the following log entry. "
            "Respond with the classification and a brief reason.\n\n"
            f"Log: {log_text}"
        ),
        "response": label,
        "category": label.split(":")[0] if ":" in label else label,
    }


def sample_threats(client, limit: int):
    """Sample rows where threat_matched=1."""
    query = f"""
    SELECT {", ".join(COLUMNS)}
    FROM fortigate_traffic
    WHERE threat_matched = 1 AND threat_type != ''
    ORDER BY rand() LIMIT {limit}
    """
    return client.query(query).named_results()


def sample_suspicious(client, limit: int):
    """Sample denied/timeout rows (potential scans, attacks)."""
    query = f"""
    SELECT {", ".join(COLUMNS)}
    FROM fortigate_traffic
    WHERE action IN ('deny', 'timeout')
      AND threat_matched = 0
    ORDER BY rand() LIMIT {limit}
    """
    return client.query(query).named_results()


def sample_normal(client, limit: int):
    """Sample normal traffic (accept/pass/close)."""
    query = f"""
    SELECT {", ".join(COLUMNS)}
    FROM fortigate_traffic
    WHERE action IN ('accept', 'pass', 'close')
      AND threat_matched = 0
    ORDER BY rand() LIMIT {limit}
    """
    return client.query(query).named_results()


def export_jsonl(records: list, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Extract dataset from v3x ClickHouse")
    parser.add_argument("--config", default="configs/data_prep.yaml")
    parser.add_argument("--output-dir", default="./datasets")
    args = parser.parse_args()

    cfg = load_config(args.config)
    sampling = cfg["sampling"]
    client = get_client(cfg)

    print("=== PPLUS Log Fine-Tune Dataset Prep ===")
    print(f"Sampling from {cfg['clickhouse']['host']}:{cfg['clickhouse']['port']}")

    all_records = []

    # Sample each category
    categories = [
        ("threat", sample_threats, sampling["splits"]["threat"]),
        ("suspicious", sample_suspicious, sampling["splits"]["suspicious"]),
        ("normal", sample_normal, sampling["splits"]["normal"]),
    ]

    for name, sampler_fn, count in categories:
        print(f"\nSampling {count} {name} rows...")
        rows = list(sampler_fn(client, count))
        print(f"  Got {len(rows)} rows")

        for row in rows:
            row_dict = {k: str(v) if v is not None else "" for k, v in dict(row).items()}
            log_text = format_log_row(row_dict)
            label = classify_label(row_dict)
            record = build_instruction(log_text, label)
            all_records.append(record)

    print(f"\nTotal records: {len(all_records)}")

    # Shuffle
    import random
    random.seed(42)
    random.shuffle(all_records)

    # Split
    n = len(all_records)
    n_train = int(n * sampling["train_ratio"])
    n_val = int(n * sampling["val_ratio"])

    train = all_records[:n_train]
    val = all_records[n_train:n_train + n_val]
    test = all_records[n_train + n_val:]

    out_dir = args.output_dir
    export_jsonl(train, f"{out_dir}/train.jsonl")
    export_jsonl(val, f"{out_dir}/val.jsonl")
    export_jsonl(test, f"{out_dir}/test.jsonl")

    # Stats
    from collections import Counter
    label_counts = Counter(r["category"] for r in all_records)
    print(f"\n=== Exported ===")
    print(f"Train: {len(train)} -> {out_dir}/train.jsonl")
    print(f"Val:   {len(val)} -> {out_dir}/val.jsonl")
    print(f"Test:  {len(test)} -> {out_dir}/test.jsonl")
    print(f"\nLabel distribution:")
    for label, count in label_counts.most_common():
        print(f"  {label}: {count}")


if __name__ == "__main__":
    main()
