"""
Data extraction from v3x ClickHouse.
Samples balanced dataset from fortigate_traffic and exports as JSONL.

Label scheme based on actual data:
  - THREAT       : subtype IN (ips, virus, waf, anomaly, file-filter) with threat_name
  - BLOCKED      : action IN (block, dropped, blocked) — app-ctrl/filter blocks
  - DENIED       : action=deny — firewall denials (could be scans, policy violations)
  - SUSPICIOUS   : action IN (client-rst, server-rst, timeout) — potential issues
  - NORMAL       : action IN (accept, pass, close) — clean traffic

Uses deterministic sampling (LIMIT + offset patterns) to avoid slow ORDER BY rand().
"""
import os
import json
import random
import argparse
from pathlib import Path
from collections import Counter

import clickhouse_connect
import yaml


def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
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
        connect_timeout=30,
        send_receive_timeout=300,
    )


# Columns to include in formatted log text
LOG_COLUMNS = [
    "timestamp", "tenant_name", "device", "subtype", "action",
    "src_ip", "dst_ip", "src_port", "dst_port",
    "protocol_name", "service", "policy_name",
    "bytes_sent", "bytes_recv", "duration",
    "app_name", "app_cat", "app_risk",
    "src_country", "dst_country",
    "threat_type", "threat_name", "threat_level", "threat_feed",
    "url_category", "url", "level", "message",
]


def format_log_text(row: dict) -> str:
    """Format a ClickHouse row as a compact FortiGate-style log line."""
    parts = []
    for col in LOG_COLUMNS:
        val = row.get(col, "")
        if val is not None and str(val) != "":
            parts.append(f"{col}={val}")
    return " ".join(parts)


def classify_label(row: dict) -> str:
    """Determine the classification label for a log row."""
    subtype = row.get("subtype", "")
    action = row.get("action", "")
    threat_name = row.get("threat_name", "")
    threat_type = row.get("threat_type", "")

    # Real threats: IPS, virus, WAF, anomaly, file-filter with names
    if subtype in ("ips", "virus", "waf", "anomaly", "file-filter") and threat_name:
        return f"THREAT:{subtype}:{threat_type or threat_name}"

    # App-ctrl / webfilter blocks
    if action in ("block", "blocked", "dropped"):
        return f"BLOCKED:{subtype}"

    # Firewall denials
    if action == "deny":
        return "DENIED"

    # Resets and timeouts — suspicious
    if action in ("client-rst", "server-rst", "timeout"):
        return "SUSPICIOUS"

    # Clean traffic
    if action in ("accept", "pass", "close", "passthrough"):
        return "NORMAL"

    return "OTHER"


def build_instruction(log_text: str, label: str) -> dict:
    """Build instruction-response pair for fine-tuning."""
    category = label.split(":")[0] if ":" in label else label
    return {
        "instruction": (
            "You are a FortiGate firewall log analyst. "
            "Classify the following log entry into one of these categories: "
            "THREAT, BLOCKED, DENIED, SUSPICIOUS, NORMAL, OTHER. "
            "If it is a THREAT, specify the subtype.\n\n"
            f"Log: {log_text}"
        ),
        "response": label,
        "category": category,
    }


# --- Sampling queries (use LIMIT to avoid full table scans) ---

def sample_threats(client, limit: int):
    """IPS, virus, WAF, anomaly, file-filter with threat names."""
    query = f"""
    SELECT {", ".join(LOG_COLUMNS)}
    FROM fortigate_traffic
    WHERE subtype IN ('ips', 'virus', 'waf', 'anomaly', 'file-filter')
      AND threat_name != ''
    LIMIT {limit}
    """
    return list(client.query(query).named_results())


def sample_blocked(client, limit: int):
    """App-ctrl / webfilter blocks."""
    query = f"""
    SELECT {", ".join(LOG_COLUMNS)}
    FROM fortigate_traffic
    WHERE action IN ('block', 'blocked', 'dropped')
      AND subtype NOT IN ('ips', 'virus', 'waf', 'anomaly', 'file-filter')
    LIMIT {limit}
    """
    return list(client.query(query).named_results())


def sample_denied(client, limit: int):
    """Firewall denials (potential scans)."""
    query = f"""
    SELECT {", ".join(LOG_COLUMNS)}
    FROM fortigate_traffic
    WHERE action = 'deny'
    LIMIT {limit}
    """
    return list(client.query(query).named_results())


def sample_suspicious(client, limit: int):
    """Resets and timeouts."""
    query = f"""
    SELECT {", ".join(LOG_COLUMNS)}
    FROM fortigate_traffic
    WHERE action IN ('client-rst', 'server-rst', 'timeout')
    LIMIT {limit}
    """
    return list(client.query(query).named_results())


def sample_normal(client, limit: int):
    """Normal clean traffic."""
    query = f"""
    SELECT {", ".join(LOG_COLUMNS)}
    FROM fortigate_traffic
    WHERE action IN ('accept', 'pass', 'close')
    LIMIT {limit}
    """
    return list(client.query(query).named_results())


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
    print(f"Source: {cfg['clickhouse']['host']}:{cfg['clickhouse']['port']}/{cfg['clickhouse']['database']}")

    all_records = []

    categories = [
        ("threat",     sample_threats,     sampling["splits"].get("threat", 50000)),
        ("blocked",    sample_blocked,     sampling["splits"].get("blocked", 50000)),
        ("denied",     sample_denied,      sampling["splits"].get("denied", 50000)),
        ("suspicious", sample_suspicious,  sampling["splits"].get("suspicious", 50000)),
        ("normal",     sample_normal,      sampling["splits"].get("normal", 100000)),
    ]

    for name, sampler_fn, count in categories:
        print(f"\nSampling {count} {name} rows...")
        try:
            rows = list(sampler_fn(client, count))
            print(f"  Got {len(rows)} rows")

            for row in rows:
                row_dict = {k: str(v) if v is not None else "" for k, v in dict(row).items()}
                log_text = format_log_text(row_dict)
                label = classify_label(row_dict)
                record = build_instruction(log_text, label)
                all_records.append(record)
        except Exception as e:
            print(f"  WARNING: Failed to sample {name}: {e}")

    print(f"\nTotal records: {len(all_records)}")

    if len(all_records) == 0:
        print("ERROR: No records collected. Check ClickHouse connection and credentials.")
        return

    # Shuffle
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
    label_counts = Counter(r["category"] for r in all_records)
    print(f"\n=== Exported ===")
    print(f"Train: {len(train)} -> {out_dir}/train.jsonl")
    print(f"Val:   {len(val)} -> {out_dir}/val.jsonl")
    print(f"Test:  {len(test)} -> {out_dir}/test.jsonl")
    print(f"\nLabel distribution:")
    for label, count in label_counts.most_common():
        print(f"  {label}: {count}")

    # Detailed label breakdown
    detail_counts = Counter(r["response"] for r in all_records)
    print(f"\nDetailed labels (top 20):")
    for label, count in detail_counts.most_common(20):
        print(f"  {label}: {count}")


if __name__ == "__main__":
    main()
