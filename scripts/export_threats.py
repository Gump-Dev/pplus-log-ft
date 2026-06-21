"""Export threat samples from ClickHouse, querying each subtype separately."""
import json
import clickhouse_connect

COLS = [
    'timestamp', 'tenant_name', 'device', 'subtype', 'action',
    'src_ip', 'dst_ip', 'src_port', 'dst_port',
    'protocol_name', 'service', 'policy_name',
    'bytes_sent', 'bytes_recv', 'duration',
    'app_name', 'app_cat', 'app_risk',
    'src_country', 'dst_country',
    'threat_type', 'threat_name', 'threat_level', 'threat_feed',
    'url_category', 'url', 'level', 'message',
]

client = clickhouse_connect.get_client(
    host='localhost', port=18123,
    username='pplus', password='pplus2026',
    database='logs',
    connect_timeout=30, send_receive_timeout=300,
)

col_list = ', '.join(COLS)
threats = []

for subtype in ['virus', 'ips', 'waf', 'anomaly', 'file-filter']:
    print(f'Sampling {subtype}...', flush=True)
    try:
        q = f"SELECT {col_list} FROM fortigate_traffic WHERE subtype = '{subtype}' LIMIT 10000"
        rows = list(client.query(q).named_results())
        print(f'  Got {len(rows)}', flush=True)
        for row in rows:
            d = {k: str(v) if v is not None else '' for k, v in dict(row).items()}
            threats.append(d)
    except Exception as e:
        print(f'  ERROR: {e}', flush=True)

print(f'Total threat records: {len(threats)}', flush=True)

with open('datasets/threat_samples.jsonl', 'w') as f:
    for t in threats:
        parts = [f'{c}={t.get(c, "")}' for c in COLS if t.get(c, '')]
        log_text = ' '.join(parts)
        label = f"THREAT:{t.get('subtype', '')}:{t.get('threat_type', '')}"
        rec = {
            'instruction': (
                'You are a FortiGate firewall log analyst. '
                'Classify the following log entry into one of these categories: '
                'THREAT, BLOCKED, DENIED, SUSPICIOUS, NORMAL, OTHER. '
                'If it is a THREAT, specify the subtype.\n\n'
                f'Log: {log_text}'
            ),
            'response': label,
            'category': 'THREAT',
        }
        f.write(json.dumps(rec, ensure_ascii=False) + '\n')

print('Written to datasets/threat_samples.jsonl', flush=True)
