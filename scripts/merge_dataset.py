"""Merge threat samples with existing train/val/test and re-split."""
import json
import random
from collections import Counter

# Load existing
existing = []
for split in ['train', 'val', 'test']:
    with open(f'datasets/{split}.jsonl') as f:
        for line in f:
            existing.append(json.loads(line))
print(f'Existing records: {len(existing)}')

# Load threats
threats = []
with open('datasets/threat_samples.jsonl') as f:
    for line in f:
        threats.append(json.loads(line))
print(f'Threat records: {len(threats)}')

# Merge
all_records = existing + threats
print(f'Total: {len(all_records)}')

# Shuffle
random.seed(42)
random.shuffle(all_records)

# Split 85/10/5
n = len(all_records)
n_train = int(n * 0.85)
n_val = int(n * 0.10)

train = all_records[:n_train]
val = all_records[n_train:n_train + n_val]
test = all_records[n_train + n_val:]

for name, data in [('train', train), ('val', val), ('test', test)]:
    with open(f'datasets/{name}.jsonl', 'w') as f:
        for rec in data:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')

print(f'\nFinal splits:')
print(f'  Train: {len(train)}')
print(f'  Val:   {len(val)}')
print(f'  Test:  {len(test)}')

# Label distribution
cats = Counter(r['category'] for r in all_records)
print(f'\nLabel distribution:')
for label, count in cats.most_common():
    pct = count / len(all_records) * 100
    print(f'  {label}: {count} ({pct:.1f}%)')

# Detailed threats
threat_details = Counter(r['response'] for r in all_records if r['category'] == 'THREAT')
print(f'\nThreat subtypes:')
for label, count in threat_details.most_common():
    print(f'  {label}: {count}')

# File sizes
import os
for name in ['train', 'val', 'test']:
    size_mb = os.path.getsize(f'datasets/{name}.jsonl') / 1e6
    print(f'  {name}.jsonl: {size_mb:.1f} MB')
