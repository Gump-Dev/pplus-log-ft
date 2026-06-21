# PPLUS Log Fine-Tune

Fine-tune local LLM (Qwen3-8B) on FortiGate log data from PPLUS Log Portal v3x for threat classification and log analysis.

## Objective

Use the 1.77B row FortiGate traffic dataset from v3x ClickHouse to fine-tune a model that can:
1. **Threat Classification** — Classify log lines as threat/suspicious/normal
2. **NL → Analysis** — Answer natural language questions about log data
3. **Auto Summarization** — Summarize daily security events

## Hardware

- **DGX Spark** `10.1.10.18` (spark-7903)
  - NVIDIA GB10 (Blackwell, 130.7 GB VRAM)
  - 20-core aarch64, 121 GB RAM
  - PyTorch 2.11.0+cu130, CUDA 13.0, vLLM 0.21.0
  - Models: Qwen3.6-35B-A3B-FP8, Qwen3-8B-FP8

## Dataset

- Source: `fortigate_traffic` table in v3x ClickHouse (`10.1.10.13:18123`)
- ~1.77B rows, 50 columns, 29 tenants
- Date range: 2026-06-14 to 2026-06-21 (7 days)
- Action distribution: accept 685M, close 326M, pass 225M, deny 104M, timeout 82M

## Project Structure

```
pplus-log-ft/
├── README.md
├── configs/          # Training configs, hyperparameters
├── data_prep/        # Dataset extraction, sampling, formatting
├── training/         # Fine-tune scripts (LoRA/QLoRA)
├── eval/             # Evaluation scripts
├── scripts/          # Utility scripts
└── requirements.txt
```

## Phases

### Phase 1: Data Prep
- Sample balanced dataset from ClickHouse (~100K-500K rows)
- Format as instruction-response pairs
- Split train/val/test

### Phase 2: Setup
- Install training deps on DGX (peft, trl, accelerate, datasets)
- Prepare Qwen3-8B-FP8 for LoRA fine-tune

### Phase 3: Fine-tune
- LoRA fine-tune on Qwen3-8B
- ~2-4h/epoch on GB10
- Full train can be run as staged lots:
  - `scripts/run_lot_training.sh` splits `datasets/train.jsonl` into 20K-row lots by default.
  - Each lot saves its own adapter under `outputs/qwen3-8b-ft-v1-lots/lot-XXX`.
  - Later lots continue from the previous lot with `--resume-adapter`.
  - `scripts/wait_then_run_lots.sh` can wait for a running pilot job, then continue remaining lots.

### Phase 4: Eval + Deploy
- Compare vs base model
- Serve via vLLM
