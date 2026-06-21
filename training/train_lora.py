"""
LoRA fine-tune script for Qwen3-8B on FortiGate log classification data.
Designed to run on DGX Spark (NVIDIA GB10, aarch64).

Usage:
  /home/travix3/vllm-install/.vllm/bin/python3 training/train_lora.py \
    --config configs/training.yaml \
    --data-dir ./datasets \
    --output-dir ./outputs/qwen3-8b-ft-v1
"""
import json
import argparse

import yaml
import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_jsonl(path: str) -> list:
    with open(path) as f:
        return [json.loads(line) for line in f]


def as_float(value) -> float:
    return float(value)


def as_int(value) -> int:
    return int(value)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen3-8B with LoRA")
    parser.add_argument("--config", default="configs/training.yaml")
    parser.add_argument("--data-dir", default="./datasets")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-train", type=int, default=None, help="Limit train samples (for testing)")
    parser.add_argument("--max-val", type=int, default=None, help="Limit val samples")
    parser.add_argument("--max-steps", type=int, default=-1, help="Override max training steps")
    parser.add_argument("--batch-size", type=int, default=None, help="Override per-device train/eval batch size")
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=None,
        help="Override gradient accumulation steps",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        choices=("true", "false"),
        default=None,
        help="Override gradient checkpointing",
    )
    parser.add_argument("--train-offset", type=int, default=0, help="Start index after optional shuffle")
    parser.add_argument("--shuffle-seed", type=int, default=None, help="Deterministically shuffle before slicing")
    parser.add_argument("--resume-adapter", default=None, help="LoRA adapter directory to continue training from")
    parser.add_argument("--resume-from-checkpoint", default=None, help="Trainer checkpoint for interrupted same-lot resume")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_cfg = cfg["model"]
    lora_cfg = cfg["lora"]
    train_cfg = cfg["training"]
    hw_cfg = cfg["hardware"]

    output_dir = args.output_dir or model_cfg["output_dir"]

    print("=== PPLUS Log Fine-Tune ===")
    print(f"Base model: {model_cfg['base_model']}")
    print(f"Output:     {output_dir}")
    print(f"Device:     {hw_cfg['device']}")
    print(f"VRAM free:  {torch.cuda.mem_get_info()[0] / 1e9:.1f} GB")

    # --- Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["base_model"],
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Format data ---
    def format_chat(example):
        """Format as Qwen chat template."""
        # Extract just the log portion from instruction
        instruction = example["instruction"]
        if "\n\nLog: " in instruction:
            log_part = instruction.split("\n\nLog: ", 1)[1]
        else:
            log_part = instruction

        messages = [
            {"role": "system", "content": "You are a FortiGate firewall log analyst. Classify log entries accurately."},
            {"role": "user", "content": f"Classify this FortiGate log entry:\n{log_part}"},
            {"role": "assistant", "content": example["response"]},
        ]
        return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}

    # --- Load datasets ---
    print("\nLoading datasets...")
    train_data = load_jsonl(f"{args.data_dir}/train.jsonl")
    val_data = load_jsonl(f"{args.data_dir}/val.jsonl")

    if args.shuffle_seed is not None:
        import random

        rng = random.Random(args.shuffle_seed)
        rng.shuffle(train_data)

    if args.train_offset:
        train_data = train_data[args.train_offset:]
    if args.max_train:
        train_data = train_data[:args.max_train]
    if args.max_val:
        val_data = val_data[:args.max_val]

    train_ds = Dataset.from_list([format_chat(ex) for ex in train_data])
    val_ds = Dataset.from_list([format_chat(ex) for ex in val_data])

    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    # Sample
    print(f"\nSample train entry:\n{train_ds[0]['text'][:500]}...")

    # --- Load model ---
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["base_model"],
        torch_dtype=torch.bfloat16,
        device_map=hw_cfg["device"],
        trust_remote_code=True,
    )
    model.config.use_cache = False

    # --- LoRA ---
    if args.resume_adapter:
        print(f"\nLoading trainable LoRA adapter from {args.resume_adapter}")
        model = PeftModel.from_pretrained(model, args.resume_adapter, is_trainable=True)
    else:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_cfg["r"],
            lora_alpha=lora_cfg["alpha"],
            lora_dropout=lora_cfg["dropout"],
            target_modules=lora_cfg["target_modules"],
            bias="none",
        )
        model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    save_steps = as_int(train_cfg["save_steps"])
    eval_steps = as_int(train_cfg["eval_steps"])
    batch_size = args.batch_size or as_int(train_cfg["batch_size"])
    gradient_accumulation_steps = (
        args.gradient_accumulation_steps or as_int(train_cfg["gradient_accumulation_steps"])
    )
    gradient_checkpointing = train_cfg["gradient_checkpointing"]
    if args.gradient_checkpointing is not None:
        gradient_checkpointing = args.gradient_checkpointing == "true"
    load_best_model = True
    if args.max_steps and 0 < args.max_steps < min(save_steps, eval_steps):
        save_steps = args.max_steps
        eval_steps = args.max_steps
        load_best_model = False

    # --- Training config ---
    sft_config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=as_float(train_cfg["epochs"]),
        max_steps=args.max_steps,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=as_float(train_cfg["learning_rate"]),
        warmup_ratio=as_float(train_cfg["warmup_ratio"]),
        lr_scheduler_type=train_cfg["lr_scheduler"],
        max_length=as_int(train_cfg["max_seq_length"]),
        save_steps=save_steps,
        eval_steps=eval_steps,
        eval_strategy="steps",
        save_total_limit=3,
        bf16=train_cfg["bf16"],
        gradient_checkpointing=gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if gradient_checkpointing else None,
        logging_steps=10,
        report_to="none",
        load_best_model_at_end=load_best_model,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataset_text_field="text",
    )

    # --- Trainer ---
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
    )

    # --- Train ---
    print("\n=== Starting training ===")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # --- Save ---
    print(f"\n=== Saving to {output_dir} ===")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("\n=== Done! ===")
    print(f"LoRA adapter: {output_dir}")
    print(f"Merge with base model to deploy via vLLM")


if __name__ == "__main__":
    main()
