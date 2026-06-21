"""
LoRA fine-tune script for Qwen3-8B on FortiGate log classification data.
Designed to run on DGX Spark (NVIDIA GB10, aarch64).
"""
import os
import json
import argparse
from pathlib import Path

import yaml
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from trl import SFTTrainer, SFTConfig


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_jsonl(path: str) -> list:
    with open(path) as f:
        return [json.loads(line) for line in f]


def format_chat example):
    """Format as Qwen chat template."""
    messages = [
        {"role": "system", "content": "You are a FortiGate firewall log analyst. Classify log entries accurately."},
        {"role": "user", "content": example["instruction"].split("\n\nLog: ")[-1]},
        {"role": "assistant", "content": example["response"]},
    ]
    return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen3-8B with LoRA")
    parser.add_argument("--config", default="configs/training.yaml")
    parser.add_argument("--data-dir", default="./datasets")
    parser.add_argument("--output-dir", default=None)
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

    # Load tokenizer
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["base_model"],
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["base_model"],
        torch_dtype=torch.bfloat16,
        device_map=hw_cfg["device"],
        trust_remote_code=True,
    )
    model.config.use_cache = False

    # LoRA
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

    # Load datasets
    print("\nLoading datasets...")
    train_data = load_jsonl(f"{args.data_dir}/train.jsonl")
    val_data = load_jsonl(f"{args.data_dir}/val.jsonl")

    train_ds = Dataset.from_list([format_chat(ex) for ex in train_data])
    val_ds = Dataset.from_list([format_chat(ex) for ex in val_data])

    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    # Training config
    sft_config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=train_cfg["epochs"],
        per_device_train_batch_size=train_cfg["batch_size"],
        per_device_eval_batch_size=train_cfg["batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        warmup_ratio=train_cfg["warmup_ratio"],
        lr_scheduler_type=train_cfg["lr_scheduler"],
        max_seq_length=train_cfg["max_seq_length"],
        save_steps=train_cfg["save_steps"],
        eval_steps=train_cfg["eval_steps"],
        eval_strategy="steps",
        save_total_limit=3,
        bf16=train_cfg["bf16"],
        gradient_checkpointing=train_cfg["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=10,
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
    )

    # Train
    print("\n=== Starting training ===")
    trainer.train()

    # Save
    print(f"\n=== Saving to {output_dir} ===")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("\n=== Done! ===")
    print(f"LoRA adapter: {output_dir}")
    print(f"Merge with base model to deploy via vLLM")


if __name__ == "__main__":
    main()
