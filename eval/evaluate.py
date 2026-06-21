"""
Evaluate fine-tuned model vs base model on test set.
"""
import json
import argparse
from pathlib import Path
from collections import Counter, defaultdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def run_inference(model, tokenizer, prompt, device="cuda:0"):
    messages = [
        {"role": "system", "content": "You are a FortiGate firewall log analyst. Classify log entries accurately."},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=64,
            temperature=0.0,
            do_sample=False,
        )

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response.strip()


def parse_prediction(text):
    """Extract classification from model response."""
    text = text.upper().strip()
    for prefix in ["THREAT:", "SUSPICIOUS:", "NORMAL"]:
        if text.startswith(prefix) or prefix in text:
            return prefix.rstrip(":")
    # Fuzzy match
    if "THREAT" in text:
        return "THREAT"
    if "SUSPICIOUS" in text or "DENY" in text or "DENIED" in text:
        return "SUSPICIOUS"
    if "NORMAL" in text or "ACCEPT" in text:
        return "NORMAL"
    return "UNKNOWN"


def compute_metrics(predictions, labels):
    """Compute accuracy, precision, recall, F1 per class."""
    classes = sorted(set(labels) | set(predictions))
    results = {}

    total = len(labels)
    correct = sum(1 for p, l in zip(predictions, labels) if p == l)
    results["accuracy"] = correct / total if total > 0 else 0

    for cls in classes:
        tp = sum(1 for p, l in zip(predictions, labels) if p == cls and l == cls)
        fp = sum(1 for p, l in zip(predictions, labels) if p == cls and l != cls)
        fn = sum(1 for p, l in zip(predictions, labels) if p != cls and l == cls)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        results[cls] = {"precision": precision, "recall": recall, "f1": f1, "support": fn + tp}

    return results


def build_confusion_matrix(predictions, labels):
    classes = sorted(set(labels) | set(predictions))
    matrix = {
        true_cls: {
            pred_cls: sum(1 for p, l in zip(predictions, labels) if l == true_cls and p == pred_cls)
            for pred_cls in classes
        }
        for true_cls in classes
    }
    return classes, matrix


def write_jsonl(records, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned model")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter-path", default=None, help="LoRA adapter path (None = base only)")
    parser.add_argument("--test-data", default="./datasets/test.jsonl")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output-json", default=None, help="Write machine-readable metrics JSON")
    parser.add_argument("--predictions-jsonl", default=None, help="Write per-sample predictions JSONL")
    args = parser.parse_args()

    print(f"=== Evaluation ===")
    print(f"Base: {args.base_model}")
    print(f"Adapter: {args.adapter_path or 'None (base only)'}")

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        trust_remote_code=True,
    )

    if args.adapter_path:
        model = PeftModel.from_pretrained(model, args.adapter_path)
        model = model.merge_and_unload()

    model.eval()

    # Load test data
    test_data = load_jsonl(args.test_data)
    if args.max_samples:
        test_data = test_data[:args.max_samples]

    print(f"Test samples: {len(test_data)}")

    predictions = []
    labels = []
    prediction_records = []

    for i, ex in enumerate(test_data):
        prompt = ex["instruction"].split("\n\nLog: ")[-1]
        true_label = ex["category"]

        response = run_inference(model, tokenizer, prompt, args.device)
        pred_label = parse_prediction(response)

        predictions.append(pred_label)
        labels.append(true_label)
        prediction_records.append(
            {
                "index": i,
                "label": true_label,
                "prediction": pred_label,
                "response": response,
                "expected_response": ex.get("response"),
                "correct": pred_label == true_label,
            }
        )

        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(test_data)}...")

    # Metrics
    metrics = compute_metrics(predictions, labels)
    classes, confusion_matrix = build_confusion_matrix(predictions, labels)

    print(f"\n=== Results ===")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    for cls in sorted(k for k in metrics if k != "accuracy"):
        m = metrics[cls]
        print(f"  {cls}: P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} (n={m['support']})")

    # Confusion matrix
    print(f"\n=== Confusion Matrix ===")
    header = "True\\Pred".ljust(15) + "".join(c[:12].ljust(13) for c in classes)
    print(header)
    for true_cls in classes:
        row = true_cls[:14].ljust(15)
        for pred_cls in classes:
            count = confusion_matrix[true_cls][pred_cls]
            row += str(count).ljust(13)
        print(row)

    if args.output_json:
        output = {
            "base_model": args.base_model,
            "adapter_path": args.adapter_path,
            "test_data": args.test_data,
            "sample_count": len(test_data),
            "metrics": metrics,
            "classes": classes,
            "confusion_matrix": confusion_matrix,
        }
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
        print(f"\nMetrics JSON: {args.output_json}")

    if args.predictions_jsonl:
        write_jsonl(prediction_records, args.predictions_jsonl)
        print(f"Predictions JSONL: {args.predictions_jsonl}")


if __name__ == "__main__":
    main()
