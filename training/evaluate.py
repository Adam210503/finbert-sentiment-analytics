import os
import sys
import csv
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # headless — no display needed, safe on MPS
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from datasets import DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.helpers import seed_everything

seed_everything(42)

# ── Constants ──────────────────────────────────────────────────────────────────
TOKENIZER_NAME  = "ProsusAI/finbert"
BASE_MODEL_NAME = "ProsusAI/finbert"
FT_MODEL_PATH   = os.path.join("training", "finetuned_finbert")
DATASET_PATH    = os.path.join("training", "processed_dataset")
CSV_OUT         = os.path.join("training", "evaluation_results.csv")
CM_BASE_OUT     = os.path.join("training", "confusion_base.png")
CM_FT_OUT       = os.path.join("training", "confusion_finetuned.png")

BATCH_SIZE  = 16
MAX_LENGTH  = 128
LABEL_NAMES = ["Negative", "Neutral", "Positive"]

# 0=negative  1=neutral  2=positive — matches prepare_data.py and train.py
CANONICAL = {"negative": 0, "neutral": 1, "positive": 2}


# ── Core helpers ───────────────────────────────────────────────────────────────

def build_dataloader(sentences, labels, tokenizer):
    enc = tokenizer(
        list(sentences),
        max_length=MAX_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    token_type_ids = enc.get("token_type_ids")
    if token_type_ids is None:
        token_type_ids = torch.zeros_like(enc["input_ids"])
    ds = TensorDataset(
        enc["input_ids"],
        enc["attention_mask"],
        token_type_ids,
        torch.tensor(list(labels), dtype=torch.long),
    )
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)


def build_label_remap(model):
    """Return a dict remapping the model's label indices to our canonical scheme.
    Returns None when the model already uses the canonical ordering."""
    id2label = {int(k): v.lower() for k, v in model.config.id2label.items()}
    remap = {
        model_idx: CANONICAL[lbl]
        for model_idx, lbl in id2label.items()
        if lbl in CANONICAL
    }
    if all(remap.get(i, i) == i for i in range(len(LABEL_NAMES))):
        return None
    return remap


def run_inference(model, dataloader, device, label_remap=None):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for input_ids, attention_mask, token_type_ids, batch_labels in dataloader:
            outputs = model(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
                token_type_ids=token_type_ids.to(device),
            )
            preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
            if label_remap is not None:
                preds = np.array([label_remap[int(p)] for p in preds])
            all_preds.extend(preds)
            all_labels.extend(batch_labels.numpy())
    return np.array(all_preds), np.array(all_labels)


def compute_metrics(preds, labels):
    acc      = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    p, r, f, _ = precision_recall_fscore_support(
        labels, preds, labels=[0, 1, 2], zero_division=0
    )
    cm = confusion_matrix(labels, preds, labels=[0, 1, 2])
    return {
        "accuracy":  acc,
        "macro_f1":  macro_f1,
        "per_class": {
            name: {
                "precision": float(p[i]),
                "recall":    float(r[i]),
                "f1":        float(f[i]),
            }
            for i, name in enumerate(LABEL_NAMES)
        },
        "confusion_matrix": cm,
    }


# ── Output helpers ─────────────────────────────────────────────────────────────

def plot_confusion_matrix(cm, title, save_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set(
        xticks=range(len(LABEL_NAMES)),
        yticks=range(len(LABEL_NAMES)),
        xticklabels=LABEL_NAMES,
        yticklabels=LABEL_NAMES,
        xlabel="Predicted",
        ylabel="True",
        title=title,
    )
    thresh = cm.max() / 2.0
    for i in range(len(LABEL_NAMES)):
        for j in range(len(LABEL_NAMES)):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  └─ Saved → {save_path}")


def print_comparison_table(base, ft):
    W   = 62
    SEP = "─" * W
    print(f"\n{SEP}")
    print(f"  {'METRIC':<30}  {'BASE':>12}  {'FINE-TUNED':>12}")
    print(f"  {'':30}  {'ProsusAI/finbert':>12}  {'finetuned_finbert':>12}")
    print(f"  {'':30}  {'(pretrained on':>12}  {'(fine-tuned on':>12}")
    print(f"  {'':30}  {'financial text)':>12}  {'Fin.PhraseBank)':>12}")
    print(SEP)

    def row(label, b_val, f_val):
        marker = " ◄" if f_val > b_val else ("   " if abs(f_val - b_val) < 1e-8 else " ▼")
        print(f"  {label:<30}  {b_val:>12.4f}  {f_val:>12.4f}{marker}")

    row("Accuracy",  base["accuracy"],  ft["accuracy"])
    row("Macro F1",  base["macro_f1"],  ft["macro_f1"])
    print(f"  {SEP}")
    for cls in LABEL_NAMES:
        row(f"{cls} — Precision", base["per_class"][cls]["precision"],
                                  ft["per_class"][cls]["precision"])
        row(f"{cls} — Recall",    base["per_class"][cls]["recall"],
                                  ft["per_class"][cls]["recall"])
        row(f"{cls} — F1",        base["per_class"][cls]["f1"],
                                  ft["per_class"][cls]["f1"])
    print(SEP)
    print("  ◄ = fine-tuned leads   ▼ = base leads")
    print(SEP)


def export_csv(base, ft, path):
    fieldnames = ["model", "accuracy", "macro_f1", "neg_f1", "neu_f1", "pos_f1"]
    rows = [
        {
            "model":    "ProsusAI/finbert (base)",
            "accuracy": f"{base['accuracy']:.6f}",
            "macro_f1": f"{base['macro_f1']:.6f}",
            "neg_f1":   f"{base['per_class']['Negative']['f1']:.6f}",
            "neu_f1":   f"{base['per_class']['Neutral']['f1']:.6f}",
            "pos_f1":   f"{base['per_class']['Positive']['f1']:.6f}",
        },
        {
            "model":    "fine-tuned (training/finetuned_finbert)",
            "accuracy": f"{ft['accuracy']:.6f}",
            "macro_f1": f"{ft['macro_f1']:.6f}",
            "neg_f1":   f"{ft['per_class']['Negative']['f1']:.6f}",
            "neu_f1":   f"{ft['per_class']['Neutral']['f1']:.6f}",
            "pos_f1":   f"{ft['per_class']['Positive']['f1']:.6f}",
        },
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  └─ CSV exported → {path}")


def print_result_block(base, ft):
    b_f1  = base["macro_f1"]
    f_f1  = ft["macro_f1"]
    delta = f_f1 - b_f1
    print("\n" + "=" * 62)
    print("  [RESULT]")
    print("=" * 62)
    if abs(delta) < 1e-6:
        print("  Both models are tied on Macro F1.")
        print(f"  Macro F1 : {b_f1:.4f}")
    elif delta > 0:
        print(f"  Winner     : Fine-tuned checkpoint")
        print(f"  Margin     : +{delta:.4f} Macro F1 over base")
        print(f"  Base F1    : {b_f1:.4f}")
        print(f"  Fine-tuned : {f_f1:.4f}")
    else:
        print(f"  Winner     : Base model (ProsusAI/finbert)")
        print(f"  Margin     : +{-delta:.4f} Macro F1 over fine-tuned")
        print(f"  Base F1    : {b_f1:.4f}")
        print(f"  Fine-tuned : {f_f1:.4f}")
    print("=" * 62 + "\n")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    print("=" * 62)
    print("      FINBERT EVALUATION — BASE vs. FINE-TUNED")
    print("=" * 62)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"\nExecution Backend : {device.upper()}")

    # 1. Load test split
    print(f"\n[1/5] Loading test split from {DATASET_PATH}...")
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"Missing dataset at '{DATASET_PATH}'. Run prepare_data.py first."
        )
    dataset    = DatasetDict.load_from_disk(DATASET_PATH)
    test_split = dataset["test"]
    sentences  = test_split["sentence"]
    labels     = test_split["label"]
    print(f"  └─ {len(sentences)} test examples loaded")

    # 2. Tokenize once — same tokenizer, same batches for both models
    print(f"\n[2/5] Tokenizing with {TOKENIZER_NAME} (max_length={MAX_LENGTH})...")
    tokenizer  = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    dataloader = build_dataloader(sentences, labels, tokenizer)
    print(f"  └─ {len(dataloader)} batches  |  batch_size={BATCH_SIZE}")

    # 3. Base model
    print(f"\n[3/5] Inference — Base: {BASE_MODEL_NAME}...")
    base_model  = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL_NAME)
    base_model  = base_model.to(device)
    label_remap = build_label_remap(base_model)
    if label_remap:
        readable = {LABEL_NAMES[v]: f"model[{k}]" for k, v in label_remap.items()}
        print(f"  ├─ Label remap applied (base uses different ordering): {label_remap}")
    base_preds, base_labels = run_inference(base_model, dataloader, device, label_remap)
    base_metrics = compute_metrics(base_preds, base_labels)
    print(f"  └─ Accuracy: {base_metrics['accuracy']:.4f}  |  Macro F1: {base_metrics['macro_f1']:.4f}")
    del base_model
    if device == "mps":
        torch.mps.empty_cache()

    # 4. Fine-tuned model
    print(f"\n[4/5] Inference — Fine-tuned: {FT_MODEL_PATH}/...")
    if not os.path.exists(FT_MODEL_PATH):
        raise FileNotFoundError(
            f"Missing fine-tuned checkpoint at '{FT_MODEL_PATH}'. Run train.py first."
        )
    ft_model = AutoModelForSequenceClassification.from_pretrained(FT_MODEL_PATH)
    ft_model = ft_model.to(device)
    ft_preds, ft_labels = run_inference(ft_model, dataloader, device, label_remap=None)
    ft_metrics = compute_metrics(ft_preds, ft_labels)
    print(f"  └─ Accuracy: {ft_metrics['accuracy']:.4f}  |  Macro F1: {ft_metrics['macro_f1']:.4f}")
    del ft_model
    if device == "mps":
        torch.mps.empty_cache()

    # 5. Outputs
    print("\n[5/5] Generating outputs...")
    print_comparison_table(base_metrics, ft_metrics)
    export_csv(base_metrics, ft_metrics, CSV_OUT)
    plot_confusion_matrix(
        base_metrics["confusion_matrix"],
        "Base: ProsusAI/finbert",
        CM_BASE_OUT,
    )
    plot_confusion_matrix(
        ft_metrics["confusion_matrix"],
        "Fine-tuned: finetuned_finbert",
        CM_FT_OUT,
    )

    print_result_block(base_metrics, ft_metrics)


if __name__ == "__main__":
    main()
