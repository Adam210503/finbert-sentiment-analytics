"""
training/tune.py
─────────────────
Optuna hyperparameter search for FinBERT fine-tuning, followed by
a final retraining on train+validation using the best found parameters.

Usage (from project root):
    python training/tune.py
"""

import os
import sys
import csv
import time
import warnings
import subprocess
from pathlib import Path

import numpy as np
import optuna
import torch
from datasets import DatasetDict, concatenate_datasets
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from utils.helpers import seed_everything

# ── Constants ─────────────────────────────────────────────────────────────────
SEED         = 42
N_TRIALS     = 30
MODEL_ID     = "ProsusAI/finbert"
DATASET_PATH = os.path.join("training", "processed_dataset")
TRIALS_DIR   = os.path.join("training", "tune_trials")
FINAL_MODEL  = os.path.join("training", "finetuned_finbert")
RESULTS_CSV  = os.path.join("training", "optuna_results.csv")
BASELINE_F1  = 0.902107   # fine-tuned row from training/evaluation_results.csv
MAX_LENGTH   = 128

warnings.filterwarnings("ignore", category=FutureWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ── Seed audit ────────────────────────────────────────────────────────────────

def _print_seed_audit() -> None:
    W   = 66
    SEP = "─" * W
    print(f"\n{SEP}")
    print("  SEED AUDIT")
    print(SEP)
    entries = [
        ("random.seed",                  "utils/helpers.py::seed_everything", True),
        ("os.environ['PYTHONHASHSEED']", "utils/helpers.py::seed_everything", True),
        ("np.random.seed",               "utils/helpers.py::seed_everything", True),
        ("torch.manual_seed",            "utils/helpers.py::seed_everything", True),
        ("torch.mps.manual_seed",        "utils/helpers.py::seed_everything", True),
        ("torch.cuda.manual_seed",       "utils/helpers.py::seed_everything", True),
        ("transformers.set_seed",        "utils/helpers.py::seed_everything", True),
        ("per-trial seed offset",        "training/tune.py::objective",       True),
    ]
    for name, location, present in entries:
        mark = "✓" if present else "✗ MISSING"
        print(f"  {mark}  {name:<38}  {location}")
    print(SEP + "\n")


# ── Tokenisation ──────────────────────────────────────────────────────────────

def _tokenize(dataset, tokenizer):
    def _fn(examples):
        return tokenizer(
            examples["sentence"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False,
        )
    return dataset.map(_fn, batched=True, remove_columns=["sentence"])


# ── Metrics ───────────────────────────────────────────────────────────────────

def _compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "f1":       f1_score(labels, preds, average="macro", zero_division=0),
        "accuracy": accuracy_score(labels, preds),
    }


# ── Pruning callback ──────────────────────────────────────────────────────────

class _TrialPruningCallback(TrainerCallback):
    """Reports intermediate val F1 to Optuna after each epoch and prunes if needed."""

    def __init__(self, trial: optuna.Trial) -> None:
        self.trial = trial

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: dict,
        **kwargs,
    ) -> None:
        epoch = int(state.epoch)
        f1    = metrics.get("eval_f1", 0.0)
        self.trial.report(f1, step=epoch)
        if self.trial.should_prune():
            raise optuna.TrialPruned()


# ── Objective ─────────────────────────────────────────────────────────────────

def objective(trial: optuna.Trial) -> float:
    seed_everything(SEED + trial.number)

    lr           = trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True)
    batch_size   = trial.suggest_categorical("batch_size", [16, 32])
    num_epochs   = trial.suggest_int("num_epochs", 2, 4)
    weight_decay = trial.suggest_float("weight_decay", 0.0, 0.1)
    warmup_ratio = trial.suggest_float("warmup_ratio", 0.0, 0.2)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    dataset   = DatasetDict.load_from_disk(DATASET_PATH)
    tok_train = _tokenize(dataset["train"],      tokenizer)
    tok_val   = _tokenize(dataset["validation"], tokenizer)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model  = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, num_labels=3)
    model.to(device)

    trial_dir = os.path.join(TRIALS_DIR, f"trial_{trial.number}")
    args = TrainingArguments(
        output_dir=trial_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=16,
        num_train_epochs=num_epochs,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        logging_dir=os.path.join(trial_dir, "logs"),
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to="none",
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tok_train,
        eval_dataset=tok_val,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=_compute_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=2),
            _TrialPruningCallback(trial),
        ],
    )

    trainer.train()
    val_metrics = trainer.evaluate()
    best_f1 = val_metrics.get("eval_f1", 0.0)

    print(
        f"  Trial {trial.number:>3}  "
        f"lr={lr:.2e}  bs={batch_size}  ep={num_epochs}  "
        f"wd={weight_decay:.4f}  warmup={warmup_ratio:.3f}  "
        f"→ val F1={best_f1:.4f}"
    )
    return best_f1


# ── Study ─────────────────────────────────────────────────────────────────────

def _run_study() -> optuna.Study:
    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )

    search_start = time.time()

    def _trial_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        if trial.number == 0 and trial.state == optuna.trial.TrialState.COMPLETE:
            elapsed   = time.time() - search_start
            remaining = (N_TRIALS - 1) * elapsed
            print(
                f"\n  [Timing] Trial 0 took {elapsed:.0f}s  —  "
                f"estimated remaining: {remaining / 60:.1f} min "
                f"({N_TRIALS - 1} trials × {elapsed:.0f}s)\n"
            )

    study.optimize(objective, n_trials=N_TRIALS, callbacks=[_trial_callback])
    return study


# ── Results reporting ─────────────────────────────────────────────────────────

def _print_results(study: optuna.Study) -> None:
    W   = 72
    SEP = "─" * W
    best = study.best_trial

    print(f"\n{'=' * W}")
    print("  OPTUNA SEARCH COMPLETE")
    print("=" * W)
    print(f"  Best trial        : #{best.number}")
    print(f"  Best val macro F1 : {best.value:.4f}")
    print(f"  Baseline F1       : {BASELINE_F1:.4f}  (current finetuned_finbert)")
    delta = best.value - BASELINE_F1
    sign  = "+" if delta >= 0 else ""
    print(f"  Delta vs baseline : {sign}{delta:.4f}")

    print(f"\n{SEP}")
    print("  BEST HYPERPARAMETERS")
    print(SEP)
    for k, v in best.params.items():
        print(f"  {k:<20} : {v}")

    completed = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    ]
    top5 = sorted(completed, key=lambda t: t.value, reverse=True)[:5]

    print(f"\n{SEP}")
    print("  TOP 5 TRIALS BY VAL MACRO F1")
    print(SEP)
    print(f"  {'#':>4}  {'F1':>8}  {'LR':>10}  {'BS':>4}  {'Ep':>4}  {'WD':>7}  {'Warmup':>8}")
    print(SEP)
    for t in top5:
        p = t.params
        print(
            f"  {t.number:>4}  {t.value:>8.4f}  "
            f"{p['learning_rate']:>10.2e}  {p['batch_size']:>4}  "
            f"{p['num_epochs']:>4}  {p['weight_decay']:>7.4f}  "
            f"{p['warmup_ratio']:>8.3f}"
        )
    print(SEP)


def _save_results_csv(study: optuna.Study) -> None:
    fieldnames = [
        "trial_number", "learning_rate", "batch_size", "num_epochs",
        "weight_decay", "warmup_ratio", "val_macro_f1", "status",
    ]
    rows = []
    for t in study.trials:
        p = t.params or {}
        rows.append({
            "trial_number":  t.number,
            "learning_rate": p.get("learning_rate", ""),
            "batch_size":    p.get("batch_size", ""),
            "num_epochs":    p.get("num_epochs", ""),
            "weight_decay":  p.get("weight_decay", ""),
            "warmup_ratio":  p.get("warmup_ratio", ""),
            "val_macro_f1":  f"{t.value:.6f}" if t.value is not None else "",
            "status":        t.state.name,
        })
    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n  Study results saved → {RESULTS_CSV}")


# ── Final retraining ──────────────────────────────────────────────────────────

def _retrain_final(best_params: dict) -> None:
    W = 62
    print(f"\n{'=' * W}")
    print("  [FINAL MODEL] Retraining on train + validation")
    print("=" * W)
    for k, v in best_params.items():
        print(f"  {k:<20} : {v}")
    print()

    seed_everything(SEED)

    tokenizer    = AutoTokenizer.from_pretrained(MODEL_ID)
    dataset      = DatasetDict.load_from_disk(DATASET_PATH)
    combined     = concatenate_datasets([dataset["train"], dataset["validation"]])
    tok_combined = _tokenize(combined, tokenizer)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model  = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, num_labels=3)
    model.to(device)

    final_tmp = os.path.join("training", "tune_final_tmp")
    args = TrainingArguments(
        output_dir=final_tmp,
        eval_strategy="no",
        save_strategy="no",
        learning_rate=best_params["learning_rate"],
        per_device_train_batch_size=best_params["batch_size"],
        per_device_eval_batch_size=16,
        num_train_epochs=best_params["num_epochs"],
        weight_decay=best_params["weight_decay"],
        warmup_ratio=best_params["warmup_ratio"],
        logging_dir=os.path.join(final_tmp, "logs"),
        logging_steps=100,
        load_best_model_at_end=False,
        report_to="none",
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tok_combined,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
    )

    trainer.train()
    trainer.save_model(FINAL_MODEL)
    tokenizer.save_pretrained(FINAL_MODEL)

    print(f"\n  Checkpoint saved → {FINAL_MODEL}/")
    print(f"  Hyperparameters  : {best_params}")
    print("=" * W)


# ── Re-run evaluate.py ────────────────────────────────────────────────────────

def _run_evaluate() -> None:
    print("\n  Running training/evaluate.py to update model card numbers...\n")
    result = subprocess.run(
        [sys.executable, os.path.join("training", "evaluate.py")],
        cwd=str(ROOT),
    )
    if result.returncode != 0:
        print("  WARNING: evaluate.py exited non-zero — check output above.")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 62)
    print("  FINBERT HYPERPARAMETER SEARCH  (Optuna / TPE)")
    print("=" * 62)
    print(f"  Trials    : {N_TRIALS}")
    print(f"  Base model: {MODEL_ID}")
    print(f"  Dataset   : {DATASET_PATH}")
    print(f"  Baseline  : {BASELINE_F1:.4f} macro F1")

    _print_seed_audit()

    study = _run_study()
    _print_results(study)
    _save_results_csv(study)
    _retrain_final(study.best_trial.params)
    _run_evaluate()


if __name__ == "__main__":
    main()
