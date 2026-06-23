import os
import sys
import numpy as np
import torch
from datasets import DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.helpers import seed_everything

seed_everything(42)

def main():
    print("==================================================")
    print("      PHASE 3 & 4: TOKENIZATION & TRAINING        ")
    print("==================================================")

    # Automatically route workloads to Apple Silicon GPU (MPS) if available
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"🚀 Execution Backend: {device.upper()}")

    # 1. Load Processed Splits from Disk
    print("\n[1/5] Hydrating dataset partitions from Phase 2...")
    dataset_path = os.path.join("training", "processed_dataset")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"❌ Missing processed dataset at '{dataset_path}'. Execute prepare_data.py first.")
    
    dataset = DatasetDict.load_from_disk(dataset_path)

    # 2. Tokenization Pipeline
    # Using base uncased BERT as the raw architecture foundational weights
    model_checkpoint = "bert-base-uncased" 
    print(f"\n[2/5] Initializing tokenizer ({model_checkpoint})...")
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    def tokenize_function(examples):
        # Truncate long sequences; padding is handled dynamically per batch later
        return tokenizer(examples["sentence"], truncation=True, padding=False)

    print("  ├─ Mapping tokens across Train, Validation, and Test splits...")
    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["sentence"])

    # 3. Custom Evaluation Metrics for Skewed Distributions
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        
        # 'macro' average forces equal weighting across minority classes (Neg/Pos)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average="macro", zero_division=0
        )
        acc = accuracy_score(labels, predictions)
        
        return {
            "accuracy": acc,
            "f1": f1,
            "precision": precision,
            "recall": recall
        }

    # 4. Initialize Sequence Classification Architecture
    print(f"\n[3/5] Instantiating sequence classifier (3 targets)...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_checkpoint, 
        num_labels=3  # 0: Negative, 1: Neutral, 2: Positive
    )
    model.to(device)

    # 5. Configure Training Infrastructure
    print("\n[4/5] Staging Hyperparameters and TrainingArguments...")
    training_args = TrainingArguments(
        output_dir=os.path.join("training", "results"),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,               # Classic learning rate for BERT fine-tuning
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,               # 3 iterations over entire dataset
        weight_decay=0.01,
        logging_dir=os.path.join("training", "logs"),
        logging_steps=10,
        load_best_model_at_end=True,      # Rewind to the optimal epoch iteration at the end
        metric_for_best_model="f1",       # Optimize strictly against Macro-F1 performance
        greater_is_better=True,
        report_to="none",                 # Suppresses third-party API logging prompts
        dataloader_num_workers=0          # Mandated 0 to prevent MPS memory cross-talk spikes
    )

    # Data collator acts as a smart padder, dynamically resizing batches on the fly
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        processing_class=tokenizer,        # 💡 CHANGED: renamed from tokenizer=tokenizer
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # 6. Execute Fine-Tuning
    print("\n[5/5] Spinning up Transformer core training loops...")
    trainer.train()

    # 7. Unbiased Evaluation on Gold-Standard Test Set
    print("\n==================================================")
    print("          EVALUATING ON UNSEEN TEST SPLIT         ")
    print("==================================================")
    test_results = trainer.evaluate(tokenized_datasets["test"])
    print(f"  ├─ Gold Test Accuracy : {test_results['eval_accuracy']:.4f}")
    print(f"  ├─ Gold Test F1-Score : {test_results['eval_f1']:.4f}")
    print(f"  └─ Gold Test Precision: {test_results['eval_precision']:.4f}")

    # Save the polished model directory
    final_model_path = os.path.join("training", "finetuned_finbert")
    trainer.save_model(final_model_path)
    print(f"\n🎉 Model checkpoint saved successfully to: {final_model_path}/")
    print("==================================================")

if __name__ == "__main__":
    main()