import os
import sys
import requests
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.helpers import seed_everything
seed_everything(42)

def prepare_financial_dataset():
    print("==================================================")
    print("      PHASE 2: DATA PREPARATION PIPELINE         ")
    print("==================================================")

    # -------------------------------------------------------
    # 1. Load Financial PhraseBank via raw HTTP request
    #    Avoids pandas.read_csv pickle machinery entirely,
    #    which conflicts with certain pandas 2.x / Python 3.12
    #    builds on macOS ARM.
    # -------------------------------------------------------
    print("\n[1/4] Downloading Financial PhraseBank from stable source...")
    try:
        url = (
            "https://raw.githubusercontent.com/"
            "maxwellsarpong/NLP-financial-text-processing-dataset/"
            "master/Sentences_AllAgree.txt"
        )

        response = requests.get(url, timeout=30)
        response.raise_for_status()          # Surface HTTP errors immediately
        response.encoding = "ISO-8859-1"     # Canonical encoding for this file

        lines = response.text.strip().split("\n")

        rows = []
        skipped = 0
        for line in lines:
            line = line.strip()
            if not line or "@" not in line:
                skipped += 1
                continue
            # rsplit from right with maxsplit=1 — handles @ inside sentence body
            # e.g. "Email ceo@company.com for details@positive" parses correctly
            sentence, label_text = line.rsplit("@", 1)
            rows.append({
                "sentence":   sentence.strip(),
                "label_text": label_text.strip().lower(),
            })

        if skipped:
            print(f"  ├─ Skipped {skipped} malformed/empty line(s)")

        df = pd.DataFrame(rows)

        # Map to canonical FinBERT label indices
        # 0 → negative   1 → neutral   2 → positive
        label_mapping = {"negative": 0, "neutral": 1, "positive": 2}
        df["label"] = df["label_text"].map(label_mapping)

        # Drop rows where the label didn't match any known class
        before = len(df)
        df = df.dropna(subset=["label"]).drop(columns=["label_text"])
        df["label"] = df["label"].astype(int)
        after = len(df)
        if before != after:
            print(f"  ├─ Dropped {before - after} row(s) with unrecognised labels")

        raw_dataset = Dataset.from_pandas(df, preserve_index=False)
        print(f"  └─ Loaded {len(raw_dataset)} samples successfully")

    except requests.exceptions.RequestException as e:
        print(f"❌ Network error: {e}")
        print("💡 Tip: Check your internet connection and try again.")
        return
    except Exception as e:
        print(f"❌ Unexpected error during loading: {e}")
        raise

    # -------------------------------------------------------
    # 2. Audit class distribution — surface training skew
    # -------------------------------------------------------
    print("\n[2/4] Auditing class distributions for training skew...")
    labels       = raw_dataset["label"]
    total        = len(labels)
    label_names  = {0: "Negative", 1: "Neutral", 2: "Positive"}
    unique, counts = np.unique(labels, return_counts=True)

    for lbl, cnt in zip(unique, counts):
        pct = (cnt / total) * 100
        bar = "█" * int(pct / 2)   # simple ASCII bar for quick visual scan
        print(f"  ├─ {label_names[lbl]:>8} (label {lbl}): {cnt:>4} samples "
              f"({pct:5.1f}%)  {bar}")

    # -------------------------------------------------------
    # 3. Stratified split — 70 / 15 / 15
    #    Fixed seed (42) guarantees reproducibility across runs.
    #    Stratification forces class ratios to be preserved in
    #    every split, preventing a skewed validation or test set.
    # -------------------------------------------------------
    print("\n[3/4] Executing stratified splitting (70 / 15 / 15)...")
    df = raw_dataset.to_pandas()

    # Step A: carve out train (70%) and a temporary block (30%)
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=42,
        stratify=df["label"],
    )

    # Step B: split the 30% block equally into val (15%) and test (15%)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=42,
        stratify=temp_df["label"],
    )

    print(f"  ├─ Training set   : {len(train_df):>4} rows")
    print(f"  ├─ Validation set : {len(val_df):>4} rows")
    print(f"  └─ Test set       : {len(test_df):>4} rows")

    # Verify stratification held — class ratios should be ~identical
    print("\n  Stratification check (class % per split):")
    for split_name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        ratios = split_df["label"].value_counts(normalize=True).sort_index()
        ratio_str = "  ".join(
            f"{label_names[i]}: {ratios.get(i, 0)*100:.1f}%" for i in sorted(label_names)
        )
        print(f"  ├─ {split_name:<5}: {ratio_str}")

    # -------------------------------------------------------
    # 4. Serialise to Hugging Face DatasetDict on disk
    #    Saves re-downloading on every training run.
    # -------------------------------------------------------
    print("\n[4/4] Serializing splits to disk...")
    final_dataset = DatasetDict({
        "train":      Dataset.from_pandas(train_df.reset_index(drop=True)),
        "validation": Dataset.from_pandas(val_df.reset_index(drop=True)),
        "test":       Dataset.from_pandas(test_df.reset_index(drop=True)),
    })

    save_path = os.path.join("training", "processed_dataset")
    os.makedirs(save_path, exist_ok=True)
    final_dataset.save_to_disk(save_path)

    print(f"\n✅ Done — splits saved to: {save_path}/")
    print("   Load later with: DatasetDict.load_from_disk('training/processed_dataset')")
    print("==================================================")


if __name__ == "__main__":
    prepare_financial_dataset()
