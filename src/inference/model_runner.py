"""
src/inference/model_runner.py
──────────────────────────────
Scores pending headlines with the fine-tuned FinBERT checkpoint.

Reads headlines where sentiment_label IS NULL via
DatabaseManager.get_unscored_headlines(), runs them through the model
saved at training/finetuned_finbert/, and writes the predicted label,
confidence, and an attention-derived keyword back via
DatabaseManager.update_sentiment().

Usage:
    python src/inference/model_runner.py
"""

import logging
import sys
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ── Path setup: allow imports from project root ──────────────────
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from config.logging_config import setup_logging
from config.settings import DB_PATH, LOG_FILE, MODEL_PATH, SCORING_BATCH_SIZE
from src.storage.db_manager import DatabaseManager

logger = logging.getLogger(__name__)

# Matches the label_mapping used in training/prepare_data.py
ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}

# Forward-pass batch size (kept separate from the larger DB fetch size
# so a single inference batch fits comfortably in memory).
_INFERENCE_BATCH_SIZE = 16

_device = "mps" if torch.backends.mps.is_available() else "cpu"
_model = None
_tokenizer = None


# ─────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────

def run_scoring_job(db: DatabaseManager) -> int:
    """
    Score every currently pending headline, in batches, until none remain.

    Returns
    -------
    Total number of headlines scored in this run.
    """
    if not MODEL_PATH.exists():
        logger.error(
            "No model found at '%s'. Run training/train.py first.", MODEL_PATH
        )
        return 0

    _load_model()
    total_scored = 0

    while True:
        headlines = db.get_unscored_headlines(limit=SCORING_BATCH_SIZE)
        if not headlines:
            break

        for start in range(0, len(headlines), _INFERENCE_BATCH_SIZE):
            chunk = headlines[start : start + _INFERENCE_BATCH_SIZE]
            results = _score_batch(chunk)
            for record, result in zip(chunk, results):
                db.update_sentiment(
                    headline_hash=record["headline_hash"],
                    label=result["label"],
                    confidence=result["confidence"],
                    attention_keyword=result["attention_keyword"],
                    model_version=MODEL_PATH.name,
                )
            total_scored += len(chunk)

        logger.info("Scored %d headlines so far...", total_scored)

    logger.info("Scoring complete — %d headlines scored", total_scored)
    return total_scored


# ─────────────────────────────────────────────────────────────────
# Internal
# ─────────────────────────────────────────────────────────────────

def _load_model() -> None:
    global _model, _tokenizer
    if _model is not None:
        return

    logger.info("Loading FinBERT checkpoint from %s (%s)", MODEL_PATH, _device)
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    _model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH, attn_implementation="eager"
    )
    _model.to(_device)
    _model.eval()


def _score_batch(records: list[dict]) -> list[dict]:
    """Run one forward pass and return label/confidence/keyword per record."""
    texts = [r["headline"] for r in records]
    encoded = _tokenizer(
        texts, truncation=True, padding=True, return_tensors="pt"
    ).to(_device)

    with torch.no_grad():
        outputs = _model(**encoded, output_attentions=True)

    probs = torch.softmax(outputs.logits, dim=-1)
    confidences, predictions = probs.max(dim=-1)

    # Average attention over heads in the last layer; row 0 is the
    # [CLS] token's attention to every other token in the sequence,
    # used here as a cheap proxy for "the word the model focused on".
    last_layer_attn = outputs.attentions[-1].mean(dim=1)

    results = []
    for i in range(len(records)):
        label = ID2LABEL[predictions[i].item()]
        confidence = confidences[i].item()
        keyword = _extract_attention_keyword(
            encoded["input_ids"][i], encoded["attention_mask"][i], last_layer_attn[i]
        )
        results.append(
            {"label": label, "confidence": confidence, "attention_keyword": keyword}
        )

    return results


def _extract_attention_keyword(
    input_ids: torch.Tensor, attention_mask: torch.Tensor, cls_attention: torch.Tensor
) -> str | None:
    """Return the non-special token most attended to by [CLS]."""
    seq_len = int(attention_mask.sum().item())
    if seq_len <= 2:   # only [CLS] and [SEP] — nothing to extract
        return None

    scores = cls_attention[0].clone()
    scores[0] = float("-inf")              # exclude [CLS] itself
    scores[seq_len - 1 :] = float("-inf")  # exclude [SEP] and padding

    best_idx = int(scores.argmax().item())
    token = _tokenizer.convert_ids_to_tokens(int(input_ids[best_idx].item()))
    return token.lstrip("##")


if __name__ == "__main__":
    setup_logging(LOG_FILE)
    database = DatabaseManager(DB_PATH)
    run_scoring_job(database)
