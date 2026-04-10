"""
predictor.py
InLegalBERT case outcome predictor.
Has full error handling — never silently fails.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import re


# ── Fallback predictor (no ML needed) ─────────────────────────────────────────

def _keyword_predict(text):
    """
    Simple keyword-based fallback prediction when the ML model
    is not available or fails to load.
    Works by counting legal outcome keywords.
    """
    text_lower = text.lower()

    accepted_keywords = [
        "appeal allowed", "petition allowed", "allowed", "quashed",
        "set aside", "granted", "accepted", "acquitted", "acquittal",
        "in favour", "relief granted", "order quashed", "writ allowed",
        "partly allowed", "disposed of in favour",
    ]
    rejected_keywords = [
        "appeal dismissed", "petition dismissed", "dismissed",
        "rejected", "upheld conviction", "conviction upheld",
        "no merit", "without merit", "no grounds", "liable",
        "guilty", "convicted", "sentence upheld", "affirmed",
    ]

    acc_score = sum(1 for kw in accepted_keywords if kw in text_lower)
    rej_score = sum(1 for kw in rejected_keywords if kw in text_lower)

    total = acc_score + rej_score + 1   # +1 avoids division by zero
    acc_prob = round(acc_score / total, 4)
    rej_prob = round(rej_score / total, 4)

    # Normalise so they add to 1
    if acc_prob + rej_prob == 0:
        acc_prob, rej_prob = 0.5, 0.5
    else:
        s = acc_prob + rej_prob
        acc_prob = round(acc_prob / s, 4)
        rej_prob = round(1 - acc_prob, 4)

    label_id = 1 if acc_prob >= rej_prob else 0
    label    = ["Rejected", "Accepted"][label_id]
    conf     = acc_prob if label_id == 1 else rej_prob

    return {
        "label"        : label,
        "label_id"     : label_id,
        "confidence"   : round(float(conf), 4),
        "probabilities": {"Rejected": float(rej_prob), "Accepted": float(acc_prob)},
        "method"       : "keyword_fallback",
    }


# ── ML predictor ──────────────────────────────────────────────────────────────

_model     = None
_tokenizer = None
_ml_failed = False    # once ML fails, skip it and use fallback


def _load_ml_model():
    global _model, _tokenizer, _ml_failed

    if _ml_failed:
        return None, None
    if _model is not None:
        return _model, _tokenizer

    try:
        import torch
        import torch.nn as nn
        from transformers import AutoModel, AutoTokenizer
        from config import PRED_MODEL, DEVICE

        print(f"[Predictor] Loading {PRED_MODEL} on {DEVICE} ...")
        _tokenizer = AutoTokenizer.from_pretrained(PRED_MODEL)

        class _Classifier(nn.Module):
            def __init__(self):
                super().__init__()
                self.bert       = AutoModel.from_pretrained(PRED_MODEL)
                self.dropout    = nn.Dropout(0.1)
                self.classifier = nn.Linear(self.bert.config.hidden_size, 2)

            def forward(self, input_ids, attention_mask):
                out    = self.bert(input_ids=input_ids, attention_mask=attention_mask)
                pooled = self.dropout(out.last_hidden_state[:, 0, :])
                return self.classifier(pooled)

        _model = _Classifier().to(DEVICE)
        _model.eval()
        print("[Predictor] InLegalBERT ready.")
        return _model, _tokenizer

    except Exception as e:
        print(f"[Predictor] ML model failed to load: {e}")
        print("[Predictor] Using keyword-based fallback predictor.")
        _ml_failed = True
        return None, None


def _ml_predict(text):
    """Run InLegalBERT prediction."""
    import torch
    from config import MAX_LEN, CHUNK_SIZE, CHUNK_OVERLAP, LABELS, DEVICE

    model, tok = _load_ml_model()
    if model is None:
        return None   # signal to use fallback

    raw_ids = tok.encode(text, add_special_tokens=False)

    def _encode(t):
        return tok(t, max_length=MAX_LEN, padding="max_length",
                   truncation=True, return_tensors="pt")

    with torch.no_grad():
        if len(raw_ids) <= MAX_LEN - 2:
            enc    = _encode(text)
            logits = model(enc["input_ids"].to(DEVICE),
                           enc["attention_mask"].to(DEVICE))
            method = "single_pass"
        else:
            print(f"[Predictor] Long doc ({len(raw_ids)} tokens) — chunk pooling")
            chunks   = []
            i        = 0
            while i < len(raw_ids):
                end = min(i + CHUNK_SIZE, len(raw_ids))
                chunks.append(raw_ids[i:end])
                if end == len(raw_ids):
                    break
                i = end - CHUNK_OVERLAP

            cls_vecs = []
            for ch in chunks:
                dec = tok.decode(ch, skip_special_tokens=True)
                enc = _encode(dec)
                out = model.bert(
                    input_ids=enc["input_ids"].to(DEVICE),
                    attention_mask=enc["attention_mask"].to(DEVICE),
                )
                cls_vecs.append(out.last_hidden_state[:, 0, :])

            pooled = torch.stack(cls_vecs).mean(0)
            logits = model.classifier(model.dropout(pooled))
            method = "chunk_pooling"

    probs    = torch.softmax(logits, dim=-1).squeeze().cpu().tolist()
    label_id = int(torch.argmax(logits, dim=-1).item())

    return {
        "label"        : LABELS[label_id],
        "label_id"     : label_id,
        "confidence"   : round(float(probs[label_id]), 4),
        "probabilities": {LABELS[i]: round(float(p), 4) for i, p in enumerate(probs)},
        "method"       : method,
    }


# ── PUBLIC API ────────────────────────────────────────────────────────────────

def predict(text):
    """
    Predict case outcome.
    Tries ML model first, falls back to keyword analysis if ML fails.

    Returns
    -------
    dict: label, label_id, confidence, probabilities, method
    """
    if not text or not text.strip():
        return {
            "label": "Unknown", "label_id": -1,
            "confidence": 0.0, "probabilities": {}, "method": "none",
        }

    try:
        result = _ml_predict(text)
        if result is not None:
            return result
    except Exception as e:
        print(f"[Predictor] ML prediction error: {e} — using keyword fallback")

    return _keyword_predict(text)