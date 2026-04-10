from pathlib import Path

ROOT    = Path(__file__).resolve().parent.parent
SUM_DIR = ROOT / "outputs" / "summaries"
RPT_DIR = ROOT / "outputs" / "reports"

for p in [SUM_DIR, RPT_DIR]:
    p.mkdir(parents=True, exist_ok=True)

LABELS        = ["Rejected", "Accepted"]
SUM_MODEL     = "facebook/bart-large-cnn"
SUM_MAX_WORDS = 800
SUM_MIN_TOK   = 60
SUM_MAX_TOK   = 220
SUM_BEAMS     = 4
PRED_MODEL    = "law-ai/InLegalBERT"
MAX_LEN       = 512
CHUNK_SIZE    = 450
CHUNK_OVERLAP = 50

try:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    DEVICE = "cpu"