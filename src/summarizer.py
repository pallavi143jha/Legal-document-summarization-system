"""
summarizer.py
Extractive (TextRank) + Abstractive (BART) summarisation.
Every function has a try/except so errors show up clearly.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import re


def _trunc_words(text, max_words=800):
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])


def _ensure_nltk():
    """Download required NLTK data quietly."""
    try:
        import nltk
        for pkg in ["punkt", "punkt_tab", "stopwords"]:
            try:
                nltk.data.find(f"tokenizers/{pkg}")
            except LookupError:
                nltk.download(pkg, quiet=True)
    except Exception as e:
        print(f"[NLTK] Warning: {e}")


# ── EXTRACTIVE ────────────────────────────────────────────────────────────────

def extractive_summarize(text, n_sentences=8):
    """
    Use TextRank to pick the most important sentences.
    Returns a plain string, never raises — returns error message on failure.
    """
    try:
        _ensure_nltk()

        from sumy.parsers.plaintext     import PlaintextParser
        from sumy.nlp.tokenizers        import Tokenizer
        from sumy.summarizers.text_rank import TextRankSummarizer
        from sumy.nlp.stemmers          import Stemmer
        from sumy.utils                 import get_stop_words

        LANG    = "english"
        parser  = PlaintextParser.from_string(text, Tokenizer(LANG))
        stemmer = Stemmer(LANG)
        sumr    = TextRankSummarizer(stemmer)
        sumr.stop_words = get_stop_words(LANG)

        sentences = sumr(parser.document, n_sentences)
        result    = " ".join(str(s) for s in sentences)

        if not result.strip():
            return _fallback_extract(text, n_sentences)

        return result

    except ImportError:
        print("[Summarizer] sumy not installed — using fallback extractor")
        return _fallback_extract(text, n_sentences)
    except Exception as e:
        print(f"[Summarizer] TextRank error: {e} — using fallback")
        return _fallback_extract(text, n_sentences)


def _fallback_extract(text, n=8):
    """
    Simple fallback: split into sentences, score by length,
    return top-n. No external library needed.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    # score = sentence length (longer = more content)
    scored = sorted(
        [(len(s.split()), i, s) for i, s in enumerate(sentences) if len(s.split()) > 5],
        key=lambda x: x[0],
        reverse=True,
    )
    top = sorted(scored[:n], key=lambda x: x[1])   # restore original order
    return " ".join(s for _, _, s in top)


# ── ABSTRACTIVE ───────────────────────────────────────────────────────────────

_bart_pipe = None


def _load_bart():
    global _bart_pipe
    if _bart_pipe is not None:
        return _bart_pipe

    from config import SUM_MODEL, DEVICE
    from transformers import pipeline

    print(f"[Summarizer] Loading BART model ({SUM_MODEL}) ...")
    print("[Summarizer] First run: downloading ~1.6 GB. Please wait ...")

    _bart_pipe = pipeline(
        "summarization",
        model=SUM_MODEL,
        device=0 if DEVICE == "cuda" else -1,
    )
    print("[Summarizer] BART ready.")
    return _bart_pipe


def abstractive_summarize(text):
    """
    Generate a fluent paragraph using BART.
    Falls back to extractive if BART fails or is not installed.
    """
    from config import SUM_MIN_TOK, SUM_MAX_TOK, SUM_BEAMS

    try:
        pipe   = _load_bart()
        src    = _trunc_words(text, 800)
        result = pipe(
            src,
            min_length=SUM_MIN_TOK,
            max_length=SUM_MAX_TOK,
            num_beams=SUM_BEAMS,
            do_sample=False,
            truncation=True,
        )
        return result[0]["summary_text"].strip()

    except Exception as e:
        print(f"[Summarizer] BART error: {e}")
        return (
            "Note: BART abstractive model failed to load. "
            "Showing extractive summary only.\n\n"
            + extractive_summarize(text, 10)
        )


# ── PUBLIC API ────────────────────────────────────────────────────────────────

def summarize(text, mode="both", n_sentences=8):
    """
    Main function called by app.py.

    Parameters
    ----------
    text        : full document text (string)
    mode        : "extractive" | "abstractive" | "both"
    n_sentences : number of sentences for extractive mode

    Returns
    -------
    dict with keys: extractive, abstractive, ext_words, abst_words
    """
    if not text or not text.strip():
        empty = "No text was provided."
        return {"extractive": empty, "abstractive": empty,
                "ext_words": 0, "abst_words": 0}

    ext_sum  = ""
    abst_sum = ""

    if mode in ("extractive", "both"):
        print(f"[Summarizer] Running extractive ({n_sentences} sentences) ...")
        ext_sum = extractive_summarize(text, n_sentences)
        print(f"[Summarizer] Extractive done: {len(ext_sum.split())} words")

    if mode in ("abstractive", "both"):
        print("[Summarizer] Running abstractive (BART) ...")
        src      = ext_sum if ext_sum.strip() else text
        abst_sum = abstractive_summarize(src)
        print(f"[Summarizer] Abstractive done: {len(abst_sum.split())} words")

    return {
        "extractive" : ext_sum  if ext_sum.strip()  else "Extractive mode not selected.",
        "abstractive": abst_sum if abst_sum.strip() else "Abstractive mode not selected.",
        "ext_words"  : len(ext_sum.split())  if ext_sum  else 0,
        "abst_words" : len(abst_sum.split()) if abst_sum else 0,
    }