import re


def read_pdf(pdf_path):
    """Extract clean text from any PDF file."""
    try:
        import pdfplumber
    except ImportError:
        raise ImportError("pdfplumber not installed. Run: pip install pdfplumber")

    pages_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                pages_text.append(t)

    raw = "\n".join(pages_text)
    return _clean(raw)


def get_pdf_meta(pdf_path):
    """Return basic stats about the PDF."""
    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            num_pages = len(pdf.pages)
    except Exception:
        num_pages = 0

    try:
        text      = read_pdf(pdf_path)
        words     = len(text.split())
        sentences = len(re.split(r"(?<=[.!?])\s+", text.strip()))
    except Exception:
        words     = 0
        sentences = 0

    return {"pages": num_pages, "words": words, "sentences": sentences}


def _clean(text):
    """Remove noise from raw PDF text."""
    text = re.sub(r"[^\x09\x0A\x20-\x7E\u00A0-\uFFFF]", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"(?m)^\s*\d{1,4}\s*$", "", text)
    text = re.sub(r"(?m)^.*JUDIS\.NIC\.IN.*$", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()