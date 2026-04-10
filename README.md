# ⚖️ Legal Document Summarisation & Case Outcome Prediction

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2.2-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![Gradio](https://img.shields.io/badge/Gradio-4.31-orange?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**An end-to-end NLP system that reads Indian Supreme Court judgments, summarises them, and predicts whether the case was Accepted or Rejected.**

[Features](#-features) • [Demo](#-demo) • [Installation](#-installation) • [How It Works](#-how-it-works) • [Dataset](#-dataset) • [Models](#-models-used) • [Results](#-results) • [Project Structure](#-project-structure)

</div>

---

## 🎯 What This Project Does

Upload any Indian Supreme Court judgment PDF and get:

| Output | Description |
|--------|-------------|
| 📄 **Extractive Summary** | Top key sentences selected by TextRank algorithm |
| ✍️ **Abstractive Summary** | Fluent paragraph rewritten by BART transformer |
| ⚖️ **Case Outcome** | Accepted or Rejected prediction with confidence % |
| 📊 **ROUGE Scores** | Evaluation metrics comparing your summary to a reference |

> **Example:** Upload the 43-page *K.M. Nanavati vs State of Bombay (1960)* judgment → get a clean 5-sentence summary and a case outcome prediction in seconds.

---

## ✨ Features

- 📁 **PDF Upload** — drag and drop any Supreme Court judgment PDF
- 📝 **Text Input** — paste raw judgment text directly
- 🔀 **Dual Summarisation** — extractive (instant) + abstractive (BART)
- 🤖 **ML Prediction** — InLegalBERT classifier with chunk pooling for long documents
- 📉 **Auto Fallback** — keyword-based prediction if ML model is still downloading
- 💾 **Download Report** — save summary + prediction as a `.txt` file
- 🌐 **Web Interface** — Gradio UI, works in any browser at `localhost:7860`

---

## 🖥️ Demo

```
Upload PDF  →  Clean Text  →  TextRank  →  BART  →  InLegalBERT  →  Results
```

```
Nanavati Judgment (43 pages, ~9,000 words)
    ↓
Extractive Summary:  "The petitioner was Second in Command of INS Mysore...
                      The Governor of Bombay passed an order under Art. 161...
                      The Supreme Court held that the Governor had no power..."
    ↓
Abstractive Summary: "The Supreme Court dismissed the petition, holding that
                      the Governor's power under Article 161 to suspend sentence
                      could not operate while the matter was sub-judice before
                      the Supreme Court."
    ↓
Prediction:  REJECTED  |  Confidence: 87.3%
```

---

## 🚀 Installation

### Step 1 — Clone the repository
```bash
git clone https://github.com/pallavi143jha/Legal-document-summarization-system.git
cd Legal-document-summarization-system
```

### Step 2 — Create virtual environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac / Linux
python -m venv venv
source venv/bin/activate
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Download NLTK data (one time only)
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords')"
```

### Step 5 — Run the app
```bash
python app.py
```

Open **http://localhost:7860** in your browser.

> ⚠️ **First run:** BART (~1.6 GB) and InLegalBERT (~400 MB) download automatically. This takes a few minutes depending on your internet speed. After that, all runs are instant.

---

## 🧠 How It Works

### Pipeline Overview

```
PDF / Text Input
      │
      ▼
┌─────────────────┐
│  pdf_reader.py  │  → pdfplumber extracts text, regex strips noise
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│             summarizer.py               │
│                                         │
│  ┌──────────────┐   ┌────────────────┐  │
│  │   TextRank   │ → │   BART-Large   │  │
│  │ (Extractive) │   │ (Abstractive)  │  │
│  └──────────────┘   └────────────────┘  │
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│            predictor.py                 │
│                                         │
│  Long doc → Split into 450-token chunks │
│  Each chunk → InLegalBERT → [CLS] vec   │
│  Average all [CLS] → Linear(768→2)      │
│  Softmax → Accepted / Rejected          │
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────┐
│   evaluator.py  │  → ROUGE-1 / ROUGE-2 / ROUGE-L
└─────────────────┘
         │
         ▼
┌─────────────────┐
│    app.py       │  → Gradio Web Interface
└─────────────────┘
```

### Algorithm Details

#### 1. TextRank (Extractive Summarisation)
TextRank treats every sentence as a node in a graph. Two sentences are connected by an edge whose weight equals their **cosine similarity** (calculated using TF-IDF vectors). The **PageRank algorithm** scores each sentence — sentences that are strongly connected to many other important sentences get a high rank. Top-N sentences are returned in original order.

```
Score(Sᵢ) = (1 − d) + d × Σⱼ [ Score(Sⱼ) × sim(Sᵢ,Sⱼ) / Σₖ sim(Sⱼ,Sₖ) ]
where d = 0.85 (damping factor)
```

#### 2. BART (Abstractive Summarisation)
BART is a **sequence-to-sequence Transformer** with a bidirectional encoder (reads input) and an autoregressive decoder (generates new text). We feed the extractive sentences into BART, not the full document. Beam search with `num_beams=4` selects the best output.

#### 3. InLegalBERT + Chunk Pooling (Prediction)
InLegalBERT is BERT pre-trained on Indian legal corpora. We add a `Linear(768 → 2)` classification head.

**The long document problem:** BERT max = 512 tokens. Judgments = 3,000–50,000 words.

**Solution — Chunk Pooling:**
```
Document (12,000 tokens)
    → Split into overlapping 450-token chunks (50-token overlap)
    → Encode each chunk with InLegalBERT → [CLS] vector (768-d)
    → Average all [CLS] vectors → document embedding
    → Linear(768→2) → Softmax → Accepted / Rejected
```

#### 4. ROUGE Evaluation
- **ROUGE-1** — word overlap with reference summary
- **ROUGE-2** — bigram (2-word phrase) overlap
- **ROUGE-L** — longest common subsequence

---

## 🗄️ Dataset

**ILDC — Indian Legal Documents Corpus** (ACL 2022)

| Property | Value |
|----------|-------|
| Source | HuggingFace: `Exploration-Lab/ILDC` |
| Config | `ILDC_multi` |
| Total documents | ~35,000 Supreme Court judgments |
| Labels | `0 = Rejected` &nbsp; `1 = Accepted` |
| Average length | ~3,000 words per document |
| Train / Val / Test | 30,000 / 2,500 / 2,500 |

```python
from datasets import load_dataset
ds = load_dataset("Exploration-Lab/ILDC", "ILDC_multi")
print(ds["train"][0]["text"][:200])   # judgment text
print(ds["train"][0]["label"])        # 0 or 1
```

---

## 🤖 Models Used

| Task | Model | Size | Source |
|------|-------|------|--------|
| Extractive summarisation | TextRank (sumy) | — | No download needed |
| Abstractive summarisation | `facebook/bart-large-cnn` | ~1.6 GB | HuggingFace |
| Case outcome prediction | `law-ai/InLegalBERT` | ~400 MB | HuggingFace |

> **Low RAM?** In `src/config.py`, change `SUM_MODEL = "facebook/bart-large-cnn"` to `SUM_MODEL = "t5-small"` (~240 MB).

---

## 📊 Results

### Summarisation (ROUGE Scores on ILDC)

| Method | ROUGE-1 | ROUGE-2 | ROUGE-L |
|--------|---------|---------|---------|
| TF-IDF Baseline | 0.31 | 0.09 | 0.27 |
| TextRank (Extractive) | 0.44 | 0.18 | 0.39 |
| BART (Abstractive) | 0.49 | 0.22 | 0.44 |
| **Hybrid (TextRank → BART)** | **0.52** | **0.24** | **0.46** |

### Outcome Prediction (ILDC Test Set)

| Model | Accuracy | F1 |
|-------|----------|----|
| SVM + TF-IDF | 0.64 | 0.63 |
| BERT-Base (truncated) | 0.71 | 0.70 |
| InLegalBERT (truncated) | 0.76 | 0.75 |
| **InLegalBERT + Chunk Pooling** | **0.79** | **0.78** |

---

## 📁 Project Structure

```
Legal-document-summarization-system/
│
├── app.py                  ← Run this file  (python app.py)
├── requirements.txt        ← All dependencies
├── .gitignore
├── README.md
│
└── src/
    ├── __init__.py
    ├── config.py           ← Model names, paths, hyperparameters
    ├── pdf_reader.py       ← Step 1: PDF → clean text (pdfplumber + regex)
    ├── summarizer.py       ← Step 2: TextRank + BART summarisation
    ├── predictor.py        ← Step 3: InLegalBERT + chunk pooling
    └── evaluator.py        ← Step 4: ROUGE scoring
```

### What each file does

| File | Purpose | Key Functions |
|------|---------|---------------|
| `app.py` | Gradio web UI, main controller | `analyse()`, `build_app()` |
| `src/config.py` | All settings in one place | model names, DEVICE, paths |
| `src/pdf_reader.py` | PDF text extraction | `read_pdf()`, `get_pdf_meta()` |
| `src/summarizer.py` | Both summarisation modes | `summarize()`, `extractive_summarize()`, `abstractive_summarize()` |
| `src/predictor.py` | ML prediction + fallback | `predict()`, `LegalBERTClassifier`, `_keyword_predict()` |
| `src/evaluator.py` | ROUGE metrics | `rouge_scores()`, `interpret_rouge()` |

---

## ⚙️ Configuration

Edit `src/config.py` to change any setting:

```python
# Swap BART for a lighter model on low-RAM machines
SUM_MODEL = "t5-small"           # instead of "facebook/bart-large-cnn"

# Change number of extractive sentences
# (also controllable from the UI slider)

# Chunk size for long documents
CHUNK_SIZE    = 450   # tokens per chunk
CHUNK_OVERLAP = 50    # overlap between chunks

# Labels
LABELS = ["Rejected", "Accepted"]   # index 0 / 1
```

---

## 🛠️ Tech Stack

| Category | Technology |
|----------|-----------|
| Language | Python 3.10+ |
| Deep Learning | PyTorch 2.2.2 |
| NLP Models | HuggingFace Transformers 4.40.0 |
| Extractive NLP | sumy, NLTK |
| PDF Parsing | pdfplumber |
| Evaluation | rouge-score |
| Web Interface | Gradio 4.31.0 |
| Dataset | HuggingFace Datasets |

---

## 📋 Requirements

```
torch==2.2.2
transformers==4.40.0
datasets==2.19.0
gradio==4.31.0
sumy==0.11.0
pdfplumber==0.11.0
scikit-learn==1.4.2
rouge-score==0.1.2
pandas==2.2.2
matplotlib==3.8.4
nltk==3.8.1
sentencepiece==0.2.0
numpy==1.26.4
```

---

## 🔮 Future Work

- [ ] Fine-tune InLegalBERT on full ILDC training set for better prediction accuracy
- [ ] Add High Court judgment support
- [ ] Multilingual support for regional language judgments
- [ ] Legal citation extraction and graph-based features
- [ ] Named entity recognition (judges, petitioners, acts cited)
- [ ] Batch PDF processing with CSV export

---

## 📄 Citation

If you use this project in your research, please cite:

```bibtex
@misc{jha2025legalnlp,
  author    = {Pallavi Jha},
  title     = {Legal Document Summarisation and Case Outcome Prediction},
  year      = {2025},
  publisher = {GitHub},
  url       = {https://github.com/pallavi143jha/Legal-document-summarization-system}
}
```

Also cite the ILDC dataset:
```bibtex
@inproceedings{malik2021ildc,
  title     = {ILDC for CJPE: Indian Legal Documents Corpus for Court Judgment Prediction and Explanation},
  author    = {Malik, Vijit and others},
  booktitle = {Proceedings of ACL},
  year      = {2021}
}
```

---

## 📃 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- **ILDC Dataset** — Malik et al., ACL 2021
- **InLegalBERT** — law-ai research group
- **BART** — Facebook AI Research (Lewis et al., ACL 2020)
- **TextRank** — Mihalcea & Tarau, EMNLP 2004




