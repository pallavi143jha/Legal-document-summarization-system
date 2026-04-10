"""
app.py  —  Run this file to start the project:
    python app.py
Then open: http://localhost:7860
"""

# ── THIS BLOCK MUST BE AT THE VERY TOP — fixes the ImportError ────────────────
import sys
import os
from pathlib import Path

# Add the src/ folder to Python's search path
SRC_PATH = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC_PATH))
# ─────────────────────────────────────────────────────────────────────────────

import gradio as gr

# Now Python can find these because we added src/ to the path above
from pdf_reader import read_pdf, get_pdf_meta
from summarizer import summarize
from predictor  import predict
from evaluator  import rouge_scores, interpret_rouge
from config     import SUM_DIR


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN ANALYSIS FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def analyse(file_obj, pasted_text, mode, n_sent, progress=gr.Progress()):
    progress(0.05, desc="Reading document ...")

    text = ""
    meta = {}

    # Step 1: Read the file
    if file_obj is not None:
        fpath = file_obj.name if hasattr(file_obj, "name") else str(file_obj)
        ext   = Path(fpath).suffix.lower()

        if ext == ".pdf":
            try:
                text = read_pdf(fpath)
                meta = get_pdf_meta(fpath)
            except Exception as e:
                return _error(f"Could not read PDF: {e}")
        elif ext in (".txt", ".text"):
            with open(fpath, encoding="utf-8", errors="replace") as f:
                text = f.read().strip()
        else:
            return _error("Please upload a PDF or TXT file.")

    if not text.strip():
        text = pasted_text.strip() if pasted_text else ""

    if not text.strip():
        return _error("No text found. Upload a PDF/TXT or paste your document text.")

    if len(text.split()) < 40:
        return _error("Document too short. Please provide a complete legal document.")

    # Step 2: Summarise
    progress(0.30, desc="Running TextRank extractive summarisation ...")
    mode_map = {"Extractive": "extractive", "Abstractive": "abstractive", "Both": "both"}
    sums     = summarize(text, mode=mode_map[mode], n_sentences=int(n_sent))
    ext_sum  = sums["extractive"]
    abst_sum = sums["abstractive"]

    # Step 3: Predict
    progress(0.75, desc="Running InLegalBERT prediction ...")
    pred = predict(text)

    # Save to file
    progress(0.95, desc="Saving results ...")
    out_path = SUM_DIR / "latest_summary.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        wc = len(text.split())
        f.write(f"PREDICTION : {pred['label']}  ({round(pred['confidence']*100,1)}%)\n")
        f.write(f"WORD COUNT : {wc:,}\n\n")
        f.write("=== EXTRACTIVE SUMMARY ===\n\n" + ext_sum + "\n\n")
        f.write("=== ABSTRACTIVE SUMMARY ===\n\n" + abst_sum + "\n")

    progress(1.0)

    wc     = len(text.split())
    status = f"Done  —  {wc:,} words processed  |  {meta.get('pages','?')} pages"

    return (
        status,
        ext_sum,
        abst_sum,
        _stats_html(text, meta, sums),
        _pred_html(pred),
        str(out_path),
    )


def _error(msg):
    return (f"  {msg}", "", "", "", _pred_html(None), None)


# ─────────────────────────────────────────────────────────────────────────────
#  EVALUATE TAB
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(doc_text, ref_text):
    if not doc_text.strip():
        return "Paste your legal document on the left."
    if not ref_text.strip():
        return "Paste a reference summary on the right to compare against."

    sums  = summarize(doc_text.strip(), mode="both")
    ext   = sums["extractive"]
    abst  = sums["abstractive"]
    r_ext  = rouge_scores(ext,  ref_text.strip())
    r_abst = rouge_scores(abst, ref_text.strip())
    pred   = predict(doc_text.strip())

    return (
        f"ROUGE SCORES\n\n"
        f"Extractive summary:\n"
        f"  ROUGE-1 : {r_ext['rouge1']}  (word overlap)\n"
        f"  ROUGE-2 : {r_ext['rouge2']}  (bigram overlap)\n"
        f"  ROUGE-L : {r_ext['rougeL']}  (longest sequence)\n"
        f"  Quality : {interpret_rouge(r_ext)}\n\n"
        f"Abstractive summary:\n"
        f"  ROUGE-1 : {r_abst['rouge1']}\n"
        f"  ROUGE-2 : {r_abst['rouge2']}\n"
        f"  ROUGE-L : {r_abst['rougeL']}\n"
        f"  Quality : {interpret_rouge(r_abst)}\n\n"
        f"PREDICTION\n\n"
        f"  Outcome    : {pred['label']}\n"
        f"  Confidence : {round(pred['confidence']*100, 1)}%\n"
        f"  Rejected   : {round(pred['probabilities'].get('Rejected', 0)*100, 1)}%\n"
        f"  Accepted   : {round(pred['probabilities'].get('Accepted', 0)*100, 1)}%\n"
    )


# ─────────────────────────────────────────────────────────────────────────────
#  HTML CARD BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

def _stats_html(text, meta, sums):
    ow  = len(text.split())
    sw  = max(sums.get("ext_words", 0), sums.get("abst_words", 0))
    red = round((1 - sw / ow) * 100) if ow else 0

    def card(label, value, sub):
        return (
            f'<div style="background:#111827;border:1px solid #1f3a5f;border-radius:10px;'
            f'padding:12px 14px;text-align:center;">'
            f'<div style="color:#4b6080;font-size:9px;letter-spacing:1.5px;'
            f'text-transform:uppercase;margin-bottom:4px;">{label}</div>'
            f'<div style="color:#e2e8f0;font-size:20px;font-weight:700;">{value}</div>'
            f'<div style="color:#1e3a55;font-size:9px;margin-top:2px;">{sub}</div></div>'
        )

    return (
        '<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-bottom:12px;">'
        + card("Pages",   str(meta.get("pages", "?")), "in PDF")
        + card("Words",   f"{ow:,}", "original")
        + card("Summary", f"{sw:,}", "words")
        + card("Reduced", f"{red}%", "shorter")
        + '</div>'
    )


def _pred_html(pred):
    if pred is None:
        return (
            '<div style="background:#0a1520;border:1px solid #1a3050;border-radius:12px;'
            'padding:20px;text-align:center;color:#1e3a55;font-size:12px;">'
            'Upload a document to see the prediction.'
            '</div>'
        )

    lbl   = pred["label"]
    conf  = pred["confidence"]
    probs = pred["probabilities"]
    color = "#10b981" if lbl == "Accepted" else "#ef4444"
    rej   = round(probs.get("Rejected", 0) * 100, 1)
    acc   = round(probs.get("Accepted", 0) * 100, 1)

    return f"""
<div style="background:#0a1520;border:1.5px solid {color}44;border-radius:12px;padding:20px;">

  <div style="text-align:center;margin-bottom:16px;">
    <span style="display:inline-block;background:{color}18;border:1.5px solid {color};
                 color:{color};border-radius:6px;padding:7px 26px;font-size:18px;
                 font-weight:700;letter-spacing:3px;">{lbl.upper()}</span>
  </div>

  <div style="margin-bottom:14px;">
    <div style="display:flex;justify-content:space-between;color:#4b6080;
                font-size:10px;margin-bottom:4px;">
      <span>Confidence</span><span>{round(conf*100,1)}%</span>
    </div>
    <div style="background:#111827;border-radius:4px;height:7px;overflow:hidden;">
      <div style="width:{round(conf*100)}%;height:100%;background:{color};border-radius:4px;"></div>
    </div>
  </div>

  <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;">
    <div style="background:#111827;border-radius:8px;padding:10px;text-align:center;">
      <div style="color:#ef4444;font-size:16px;font-weight:700;">{rej}%</div>
      <div style="color:#4b6080;font-size:9px;text-transform:uppercase;
                  letter-spacing:1px;margin-top:2px;">Rejected</div>
      <div style="background:#1f2937;border-radius:2px;height:4px;margin-top:6px;overflow:hidden;">
        <div style="width:{rej}%;height:100%;background:#ef4444;"></div></div>
    </div>
    <div style="background:#111827;border-radius:8px;padding:10px;text-align:center;">
      <div style="color:#10b981;font-size:16px;font-weight:700;">{acc}%</div>
      <div style="color:#4b6080;font-size:9px;text-transform:uppercase;
                  letter-spacing:1px;margin-top:2px;">Accepted</div>
      <div style="background:#1f2937;border-radius:2px;height:4px;margin-top:6px;overflow:hidden;">
        <div style="width:{acc}%;height:100%;background:#10b981;"></div></div>
    </div>
  </div>

  <div style="color:#1e3a55;font-size:9px;text-align:center;margin-top:12px;
              border-top:1px solid #111827;padding-top:8px;">
    AI prediction only — not legal advice
  </div>
</div>"""


# ─────────────────────────────────────────────────────────────────────────────
#  CSS
# ─────────────────────────────────────────────────────────────────────────────

CSS = """
body, .gradio-container {
    background: #060f1a !important;
    font-family: Georgia, serif !important;
}
.gr-panel, .gr-box {
    background: #0a1520 !important;
    border: 1px solid #1a3050 !important;
    border-radius: 10px !important;
}
textarea, input[type=text] {
    background: #0a1520 !important;
    border: 1px solid #1a3050 !important;
    color: #8ab4c8 !important;
    font-family: Georgia, serif !important;
    font-size: 13px !important;
}
label span {
    color: #2a5070 !important;
    font-size: 10px !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
}
.gr-button-primary {
    background: #0d2040 !important;
    border: 1px solid #1a5070 !important;
    color: #10b981 !important;
    font-size: 13px !important;
    border-radius: 6px !important;
}
.gr-button-primary:hover {
    border-color: #10b981 !important;
}
"""

HOW_IT_WORKS = """
<div style="font-family:Georgia,serif;color:#8ab4c8;line-height:1.9;font-size:13px;padding:4px 0;">

<div style="color:#10b981;font-size:13px;letter-spacing:1px;margin-bottom:18px;">
  Complete pipeline — what happens when you upload a PDF
</div>

<div style="border-left:2px solid #3b82f6;padding:10px 14px;margin-bottom:12px;background:#0d1f35;border-radius:0 8px 8px 0;">
  <div style="color:#3b82f6;font-size:9px;letter-spacing:2px;margin-bottom:5px;">STEP 1 — pdf_reader.py</div>
  <div style="color:#d4e4f4;font-weight:500;margin-bottom:4px;">Extract text from the PDF</div>
  <div style="color:#6a9ab8;">pdfplumber opens the PDF page-by-page. Each page's text is extracted and joined. Then noise is stripped: JUDIS watermarks, page numbers, non-printable characters, extra blank lines. Output: one clean string.</div>
</div>

<div style="border-left:2px solid #8b5cf6;padding:10px 14px;margin-bottom:12px;background:#0d1f35;border-radius:0 8px 8px 0;">
  <div style="color:#8b5cf6;font-size:9px;letter-spacing:2px;margin-bottom:5px;">STEP 2A — summarizer.py (TextRank)</div>
  <div style="color:#d4e4f4;font-weight:500;margin-bottom:4px;">Extractive summary — pick key sentences</div>
  <div style="color:#6a9ab8;">sumy builds a graph where every sentence is a node. Shared important words create edges. PageRank scores each sentence. Top-N sentences are returned in original order. Runs in under 1 second, no GPU needed.</div>
</div>

<div style="border-left:2px solid #f59e0b;padding:10px 14px;margin-bottom:12px;background:#0d1f35;border-radius:0 8px 8px 0;">
  <div style="color:#f59e0b;font-size:9px;letter-spacing:2px;margin-bottom:5px;">STEP 2B — summarizer.py (BART)</div>
  <div style="color:#d4e4f4;font-weight:500;margin-bottom:4px;">Abstractive summary — rewrite as a paragraph</div>
  <div style="color:#6a9ab8;">facebook/bart-large-cnn reads the extractive sentences and generates brand-new text using beam search. First run downloads ~1.6 GB. Subsequent runs use the cache.</div>
</div>

<div style="border-left:2px solid #ef4444;padding:10px 14px;margin-bottom:12px;background:#0d1f35;border-radius:0 8px 8px 0;">
  <div style="color:#ef4444;font-size:9px;letter-spacing:2px;margin-bottom:5px;">STEP 3 — predictor.py (InLegalBERT)</div>
  <div style="color:#d4e4f4;font-weight:500;margin-bottom:4px;">Predict: Accepted or Rejected</div>
  <div style="color:#6a9ab8;">InLegalBERT = BERT pre-trained on Indian legal text. The [CLS] token (768-d vector) passes through Linear(768→2) → softmax → probabilities. Long documents are split into 450-token chunks, each [CLS] vector is extracted, averaged, then classified. First run downloads ~400 MB.</div>
</div>

<div style="border-left:2px solid #06b6d4;padding:10px 14px;background:#0d1f35;border-radius:0 8px 8px 0;">
  <div style="color:#06b6d4;font-size:9px;letter-spacing:2px;margin-bottom:5px;">STEP 4 — evaluator.py (ROUGE)</div>
  <div style="color:#d4e4f4;font-weight:500;margin-bottom:4px;">Measure summary quality</div>
  <div style="color:#6a9ab8;">ROUGE-1 = word overlap with reference. ROUGE-2 = bigram overlap. ROUGE-L = longest common sequence. Scores 0–1, higher is better. Use the Evaluate tab to compute these against your own reference.</div>
</div>

</div>
"""


# ─────────────────────────────────────────────────────────────────────────────
#  BUILD GRADIO UI
# ─────────────────────────────────────────────────────────────────────────────

def build_app():
    with gr.Blocks(
        css=CSS,
        title="Legal NLP Analyser",
        theme=gr.themes.Base(primary_hue="blue", neutral_hue="slate"),
    ) as demo:

        gr.HTML("""
        <div style="background:#06111e;border-bottom:1px solid #0d2035;
                    padding:24px 36px 16px;text-align:center;">
          <h1 style="color:#d4e4f4;font-family:Georgia,serif;font-size:2rem;
                     font-weight:600;margin:0 0 6px;letter-spacing:1px;">
            Legal Document Analyser
          </h1>
          <p style="color:#1e3a55;font-size:11px;letter-spacing:2px;margin:0;">
            PDF UPLOAD  ·  SUMMARISATION  ·  OUTCOME PREDICTION
          </p>
        </div>
        """)

        with gr.Tabs():

            # ── TAB 1: ANALYSE ──────────────────────────────────────────────
            with gr.Tab("Analyse"):
                with gr.Row():

                    with gr.Column(scale=1):
                        gr.HTML('<p style="color:#1a4060;font-size:9px;letter-spacing:2px;'
                                'text-transform:uppercase;border-bottom:1px solid #0d2035;'
                                'padding-bottom:5px;margin-bottom:10px;">Document Input</p>')

                        file_in = gr.File(
                            label="Upload PDF or TXT (e.g. the Nanavati judgment)",
                            file_types=[".pdf", ".txt"],
                            height=120,
                        )
                        gr.HTML('<p style="text-align:center;color:#0d2035;font-size:10px;'
                                'margin:5px 0;">— or paste text below —</p>')
                        text_in = gr.Textbox(
                            label="Paste document text",
                            placeholder="Paste any Supreme Court judgment text here ...",
                            lines=8,
                        )

                        gr.HTML('<p style="color:#1a4060;font-size:9px;letter-spacing:2px;'
                                'text-transform:uppercase;border-bottom:1px solid #0d2035;'
                                'padding-bottom:5px;margin:12px 0 10px;">Options</p>')

                        mode   = gr.Radio(
                            choices=["Extractive", "Abstractive", "Both"],
                            value="Both",
                            label="Summarisation mode",
                        )
                        n_sent = gr.Slider(3, 15, step=1, value=8,
                                           label="Extractive sentences")

                        with gr.Row():
                            run_btn   = gr.Button("Analyse", variant="primary", size="lg")
                            clear_btn = gr.Button("Clear",   size="lg")

                        status_out = gr.Textbox(label="Status", interactive=False, lines=1)

                    with gr.Column(scale=2):
                        stats_out = gr.HTML()

                        gr.HTML('<p style="color:#1a4060;font-size:9px;letter-spacing:2px;'
                                'text-transform:uppercase;border-bottom:1px solid #0d2035;'
                                'padding-bottom:5px;margin:8px 0 10px;">Prediction</p>')
                        pred_out = gr.HTML(_pred_html(None))

                        gr.HTML('<p style="color:#1a4060;font-size:9px;letter-spacing:2px;'
                                'text-transform:uppercase;border-bottom:1px solid #0d2035;'
                                'padding-bottom:5px;margin:12px 0 10px;">Summaries</p>')

                        with gr.Tabs():
                            with gr.Tab("Extractive (TextRank)"):
                                ext_out = gr.Textbox(
                                    label="Key sentences selected by TextRank",
                                    lines=9, interactive=False,
                                )
                            with gr.Tab("Abstractive (BART)"):
                                abst_out = gr.Textbox(
                                    label="Fluent paragraph generated by BART",
                                    lines=9, interactive=False,
                                )

                        dl_btn = gr.File(label="Download summary report (.txt)")

                run_btn.click(
                    fn=analyse,
                    inputs=[file_in, text_in, mode, n_sent],
                    outputs=[status_out, ext_out, abst_out, stats_out, pred_out, dl_btn],
                )
                clear_btn.click(
                    fn=lambda: [None, "", "Both", 8, "", "", "", "", _pred_html(None), None],
                    outputs=[file_in, text_in, mode, n_sent,
                             status_out, ext_out, abst_out, stats_out, pred_out, dl_btn],
                )

            # ── TAB 2: HOW IT WORKS ─────────────────────────────────────────
            with gr.Tab("How It Works"):
                gr.HTML(HOW_IT_WORKS)

            # ── TAB 3: EVALUATE ─────────────────────────────────────────────
            with gr.Tab("Evaluate (ROUGE)"):
                gr.HTML('<p style="color:#2a5070;font-size:11px;margin-bottom:12px;">'
                        'Paste a document and your own reference summary to get ROUGE scores.</p>')
                with gr.Row():
                    with gr.Column():
                        eval_doc = gr.Textbox(label="Legal document text",
                                              placeholder="Paste judgment text ...", lines=9)
                    with gr.Column():
                        eval_ref = gr.Textbox(label="Your reference summary",
                                              placeholder="Write a brief summary yourself ...",
                                              lines=9)
                eval_btn = gr.Button("Evaluate", variant="primary")
                eval_out = gr.Textbox(label="Results", lines=14, interactive=False)
                eval_btn.click(fn=evaluate, inputs=[eval_doc, eval_ref], outputs=eval_out)

        gr.HTML('<p style="text-align:center;padding:12px 0 6px;color:#0d2035;'
                'font-size:9px;letter-spacing:1px;border-top:1px solid #060f1a;margin-top:6px;">'
                'ILDC Dataset  ·  InLegalBERT + BART  ·  Academic research only</p>')

    return demo


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*50)
    print("  Legal NLP Analyser")
    print("  Opening at: http://localhost:7860")
    print("="*50 + "\n")
    build_app().launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True,
        show_error=True,
    )