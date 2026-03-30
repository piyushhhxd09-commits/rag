import gradio as gr
import fitz  # PyMuPDF
import numpy as np
import faiss
import re
import math
from collections import Counter
from sentence_transformers import SentenceTransformer
from PIL import Image
import os  # ✅ added (only for render)

# =========================
# MODELS
# =========================
model = SentenceTransformer("all-MiniLM-L6-v2")

# =========================
# GLOBAL STORAGE
# =========================
text_chunks = []
images = []
tables = []

text_index = None
image_index = None
table_index = None

WORD_FREQ = None
TOTAL_WORDS = 1

# =========================
# TEXT CLEAN
# =========================
def clean_text(text):
    return " ".join(text.split())

# =========================
# PROCESS PDF
# =========================
def process_pdf(file):
    global text_chunks, images, tables
    global text_index, image_index, table_index
    global WORD_FREQ, TOTAL_WORDS

    text_chunks = []
    images = []
    tables = []

    # ✅ file handling (your logic intact)
    if isinstance(file, str):
        doc = fitz.open(file)
    else:
        doc = fitz.open(stream=file.read(), filetype="pdf")

    # -------- TEXT --------
    for page_num, page in enumerate(doc):
        blocks = page.get_text("blocks")
        for b in blocks:
            txt = clean_text(b[4])
            if len(txt) > 40:
                text_chunks.append(txt)

    # -------- WORD IMPORTANCE --------
    word_freq = Counter()
    total_words = 0

    for chunk in text_chunks:
        words = re.findall(r"\b\w+\b", chunk.lower())
        word_freq.update(words)
        total_words += len(words)

    WORD_FREQ = word_freq
    TOTAL_WORDS = total_words if total_words > 0 else 1

    # -------- IMAGES --------
    for page_num, page in enumerate(doc):
        for img in page.get_images(full=True):
            xref = img[0]
            pix = fitz.Pixmap(doc, xref)

            if pix.n >= 4:
                pix = fitz.Pixmap(fitz.csRGB, pix)

            img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, pix.n
            )

            rect = page.rect
            blocks = page.get_text("blocks")

            context = ""
            for b in blocks:
                if abs(b[1] - rect.y1) < 300:
                    context += " " + b[4]

            images.append({
                "image": img_data,
                "context": context.lower()
            })

    # -------- TABLES --------
    for page_num, page in enumerate(doc):
        blocks = page.get_text("blocks")

        for b in blocks:
            txt = b[4].lower()

            if len(txt) > 80 and (
                "|" in txt or "table" in txt or "no." in txt
            ):
                rect = fitz.Rect(b[:4])
                pix = page.get_pixmap(clip=rect)

                img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                    pix.height, pix.width, pix.n
                )

                tables.append({
                    "image": img_data,
                    "context": txt
                })

    # -------- EMBEDDINGS --------
    if text_chunks:
        emb = model.encode(text_chunks)
        text_index = faiss.IndexFlatL2(emb.shape[1])
        text_index.add(np.array(emb))

    if images:
        emb = model.encode([i["context"] for i in images])
        image_index = faiss.IndexFlatL2(emb.shape[1])
        image_index.add(np.array(emb))

    if tables:
        emb = model.encode([t["context"] for t in tables])
        table_index = faiss.IndexFlatL2(emb.shape[1])
        table_index.add(np.array(emb))

    return "PDF loaded successfully ✅"

# =========================
# IMPORTANCE
# =========================
def word_importance(word):
    freq = WORD_FREQ.get(word, 1)
    return math.log((TOTAL_WORDS + 1) / freq)

# =========================
# MATCHING
# =========================
def strong_match(query, text):
    score = 0
    q = query.lower()

    if q in text:
        score += 5

    for w in q.split():
        if w in text:
            score += 1

    return score

def smart_score(query, text):
    base = strong_match(query, text)
    text_lower = text.lower()

    words = query.lower().split()
    dynamic = 0

    for w in words:
        if w in text_lower:
            dynamic += word_importance(w)

    return base + dynamic

# =========================
# SEARCH
# =========================
def search(query):
    results_text = []
    results_images = []
    results_tables = []

    # TEXT
    if text_index:
        q_emb = model.encode([query])
        _, idx = text_index.search(np.array(q_emb), 5)

        for i in idx[0]:
            txt = text_chunks[i]
            if smart_score(query, txt) > 2:
                results_text.append(txt)

    # IMAGES
    if image_index:
        q_emb = model.encode([query])
        _, idx = image_index.search(np.array(q_emb), 6)

        scored = []
        for i in idx[0]:
            item = images[i]
            score = smart_score(query, item["context"])
            if score > 2:
                scored.append((score, item))

        scored.sort(reverse=True, key=lambda x: x[0])
        results_images = [x[1]["image"] for x in scored[:3]]

    # TABLES
    if table_index:
        q_emb = model.encode([query])
        _, idx = table_index.search(np.array(q_emb), 6)

        scored = []
        for i in idx[0]:
            item = tables[i]
            score = smart_score(query, item["context"])
            if score > 2:
                scored.append((score, item))

        scored.sort(reverse=True, key=lambda x: x[0])
        results_tables = [x[1]["image"] for x in scored[:3]]

    return results_text, results_images, results_tables

# =========================
# UI
# =========================
with gr.Blocks() as demo:
    gr.Markdown("# 📄 PDF Assistant")

    with gr.Row():
        with gr.Column(scale=1):
            file = gr.File(label="Upload PDF")
            load_btn = gr.Button("Load PDF")
            status = gr.Textbox(label="Status")

            query = gr.Textbox(label="Ask")
            search_btn = gr.Button("Search")

        with gr.Column(scale=2):
            text_output = gr.Textbox(label="Key Points", lines=15)
            img_output = gr.Gallery(label="Images")
            table_output = gr.Gallery(label="Tables")

    load_btn.click(process_pdf, inputs=file, outputs=status)

    def run_search(q):
        txt, imgs, tbls = search(q)
        return "\n\n".join(txt), imgs, tbls

    search_btn.click(run_search, inputs=query,
                     outputs=[text_output, img_output, table_output])

# =========================
# 🚀 RENDER FIX ONLY
# =========================
port = int(os.environ.get("PORT", 10000))

demo.launch(
    server_name="0.0.0.0",
    server_port=port
)
