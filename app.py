import gradio as gr
import fitz
import numpy as np
import re
import math
from collections import Counter
from sentence_transformers import SentenceTransformer
from PIL import Image
import io
import os  # ✅ only added

# =========================
# MODEL
# =========================
model = SentenceTransformer("all-MiniLM-L6-v2")

# =========================
# STORAGE
# =========================
text_chunks = []
images = []
tables = []

text_embeddings = None
image_embeddings = None
table_embeddings = None

WORD_FREQ = None
TOTAL_WORDS = 1

# =========================
# CLEAN
# =========================
def clean_text(text):
    return " ".join(text.split())

# =========================
# SAFE IMAGE CONVERT
# =========================
def pix_to_pil(pix):
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    
    max_size = 1200
    if img.width > max_size or img.height > max_size:
        img.thumbnail((max_size, max_size))
    
    return img

# =========================
# PROCESS PDF
# =========================
def process_pdf(file):
    global text_chunks, images, tables
    global text_embeddings, image_embeddings, table_embeddings
    global WORD_FREQ, TOTAL_WORDS

    text_chunks, images, tables = [], [], []

    if isinstance(file, str):
        doc = fitz.open(file)
    else:
        doc = fitz.open(stream=file.read(), filetype="pdf")

    # -------- TEXT --------
    for page in doc:
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
    TOTAL_WORDS = max(total_words, 1)

    # -------- IMAGES --------
    for page in doc:
        for img in page.get_images(full=True):
            xref = img[0]
            pix = fitz.Pixmap(doc, xref)

            if pix.n >= 4:
                pix = fitz.Pixmap(fitz.csRGB, pix)

            pil_img = pix_to_pil(pix)

            context = page.get_text().lower()

            images.append({
                "image": pil_img,
                "context": context
            })

    # -------- TABLES --------
    for page in doc:
        blocks = page.get_text("blocks")
        for b in blocks:
            txt = b[4].lower()

            if len(txt) > 80 and ("table" in txt or "|" in txt):
                rect = fitz.Rect(b[:4])
                pix = page.get_pixmap(clip=rect)

                pil_img = pix_to_pil(pix)

                tables.append({
                    "image": pil_img,
                    "context": txt
                })

    # -------- EMBEDDINGS --------
    if text_chunks:
        text_embeddings = model.encode(text_chunks)

    if images:
        image_embeddings = model.encode([i["context"] for i in images])

    if tables:
        table_embeddings = model.encode([t["context"] for t in tables])

    return "PDF loaded successfully ✅"

# =========================
# SCORING
# =========================
def word_importance(word):
    freq = WORD_FREQ.get(word, 1)
    return math.log((TOTAL_WORDS + 1) / freq)

def smart_score(query, text):
    score = 0
    q = query.lower()

    if q in text:
        score += 5

    for w in q.split():
        if w in text:
            score += word_importance(w)

    return score

# =========================
# SEARCH
# =========================
def search(query):
    q_emb = model.encode([query])[0]

    output = ""

    # -------- TEXT --------
    if text_embeddings is not None:
        scores = np.dot(text_embeddings, q_emb)
        idx = np.argsort(scores)[::-1][:5]

        for i in idx:
            txt = text_chunks[i]
            if smart_score(query, txt) > 2:
                output += f"• {txt}\n\n"

    # -------- TABLES --------
    table_imgs = []
    if table_embeddings is not None:
        scores = np.dot(table_embeddings, q_emb)
        idx = np.argsort(scores)[::-1][:5]

        for i in idx:
            item = tables[i]
            if smart_score(query, item["context"]) > 2:
                table_imgs.append(item["image"])

    # -------- IMAGES --------
    image_imgs = []
    if image_embeddings is not None:
        scores = np.dot(image_embeddings, q_emb)
        idx = np.argsort(scores)[::-1][:6]

        scored = []
        for i in idx:
            item = images[i]
            s = smart_score(query, item["context"])
            if s > 2:
                scored.append((s, item))

        scored.sort(reverse=True, key=lambda x: x[0])

        if scored:
            best = scored[0][0]
            scored = [x for x in scored if x[0] >= best * 0.75]

        image_imgs = [x[1]["image"] for x in scored]

    all_images = table_imgs + image_imgs

    return output, all_images

# =========================
# UI
# =========================
with gr.Blocks() as demo:
    gr.Markdown("# 📄 PDF Assistant")

    file = gr.File()
    load_btn = gr.Button("Load PDF")
    status = gr.Textbox()

    query = gr.Textbox()
    search_btn = gr.Button("Search")

    text_output = gr.Textbox(label="Answer", lines=15)
    gallery = gr.Gallery(label="Results")

    load_btn.click(process_pdf, inputs=file, outputs=status)
    search_btn.click(search, inputs=query, outputs=[text_output, gallery])

# =========================
# 🚀 RENDER FIX ONLY
# =========================
port = int(os.environ.get("PORT", 10000))

demo.launch(
    server_name="0.0.0.0",
    server_port=port
)
