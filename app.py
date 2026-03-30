import gradio as gr
import fitz
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from PIL import Image
import io

# =========================
# MODEL
# =========================
model = SentenceTransformer("all-MiniLM-L6-v2")

TEXT_DB = []
IMG_DB = []
text_index = None

# =========================
# IMPORTANCE SCORE
# =========================
def importance_score(text, query):
    score = 0
    text_l = text.lower()
    query_l = query.lower()

    for w in query_l.split():
        if w in text_l:
            score += 3

    if "defined" in text_l or "is" in text_l:
        score += 2

    if any(char.isdigit() for char in text_l):
        score += 1

    if len(text.split()) > 12:
        score += 2

    return score

# =========================
# QUERY FILTER
# =========================
def filter_relevant(query):
    keywords = query.lower().split()

    filtered = [
        t for t in TEXT_DB
        if any(k in t["text"].lower() for k in keywords)
    ]

    return filtered if filtered else TEXT_DB

# =========================
# 🔥 DEEP SEARCH
# =========================
def deep_search(query):
    global text_index

    query_vec = model.encode([query])
    D, I = text_index.search(np.array(query_vec), 25)  # 🔥 increased

    filtered = filter_relevant(query)

    collected = []

    for idx in I[0]:
        if idx < len(TEXT_DB):
            text = TEXT_DB[idx]["text"]

            # only keep filtered ones
            if text in [f["text"] for f in filtered]:
                score = importance_score(text, query)
                if score > 2:
                    collected.append((score, text))

    # sort
    collected = sorted(collected, reverse=True)

    # remove duplicates
    seen = set()
    final = []
    for _, t in collected:
        if t not in seen:
            final.append(t)
            seen.add(t)
        if len(final) >= 10:
            break

    return final

# =========================
# FORMAT OUTPUT
# =========================
def format_answer(texts):
    return "\n".join([f"- {t}" for t in texts])

# =========================
# PROCESS PDF
# =========================
def process_pdf(file):
    global TEXT_DB, IMG_DB, text_index

    TEXT_DB = []
    IMG_DB = []

    doc = fitz.open(file.name)

    for page_num, page in enumerate(doc):

        blocks = page.get_text("blocks")

        for b in blocks:
            text = b[4].strip()

            if len(text) > 40:
                TEXT_DB.append({
                    "text": text,
                    "page": page_num
                })

        for img in page.get_images(full=True):
            xref = img[0]
            base = doc.extract_image(xref)

            try:
                img_bytes = base["image"]
                image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                IMG_DB.append(image)
            except:
                continue

    # embeddings
    embeddings = model.encode([t["text"] for t in TEXT_DB])
    dim = embeddings.shape[1]

    text_index = faiss.IndexFlatL2(dim)
    text_index.add(np.array(embeddings))

    return "PDF loaded successfully"

# =========================
# SEARCH
# =========================
def search(query):
    if not query.strip():
        return "Enter query", []

    results = deep_search(query)

    if not results:
        return "No relevant information found", []

    answer = format_answer(results)

    return answer, IMG_DB[:3]

# =========================
# UI
# =========================
with gr.Blocks() as app:
    gr.Markdown("# 📄 PDF Assistant")

    file = gr.File(label="Upload PDF")
    load_btn = gr.Button("Load PDF")
    status = gr.Textbox(label="Status")

    query = gr.Textbox(label="Ask")
    search_btn = gr.Button("Search")

    output_text = gr.Textbox(label="Answer")
    output_img = gr.Gallery(label="Images")

    load_btn.click(process_pdf, inputs=file, outputs=status)
    search_btn.click(search, inputs=query, outputs=[output_text, output_img])

app.launch(server_name="0.0.0.0", server_port=7860)
