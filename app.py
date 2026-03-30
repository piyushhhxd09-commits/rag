import gradio as gr
import fitz
import numpy as np
from PIL import Image
import io
import re
import base64
from sentence_transformers import SentenceTransformer
import os  # ✅ only added

# MODEL
model = SentenceTransformer("all-MiniLM-L6-v2")

# STORAGE
text_chunks, images, tables = [], [], []
text_embeddings = image_embeddings = table_embeddings = None
chat_history = []

# SAFE IMAGE
def pix_to_base64(pix):
    try:
        if pix.n >= 4:
            pix = fitz.Pixmap(fitz.csRGB, pix)

        img_bytes = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_bytes))
        img.thumbnail((900, 900))

        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode()
    except:
        return None

# SCORING
def smart_score(query, text):
    q_words = re.findall(r"\w+", query.lower())
    t_words = re.findall(r"\w+", text.lower())

    if not q_words:
        return 0

    exact = sum(1 for w in q_words if w in t_words)
    partial = sum(1 for w in q_words if any(w in tw for tw in t_words))

    density = exact / len(q_words)

    return (exact * 3 + partial) * density

# PROCESS PDF
def process_pdf(file):
    global text_chunks, images, tables
    global text_embeddings, image_embeddings, table_embeddings

    text_chunks, images, tables = [], [], []

    doc = fitz.open(file.name)

    for page in doc:
        blocks = page.get_text("blocks")

        for b in blocks:
            txt = b[4].strip()
            if len(txt) > 40:
                text_chunks.append(txt)

        page_text = page.get_text().lower()

        # IMAGES
        for img in page.get_images(full=True):
            try:
                xref = img[0]
                pix = fitz.Pixmap(doc, xref)
                b64 = pix_to_base64(pix)

                if b64:
                    images.append({"image": b64, "context": page_text})
            except:
                continue

        # TABLES
        if "table" in page_text:
            pix = page.get_pixmap()
            b64 = pix_to_base64(pix)

            if b64:
                tables.append({"image": b64, "context": page_text})

    if text_chunks:
        text_embeddings = model.encode(text_chunks)

    if images:
        image_embeddings = model.encode([i["context"] for i in images])

    if tables:
        table_embeddings = model.encode([t["context"] for t in tables])

    return "✅ PDF loaded"

# SEARCH
def chat_fn(query):
    global chat_history

    q_emb = model.encode([query])[0]

    answer = ""
    imgs = []

    # TEXT
    if text_embeddings is not None:
        scores = np.dot(text_embeddings, q_emb)
        idx = np.argsort(scores)[::-1][:5]

        for i in idx:
            txt = text_chunks[i]
            if smart_score(query, txt) > 2:
                answer += f"• {txt}\n\n"

    # TABLES
    if table_embeddings is not None:
        scores = np.dot(table_embeddings, q_emb)
        idx = np.argsort(scores)[::-1][:5]

        for i in idx:
            if smart_score(query, tables[i]["context"]) > 2:
                imgs.append(tables[i]["image"])

    # IMAGES
    if image_embeddings is not None:
        scores = np.dot(image_embeddings, q_emb)
        idx = np.argsort(scores)[::-1][:6]

        scored = []
        for i in idx:
            s = smart_score(query, images[i]["context"])
            if s > 2:
                scored.append((s, images[i]))

        scored.sort(reverse=True)

        if scored:
            best = scored[0][0]
            scored = [x for x in scored if x[0] >= best * 0.75]

        imgs += [x[1]["image"] for x in scored]

    if not answer and not imgs:
        answer = "The answer is not available in the document."

    content = answer
    for img in imgs:
        content += f'<br><img src="data:image/png;base64,{img}" width="400"/>'

    chat_history.append((query, content))
    return chat_history, ""

# UI
with gr.Blocks() as demo:
    gr.Markdown("# 🤖 PDF Assistant")

    file = gr.File()
    load = gr.Button("Load PDF")
    status = gr.Textbox()

    chatbot = gr.Chatbot(height=500)

    query = gr.Textbox(placeholder="Ask from PDF...")
    send = gr.Button("Send")

    load.click(process_pdf, inputs=file, outputs=status)
    send.click(chat_fn, inputs=query, outputs=[chatbot, query])

# =========================
# 🚀 RENDER FIX ONLY
# =========================
port = int(os.environ.get("PORT", 10000))

demo.launch(
    server_name="0.0.0.0",
    server_port=port
)
