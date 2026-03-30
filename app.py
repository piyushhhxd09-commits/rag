import gradio as gr
import fitz
import io
from PIL import Image
import base64
import os

# =========================
# STORAGE
# =========================
TEXT_DB = []
TABLE_DB = []
IMAGE_DB = []

# =========================
# HELPERS
# =========================

def normalize(text):
    return text.lower().strip()

def expand_query(query):
    words = query.split()
    expanded = set(words)

    for w in words:
        expanded.add(w)
        expanded.add(w + "s")
        expanded.add(w.rstrip("s"))
        expanded.add(w[:4])

    return " ".join(expanded)

def strong_match(query, text):
    q_words = set(query.split())
    t_words = set(text.split())

    overlap = len(q_words & t_words)

    soft = 0
    for q in q_words:
        for t in t_words:
            if q in t or t in q:
                soft += 1

    return overlap + (soft * 0.3)

def encode_image(pix):
    if pix.colorspace is None or pix.colorspace.n != 3:
        pix = fitz.Pixmap(fitz.csRGB, pix)

    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    img = img.resize((500, int(500 * img.height / img.width)))

    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=85)

    return base64.b64encode(buffer.getvalue()).decode()

# =========================
# PDF PROCESSING
# =========================

def process_pdf(file):
    global TEXT_DB, TABLE_DB, IMAGE_DB
    TEXT_DB, TABLE_DB, IMAGE_DB = [], [], []

    try:
        if hasattr(file, "name"):
            doc = fitz.open(file.name)
        elif isinstance(file, str):
            doc = fitz.open(file)
        else:
            doc = fitz.open(stream=file.read(), filetype="pdf")
    except Exception as e:
        return f"Error: {str(e)}"

    for page_num, page in enumerate(doc):

        blocks = page.get_text("blocks")
        page_text = ""

        # -------- TEXT + TABLE CAPTION --------
        for b in blocks:
            x0, y0, x1, y1, text, *_ = b
            text = text.strip()
            if not text:
                continue

            norm = normalize(text)
            page_text += " " + norm

            TEXT_DB.append({
                "text": norm,
                "page": page_num
            })

            if "table" in norm:
                TABLE_DB.append({
                    "text": norm,
                    "page": page_num,
                    "bbox": (x0, y0, x1, y1),
                    "type": "caption"
                })

        # -------- IMAGE --------
        for img in page.get_images(full=True):
            xref = img[0]
            pix = fitz.Pixmap(doc, xref)

            IMAGE_DB.append({
                "image": encode_image(pix),
                "page": page_num,
                "context": page_text[:500]
            })

        # -------- TABLE REGION --------
        drawings = page.get_drawings()

        for d in drawings:
            rect = fitz.Rect(d["rect"])

            if rect.width > 200 and rect.height > 100:
                pix = page.get_pixmap(clip=rect)

                TABLE_DB.append({
                    "image": encode_image(pix),
                    "page": page_num,
                    "type": "table_image",
                    "context": page_text[:500]
                })

    return "PDF loaded successfully"

# =========================
# SEARCH
# =========================

def search(query):
    if not query.strip():
        return "Enter query", ""

    query = normalize(query)
    expanded = expand_query(query)

    base_threshold = max(2, len(query.split()) * 0.7)

    text_results = []
    table_results = []
    image_results = []

    # -------- TEXT --------
    for item in TEXT_DB:
        score = strong_match(expanded, item["text"])
        if score >= base_threshold:
            text_results.append((score, item))

    # -------- TABLE --------
    for item in TABLE_DB:
        if item.get("type") == "caption":
            score = strong_match(expanded, item["text"])
        else:
            score = strong_match(expanded, item["context"])

        if score >= 1.5:
            table_results.append((score, item))

    # -------- IMAGE --------
    for item in IMAGE_DB:
        score = strong_match(expanded, item["context"])

        if score >= base_threshold + 1:
            image_results.append((score, item))

    text_results.sort(reverse=True, key=lambda x: x[0])
    table_results.sort(reverse=True, key=lambda x: x[0])
    image_results.sort(reverse=True, key=lambda x: x[0])

    # -------- OUTPUT --------
    output = ""

    if text_results:
        output += "### Key Points\n"
        for _, item in text_results[:3]:
            output += f"- {item['text']}\n"

    if table_results:
        output += "\n### Tables\n"

        for _, item in table_results:
            if item.get("type") == "table_image":
                output += f"<img src='data:image/jpeg;base64,{item['image']}' width='500'><br><br>"
            else:
                output += f"{item['text']}\n\n"

    images_html = ""
    for _, item in image_results:
        images_html += f"<img src='data:image/jpeg;base64,{item['image']}' width='500'><br><br>"

    return output, images_html

# =========================
# UI
# =========================

with gr.Blocks(theme=gr.themes.Soft()) as app:

    gr.Markdown("# 📄 PDF Assistant")

    with gr.Row():
        with gr.Column():
            file = gr.File(label="Upload PDF")
            load_btn = gr.Button("Load PDF")
            status = gr.Textbox(label="Status")

            query = gr.Textbox(label="Ask")
            search_btn = gr.Button("Search")

        with gr.Column():
            output_text = gr.Markdown()
            output_images = gr.HTML()

    load_btn.click(process_pdf, inputs=file, outputs=status)
    search_btn.click(search, inputs=query, outputs=[output_text, output_images])

# =========================
# 🚀 RENDER FIX ONLY
# =========================

port = int(os.environ.get("PORT", 10000))

app.launch(
    server_name="0.0.0.0",
    server_port=port
)
