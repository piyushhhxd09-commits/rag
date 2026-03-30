import gradio as gr
import fitz  # PyMuPDF
import io
from PIL import Image
import base64
import tempfile
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

def overlap(a, b):
    a_words = set(a.split())
    b_words = set(b.split())
    return len(a_words & b_words)

def encode_image(pix):
    img = Image.open(io.BytesIO(pix.tobytes("jpeg")))
    img = img.resize((500, int(500 * img.height / img.width)))
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=60)
    return base64.b64encode(buffer.getvalue()).decode()

# =========================
# PDF PROCESSING (FIXED FOR RENDER)
# =========================

def process_pdf(file):
    global TEXT_DB, TABLE_DB, IMAGE_DB
    TEXT_DB, TABLE_DB, IMAGE_DB = [], [], []

    # 🔥 FIX: Save temp file (Render-safe)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file.read())
        tmp_path = tmp.name

    doc = fitz.open(tmp_path)

    for page_num, page in enumerate(doc):
        blocks = page.get_text("blocks")

        page_text = ""

        for b in blocks:
            text = b[4].strip()
            if not text:
                continue

            norm = normalize(text)
            page_text += " " + norm

            # TEXT
            TEXT_DB.append({
                "text": norm,
                "page": page_num
            })

            # TABLE DETECTION (UNCHANGED)
            if (
                "|" in text
                or text.count("  ") > 3
                or (len(text.split()) > 15 and text.count(" ") / len(text.split()) > 1.5)
            ):
                TABLE_DB.append({
                    "text": norm,
                    "page": page_num
                })

        # IMAGE EXTRACTION (UNCHANGED)
        for img in page.get_images(full=True):
            xref = img[0]
            pix = fitz.Pixmap(doc, xref)

            if pix.n > 4:
                pix = fitz.Pixmap(fitz.csRGB, pix)

            img_base64 = encode_image(pix)

            IMAGE_DB.append({
                "image": img_base64,
                "page": page_num,
                "caption": page_text[:500],
                "keywords": " ".join(page_text.split()[:20])
            })

    return "PDF loaded successfully"

# =========================
# SEARCH (UNCHANGED)
# =========================

def search(query):
    if not query.strip():
        return "Enter a query", ""

    query = normalize(query)
    query = expand_query(query)

    text_results = []
    table_results = []
    image_results = []

    # TEXT SEARCH
    for item in TEXT_DB:
        score = overlap(query, item["text"])
        if score > 0:
            text_results.append((score, item))

    # TABLE SEARCH
    for item in TABLE_DB:
        score = overlap(query, item["text"])
        if score > 0:
            table_results.append((score, item))

    # IMAGE SEARCH
    for item in IMAGE_DB:
        score = overlap(query, item["caption"] + " " + item["keywords"])
        if score > 0:
            image_results.append((score, item))

    # SORT
    text_results.sort(reverse=True, key=lambda x: x[0])
    table_results.sort(reverse=True, key=lambda x: x[0])
    image_results.sort(reverse=True, key=lambda x: x[0])

    # LIMIT RESULTS
    text_results = text_results[:5]
    table_results = table_results[:3]
    image_results = image_results[:4]

    # OUTPUT
    output = ""

    if text_results:
        output += "### Key Points\n"
        for _, item in text_results:
            output += f"- {item['text']}\n"

    if table_results:
        output += "\n### Tables\n"
        for _, item in table_results:
            output += f"{item['text']}\n\n"

    images_html = ""
    for _, item in image_results:
        images_html += f"<img src='data:image/jpeg;base64,{item['image']}' width='500'><br><br>"

    return output, images_html

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

    output_text = gr.Markdown()
    output_images = gr.HTML()

    load_btn.click(process_pdf, inputs=file, outputs=status)
    search_btn.click(search, inputs=query, outputs=[output_text, output_images])

# =========================
# LAUNCH (RENDER SAFE)
# =========================

app.launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 7860))
)
