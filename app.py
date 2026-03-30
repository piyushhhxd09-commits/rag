import gradio as gr
import fitz
import io
from PIL import Image
import base64
import os
import tempfile

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
    return len(set(a.split()) & set(b.split()))

# 🔥 NEW: importance scoring
def importance_score(text, query):
    score = overlap(query, text)

    # longer meaningful sentences
    if len(text.split()) > 8:
        score += 2

    # definition-like sentences
    if "is defined as" in text or "refers to" in text:
        score += 3

    # contains numbers (often important)
    if any(char.isdigit() for char in text):
        score += 1

    return score

def encode_image(pix):
    if pix.n >= 4:
        pix = fitz.Pixmap(fitz.csRGB, pix)

    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    img = img.resize((500, int(500 * img.height / img.width)))

    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=80)

    return base64.b64encode(buffer.getvalue()).decode()

# =========================
# PDF PROCESSING
# =========================

def process_pdf(file):
    global TEXT_DB, TABLE_DB, IMAGE_DB
    TEXT_DB, TABLE_DB, IMAGE_DB = [], [], []

    try:
        if isinstance(file, str):
            doc = fitz.open(file)
        else:
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

                TEXT_DB.append({
                    "text": norm,
                    "page": page_num
                })

                if (
                    "|" in text
                    or text.count("  ") > 3
                    or (len(text.split()) > 15 and text.count(" ") / len(text.split()) > 1.5)
                ):
                    TABLE_DB.append({
                        "text": norm,
                        "page": page_num
                    })

            for img in page.get_images(full=True):
                xref = img[0]
                pix = fitz.Pixmap(doc, xref)

                IMAGE_DB.append({
                    "image": encode_image(pix),
                    "page": page_num,
                    "caption": page_text[:500],
                    "keywords": " ".join(page_text.split()[:20])
                })

        return "PDF loaded successfully"

    except Exception as e:
        return f"Error: {str(e)}"

# =========================
# SEARCH (IMPROVED TEXT)
# =========================

def search(query):
    if not query.strip():
        return "Enter a query", ""

    query = normalize(query)
    query = expand_query(query)

    text_results = []
    table_results = []
    image_results = []

    for item in TEXT_DB:
        score = importance_score(item["text"], query)
        if score > 0:
            text_results.append((score, item))

    for item in TABLE_DB:
        score = overlap(query, item["text"])
        if score > 0:
            table_results.append((score, item))

    for item in IMAGE_DB:
        score = overlap(query, item["caption"] + " " + item["keywords"])
        if score > 0:
            image_results.append((score, item))

    text_results.sort(reverse=True, key=lambda x: x[0])
    table_results.sort(reverse=True, key=lambda x: x[0])
    image_results.sort(reverse=True, key=lambda x: x[0])

    # 🔥 remove duplicates
    seen = set()
    final_text = []
    for _, item in text_results:
        if item["text"] not in seen:
            final_text.append(item["text"])
            seen.add(item["text"])
        if len(final_text) >= 5:
            break

    table_results = table_results[:3]
    image_results = image_results[:4]

    output = ""

    if final_text:
        output += "### Key Explanation\n"
        for t in final_text:
            output += f"- {t}\n"

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
# LAUNCH
# =========================

app.launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 7860))
)
