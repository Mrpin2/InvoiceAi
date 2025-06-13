import streamlit as st
st.set_page_config(layout="wide")  # MUST be the first Streamlit call

from PIL import Image
import fitz  # PyMuPDF
import io
import pandas as pd
import base64
import requests
import traceback
from streamlit_lottie import st_lottie
from openai import OpenAI
import tempfile
import os

# ---------- Load Animations ----------
hello_lottie = "https://raw.githubusercontent.com/Mrpin2/InvoiceAi/refs/heads/main/Animation%20-%201749845212531.json"
completed_lottie = "https://raw.githubusercontent.com/Mrpin2/InvoiceAi/refs/heads/main/Animation%20-%201749845303699.json"

def load_lottie_json_safe(url: str):
    """Return JSON for a Lottie animation, or None on failure."""
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None

hello_json = load_lottie_json_safe(hello_lottie)
completed_json = load_lottie_json_safe(completed_lottie)

# ---------- UI HEADER ----------
if "files_uploaded" not in st.session_state:
    if hello_json:
        st_lottie(hello_json, height=200, key="hello")

st.markdown(
    "<h2 style='text-align: center;'>📄 AI Invoice Extractor (ChatGPT Vision)</h2>",
    unsafe_allow_html=True,
)
st.markdown("Upload scanned PDF invoices and extract structured finance data.")
st.markdown("---")

# ---------- Table Columns ----------
columns = [
    "File Name", "Vendor Name", "Invoice No", "GSTIN", "HSN/SAC", "Buyer Name",
    "Place of Supply", "Invoice Date", "Expense Ledger", "GST Type", "Tax Rate",
    "Basic Amount", "CGST", "SGST", "IGST", "Total Payable", "Narration",
    "GST Input Eligible", "TDS Applicable", "TDS Rate"
]

# ---------- Session State ----------
for key in ("processed_results", "processing_status", "summary_rows"):
    st.session_state.setdefault(key, {} if "results" in key else [])

# ---------- Sidebar Config ----------
st.sidebar.header("🔐 AI Config")
passcode = st.sidebar.text_input("Admin Passcode", type="password")
admin_unlocked = passcode == "Essenbee"

# API key input
openai_api_key: str | None
if admin_unlocked:
    st.sidebar.success("🔓 Admin access granted.")
    openai_api_key = st.secrets.get("OPENAI_API_KEY")
    if not openai_api_key:
        st.sidebar.error("OPENAI_API_KEY missing in secrets.")
        st.stop()
else:
    openai_api_key = st.sidebar.text_input("🔑 Enter your OpenAI API Key", type="password")
    if not openai_api_key:
        st.sidebar.warning("Please enter a valid API key to continue.")
        st.stop()

client = OpenAI(api_key=openai_api_key)

# ---------- Extraction Prompt ----------
main_prompt = """
You are an invoice-extraction assistant.

Reply **exactly one** of:
• NOT AN INVOICE — if the document is clearly **not** an invoice  
• A single comma-separated line in the field order below, with **no headers or extra text**

Field order  
Vendor Name, Invoice No, Tax ID (GSTIN/EIN/VAT), HSN/SAC, Buyer Name, Place of Supply,
Invoice Date, Expense Ledger, Tax Type, Tax Rate %, Basic Amount, CGST, SGST,
IGST/Sales Tax, Total Payable, Narration, GST Input Eligible (Yes/No/Uncertain),
TDS Applicable (Yes/No/Section/Uncertain), TDS Rate

Rules
DATES  
• Indian vendor (Indian address or valid GSTIN) → convert to DD/MM/YYYY  
• Otherwise keep the visible format (MM/DD/YYYY or YYYY-MM-DD)

TAX ID  
• GSTIN → exactly 15 alphanumeric chars; else MISSING  
• EIN → 9-digit NN-NNNNNNN; else MISSING  
• VAT → use if explicitly labelled  
• Never output GSTIN for non-Indian vendors

TAX TYPE & BREAKDOWN  
• India → GST; extract CGST, SGST, IGST separately  
• International → VAT or Sales Tax; put total tax in IGST/Sales Tax column

HSN/SAC  
• Code starts “99” **or** description mentions “service/consulting/professional” → Service (SAC)  
• Otherwise → Goods (HSN)  
• Leave blank if unknown

EXPENSE LEDGER  
• Suggest based on narration, e.g. “Professional Fees”, “Cloud Hosting”, etc.

MISSING DATA  
• Required & not found → MISSING  
• Optional & not found → empty string ""  
• Zero/blank amounts → 0.0

OTHER  
• Ignore logos, footers, boilerplate  
• Invoice No must be unique; if only word “Invoice” appears → MISSING  
• Extract only visible data; never invent
"""

# ---------- Helper Functions ----------
def is_placeholder_row(text: str) -> bool:
    """Detects if the model returned the column header line by mistake."""
    placeholder_keywords = ("Vendor Name", "Invoice No", "Invoice Date", "Expense Ledger")
    return all(k.lower() in text.lower() for k in placeholder_keywords)

def convert_pdf_first_page(pdf_bytes: bytes) -> Image.Image:
    """Render first page of PDF to a PIL Image."""
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        page = doc.load_page(0)
        pix = page.get_pixmap(dpi=300)
    return Image.open(io.BytesIO(pix.tobytes("png")))

# ---------- PDF Upload ----------
uploaded_files = st.file_uploader(
    "📤 Upload scanned invoice PDFs", type=["pdf"], accept_multiple_files=True
)

if uploaded_files:
    st.session_state["files_uploaded"] = True
    total_files = len(uploaded_files)
    completed_count = 0

    for idx, file in enumerate(uploaded_files):
        file_name = file.name

        # Skip if already processed this session
        if file_name in st.session_state["processed_results"]:
            continue

        st.markdown(f"**Processing file {idx+1}/{total_files}: {file_name}**")
        st.session_state["processing_status"][file_name] = "⏳ Pending..."
        st.info(f"{file_name}: ⏳ Pending...")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.read())
            temp_path = tmp.name

        try:
            pdf_data = open(temp_path, "rb").read()
            first_image = convert_pdf_first_page(pdf_data)

            with st.spinner("🧠 Extracting data with ChatGPT Vision…"):
                img_buf = io.BytesIO()
                first_image.save(img_buf, format="PNG")
                base64_image = base64.b64encode(img_buf.getvalue()).decode()

                chat_prompt = [
                    {"role": "system", "content": "You are a finance assistant."},
                    {"role": "user", "content": [
                        {"type": "text", "text": main_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                    ]}
                ]

                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=chat_prompt,
                    max_tokens=1000
                )
                csv_line = response.choices[0].message.content.strip()

                # Parse response
                if csv_line.upper().startswith("NOT AN INVOICE") or is_placeholder_row(csv_line):
                    result_row = [file_name] + ["NOT AN INVOICE"] + ["-"] * (len(columns) - 2)
                else:
                    result_row = [file_name] + [
                        x.strip().strip('"') for x in csv_line.split(",")
                    ][:len(columns) - 1]

                st.session_state["processed_results"][file_name] = result_row
                st.session_state["processing_status"][file_name] = "✅ Done"
                completed_count += 1
                st.success(f"{file_name}: ✅ Done")
                st.info(f"🤖 Processed {completed_count} / {total_files}")

        except Exception as e:
            st.session_state["processed_results"][file_name] = \
                [file_name] + ["NOT AN INVOICE"] + ["-"] * (len(columns) - 2)
            st.session_state["processing_status"][file_name] = "❌ Error"
            st.error(f"❌ Error processing {file_name}: {e}")
            st.text_area(f"Traceback ({file_name})", traceback.format_exc())

        finally:
            os.remove(temp_path)

# ---------- Display Results ----------
results = list(st.session_state["processed_results"].values())
if results:
    if completed_json:
        st_lottie(completed_json, height=200, key="done_animation")

    st.markdown(
        "<h3 style='text-align: center;'>🎉 All invoices processed!</h3>",
        unsafe_allow_html=True,
    )

    df = pd.DataFrame(results, columns=columns)
    df.insert(0, "S. No", range(1, len(df) + 1))
    st.dataframe(df, use_container_width=True)

    csv_data = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Download results as CSV", csv_data, "invoice_results.csv", "text/csv"
    )

    st.markdown("---")
    if st.session_state["summary_rows"]:
        st.balloons()
else:
    st.info("Upload one or more scanned invoices to get started.")
