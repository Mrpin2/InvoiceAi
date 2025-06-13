import streamlit as st st.set_page_config(layout="wide")  # MUST be first

from PIL import Image import fitz  # PyMuPDF import io import pandas as pd import base64 import requests import traceback from streamlit_lottie import st_lottie from openai import OpenAI import tempfile import os

---------- Load Animations ----------

hello_lottie = "https://raw.githubusercontent.com/Mrpin2/InvoiceAi/refs/heads/main/Animation%20-%201749845212531.json" completed_lottie = "https://raw.githubusercontent.com/Mrpin2/InvoiceAi/refs/heads/main/Animation%20-%201749845303699.json"

def load_lottie_json_safe(url): try: r = requests.get(url) r.raise_for_status() return r.json() except Exception: return None

hello_json = load_lottie_json_safe(hello_lottie) completed_json = load_lottie_json_safe(completed_lottie)

---------- UI HEADER ----------

if "files_uploaded" not in st.session_state: if hello_json: st_lottie(hello_json, height=200, key="hello")

st.markdown("<h2 style='text-align: center;'>📄 AI Invoice Extractor (ChatGPT)</h2>", unsafe_allow_html=True) st.markdown("Upload scanned PDF invoices and extract clean finance data using ChatGPT Vision") st.markdown("---")

---------- Table Columns ----------

columns = [ "File Name", "Vendor Name", "Invoice No", "GSTIN", "HSN/SAC", "Buyer Name", "Place of Supply", "Invoice Date", "Expense Ledger", "GST Type", "Tax Rate", "Basic Amount", "CGST", "SGST", "IGST", "Total Payable", "Narration", "GST Input Eligible", "TDS Applicable", "TDS Rate" ]

---------- Session State ----------

if "processed_results" not in st.session_state: st.session_state["processed_results"] = {} if "processing_status" not in st.session_state: st.session_state["processing_status"] = {} if "summary_rows" not in st.session_state: st.session_state["summary_rows"] = []

---------- Sidebar Config ----------

st.sidebar.header("🔐 AI Config") passcode = st.sidebar.text_input("Admin Passcode", type="password") admin_unlocked = passcode == "Essenbee"

openai_api_key = None if admin_unlocked: st.sidebar.success("🔓 Admin access granted.") openai_api_key = st.secrets.get("OPENAI_API_KEY") if not openai_api_key: st.sidebar.error("OPENAI_API_KEY missing in secrets.") st.stop() else: openai_api_key = st.sidebar.text_input("🔑 Enter your OpenAI API Key", type="password") if not openai_api_key: st.sidebar.warning("Please enter a valid API key to continue.") st.stop()

client = OpenAI(api_key=openai_api_key)

---------- Extraction Prompt ----------

main_prompt = """ You are an invoice-extraction assistant.

Reply exactly one of: • NOT AN INVOICE — if the document is clearly not an invoice • a single comma-separated line in the field order below

Field order Vendor Name, Invoice No, Tax ID (GSTIN/EIN/VAT), HSN/SAC, Buyer Name, Place of Supply, Invoice Date, Expense Ledger, Tax Type, Tax Rate %, Basic Amount, CGST, SGST, IGST/Sales Tax, Total Payable, Narration, GST Input Eligible (Yes/No/Uncertain), TDS Applicable (Yes/No/Section/Uncertain), TDS Rate

Rules

DATES
• If the vendor is Indian (Indian address or a valid GSTIN) → output date as DD/MM/YYYY
• Otherwise keep the invoice’s visible date format (MM/DD/YYYY or YYYY-MM-DD)

TAX ID VALIDATION
• GSTIN ⇒ exactly 15 alphanumeric characters; if length ≠ 15 or format wrong → MISSING
• EIN ⇒ 9 digits in the form NN-NNNNNNN
• VAT ⇒ use if explicitly labelled VAT
• Never output GSTIN when the vendor country is not India; output MISSING instead.
• If multiple tax IDs are present, choose the one that matches the vendor’s country.

TAX TYPE & BREAKDOWN
• India → Tax Type = GST and extract CGST, SGST, IGST separately
• International → Tax Type = VAT or Sales Tax and put total tax in IGST/Sales Tax column

HSN/SAC & SERVICE DETECTION
• If code is 8 digits and starts with “99” OR description contains “service”/“consulting”/“professional” → treat as Service (SAC)
• Otherwise treat as Goods (HSN).
• Leave the HSN/SAC cell blank if no code and nothing can be inferred.

EXPENSE LEDGER
• Suggest a ledger based on narration and item type, e.g., “Professional Fees”, “Cloud Hosting”, “Software Subscription”.

MISSING DATA
• Required & not found → MISSING
• Optional & not found → empty string ""
• Amounts that are zero or blank → 0.0

OTHER GUIDELINES
• Ignore logos, repeat headers/footers, and boiler-plate text.
• Invoice No must be unique; if only the word “Invoice” appears → MISSING
• Extract only what is visibly present; never invent data. """

def is_placeholder_row(text): placeholder_keywords = ["Vendor Name", "Invoice No", "Invoice Date", "Expense Ledger"] return all(x.lower() in text.lower() for x in placeholder_keywords)

def convert_pdf_first_page(pdf_bytes): doc = fitz.open(stream=pdf_bytes, filetype="pdf") page = doc.load_page(0) pix = page.get_pixmap(dpi=300) return Image.open(io.BytesIO(pix.tobytes("png")))

---------- PDF Upload ----------

uploaded_files = st.file_uploader("📤 Upload scanned invoice PDFs", type=["pdf"], accept_multiple_files=True)

if uploaded_files: st.session_state["files_uploaded"] = True

total_files = len(uploaded_files)
completed_count = 0

for idx, file in enumerate(uploaded_files):
    file_name = file.name

    if file_name in st.session_state["processed_results"]:
        continue

    st.markdown(f"**Processing file: {file_name} ({idx+1}/{total_files})**")
    st.session_state["processing_status"][file_name] = "⏳ Pending..."
    st.info(f"{file_name}: ⏳ Pending...")

    temp_file_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.read())
            temp_file_path = tmp.name

        pdf_data = open(temp_file_path, "rb").read()
        first_image = convert_pdf_first_page(pdf_data)

        with st.spinner("🧠 Extracting data using ChatGPT..."):
            img_buf = io.BytesIO()
            first_image.save(img_buf, format="PNG")
            img_buf.seek(0)
            base64_image = base64.b64encode(img_buf.read()).decode()

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

            if csv_line.upper().startswith("NOT AN INVOICE") or is_placeholder_row(csv_line):
                result_row = [file_name] + ["NOT AN INVOICE"] + ["-"] * (len(columns) - 2)
            else:
                matched = False
                for line in csv_line.strip().split("\n"):
                    try:
                        row = [x.strip().strip('"') for x in line.split(",")]
                        if len(row) >= len(columns) - 1:
                            result_row = [file_name] + row[:len(columns) - 1]
                            matched = True
                            break
                    except Exception:
                        pass
                if not matched:
                    result_row = [file_name] + ["NOT AN INVOICE"] + ["-"] * (len(columns) - 2)

            st.session_state["processed_results"][file

