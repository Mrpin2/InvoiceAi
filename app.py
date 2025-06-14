import streamlit as st 
st.set_page_config(layout="wide")  # MUST be first

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

def load_lottie_json_safe(url):
    try:
        r = requests.get(url)
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

st.markdown("<h2 style='text-align: center;'>📄 AI Invoice Extractor (ChatGPT)</h2>", unsafe_allow_html=True)
st.markdown("Upload scanned PDF invoices and extract clean finance data using ChatGPT Vision")
st.markdown("---")

# ---------- Table Columns ----------
columns = [
    "Vendor Name", "Invoice No", "GSTIN", "HSN/SAC", "Buyer Name", "Place of Supply", "Invoice Date", "Expense Ledger",
    "GST Type", "Tax Rate", "Basic Amount", "CGST", "SGST", "IGST",
    "Total Payable", "Narration", "GST Input Eligible", "TDS Applicable", "TDS Rate"
]

# ---------- Session State ----------
if "processed_results" not in st.session_state:
    st.session_state["processed_results"] = {}
if "processing_status" not in st.session_state:
    st.session_state["processing_status"] = {}
if "summary_rows" not in st.session_state:
    st.session_state["summary_rows"] = []

# ---------- Sidebar Config ----------
st.sidebar.header("🔐 AI Config")
passcode = st.sidebar.text_input("Admin Passcode", type="password")
admin_unlocked = passcode == "Essenbee"

openai_api_key = None
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
You are an intelligent invoice assistant. Analyze the attached document and extract structured invoice data only if it is a valid invoice. If it's clearly not an invoice (e.g., bank statement or request letter), return exactly:
NOT AN INVOICE

If valid, extract and return the following details clearly and precisely, adapting to the invoice's country and structure:

1. Date Format: Normalize all dates to DD/MM/YYYY regardless of source region (e.g., 06/02/2025 even if in US MM/DD/YYYY format).
2. Identifiers:
   - India: Extract GSTIN, SAC/HSN codes.
   - US/EU: Extract EIN/VAT if shown.
   - If no tax ID is visible, leave blank.
3. Amounts:
   - Total Payable (final invoice value)
   - Basic Amount (before taxes)
   - Tax components:
     - CGST, SGST, IGST (India)
     - Sales Tax or VAT (US/EU, map into IGST)
   - Tax Rate (in %)
   - Represent zero values as 0.0
4. Parties:
   - Vendor Name
   - Buyer Name
   - Place of Supply or jurisdiction (State/Region)
5. Invoice Details:
   - Invoice Number
   - Invoice Date
   - Narration (brief description of goods/services)
6. Classification & Tax:
   - Suggested Expense Ledger (e.g., 'Professional Fees', 'Software Subscription', 'Trademark Filing')
   - GST Input Eligibility: Yes, No, or Uncertain
   - TDS Applicability: e.g., Yes - Section 194J, No, Uncertain
   - RCM (Reverse Charge Mechanism): Yes, No, Uncertain
7. Missing Data Handling:
   - If a required field is missing, return "MISSING"
   - If optional, leave it as an empty string ""
   - Never make up values. Use only what is explicitly visible.

Return a single comma-separated line, in this exact order:

Vendor Name, Invoice No, Tax ID (GSTIN/EIN/VAT), HSN/SAC, Buyer Name, Place of Supply, Invoice Date, Expense Ledger, Tax Type, Tax Rate, Basic Amount, CGST, SGST, IGST/Sales Tax, Total Payable, Narration, GST Input Eligible, TDS Applicable, TDS Rate

Do not include labels, newlines, or explanation — only the data in that order.

If the document is not an invoice, return:
NOT AN INVOICE
"""

def is_placeholder_row(text):
    placeholder_keywords = ["Vendor Name", "Invoice No", "Invoice Date", "Expense Ledger"]
    return all(x.lower() in text.lower() for x in placeholder_keywords)

def convert_pdf_first_page(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc.load_page(0)
    pix = page.get_pixmap(dpi=300)
    return Image.open(io.BytesIO(pix.tobytes("png")))

# ---------- PDF Upload ----------
uploaded_files = st.file_uploader("📄 Upload scanned invoice PDFs", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    st.session_state["files_uploaded"] = True

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

            with st.spinner("🧐 Extracting data using ChatGPT..."):
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
                    result_row = ["NOT AN INVOICE"] + ["-"] * (len(columns) - 1)
                else:
                    matched = False
                    for line in csv_line.strip().split("\n"):
                        try:
                            row = [x.strip().strip('"') for x in line.split(",")]
                            if len(row) >= len(columns):
                                result_row = row[:len(columns)]
                                matched = True
                                break
                        except Exception:
                            pass
                    if not matched:
                        result_row = ["NOT AN INVOICE"] + ["-"] * (len(columns) - 1)

                st.session_state["processed_results"][file_name] = result_row
                st.session_state["processing_status"][file_name] = "✅ Done"
                completed_count += 1
                st.success(f"{file_name}: ✅ Done")
                st.info(f"🧠 {completed_count} out of {total_files} files processed")

        except Exception as e:
            st.session_state["processed_results"][file_name] = ["NOT AN INVOICE"] + ["-"] * (len(columns) - 1)
            st.session_state["processing_status"][file_name] = "❌ Error"
            st.error(f"❌ Error processing {file_name}: {e}")
            st.text_area(f"Raw Output ({file_name})", traceback.format_exc())

        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

# ---------- Display Results ----------
results = list(st.session_state["processed_results"].values())
if results:
    if completed_json:
        st_lottie(completed_json, height=200, key="done_animation")

    st.markdown("<h3 style='text-align: center;'>🎉 Yippie! All invoices processed with a smile 😊</h3>", unsafe_allow_html=True)

    sanitized_results = []
    for r in results:
        if len(r) == len(columns):
            sanitized_results.append(r)
        elif len(r) == len(columns) + 1:
            sanitized_results.append(r[1:])
        elif len(r) < len(columns):
            padded = r + ["-"] * (len(columns) - len(r))
            sanitized_results.append(padded)
        else:
            sanitized_results.append(r[:len(columns)])

    df = pd.DataFrame(sanitized_results, columns=columns)
    df.insert(0, "S. No", range(1, len(df) + 1))

    display_columns = ["S. No"] + columns
    st.dataframe(df[display_columns])

    csv_data = df[display_columns].to_csv(index=False).encode("utf-8")
    st.download_button("📅 Download Results as CSV", csv_data, "invoice_results.csv", "text/csv")

    st.markdown("---")
    if st.session_state.summary_rows:
        st.balloons()
else:
    st.info("Upload one or more scanned invoices to get started.")
