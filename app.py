import streamlit as st
from PIL import Image
import google.generativeai as genai
import openai
import fitz
import io
import pandas as pd
import base64
import requests
import traceback
import tempfile
import os
from typing import List, Optional
from pydantic import BaseModel

# ---------- Lottie Animation ----------
def load_lottie_url(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_url = "https://assets2.lottiefiles.com/packages/lf20_3vbOcw.json"
lottie_json = load_lottie_url(lottie_url)

# ---------- UI CONFIG ----------
st.set_page_config(layout="wide")
st_lottie(lottie_json, height=200, key="animation")
st.markdown("<h2 style='text-align: center;'>📄 AI Invoice Extractor</h2>", unsafe_allow_html=True)
st.markdown("Upload scanned PDF invoices and extract clean finance data using Gemini or ChatGPT")
st.markdown("---")

# ---------- Columns ----------
columns = [
    "File Name", "Vendor Name", "Invoice No", "Invoice Date", "Expense Ledger",
    "GST Type", "Tax Rate", "Basic Amount", "CGST", "SGST", "IGST",
    "Total Payable", "Narration", "GST Input Eligible", "TDS Applicable", "TDS Rate"
]

# ---------- Session Init ----------
if "processed_results" not in st.session_state:
    st.session_state["processed_results"] = {}

# ---------- Sidebar Auth ----------
st.sidebar.header("🔐 AI Config")
passcode = st.sidebar.text_input("Admin Passcode (optional)", type="password")
admin_unlocked = passcode == "Essenbee"

ai_model = None
model_choice = None
gemini_api_key = None
openai_api_key = None
openai_model = "gpt-4-vision-preview"

if admin_unlocked:
    st.sidebar.success("🔓 Admin access granted.")
    model_choice = st.sidebar.radio("Use which AI?", ["Gemini", "ChatGPT"])
    if model_choice == "Gemini":
        gemini_api_key = st.secrets.get("GEMINI_API_KEY")
        if gemini_api_key:
            genai.configure(api_key=gemini_api_key)
        else:
            st.sidebar.error("GEMINI_API_KEY missing in secrets.")
    elif model_choice == "ChatGPT":
        openai_api_key = "sk-admin-openai-key-here"
        openai.api_key = openai_api_key
else:
    model_choice = st.sidebar.radio("Choose AI Model", ["Gemini", "ChatGPT"])
    if model_choice == "Gemini":
        gemini_api_key = st.sidebar.text_input("🔑 Enter your Gemini API Key", type="password")
        if gemini_api_key:
            genai.configure(api_key=gemini_api_key)
    elif model_choice == "ChatGPT":
        openai_api_key = st.sidebar.text_input("🔑 Enter your OpenAI API Key", type="password")
        if openai_api_key:
            openai.api_key = openai_api_key

# ---------- Schema ----------
class LineItem(BaseModel):
    description: str
    quantity: float
    gross_worth: float

class Invoice(BaseModel):
    invoice_number: str
    date: str
    gstin: str
    seller_name: str
    buyer_name: str
    buyer_gstin: Optional[str] = None
    line_items: List[LineItem]
    total_gross_worth: float
    cgst: Optional[float] = None
    sgst: Optional[float] = None
    igst: Optional[float] = None
    place_of_supply: Optional[str] = None
    expense_ledger: Optional[str] = None
    tds: Optional[str] = None

# ---------- PDF to Image ----------
def convert_pdf_first_page(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc.load_page(0)
    pix = page.get_pixmap(dpi=200)
    return Image.open(io.BytesIO(pix.tobytes("png")))

# ---------- Upload ----------
uploaded_files = st.file_uploader("📤 Upload scanned invoice PDFs", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    for file in uploaded_files:
        file_name = file.name
        if file_name in st.session_state["processed_results"]:
            continue

        st.subheader(f"📄 Processing: {file_name}")
        try:
            pdf_data = file.read()
            first_image = convert_pdf_first_page(pdf_data)
        except Exception as e:
            st.error(f"❌ Error reading PDF: {e}")
            st.session_state["processed_results"][file_name] = [file_name] + ["NOT AN INVOICE"] + ["-"] * (len(columns) - 2)
            continue

        with st.spinner("🧠 Extracting data using AI..."):
            try:
                if model_choice == "Gemini" and gemini_api_key:
                    client = genai.Client(api_key=gemini_api_key)
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(pdf_data)
                        tmp_path = tmp.name

                    uploaded = client.files.upload(file=tmp_path, config={"display_name": file_name})
                    os.unlink(tmp_path)

                    prompt = (
                        "Extract invoice data as structured JSON. Use DD/MM/YYYY format. "
                        "Leave missing fields empty or null. Do not hallucinate. Focus on Indian invoice fields."
                    )
                    response = client.models.generate_content(
                        model="gemini-1.5-flash-latest",
                        contents=[prompt, uploaded],
                        config={
                            "response_mime_type": "application/json",
                            "response_schema": Invoice
                        }
                    )
                    client.files.delete(name=uploaded.name)
                    data = response.parsed

                    narration = (
                        f"Invoice {data.invoice_number} dated {data.date} issued by {data.seller_name} "
                        f"(GSTIN: {data.gstin}) to {data.buyer_name} "
                        f"(GSTIN: {data.buyer_gstin or '-'}) for ₹{data.total_gross_worth:.2f}. "
                        f"Taxes: CGST ₹{data.cgst or 0.0}, SGST ₹{data.sgst or 0.0}, IGST ₹{data.igst or 0.0}. "
                        f"Ledger: {data.expense_ledger or '-'}, POS: {data.place_of_supply or '-'}, "
                        f"TDS: {data.tds or '-'}."
                    )

                    result_row = [
                        file_name,
                        data.seller_name,
                        data.invoice_number,
                        data.date,
                        data.expense_ledger or "-",
                        "CGST+SGST" if data.cgst and data.sgst else ("IGST" if data.igst else "NA"),
                        "-",  # Tax Rate not extracted from schema yet
                        data.total_gross_worth - sum([data.cgst or 0, data.sgst or 0, data.igst or 0]),
                        data.cgst or 0.0,
                        data.sgst or 0.0,
                        data.igst or 0.0,
                        data.total_gross_worth,
                        narration,
                        "Yes" if data.expense_ledger and "travel" not in data.expense_ledger.lower() else "No",
                        "Yes" if data.tds and data.tds.lower().startswith("yes") else "No",
                        data.tds.split()[-1] if data.tds and "%" in data.tds else "0"
                    ]

                elif model_choice == "ChatGPT" and openai_api_key:
                    st.warning("⚠️ ChatGPT parsing is not set up for structured invoice schema.")
                    result_row = [file_name] + ["NOT AN INVOICE"] + ["-"] * (len(columns) - 2)

                else:
                    raise Exception("No valid API key provided.")

                st.session_state["processed_results"][file_name] = result_row

            except Exception as e:
                st.error(f"❌ Error processing {file_name}: {e}")
                st.text_area(f"Raw Output ({file_name})", traceback.format_exc())
                st.session_state["processed_results"][file_name] = [file_name] + ["NOT AN INVOICE"] + ["-"] * (len(columns) - 2)

# ---------- DISPLAY RESULTS ----------
results = list(st.session_state["processed_results"].values())
if results:
    df = pd.DataFrame(results, columns=columns)
    df.index = df.index + 1
    df.reset_index(inplace=True)
    df.rename(columns={"index": "S. No"}, inplace=True)

    st.success("✅ All invoices processed!")
    st.dataframe(df)

    csv = df.to_csv(index=False).encode()
    st.download_button("📥 Download Extracted Data", csv, "invoice_data.csv", "text/csv")
    st.balloons()
else:
    st.info("Upload one or more scanned invoices to get started.")
