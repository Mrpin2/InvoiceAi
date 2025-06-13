import streamlit as st
import pandas as pd
import tempfile
import os
import io
import base64
import fitz  # PyMuPDF
import requests
from pydantic import BaseModel
from typing import List, Optional
from PIL import Image
from streamlit_lottie import st_lottie
import csv

# ---------- Lottie Animation ----------
def load_lottie_url(url):
    r = requests.get(url)
    return r.json() if r.status_code == 200 else None

lottie_url = "https://assets2.lottiefiles.com/packages/lf20_3vbOcw.json"
lottie_json = load_lottie_url(lottie_url)

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

# ---------- UI Setup ----------
st.set_page_config(layout="wide")
st_lottie(lottie_json, height=200, key="animation")
st.markdown("<h2 style='text-align: center;'>📄 AI Invoice Extractor</h2>", unsafe_allow_html=True)
st.markdown("Upload scanned PDF invoices and extract clean finance data using Gemini or ChatGPT")
st.markdown("---")

# ---------- Columns ----------
columns = [
    "Vendor Name", "Invoice No", "Invoice Date", "Expense Ledger",
    "GST Type", "Tax Rate", "Basic Amount", "CGST", "SGST", "IGST",
    "Total Payable", "Narration", "GST Input Eligible", "TDS Applicable", "TDS Rate"
]

# ---------- Sidebar Config ----------
st.sidebar.header("🔐 AI Config")
passcode = st.sidebar.text_input("Admin Passcode (optional)", type="password")
admin_unlocked = passcode == "Essenbee"
model_choice = st.sidebar.radio("Choose AI Model", ["Gemini", "ChatGPT"])

gemini_api_key = None
openai_api_key = None
openai_model = "gpt-4-vision-preview"

if model_choice == "Gemini":
    if admin_unlocked:
        st.sidebar.caption("🔓 Using admin Gemini key")
        gemini_api_key = "AIzaSyA5Jnd7arMlbZ1x_ZpiE-AezrmsaXams7Y"
    else:
        gemini_api_key = st.sidebar.text_input("🔑 Your Gemini API Key", type="password")

elif model_choice == "ChatGPT":
    if admin_unlocked:
        st.sidebar.caption("🔓 Using admin OpenAI key")
        openai_api_key = "sk-admin-openai-key-here"  # Replace with real key
    else:
        openai_api_key = st.sidebar.text_input("🔑 Your OpenAI API Key", type="password")

# ---------- Gemini Extraction ----------
def extract_invoice_from_pdf_gemini(pdf_path, gemini_api_key, model_id="gemini-1.5-flash-latest"):
    try:
        from google import generativeai as genai
        genai.configure(api_key=gemini_api_key)

        model = genai.GenerativeModel(model_id)
        file_resource = model.upload_file(pdf_path)

        prompt = (
            "Extract all invoice fields according to the schema. "
            "Use Indian formats. If data is missing, leave it null or empty. "
        )

        response = model.generate_content(
            contents=[prompt, file_resource],
            config={
                "response_schema": Invoice,
                "response_mime_type": "application/json"
            }
        )

        model.delete_file(file_resource.name)
        return response.parsed

    except Exception as e:
        st.error(f"❌ Gemini error: {e}")
        return None

# ---------- ChatGPT Vision ----------
def extract_invoice_from_pdf_openai(image: Image.Image, prompt: str, api_key: str):
    import openai
    openai.api_key = api_key
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode()
    chat_prompt = [
        {"role": "system", "content": "You are a finance assistant."},
        {"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
        ]}
    ]
    response = openai.ChatCompletion.create(
        model="gpt-4-vision-preview",
        messages=chat_prompt,
        max_tokens=1000
    )
    return response.choices[0].message.content.strip()

# ---------- PDF Upload ----------
uploaded_files = st.file_uploader("📤 Upload scanned invoice PDFs", type=["pdf"], accept_multiple_files=True)

results = []
if uploaded_files:
    for file in uploaded_files:
        st.subheader(f"📄 Processing: {file.name}")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name

        try:
            extracted = None

            if model_choice == "Gemini" and gemini_api_key:
                extracted = extract_invoice_from_pdf_gemini(tmp_path, gemini_api_key)

            elif model_choice == "ChatGPT" and openai_api_key:
                doc = fitz.open(tmp_path)
                page = doc.load_page(0)
                pix = page.get_pixmap(dpi=200)
                img = Image.open(io.BytesIO(pix.tobytes("png")))

                prompt = """
                You are a professional finance assistant. Extract the following fields from the invoice image:
                Vendor Name, Invoice No, Invoice Date, Expense Ledger (like Office Supplies, Travel, Legal Fees, etc.),
                GST Type (IGST or CGST+SGST or NA), Tax Rate (%, single value), Basic Amount,
                CGST, SGST, IGST, Total Payable, Narration (short sentence),
                GST Input Eligible (Yes/No — No if travel, food, hotel, etc.),
                TDS Applicable (Yes/No), TDS Rate (in % if applicable).
                Respond with CSV-style values in this exact order:
                Vendor Name, Invoice No, Invoice Date, Expense Ledger,
                GST Type, Tax Rate, Basic Amount, CGST, SGST, IGST,
                Total Payable, Narration, GST Input Eligible, TDS Applicable, TDS Rate.
                """
                csv_line = extract_invoice_from_pdf_openai(img, prompt, openai_api_key)
                reader = csv.reader(io.StringIO(csv_line, newline=''))
                row = [x.strip() for x in next(reader)]

                if len(row) != len(columns):
                    raise ValueError("CSV row doesn't match expected column count")
                results.append(row)

            if model_choice == "Gemini" and extracted:
                row = [
                    extracted.seller_name, extracted.invoice_number, extracted.date, extracted.expense_ledger or "N/A",
                    "IGST" if extracted.igst else "CGST+SGST" if extracted.cgst and extracted.sgst else "NA",
                    "",  # Tax Rate not extracted yet
                    extracted.total_gross_worth or 0.0,
                    extracted.cgst or 0.0, extracted.sgst or 0.0, extracted.igst or 0.0,
                    extracted.total_gross_worth + (extracted.cgst or 0) + (extracted.sgst or 0) + (extracted.igst or 0),
                    f"Invoice {extracted.invoice_number} issued by {extracted.seller_name}",
                    "No" if (extracted.expense_ledger or "").lower() in ["travel", "food", "hotel"] else "Yes",
                    "Yes" if extracted.tds else "No",
                    extracted.tds or ""
                ]
                results.append(row)

        except Exception as e:
            st.error(f"❌ Error processing {file.name}: {e}")

        finally:
            os.unlink(tmp_path)

# ---------- Display Results ----------
if results:
    df = pd.DataFrame(results, columns=columns)
    st.success("✅ All invoices processed!")
    st.dataframe(df)

    csv_data = df.to_csv(index=False).encode()
    st.download_button("📥 Download Extracted Data", csv_data, "invoice_data.csv", "text/csv")
    st.balloons()
else:
    st.info("Upload one or more scanned invoices to get started.")
