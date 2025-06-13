import streamlit as st
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import tempfile
import io
import fitz  # PyMuPDF
import os
import requests
from streamlit_lottie import st_lottie

# ---------- Load Lottie ----------
def load_lottie_url(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_url = "https://assets2.lottiefiles.com/packages/lf20_3vbOcw.json"
lottie_json = load_lottie_url(lottie_url)

# ---------- Pydantic Schema ----------
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

# ---------- Gemini Extraction ----------
def extract_invoice(client, model_id, file_path, schema):
    display_name = os.path.basename(file_path)
    file_resource = None
    try:
        file_resource = client.files.upload(file=file_path, config={'display_name': display_name})
        prompt = (
            "Extract structured invoice data using the provided schema. "
            "Be accurate. Return only values that are clearly present. Prefer DD/MM/YYYY for dates."
        )
        response = client.models.generate_content(
            model=model_id,
            contents=[prompt, file_resource],
            config={
                'response_mime_type': 'application/json',
                'response_schema': schema
            }
        )
        return response.parsed
    except Exception as e:
        st.error(f"Gemini error: {e}")
        return None
    finally:
        if file_resource:
            try:
                client.files.delete(name=file_resource.name)
            except:
                pass

# ---------- Streamlit App ----------
st.set_page_config(layout="wide")
st_lottie(lottie_json, height=200, key="animation")
st.markdown("<h2 style='text-align: center;'>📄 Gemini Invoice Extractor</h2>", unsafe_allow_html=True)
st.markdown("Upload scanned invoice PDFs to extract structured data using Gemini.")
st.markdown("---")

st.sidebar.header("🔐 API Setup")
gemini_api_key = st.sidebar.text_input("Enter your Gemini API Key", type="password")
model_id = st.sidebar.text_input("Model ID", value="gemini-1.5-flash-latest")

uploaded_files = st.file_uploader("📤 Upload scanned invoice PDFs", type=["pdf"], accept_multiple_files=True)

if st.button("🚀 Process Invoices"):
    if not gemini_api_key or not uploaded_files:
        st.error("Please enter your API key and upload files.")
    else:
        from google import genai
        genai.configure(api_key=gemini_api_key)
        client = genai

        summary_rows = []
        for i, file in enumerate(uploaded_files):
            st.subheader(f"📄 Processing: {file.name}")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file.read())
                file_path = tmp.name

            data = extract_invoice(client, model_id, file_path, Invoice)
            os.unlink(file_path)

            if data:
                narration = (
                    f"Invoice {data.invoice_number} dated {data.date} from {data.seller_name} to "
                    f"{data.buyer_name}, total ₹{data.total_gross_worth:.2f}. "
                    f"CGST: ₹{data.cgst or 0:.2f}, SGST: ₹{data.sgst or 0:.2f}, IGST: ₹{data.igst or 0:.2f}. "
                    f"POS: {data.place_of_supply or 'N/A'}, Ledger: {data.expense_ledger or 'N/A'}, "
                    f"TDS: {data.tds or 'N/A'}."
                )
                summary_rows.append({
                    "File": file.name,
                    "Invoice No": data.invoice_number,
                    "Date": data.date,
                    "Seller": data.seller_name,
                    "Buyer": data.buyer_name,
                    "Seller GSTIN": data.gstin,
                    "Buyer GSTIN": data.buyer_gstin or "N/A",
                    "Gross Amount": data.total_gross_worth,
                    "CGST": data.cgst or 0,
                    "SGST": data.sgst or 0,
                    "IGST": data.igst or 0,
                    "Ledger": data.expense_ledger or "N/A",
                    "POS": data.place_of_supply or "N/A",
                    "TDS": data.tds or "N/A",
                    "Narration": narration
                })
            else:
                st.warning(f"No data extracted for {file.name}")

        if summary_rows:
            df = pd.DataFrame(summary_rows)
            st.success("✅ All invoices processed!")
            st.dataframe(df)

            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False)
            st.download_button("📥 Download Excel", output.getvalue(), "invoice_summary.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        else:
            st.warning("No valid invoices extracted.")
