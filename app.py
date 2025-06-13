import streamlit as st
import os
import tempfile
import io
import pandas as pd
from typing import List, Optional
from pydantic import BaseModel
from google import generativeai as genai

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

# ---------- Config ----------
st.set_page_config(layout="wide")
st.title("📄 Invoice Extractor with Gemini + Schema")

# ---------- Sidebar ----------
st.sidebar.header("API Configuration")
api_key = st.sidebar.text_input("🔑 Gemini API Key", type="password")
model_id = st.sidebar.text_input("Model ID", value="gemini-1.5-flash-latest")

# ---------- File Upload ----------
st.info("Upload one or more **PDF invoices** below and click **Process**.")
uploaded_files = st.file_uploader("Upload invoices", type="pdf", accept_multiple_files=True)

# ---------- Session Vars ----------
if "results" not in st.session_state:
    st.session_state.results = []

# ---------- Extraction Function ----------
def extract_invoice_data(client, model_id: str, file_path: str) -> Optional[Invoice]:
    try:
        file_name = os.path.basename(file_path)
        uploaded = client.files.upload(file=file_path, config={"display_name": file_name})
        prompt = (
            "Extract all invoice fields in structured JSON. Use Indian standards for dates, tax, ledgers.\n"
            "Do not hallucinate values. If any field is not present, leave it null."
        )
        response = client.models.generate_content(
            model=model_id,
            contents=[prompt, uploaded],
            config={
                "response_mime_type": "application/json",
                "response_schema": Invoice
            }
        )
        parsed = response.parsed
        client.files.delete(name=uploaded.name)
        return parsed
    except Exception as e:
        st.error(f"❌ Error: {e}")
        return None

# ---------- Process Button ----------
if st.button("🚀 Process Invoices"):
    if not api_key:
        st.error("Please enter your Gemini API key.")
    elif not uploaded_files:
        st.error("Upload at least one invoice.")
    else:
        client = genai.Client(api_key=api_key)
        st.session_state.results.clear()
        progress = st.progress(0)

        for idx, file in enumerate(uploaded_files):
            st.markdown(f"---\n📄 **{file.name}**")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file.read())
                temp_path = tmp.name

            with st.spinner(f"Processing {file.name}..."):
                data = extract_invoice_data(client, model_id, temp_path)

            os.unlink(temp_path)
            progress.progress((idx + 1) / len(uploaded_files))

            if data:
                narration = (
                    f"Invoice {data.invoice_number} dated {data.date} issued by {data.seller_name} (GSTIN: {data.gstin}) "
                    f"to {data.buyer_name} (GSTIN: {data.buyer_gstin or 'N/A'}) for ₹{data.total_gross_worth:.2f}. "
                    f"Taxes: CGST ₹{data.cgst or 0.0}, SGST ₹{data.sgst or 0.0}, IGST ₹{data.igst or 0.0}. "
                    f"Place of Supply: {data.place_of_supply or 'N/A'}. Ledger: {data.expense_ledger or 'N/A'}. "
                    f"TDS: {data.tds or 'N/A'}."
                )
                st.success("✅ Extraction successful.")
                st.session_state.results.append({
                    "File Name": file.name,
                    "Invoice No": data.invoice_number,
                    "Date": data.date,
                    "Seller": data.seller_name,
                    "Seller GSTIN": data.gstin,
                    "Buyer": data.buyer_name,
                    "Buyer GSTIN": data.buyer_gstin or "-",
                    "Ledger": data.expense_ledger or "-",
                    "Total": data.total_gross_worth,
                    "CGST": data.cgst or 0.0,
                    "SGST": data.sgst or 0.0,
                    "IGST": data.igst or 0.0,
                    "POS": data.place_of_supply or "-",
                    "TDS": data.tds or "-",
                    "Narration": narration
                })
            else:
                st.warning("⚠️ Extraction failed or no structured data returned.")

        st.balloons()

# ---------- Display Results ----------
if st.session_state.results:
    st.subheader("📊 Invoice Summary")
    df = pd.DataFrame(st.session_state.results)
    st.dataframe(df)

    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Summary")
    st.download_button("📥 Download as Excel", data=out.getvalue(),
                       file_name="invoice_summary.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
