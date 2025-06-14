import streamlit as st
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import os
import tempfile
import io
import openai
from datetime import datetime

# --- Pydantic Models ---
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

# --- Streamlit Setup ---
st.set_page_config(layout="wide")
st.title("📄 PDF Invoice Extractor (ChatGPT)")

st.sidebar.header("Configuration")
password = st.sidebar.text_input("Enter password to unlock ChatGPT API Key:", type="password")

if password != "essenbee":
    st.warning("Please enter the correct password to continue.")
    st.stop()

api_key_input = st.sidebar.text_input("Enter your OpenAI API Key:", type="password")
model_id = st.sidebar.text_input("ChatGPT Model ID:", "gpt-4-turbo")

st.info(
    "**Instructions:**\n"
    "1. Enter your OpenAI API Key in the sidebar.\n"
    "2. Upload one or more PDF invoice files.\n"
    "3. Click 'Process Invoices' to extract data.\n"
    "   The extracted data will be displayed in a table and available for download as Excel."
)

uploaded_files = st.file_uploader(
    "Choose PDF invoice files",
    type="pdf",
    accept_multiple_files=True
)

if 'summary_rows' not in st.session_state:
    st.session_state.summary_rows = []
if 'processing_status' not in st.session_state:
    st.session_state.processing_status = {}

# --- Function: Extract data using ChatGPT ---
def extract_with_chatgpt(api_key: str, model: str, file_bytes: bytes, file_name: str) -> Optional[Invoice]:
    try:
        content = file_bytes.decode('latin1', errors='ignore')

        prompt = (
            f"""
You are a strict invoice parser. Extract structured data from the below invoice text.
Return data as JSON matching this structure:

Invoice(BaseModel):
- invoice_number: str
- date: str (format: DD/MM/YYYY or DD-MM-YYYY)
- gstin: str (seller GSTIN)
- seller_name: str
- buyer_name: str
- buyer_gstin: str | null
- line_items: List of objects with: description, quantity (float), gross_worth (float)
- total_gross_worth: float
- cgst: float | null
- sgst: float | null
- igst: float | null
- place_of_supply: str | null
- expense_ledger: str | null
- tds: str | null

Only use fields clearly visible in the invoice. If unknown, set to null or empty string. Do NOT guess or hallucinate.
PDF Content:\n""" + content[:12000] + "\n--- END OF FILE ---"
        )

        openai.api_key = api_key
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            response_format="json"
        )

        parsed = response.choices[0].message.content
        return Invoice.model_validate_json(parsed)

    except Exception as e:
        st.error(f"Failed to extract data from {file_name}: {e}")
        return None

# --- Main Processing Button ---
if st.button("🚀 Process Invoices", type="primary"):
    if not api_key_input:
        st.error("Please enter your OpenAI API Key.")
    elif not uploaded_files:
        st.error("Please upload at least one PDF file.")
    else:
        st.session_state.summary_rows = []
        progress_bar = st.progress(0)
        total_files = len(uploaded_files)

        for i, uploaded_file in enumerate(uploaded_files):
            file_name = uploaded_file.name
            st.session_state.processing_status[file_name] = "⏳ Pending..."
            st.markdown("---")
            st.info(f"Processing file: {file_name} ({i+1}/{total_files})")

            temp_path = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.getvalue())
                    temp_path = tmp.name

                with open(temp_path, "rb") as f:
                    file_bytes = f.read()

                with st.spinner(f"Extracting data from {file_name}..."):
                    extracted = extract_with_chatgpt(api_key_input, model_id, file_bytes, file_name)

                if extracted:
                    st.success(f"✅ Extracted data from {file_name}")
                    cgst = extracted.cgst or 0.0
                    sgst = extracted.sgst or 0.0
                    igst = extracted.igst or 0.0
                    pos = extracted.place_of_supply or "N/A"
                    buyer_gstin = extracted.buyer_gstin or "N/A"

                    narration = (
                        f"Invoice {extracted.invoice_number} dated {extracted.date} was issued by "
                        f"{extracted.seller_name} (GSTIN: {extracted.gstin}) to {extracted.buyer_name} "
                        f"(GSTIN: {buyer_gstin}), total ₹{extracted.total_gross_worth:.2f}. Taxes - "
                        f"CGST: ₹{cgst:.2f}, SGST: ₹{sgst:.2f}, IGST: ₹{igst:.2f}. Place of supply: {pos}. "
                        f"Expense: {extracted.expense_ledger or 'N/A'}. TDS: {extracted.tds or 'N/A'}."
                    )

                    st.session_state.summary_rows.append({
                        "File Name": file_name,
                        "Invoice Number": extracted.invoice_number,
                        "Date": extracted.date,
                        "Seller Name": extracted.seller_name,
                        "Seller GSTIN": extracted.gstin,
                        "Buyer Name": extracted.buyer_name,
                        "Buyer GSTIN": buyer_gstin,
                        "Total Gross Worth": extracted.total_gross_worth,
                        "CGST": cgst,
                        "SGST": sgst,
                        "IGST": igst,
                        "Place of Supply": pos,
                        "Expense Ledger": extracted.expense_ledger,
                        "TDS": extracted.tds,
                        "Narration": narration,
                    })
                    st.session_state.processing_status[file_name] = "✅ Success"
                else:
                    st.warning(f"No data returned for {file_name}")
                    st.session_state.processing_status[file_name] = "⚠️ No Data"

            except Exception as e:
                st.error(f"Unexpected error while processing {file_name}: {e}")
                st.session_state.processing_status[file_name] = "❌ Failed"
            finally:
                if temp_path and os.path.exists(temp_path):
                    os.unlink(temp_path)
                    st.write(f"Deleted temporary file: {temp_path}")
                progress_bar.progress((i + 1) / total_files)

        st.balloons()

# --- Display Summary Table ---
if st.session_state.summary_rows:
    st.subheader("📊 Extracted Invoice Summary")
    df = pd.DataFrame(st.session_state.summary_rows)
    st.dataframe(df)

    output_excel = io.BytesIO()
    with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='InvoiceSummary')
    excel_data = output_excel.getvalue()

    st.download_button(
        label="📥 Download Summary as Excel",
        data=excel_data,
        file_name="invoice_summary.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
elif not uploaded_files:
    st.info("Upload PDF files and click 'Process Invoices' to see results.")
