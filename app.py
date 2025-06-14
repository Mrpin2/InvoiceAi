import streamlit as st
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import openai
import os
import tempfile
import io
import base64

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

# --- OpenAI Function ---
def extract_structured_data(file_bytes: bytes, openai_api_key: str) -> Optional[Invoice]:
    openai.api_key = openai_api_key
    try:
        file_base64 = base64.b64encode(file_bytes).decode("utf-8")
        prompt = (
            "You are an intelligent invoice parser. Extract fields like invoice number, date, GSTIN, seller and buyer names, line items with description, quantity and gross worth, total gross worth, taxes, place of supply, expense ledger and TDS applicability from the PDF text."
        )
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"The following is the base64-encoded content of a PDF invoice. Decode, read, and extract structured data:\n{file_base64}"}
        ]
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
            temperature=0.0,
            response_format="json"
        )
        parsed = Invoice.parse_raw(response.choices[0].message.content)
        return parsed
    except Exception as e:
        st.error(f"OpenAI extraction error: {e}")
        return None

# --- Streamlit UI ---
st.set_page_config(layout="wide")
st.title("\U0001F4C4 PDF Invoice Extractor (OpenAI GPT)")

st.sidebar.header("Configuration")
api_key_input = st.sidebar.text_input("Enter your OpenAI API Key:", type="password")

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

if st.button("\U0001F680 Process Invoices", type="primary"):
    if not api_key_input:
        st.error("Please enter your OpenAI API Key in the sidebar.")
    elif not uploaded_files:
        st.error("Please upload at least one PDF file.")
    else:
        st.session_state.summary_rows = []
        progress_bar = st.progress(0)
        total_files = len(uploaded_files)

        for i, uploaded_file_obj in enumerate(uploaded_files):
            st.markdown(f"---")
            st.info(f"Processing file: {uploaded_file_obj.name} ({i+1}/{total_files})")

            try:
                file_bytes = uploaded_file_obj.read()
                with st.spinner(f"Extracting data from {uploaded_file_obj.name}..."):
                    extracted_data = extract_structured_data(
                        file_bytes=file_bytes,
                        openai_api_key=api_key_input
                    )

                if extracted_data:
                    st.success(f"Successfully extracted data from {uploaded_file_obj.name}")
                    cgst = extracted_data.cgst if extracted_data.cgst is not None else 0.0
                    sgst = extracted_data.sgst if extracted_data.sgst is not None else 0.0
                    igst = extracted_data.igst if extracted_data.igst is not None else 0.0
                    pos = extracted_data.place_of_supply or "N/A"
                    buyer_gstin_display = extracted_data.buyer_gstin or "N/A"

                    narration = (
                        f"Invoice {extracted_data.invoice_number} dated {extracted_data.date} "
                        f"was issued by {extracted_data.seller_name} (GSTIN: {extracted_data.gstin}) "
                        f"to {extracted_data.buyer_name} (GSTIN: {buyer_gstin_display}), "
                        f"with a total value of ₹{extracted_data.total_gross_worth:.2f}. "
                        f"Taxes applied - CGST: ₹{cgst:.2f}, SGST: ₹{sgst:.2f}, IGST: ₹{igst:.2f}. "
                        f"Place of supply: {pos}. Expense: {extracted_data.expense_ledger or 'N/A'}. "
                        f"TDS: {extracted_data.tds or 'N/A'}."
                    )

                    st.session_state.summary_rows.append({
                        "File Name": uploaded_file_obj.name,
                        "Invoice Number": extracted_data.invoice_number,
                        "Date": extracted_data.date,
                        "Seller Name": extracted_data.seller_name,
                        "Seller GSTIN": extracted_data.gstin,
                        "Buyer Name": extracted_data.buyer_name,
                        "Buyer GSTIN": buyer_gstin_display,
                        "Total Gross Worth": extracted_data.total_gross_worth,
                        "CGST": cgst,
                        "SGST": sgst,
                        "IGST": igst,
                        "Place of Supply": pos,
                        "Expense Ledger": extracted_data.expense_ledger,
                        "TDS": extracted_data.tds,
                        "Narration": narration,
                    })
                else:
                    st.warning(f"Failed to extract data from {uploaded_file_obj.name}")

            except Exception as e:
                st.error(f"Error processing {uploaded_file_obj.name}: {e}")

            progress_bar.progress((i + 1) / total_files)

        st.markdown(f"---")
        if st.session_state.summary_rows:
            st.balloons()

if st.session_state.summary_rows:
    st.subheader("\U0001F4CA Extracted Invoice Summary")
    df = pd.DataFrame(st.session_state.summary_rows)
    st.dataframe(df)

    output_excel = io.BytesIO()
    with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='InvoiceSummary')
    excel_data = output_excel.getvalue()

    st.download_button(
        label="\U0001F4E5 Download Summary as Excel",
        data=excel_data,
        file_name="invoice_summary.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
elif not uploaded_files:
    st.info("Upload PDF files and click 'Process Invoices' to see results.")
