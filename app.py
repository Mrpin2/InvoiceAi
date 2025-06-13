import streamlit as st
from streamlit_lottie import st_lottie
import requests
from google import genai
from pydantic import BaseModel, Field
from typing import Optional
import pandas as pd
import os
import re
import traceback

# Page config and layout
st.set_page_config(page_title="Invoice Data Extractor", page_icon="💳", layout="centered")

# Load Lottie animation from URL (adjust URL if needed)
def load_lottie_url(url: str):
    res = requests.get(url)
    if res.status_code != 200:
        return None
    return res.json()

lottie_url = "https://assets4.lottiefiles.com/packages/lf20_TyP8dv.json"  # sample animation URL
lottie_json = load_lottie_url(lottie_url)
if lottie_json:
    st_lottie(lottie_json, height=200)

# Title and centered heading
st.title("Invoice Data Extraction")
st.markdown("<h2 style='text-align: center;'>Extract Data from Invoices</h2>", unsafe_allow_html=True)

# Sidebar for API settings
st.sidebar.header("Settings")
password = st.sidebar.text_input("Enter passcode to unlock API key input", type="password")
if password == "Essenbee":
    api_key = st.sidebar.text_input("Enter Google Gemini API Key", type="password")
else:
    api_key = None
    st.sidebar.warning("Enter the admin passcode to input the API key.")

model_choice = st.sidebar.radio("Select Model", ["Google Gemini", "OpenAI ChatGPT"], index=0)
if model_choice == "OpenAI ChatGPT":
    st.sidebar.info("ChatGPT mode is not available yet. Using Google Gemini.")
    model_choice = "Google Gemini"

# Require API key for Gemini
if model_choice == "Google Gemini" and not api_key:
    st.warning("Please enter the API key for Google Gemini in the sidebar.")
    st.stop()

# Initialize Gemini client
try:
    client = genai.Client(api_key=api_key)
except Exception as e:
    st.error("Failed to initialize Gemini API client. Check the API key.")
    st.stop()

# Define Pydantic model for structured invoice data
class InvoiceData(BaseModel):
    seller_name: Optional[str] = Field(None, description="Vendor or seller name")
    invoice_no: Optional[str] = Field(None, description="Invoice number")
    invoice_date: Optional[str] = Field(None, description="Invoice date")
    seller_gstin: Optional[str] = Field(None, description="GSTIN of the seller")
    total_gross: Optional[str] = Field(None, description="Total invoice amount (gross)")
    cgst: Optional[str] = Field(None, description="CGST amount")
    sgst: Optional[str] = Field(None, description="SGST amount")
    igst: Optional[str] = Field(None, description="IGST amount")
    buyer_gstin: Optional[str] = Field(None, description="GSTIN of the buyer")
    expense_ledger: Optional[str] = Field(None, description="Suggested expense ledger/category")
    tds: Optional[str] = Field(None, description="TDS deduction (Yes/No or percentage)")
    place_of_supply: Optional[str] = Field(None, description="Place of Supply")

# Session state to cache results
if 'processed_files' not in st.session_state:
    st.session_state['processed_files'] = {}

# File uploader (multiple PDFs)
uploaded_files = st.file_uploader("Upload Invoice PDF files", type=["pdf"], accept_multiple_files=True)

# Process extraction when button is clicked
if uploaded_files:
    if st.button("Extract Data"):
        # Create temp directory for saving uploaded files
        temp_dir = "temp_uploads"
        if not os.path.isdir(temp_dir):
            os.makedirs(temp_dir)
        for file in uploaded_files:
            file_name = file.name
            # Skip if already processed
            if file_name in st.session_state['processed_files']:
                continue
            try:
                # Save file to disk
                file_path = os.path.join(temp_dir, file_name)
                with open(file_path, "wb") as f:
                    f.write(file.getvalue())
                # Upload to Gemini Files API
                uploaded_file = client.files.upload(file=file_path, config={'display_name': file_name})
                # Prompt for extraction
                prompt = "Extract all relevant invoice details from this file."
                # Generate structured JSON output
                response = client.models.generate_content(
                    model="gemini-2.0-pro-vision",
                    contents=[prompt, uploaded_file],
                    config={
                        "response_mime_type": "application/json",
                        "response_schema": InvoiceData
                    }
                )
                result: InvoiceData = response.parsed  # Parsed Pydantic model
                if result is None:
                    # If parsing failed or no structured data, mark as not an invoice
                    st.session_state['processed_files'][file_name] = {"Vendor Name": "NOT AN INVOICE"}
                else:
                    data = result.model_dump()
                    # If key fields missing, treat as not an invoice
                    if not data.get("seller_name") or not data.get("invoice_no"):
                        st.session_state['processed_files'][file_name] = {"Vendor Name": "NOT AN INVOICE"}
                    else:
                        # Helper to parse numeric strings
                        def parse_number(val) -> float:
                            if val is None:
                                return 0.0
                            s = ''.join(ch for ch in str(val) if (ch.isdigit() or ch in '.-'))
                            if s == '' or s == '.' or s == '-' or s == '-.':
                                return 0.0
                            try:
                                return float(s)
                            except:
                                return 0.0
                        # Parse numeric fields
                        cgst_val = parse_number(data.get('cgst'))
                        sgst_val = parse_number(data.get('sgst'))
                        igst_val = parse_number(data.get('igst'))
                        total_val = parse_number(data.get('total_gross'))
                        # Compute basic amount (before GST)
                        if igst_val and igst_val > 0:
                            basic_amount = total_val - igst_val
                        else:
                            basic_amount = total_val - (cgst_val + sgst_val)
                        # Determine GST Type and Tax Rate
                        if igst_val and igst_val > 0:
                            gst_type = "IGST"
                            tax_rate = f"{round((igst_val / basic_amount) * 100)}%" if basic_amount else ""
                        elif cgst_val or sgst_val:
                            gst_type = "CGST/SGST"
                            tax_rate = f"{round(((cgst_val + sgst_val) / basic_amount) * 100)}%" if basic_amount else ""
                        else:
                            gst_type = ""
                            tax_rate = ""
                        # Determine GST Input Eligible (Yes if GST present and buyer/seller GSTIN exist)
                        gst_input_eligible = "Yes" if data.get('buyer_gstin') and data.get('seller_gstin') and (cgst_val > 0 or sgst_val > 0 or igst_val > 0) else "No"
                        # Determine TDS Applicable and Rate
                        tds_applicable = "No"
                        tds_rate = ""
                        tds_str = (data.get('tds') or "").strip()
                        if tds_str:
                            tds_lower = tds_str.lower()
                            # Check if any number in TDS field
                            match = re.search(r'(\d+)\s*%?', tds_lower)
                            if match:
                                tds_applicable = "Yes"
                                tds_rate_val = match.group(1)
                                tds_rate = tds_rate_val + "%" if "%" not in tds_str else match.group(0)
                            elif "yes" in tds_lower:
                                tds_applicable = "Yes"
                            elif "no" in tds_lower:
                                tds_applicable = "No"
                        # Calculate TDS amount if applicable
                        tds_amount = 0.0
                        if tds_applicable == "Yes" and tds_rate:
                            try:
                                tds_percent = float(tds_rate.replace('%', ''))
                                tds_amount = basic_amount * (tds_percent / 100.0)
                            except:
                                tds_amount = 0.0
                        # Compute total payable after TDS deduction
                        total_payable = total_val - tds_amount
                        # Build narration string
                        narration = f"Being {data.get('expense_ledger') or 'expense'} as per Invoice {data.get('invoice_no')} dated {data.get('invoice_date')} from {data.get('seller_name')}."
                        # Store the structured result
                        st.session_state['processed_files'][file_name] = {
                            "File Name": file_name,
                            "Vendor Name": data.get('seller_name', ''),
                            "Invoice No": data.get('invoice_no', ''),
                            "Invoice Date": data.get('invoice_date', ''),
                            "Expense Ledger": data.get('expense_ledger', ''),
                            "GST Type": gst_type,
                            "Tax Rate": tax_rate,
                            "Basic Amount": f"{basic_amount:.2f}",
                            "CGST": data.get('cgst', '') or "0",
                            "SGST": data.get('sgst', '') or "0",
                            "IGST": data.get('igst', '') or "0",
                            "Total Payable": f"{total_payable:.2f}",
                            "Narration": narration,
                            "GST Input Eligible": gst_input_eligible,
                            "TDS Applicable": tds_applicable,
                            "TDS Rate": tds_rate
                        }
            except Exception as e:
                # On any error, display traceback and mark file as processed (skip in future)
                st.error(f"Error processing file {file_name}:")
                st.exception(e)
                st.session_state['processed_files'][file_name] = {"Vendor Name": "NOT AN INVOICE"}
        # Build results table from session state for currently uploaded files
        results = []
        for file in uploaded_files:
            fname = file.name
            if fname in st.session_state['processed_files']:
                row = st.session_state['processed_files'][fname]
                if row.get("Vendor Name") == "NOT AN INVOICE":
                    # File not recognized as invoice or error
                    results.append({
                        "File Name": fname,
                        "Vendor Name": "NOT AN INVOICE",
                        "Invoice No": "",
                        "Invoice Date": "",
                        "Expense Ledger": "",
                        "GST Type": "",
                        "Tax Rate": "",
                        "Basic Amount": "",
                        "CGST": "",
                        "SGST": "",
                        "IGST": "",
                        "Total Payable": "",
                        "Narration": "",
                        "GST Input Eligible": "",
                        "TDS Applicable": "",
                        "TDS Rate": ""
                    })
                else:
                    results.append(row)
        if results:
            # Add serial number column
            for i, res in enumerate(results, start=1):
                res["S.No"] = i
            # Create DataFrame with specified column order
            cols = ["S.No", "File Name", "Vendor Name", "Invoice No", "Invoice Date", "Expense Ledger",
                    "GST Type", "Tax Rate", "Basic Amount", "CGST", "SGST", "IGST",
                    "Total Payable", "Narration", "GST Input Eligible", "TDS Applicable", "TDS Rate"]
            df = pd.DataFrame(results)[cols]
            st.success("Extraction completed!")
            st.dataframe(df, use_container_width=True)
            # Download button for CSV
            csv_data = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Results as CSV", csv_data, "invoice_results.csv", "text/csv")
            # Celebration balloons
            st.balloons()
