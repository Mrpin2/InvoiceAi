import streamlit as st
st.set_page_config(layout="wide")

import io
import pandas as pd
import base64
import requests
import tempfile
import os
import locale
from dateutil import parser
import json
from streamlit_lottie import st_lottie
from openai import OpenAI

# Import utility functions
from utils.general_utils import safe_float, format_currency
from utils.gstin_utils import is_valid_gstin, extract_gstin_from_text
from utils.tds_utils import determine_tds_rate
from utils.openai_utils import extract_json_from_response, MAIN_PROMPT
from utils.pdf_utils import convert_pdf_first_page

locale.setlocale(locale.LC_ALL, '')

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

if "files_uploaded" not in st.session_state:
    if hello_json:
        st_lottie(hello_json, height=200, key="hello")

st.markdown("<h2 style='text-align: center;'>📄 AI Invoice Extractor (OpenAI)</h2>", unsafe_allow_html=True)
st.markdown("Upload scanned PDF invoices and extract structured finance data using GPT-4 Vision")
st.markdown("---")

# Define the fields we want to extract
fields = [
    "invoice_number", "date", "gstin", "seller_name", "buyer_name", "buyer_gstin",
    "taxable_amount", "cgst", "sgst", "igst", "place_of_supply", "expense_ledger", "tds"
]

if "processed_results" not in st.session_state:
    st.session_state["processed_results"] = {}
if "processing_status" not in st.session_state:
    st.session_state["processing_status"] = {}
if "summary_rows" not in st.session_state:
    st.session_state["summary_rows"] = []

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

uploaded_files = st.file_uploader("📤 Upload scanned invoice PDFs", type=["pdf"], accept_multiple_files=True)

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

        temp_file_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file.getvalue())
                temp_file_path = tmp.name

            with open(temp_file_path, "rb") as f:
                pdf_data = f.read()
                
            first_image = convert_pdf_first_page(pdf_data)

            with st.spinner("🧠 Extracting data using GPT-4 Vision..."):
                img_buf = io.BytesIO()
                first_image.save(img_buf, format="PNG")
                img_buf.seek(0)
                base64_image = base64.b64encode(img_buf.read()).decode()

                chat_prompt = [
                    {"role": "system", "content": "You are a finance assistant specializing in Indian invoices. Pay special attention to GSTIN extraction."},
                    {"role": "user", "content": [
                        {"type": "text", "text": MAIN_PROMPT},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                    ]}
                ]

                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=chat_prompt,
                    max_tokens=1500
                )

                response_text = response.choices[0].message.content.strip()
                
                # Try to extract JSON from the response
                raw_data = extract_json_from_response(response_text)
                
                if raw_data is None:
                    if "not an invoice" in response_text.lower():
                        result_row = {
                            "File Name": file_name,
                            "Invoice Number": "NOT AN INVOICE",
                            "Date": "",
                            "Seller Name": "",
                            "Seller GSTIN": "",
                            "Buyer Name": "",
                            "Buyer GSTIN": "",
                            "Taxable Amount": 0.0,
                            "CGST": 0.0,
                            "SGST": 0.0,
                            "IGST": 0.0,
                            "Total Amount": 0.0,
                            "TDS Rate": 0.0,
                            "TDS Amount": 0.0,
                            "Amount Payable": 0.0,
                            "Place of Supply": "",
                            "Expense Ledger": "",
                            "TDS": "",
                            "Narration": "This document was identified as not an invoice."
                        }
                    else:
                        raise ValueError("GPT returned non-JSON response")
                else:
                    # Create the result row with all fields
                    invoice_number = raw_data.get("invoice_number", "")
                    date = raw_data.get("date", "")
                    seller_name = raw_data.get("seller_name", "")
                    seller_gstin = raw_data.get("gstin", "")
                    buyer_name = raw_data.get("buyer_name", "")
                    buyer_gstin = raw_data.get("buyer_gstin", "")
                    taxable_amount = safe_float(raw_data.get("taxable_amount", 0.0))
                    cgst = safe_float(raw_data.get("cgst", 0.0))
                    sgst = safe_float(raw_data.get("sgst", 0.0))
                    igst = safe_float(raw_data.get("igst", 0.0))
                    place_of_supply = raw_data.get("place_of_supply", "")
                    expense_ledger = raw_data.get("expense_ledger", "")
                    tds_str = raw_data.get("tds", "")
                    
                    # Calculate derived fields
                    total_amount = taxable_amount + cgst + sgst + igst
                    tds_rate = determine_tds_rate(expense_ledger, tds_str)
                    tds_amount = round(taxable_amount * tds_rate / 100, 2)
                    amount_payable = total_amount - tds_amount
                    
                    # Enhanced GSTIN handling
                    gstin_status = "VALID"
                    
                    # Clean and validate GSTIN
                    if seller_gstin:
                        # Clean GSTIN by removing spaces and special characters
                        seller_gstin = re.sub(r'[^A-Z0-9]', '', seller_gstin.upper())
                        
                        # Validate the cleaned GSTIN
                        if not is_valid_gstin(seller_gstin):
                            gstin_status = "INVALID"
                            
                            # Try to extract GSTIN from seller name as fallback
                            fallback_gstin = extract_gstin_from_text(seller_name)
                            if fallback_gstin:
                                seller_gstin = fallback_gstin
                    else:
                        # Try to extract GSTIN from seller name if not provided
                        fallback_gstin = extract_gstin_from_text(seller_name)
                        if fallback_gstin:
                            seller_gstin = fallback_gstin
                    
                    # Parse and format date
                    try:
                        parsed_date = parser.parse(str(date), dayfirst=True)
                        date = parsed_date.strftime("%d/%m/%Y")
                    except:
                        date = ""
                    
                    # Create narration text
                    buyer_gstin_display = buyer_gstin or "N/A"
                    narration = (
                        f"Invoice {invoice_number} dated {date} "
                        f"was issued by {seller_name} (GSTIN: {seller_gstin}) "
                        f"to {buyer_name} (GSTIN: {buyer_gstin_display}), "
                        f"with a taxable amount of ₹{taxable_amount:,.2f}. "
                        f"Taxes applied - CGST: ₹{cgst:,.2f}, SGST: ₹{sgst:,.2f}, IGST: ₹{igst:,.2f}. "
                        f"Total Amount: ₹{total_amount:,.2f}. "
                        f"Place of supply: {place_of_supply or 'N/A'}. Expense: {expense_ledger or 'N/A'}. "
                        f"TDS: {tds_str or 'N/A'} @ {tds_rate}% (₹{tds_amount:,.2f}). "
                        f"Amount Payable: ₹{amount_payable:,.2f}."
                    )
                    
                    # Create result row
                    result_row = {
                        "File Name": file_name,
                        "Invoice Number": invoice_number,
                        "Date": date,
                        "Seller Name": seller_name,
                        "Seller GSTIN": seller_gstin,
                        "Buyer Name": buyer_name,
                        "Buyer GSTIN": buyer_gstin,
                        "Taxable Amount": taxable_amount,
                        "CGST": cgst,
                        "SGST": sgst,
                        "IGST": igst,
                        "Total Amount": total_amount,
                        "TDS Rate": tds_rate,
                        "TDS Amount": tds_amount,
                        "Amount Payable": amount_payable,
                        "Place of Supply": place_of_supply,
                        "Expense Ledger": expense_ledger,
                        "TDS": tds_str,
                        "Narration": narration
                    }

                st.session_state["processed_results"][file_name] = result_row
                st.session_state["processing_status"][file_name] = "✅ Done"
                completed_count += 1
                st.success(f"{file_name}: ✅ Done")

        except Exception as e:
            error_row = {
                "File Name": file_name,
                "Invoice Number": "PROCESSING ERROR",
                "Date": "",
                "Seller Name": "",
                "Seller GSTIN": "",
                "Buyer Name": "",
                "Buyer GSTIN": "",
                "Taxable Amount": 0.0,
                "CGST": 0.0,
                "SGST": 0.0,
                "IGST": 0.0,
                "Total Amount": 0.0,
                "TDS Rate": 0.0,
                "TDS Amount": 0.0,
                "Amount Payable": 0.0,
                "Place of Supply": "",
                "Expense Ledger": "",
                "TDS": "",
                "Narration": f"Error processing file: {str(e)}"
            }
            st.session_state["processed_results"][file_name] = error_row
            st.session_state["processing_status"][file_name] = "❌ Error"
            st.error(f"❌ Error processing {file_name}: {e}")
            st.text_area(f"Raw Output ({file_name})", response_text if 'response_text' in locals() else "No response", height=200)

        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

# Get all processed results
results = list(st.session_state["processed_results"].values())

if results:
    if completed_json:
        st_lottie(completed_json, height=200, key="done_animation")

    st.markdown("<h3 style='text-align: center;'>🎉 Yippie! All invoices processed with a smile 😊</h3>", unsafe_allow_html=True)

    # Create DataFrame
    try:
        df = pd.DataFrame(results)
        
        # Format currency columns
        currency_cols = ["Taxable Amount", "CGST", "SGST", "IGST", "Total Amount", "TDS Amount", "Amount Payable"]
        for col in currency_cols:
            df[f"{col} (₹)"] = df[col].apply(format_currency)
        
        # Format TDS Rate as percentage
        df["TDS Rate (%)"] = df["TDS Rate"].apply(lambda x: f"{x}%")
        
        # Reorder columns for better display
        display_cols = [
            "File Name", "Invoice Number", "Date", "Seller Name", "Seller GSTIN", 
            "Buyer Name", "Buyer GSTIN", "Taxable Amount (₹)", 
            "CGST (₹)", "SGST (₹)", "IGST (₹)", "Total Amount (₹)", "TDS Rate (%)", 
            "TDS Amount (₹)", "Amount Payable (₹)", "Place of Supply", 
            "Expense Ledger", "TDS", "Narration"
        ]
        
        st.dataframe(df[display_cols])

        # Create download dataframe
        download_df = df[display_cols].copy()
        
        # CSV Download
        csv_data = download_df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Results as CSV", csv_data, "invoice_results.csv", "text/csv")

        # Excel Download
        excel_buffer = io.BytesIO()
        try:
            with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                download_df.to_excel(writer, index=False, sheet_name="Invoice Data")
            st.download_button(
                label="📥 Download Results as Excel",
                data=excel_buffer.getvalue(),
                file_name="invoice_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        except Exception as e:
            st.error(f"Failed to create Excel file: {str(e)}")
            
    except Exception as e:
        st.error(f"Error creating results table: {str(e)}")
        st.write("Raw results data:")
        st.json(results)

    st.markdown("---")
    if st.session_state.summary_rows:
        st.balloons()
else:
    st.info("Upload one or more scanned invoices to get started.")
