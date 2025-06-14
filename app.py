import streamlit as st
st.set_page_config(layout="wide")

from PIL import Image
import fitz
import io
import pandas as pd
import base64
import requests
import traceback
from streamlit_lottie import st_lottie
from openai import OpenAI
import tempfile
import os
import locale
import re
from dateutil import parser
import json

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
    "taxable_amount", "cgst", "sgst", "igst", "place_of_supply", "expense_ledger", "tds_applicable"
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

def convert_pdf_first_page(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc.load_page(0)
    pix = page.get_pixmap(dpi=300)
    return Image.open(io.BytesIO(pix.tobytes("png")))

def safe_float(x):
    try:
        cleaned = str(x).replace(",", "").replace("₹", "").replace("$", "").strip()
        return float(cleaned) if cleaned else 0.0
    except:
        return 0.0

def format_currency(x):
    try:
        return f"₹{safe_float(x):,.2f}"
    except:
        return "₹0.00"

def is_valid_gstin(gstin):
    """Improved GSTIN validation with OCR error tolerance"""
    if not gstin or not isinstance(gstin, str):
        return False
        
    # Clean the GSTIN: remove spaces, special characters, convert to uppercase
    cleaned = re.sub(r'[^A-Z0-9]', '', gstin.upper())
    
    # Check length
    if len(cleaned) != 15:
        return False
        
    # Validate pattern with tolerance for common OCR errors
    pattern = r"^\d{2}[A-Z0-9]{10}[A-Z]{1}[A-Z0-9]{1}Z[A-Z0-9]{1}$"
    return bool(re.match(pattern, cleaned))

def extract_gstin_from_text(text):
    """Robust GSTIN extraction from text using pattern matching"""
    # Look for GSTIN pattern in the text
    matches = re.findall(r'\b\d{2}[A-Z0-9]{10}[A-Z]{1}[A-Z0-9]{1}Z[A-Z0-9]{1}\b', text.upper())
    if matches:
        return matches[0]
    return ""

def determine_tds_rate(expense_ledger, tds_applicable, buyer_gstin, place_of_supply, seller_gstin):
    """
    Smart TDS rate determination considering:
    - TDS applicability
    - Buyer/seller location
    - Expense type
    - International transactions
    """
    # No TDS if explicitly not applicable
    if tds_applicable and "no" in tds_applicable.lower():
        return 0.0
        
    # No TDS for international transactions
    if not buyer_gstin and not seller_gstin:
        return 0.0
        
    if place_of_supply and ("international" in place_of_supply.lower() or 
                            "foreign" in place_of_supply.lower()):
        return 0.0
        
    # Check if buyer is international (no GSTIN)
    if not buyer_gstin:
        # For services to international clients, TDS might still apply in some cases
        # But generally not applicable for export services
        return 0.0
        
    # If TDS is applicable but no section specified, determine by expense ledger
    expense_ledger = expense_ledger.lower() if expense_ledger else ""
    
    # Professional services - 10%
    if "professional" in expense_ledger or "consultancy" in expense_ledger or "service" in expense_ledger:
        return 10.0
    
    # Contract work - 2%
    if "contract" in expense_ledger or "sub-contract" in expense_ledger or "work" in expense_ledger:
        return 2.0
    
    # Commission, brokerage - 5%
    if "commission" in expense_ledger or "brokerage" in expense_ledger:
        return 5.0
    
    # Rent - 10%
    if "rent" in expense_ledger:
        return 10.0
    
    # Advertisement - 1%
    if "advertis" in expense_ledger or "marketing" in expense_ledger:
        return 1.0
    
    # Default to 0 if not applicable
    return 0.0

def extract_json_from_response(text):
    """Try to extract JSON from GPT response which might have extra text"""
    try:
        # Look for JSON code block
        matches = re.findall(r'```json\s*({.*?})\s*```', text, re.DOTALL)
        if matches:
            return json.loads(matches[0])
        
        # Look for plain JSON
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1:
            return json.loads(text[start:end+1])
        
        # Try parsing the whole text
        return json.loads(text)
    except Exception:
        return None

# Enhanced prompt with specific GSTIN and TDS instructions
main_prompt = (
    "Extract structured invoice data as a JSON object with the following keys: "
    "invoice_number, date, gstin, seller_name, buyer_name, buyer_gstin, "
    "taxable_amount, cgst, sgst, igst, place_of_supply, expense_ledger, tds_applicable. "
    "Important: 'taxable_amount' is the amount BEFORE taxes. "
    "Use DD/MM/YYYY for dates. Use only values shown in the invoice. "
    "Return 'NOT AN INVOICE' if clearly not one. "
    "If a value is not available, use null. "
    
    "SPECIAL INSTRUCTIONS: "
    "1. For GSTIN: It's a 15-digit alphanumeric code (format: 22AAAAA0000A1Z5) "
    "   found near seller/buyer details. If missing, leave blank."
    "2. For TDS: "
    "   - Return 'No' if TDS is not applicable (e.g., international transactions) "
    "   - Return TDS section (e.g., '194J') if specified "
    "   - Return 'Yes' if applicable but section not specified "
    "3. For place_of_supply: Specify if international "
    
    "For expense_ledger: Classify the nature of expense "
    "(e.g., 'Office Supplies', 'Professional Fees'). "
)

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
                    {"role": "system", "content": "You are a finance assistant specializing in Indian invoices. Pay special attention to GSTIN and TDS applicability."},
                    {"role": "user", "content": [
                        {"type": "text", "text": main_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                    ]}
                ]

                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=chat_prompt,
                    max_tokens=1500  # Increased for better extraction
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
                            "TDS Applicable": "",
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
                    tds_applicable = raw_data.get("tds_applicable", "")
                    
                    # Calculate derived fields
                    total_amount = taxable_amount + cgst + sgst + igst
                    tds_rate = determine_tds_rate(
                        expense_ledger, 
                        tds_applicable,
                        buyer_gstin,
                        place_of_supply,
                        seller_gstin
                    )
                    tds_amount = round(taxable_amount * tds_rate / 100, 2) if tds_rate > 0 else 0.0
                    amount_payable = total_amount - tds_amount
                    
                    # Enhanced GSTIN handling
                    if seller_gstin:
                        # Clean GSTIN by removing spaces and special characters
                        seller_gstin = re.sub(r'[^A-Z0-9]', '', seller_gstin.upper())
                        
                        # Validate the cleaned GSTIN
                        if not is_valid_gstin(seller_gstin):
                            # Try to extract GSTIN from seller name as fallback
                            fallback_gstin = extract_gstin_from_text(seller_name)
                            if fallback_gstin:
                                seller_gstin = fallback_gstin
                    else:
                        # Try to extract GSTIN from seller name if not provided
                        fallback_gstin = extract_gstin_from_text(seller_name)
                        if fallback_gstin:
                            seller_gstin = fallback_gstin
                    
                    # Final GSTIN validation
                    if seller_gstin and not is_valid_gstin(seller_gstin):
                        seller_gstin = ""  # Clear invalid GSTIN
                    
                    # Parse and format date
                    try:
                        parsed_date = parser.parse(str(date), dayfirst=True)
                        date = parsed_date.strftime("%d/%m/%Y")
                    except:
                        date = ""
                    
                    # Clean TDS Applicable field
                    if tds_applicable and "uncertain" in tds_applicable.lower():
                        tds_applicable = ""
                    elif tds_applicable and "no" in tds_applicable.lower():
                        tds_applicable = "No"
                    elif tds_applicable and "yes" in tds_applicable.lower():
                        tds_applicable = "Yes"
                    
                    # Create narration text
                    buyer_gstin_display = buyer_gstin or "N/A"
                    narration = (
                        f"Invoice {invoice_number} dated {date} "
                        f"was issued by {seller_name} (GSTIN: {seller_gstin or 'N/A'}) "
                        f"to {buyer_name} (GSTIN: {buyer_gstin_display}), "
                        f"with a taxable amount of ₹{taxable_amount:,.2f}. "
                        f"Taxes applied - CGST: ₹{cgst:,.2f}, SGST: ₹{sgst:,.2f}, IGST: ₹{igst:,.2f}. "
                        f"Total Amount: ₹{total_amount:,.2f}. "
                        f"Place of supply: {place_of_supply or 'N/A'}. Expense: {expense_ledger or 'N/A'}. "
                        f"TDS Applicable: {tds_applicable or 'N/A'}. "
                        f"TDS Rate: {tds_rate}% (₹{tds_amount:,.2f}). "
                        f"Amount Payable: ₹{amount_payable:,.2f}."
                    )
                    
                    # Add GSTIN status to results
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
                        "TDS Applicable": tds_applicable,
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
                "TDS Applicable": "",
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
            "Buyer Name", "Buyer GSTIN", "Taxable Amount (₹)", "CGST (₹)", 
            "SGST (₹)", "IGST (₹)", "Total Amount (₹)", "TDS Rate (%)", 
            "TDS Amount (₹)", "Amount Payable (₹)", "Place of Supply", 
            "Expense Ledger", "TDS Applicable", "Narration"
        ]
        
        st.dataframe(df[display_cols])

        # CSV Download
        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Results as CSV", csv_data, "invoice_results.csv", "text/csv")

        # Excel Download
        excel_buffer = io.BytesIO()
        try:
            with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                df.to_excel(writer, index=False, sheet_name="Invoice Data")
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
