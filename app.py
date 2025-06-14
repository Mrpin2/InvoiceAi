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
import datetime

# Set locale for currency formatting
try:
    locale.setlocale(locale.LC_ALL, 'en_IN.UTF-8')
except:
    locale.setlocale(locale.LC_ALL, '')

# Lottie animations
HELLO_LOTTIE = "https://raw.githubusercontent.com/Mrpin2/InvoiceAi/refs/heads/main/Animation%20-%201749845212531.json"
COMPLETED_LOTTIE = "https://raw.githubusercontent.com/Mrpin2/InvoiceAi/refs/heads/main/Animation%20-%201749845303699.json"

def load_lottie_json(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except Exception:
        return None

hello_json = load_lottie_json(HELLO_LOTTIE)
completed_json = load_lottie_json(COMPLETED_LOTTIE)

# Initialize session state
if "files_uploaded" not in st.session_state:
    if hello_json:
        st_lottie(hello_json, height=200, key="hello")

st.markdown("<h2 style='text-align: center;'>📄 AI Invoice Extractor (OpenAI)</h2>", unsafe_allow_html=True)
st.markdown("Upload scanned PDF invoices and extract structured finance data using GPT-4 Vision")
st.markdown("---")

# Initialize session state variables
if "processed_results" not in st.session_state:
    st.session_state.processed_results = {}
if "processing_status" not in st.session_state:
    st.session_state.processing_status = {}
if "summary_rows" not in st.session_state:
    st.session_state.summary_rows = []

# Authentication
st.sidebar.header("🔐 AI Config")
passcode = st.sidebar.text_input("Admin Passcode", type="password")
admin_unlocked = passcode == "Essenbee"

openai_api_key = None
if admin_unlocked:
    st.sidebar.success("🔓 Admin access granted.")
    openai_api_key = st.secrets.get("OPENAI_API_KEY", "")
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
    """Convert first page of PDF to high-quality image"""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc.load_page(0)
    pix = page.get_pixmap(dpi=300)
    return Image.open(io.BytesIO(pix.tobytes("png")))

def safe_float(value):
    """Safely convert value to float, handling currency symbols and commas"""
    if isinstance(value, (int, float)):
        return float(value)
    
    if not isinstance(value, str):
        return 0.0
        
    cleaned = re.sub(r'[^\d.]', '', value)
    try:
        return float(cleaned) if cleaned else 0.0
    except:
        return 0.0

def format_currency(amount):
    """Format amount as Indian currency"""
    try:
        return locale.currency(amount, grouping=True)
    except:
        return "₹0.00"

def is_valid_gstin(gstin):
    """Validate GSTIN format"""
    if not gstin or not isinstance(gstin, str):
        return False
    pattern = r"^[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[1-9A-Z]{1}[Z]{1}[0-9A-Z]{1}$"
    return bool(re.match(pattern, gstin.strip()))

def is_company(seller_name):
    """Determine if seller is a company based on name"""
    if not seller_name or not isinstance(seller_name, str):
        return False
        
    seller_lower = seller_name.lower()
    company_keywords = [
        'pvt', 'ltd', 'limited', 'llp', 'inc', 'plc', 'corp', 
        'company', 'co.', 'co ', 'corporation', 'pvt. ltd.'
    ]
    return any(kw in seller_lower for kw in company_keywords)

def determine_tds_rate(tds_str, seller_name, expense_ledger, invoice_amount):
    """
    Determine TDS rate based on:
    - TDS string content
    - Seller type (company vs individual)
    - Expense ledger category
    - Invoice amount thresholds
    """
    # Return 0 if TDS is explicitly not applicable
    if tds_str and isinstance(tds_str, str):
        tds_lower = tds_str.lower()
        if "no" in tds_lower or "not applicable" in tds_lower or "n/a" in tds_lower:
            return 0.0
    
    # Try to extract rate from TDS string
    if tds_str and isinstance(tds_str, str):
        match = re.search(r'(\d+(\.\d+)?)%', tds_str)
        if match:
            return float(match.group(1))
        
        # Match by section
        section_mapping = {
            "194j": 10.0,   # Professional services
            "194c": 2.0,    # Contracts
            "194h": 5.0,    # Commission/brokerage
            "194i": 10.0,   # Rent
            "194q": 2.0,    # Purchase of goods
            "194r": 1.0     # Advertising
        }
        for section, rate in section_mapping.items():
            if section in tds_str.lower():
                return rate
    
    # Apply business rules based on seller type
    if is_company(seller_name):
        # For companies: 2% TDS
        return 2.0
    else:
        # For individuals: 1% TDS
        return 1.0

def extract_json_from_response(text):
    """Robust JSON extraction from GPT response"""
    try:
        # Handle JSON wrapped in code blocks
        matches = re.findall(r'```(?:json)?\s*({.*?})\s*```', text, re.DOTALL)
        if matches:
            return json.loads(matches[0])
        
        # Extract JSON from plain text
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start:end+1])
        
        # Try parsing the whole text
        return json.loads(text)
    except json.JSONDecodeError:
        return None
    except Exception:
        return None

# Enhanced prompt with clearer instructions
MAIN_PROMPT = """
Extract structured invoice data as a JSON object with the following keys:
- invoice_number: Invoice number/ID
- date: Invoice date in DD/MM/YYYY format
- gstin: Seller's GSTIN
- seller_name: Seller's legal name
- buyer_name: Buyer's name
- buyer_gstin: Buyer's GSTIN (if available)
- taxable_amount: Amount before taxes (taxable value)
- cgst: Central GST amount
- sgst: State GST amount
- igst: Integrated GST amount
- place_of_supply: State code or name
- expense_ledger: Expense category (e.g., 'Office Supplies', 'Professional Fees')
- tds: TDS applicability and section if mentioned

Important:
1. 'taxable_amount' should be the amount BEFORE taxes
2. Total amount = taxable_amount + cgst + sgst + igst
3. For expense_ledger, classify based on invoice content
4. For tds, note any mentioned sections like '194C', '194J'
5. Use only values shown in the invoice
6. Return 'NOT AN INVOICE' if document is clearly not an invoice
7. Use null for unavailable values
8. Dates must be in DD/MM/YYYY format
"""

# File uploader
uploaded_files = st.file_uploader(
    "📤 Upload scanned invoice PDFs", 
    type=["pdf"], 
    accept_multiple_files=True
)

if uploaded_files:
    st.session_state.files_uploaded = True
    total_files = len(uploaded_files)
    completed_count = sum(
        1 for file in uploaded_files 
        if file.name in st.session_state.processed_results
    )
    
    # Reset button
    if st.button("🔄 Reset Processing"):
        st.session_state.processed_results = {}
        st.session_state.processing_status = {}
        st.experimental_rerun()

    for idx, file in enumerate(uploaded_files):
        file_name = file.name
        if file_name in st.session_state.processed_results:
            continue

        st.subheader(f"Processing: {file_name} ({idx+1}/{total_files})")
        st.session_state.processing_status[file_name] = "⏳ Processing..."
        
        progress_bar = st.progress(0)
        temp_file_path = None
        
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file.getvalue())
                temp_file_path = tmp.name
            progress_bar.progress(20)

            # Convert to image
            with open(temp_file_path, "rb") as f:
                pdf_data = f.read()
            first_image = convert_pdf_first_page(pdf_data)
            progress_bar.progress(40)
            
            # Prepare image for API
            img_buf = io.BytesIO()
            first_image.save(img_buf, format="PNG")
            img_buf.seek(0)
            base64_image = base64.b64encode(img_buf.read()).decode()
            progress_bar.progress(60)

            # Call OpenAI API
            with st.spinner("🧠 Extracting data using GPT-4 Vision..."):
                chat_prompt = [
                    {
                        "role": "system", 
                        "content": "You are an expert Indian finance assistant. Extract invoice data accurately."
                    },
                    {
                        "role": "user", 
                        "content": [
                            {"type": "text", "text": MAIN_PROMPT},
                            {
                                "type": "image_url", 
                                "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                            }
                        ]
                    }
                ]

                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=chat_prompt,
                    max_tokens=1500,
                    response_format={"type": "json_object"}
                )
                progress_bar.progress(80)

                response_text = response.choices[0].message.content.strip()
                raw_data = extract_json_from_response(response_text)
                
                if not raw_data:
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
                            "Place of Supply": "",
                            "Expense Ledger": "",
                            "TDS": "",
                            "TDS Rate": 0.0,
                            "TDS Amount": 0.0,
                            "Net Payable": 0.0,
                            "Narration": "Document identified as not an invoice"
                        }
                    else:
                        raise ValueError("GPT returned invalid response format")
                else:
                    # Extract and validate data
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
                    
                    # Calculate amounts
                    total_amount = taxable_amount + cgst + sgst + igst
                    tds_rate = determine_tds_rate(tds_str, seller_name, expense_ledger, total_amount)
                    tds_amount = round(taxable_amount * tds_rate / 100, 2) if tds_rate > 0 else 0.0
                    net_payable = total_amount - tds_amount
                    
                    # Validate GSTIN
                    if seller_gstin and not is_valid_gstin(seller_gstin):
                        seller_gstin = "INVALID GSTIN"
                    
                    # Parse and format date
                    try:
                        parsed_date = parser.parse(date, dayfirst=True, fuzzy=True)
                        date = parsed_date.strftime("%d/%m/%Y")
                    except:
                        # Try common date formats
                        for fmt in ("%d-%m-%Y", "%d/%m/%Y", "%Y-%m-%d", "%d %b %Y"):
                            try:
                                parsed_date = datetime.datetime.strptime(date, fmt)
                                date = parsed_date.strftime("%d/%m/%Y")
                                break
                            except:
                                pass
                        else:
                            date = "INVALID DATE"
                    
                    # Create narration
                    seller_type = "Company" if is_company(seller_name) else "Individual"
                    narration = (
                        f"Invoice {invoice_number} dated {date} from {seller_name} ({seller_type}) "
                        f"(GSTIN: {seller_gstin}) to {buyer_name} (GSTIN: {buyer_gstin or 'N/A'}). "
                        f"Taxable: {format_currency(taxable_amount)}, GST: CGST {format_currency(cgst)}, "
                        f"SGST {format_currency(sgst)}, IGST {format_currency(igst)}. "
                        f"Total: {format_currency(total_amount)}. "
                        f"Place of Supply: {place_of_supply or 'N/A'}, "
                        f"Expense: {expense_ledger or 'N/A'}, "
                        f"TDS: {tds_str or 'N/A'} @ {tds_rate}% ({format_currency(tds_amount)}). "
                        f"Net Payable: {format_currency(net_payable)}."
                    )
                    
                    result_row = {
                        "File Name": file_name,
                        "Invoice Number": invoice_number,
                        "Date": date,
                        "Seller Name": seller_name,
                        "Seller Type": seller_type,
                        "Seller GSTIN": seller_gstin,
                        "Buyer Name": buyer_name,
                        "Buyer GSTIN": buyer_gstin,
                        "Taxable Amount": taxable_amount,
                        "CGST": cgst,
                        "SGST": sgst,
                        "IGST": igst,
                        "Total Amount": total_amount,
                        "Place of Supply": place_of_supply,
                        "Expense Ledger": expense_ledger,
                        "TDS": tds_str,
                        "TDS Rate": tds_rate,
                        "TDS Amount": tds_amount,
                        "Net Payable": net_payable,
                        "Narration": narration
                    }

                st.session_state.processed_results[file_name] = result_row
                st.session_state.processing_status[file_name] = "✅ Done"
                progress_bar.progress(100)
                st.success(f"✅ Successfully processed {file_name}")
                
        except Exception as e:
            error_msg = f"❌ Error processing {file_name}: {str(e)}"
            st.error(error_msg)
            
            error_row = {
                "File Name": file_name,
                "Invoice Number": "ERROR",
                "Date": "",
                "Seller Name": "",
                "Seller Type": "",
                "Seller GSTIN": "",
                "Buyer Name": "",
                "Buyer GSTIN": "",
                "Taxable Amount": 0.0,
                "CGST": 0.0,
                "SGST": 0.0,
                "IGST": 0.0,
                "Total Amount": 0.0,
                "Place of Supply": "",
                "Expense Ledger": "",
                "TDS": "",
                "TDS Rate": 0.0,
                "TDS Amount": 0.0,
                "Net Payable": 0.0,
                "Narration": error_msg
            }
            st.session_state.processed_results[file_name] = error_row
            st.session_state.processing_status[file_name] = "❌ Error"
            
            # Show debug info
            with st.expander("Error Details"):
                st.text(traceback.format_exc())
                if 'response_text' in locals():
                    st.text_area("GPT Response", response_text, height=200)
            
        finally:
            # Clean up temporary file
            if temp_file_path and os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

# Display results
results = list(st.session_state.processed_results.values())
if results:
    if completed_json:
        st_lottie(completed_json, height=200, key="done_animation")

    st.markdown("<h3 style='text-align: center;'>🎉 Processing Complete! 😊</h3>", unsafe_allow_html=True)
    
    try:
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Format currency columns
        currency_cols = ["Taxable Amount", "CGST", "SGST", "IGST", "Total Amount", "TDS Amount", "Net Payable"]
        for col in currency_cols:
            df[f"{col} (₹)"] = df[col].apply(format_currency)
        
        # Reorder columns
        display_cols = [
            "File Name", "Invoice Number", "Date", "Seller Name", "Seller Type", "Seller GSTIN",
            "Buyer Name", "Buyer GSTIN", "Taxable Amount (₹)", "CGST (₹)", 
            "SGST (₹)", "IGST (₹)", "Total Amount (₹)", "Place of Supply", 
            "Expense Ledger", "TDS", "TDS Rate", "TDS Amount (₹)", 
            "Net Payable (₹)", "Narration"
        ]
        
        # Display results
        st.dataframe(df[display_cols], height=600)
        
        # Download options
        col1, col2 = st.columns(2)
        with col1:
            csv_data = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download as CSV", 
                csv_data, 
                "invoice_results.csv", 
                "text/csv"
            )
        with col2:
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                df.to_excel(writer, index=False, sheet_name="Invoice Data")
            st.download_button(
                "📥 Download as Excel",
                excel_buffer.getvalue(),
                "invoice_results.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            
    except Exception as e:
        st.error(f"Error creating results: {str(e)}")
        with st.expander("Raw Results Data"):
            st.json(results)

    st.markdown("---")
    st.balloons()
else:
    st.info("Upload PDF invoices and process them to see results")
