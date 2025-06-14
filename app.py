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
    "taxable_amount", "cgst", "sgst", "igst", "place_of_supply",
    "expense_ledger", "tds", "hsn_sac"
]


if "processed_results" not in st.session_state:
    st.session_state["processed_results"] = {}
if "processing_status" not in st.session_state:
    st.session_state["processing_status"] = {}
if "summary_rows" not in st.session_state:
    st.session_state["summary_rows"] = []

st.sidebar.header("🔐 AI Config")
passcode = st.sidebar.text_input("Admin Passcode", type="password")
admin_unlocked = passcode == "Rajeev"

openai_api_key = None
if admin_unlocked:
    st.sidebar.success("🔓 Admin access granted.")
    # Assuming OPENAI_API_KEY is correctly set in Streamlit Secrets
    openai_api_key = st.secrets.get("OPENAI_API_KEY")
    if not openai_api_key:
        st.sidebar.error("OPENAI_API_KEY missing in Streamlit secrets.")
        st.stop()
else:
    openai_api_key = st.sidebar.text_input("🔑 Enter your OpenAI API Key", type="password")
    if not openai_api_key:
        st.sidebar.warning("Please enter a valid API key to continue.")
        st.stop()

# Initialize OpenAI client after API key is confirmed
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
        # Handle cases where x might already be formatted or non-numeric
        if isinstance(x, str) and x.startswith('₹'):
            return x # Already formatted
        return f"₹{safe_float(x):,.2f}"
    except:
        return "₹0.00"

def is_valid_gstin(gstin):
    """Validate GSTIN format with more flexibility"""
    if not gstin:
        return False
        
    # Clean the GSTIN: remove spaces, special characters, convert to uppercase
    cleaned = re.sub(r'[^A-Z0-9]', '', gstin.upper())
        
    # GSTIN must be exactly 15 characters
    if len(cleaned) != 15:
        return False
        
    # Validate pattern: 2 digits + 10 alphanumeric + 1 letter + 1 alphanumeric + 1 letter
    pattern = r"^\d{2}[A-Z0-9]{10}[A-Z]{1}[A-Z0-9]{1}[Z]{1}[A-Z0-9]{1}$"
    return bool(re.match(pattern, cleaned))

def extract_gstin_from_text(text):
    """Try to extract GSTIN from any text using pattern matching"""
    # Look for GSTIN pattern in the text
    matches = re.findall(r'\b\d{2}[A-Z0-9]{10}[A-Z]{1}[A-Z0-9]{1}Z[A-Z0-9]{1}\b', text.upper())
    if matches:
        return matches[0] # Return the first found valid GSTIN
    return ""

def determine_tds_rate(expense_ledger, tds_str=""):
    """Determine TDS rate based on expense ledger and TDS string"""
    # First check if TDS string contains a specific rate
    if tds_str and isinstance(tds_str, str):
        # Look for percentage in the TDS string
        match = re.search(r'(\d+(\.\d+)?)%', tds_str)
        if match:
            return float(match.group(1))
            
        # Check for TDS sections (GPT might return section in 'tds' field)
        section_rates = {
            "194j": 10.0,  # Professional services
            "194c": 2.0,   # Contracts (individual/HUF) or 1% (company/other)
            "194i": 10.0,  # Rent (plant/machinery/equipment) or 2% (land/building)
            "194h": 5.0,   # Commission/brokerage
            "194q": 0.1    # Purchase of goods
        }
        for section, rate in section_rates.items():
            if section in tds_str.lower():
                return rate
        
    # If no TDS string info, determine by expense ledger (simplified logic as per our chat)
    expense_ledger = expense_ledger.lower() if expense_ledger else ""
    
    if "professional" in expense_ledger or "consultancy" in expense_ledger or "service" in expense_ledger:
        return 10.0 # Common for 194J
    
    if "contract" in expense_ledger or "work" in expense_ledger:
        return 1.0 # Common rate for 194C (can be 2% for non-company) - choose a typical one or refine as needed
    
    if "rent" in expense_ledger:
        return 10.0 # Common for 194I (can be 2% for land/building) - choose a typical one or refine as needed
    
    # Default to 0 if not applicable
    return 0.0

def determine_tds_section(expense_ledger):
    """Determine TDS section based on expense ledger (simplified to 194J for Professional Fees)"""
    expense_ledger = expense_ledger.lower() if expense_ledger else ""
    if "professional" in expense_ledger or "consultancy" in expense_ledger or "service" in expense_ledger:
        return "194J"
    # Add more rules here for other sections if needed
    # elif "contract" in expense_ledger:
    #     return "194C"
    # elif "rent" in expense_ledger:
    #     return "194I"
    return None # Return None if no specific section is matched

def extract_json_from_response(text):
    """Try to extract JSON from GPT response which might have extra text"""
    try:
        # Look for JSON code block
        matches = re.findall(r'```json\s*({.*?})\s*```', text, re.DOTALL)
        if matches:
            # json.loads expects a string, so apply it to the first matched string
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

# Enhanced prompt with specific GSTIN and HSN/SAC instructions, and updated TDS instruction
main_prompt = (
    "You are an expert at extracting structured data from Indian invoices. "
    "Your task is to extract information into a JSON object with the following keys. "
    "If a key's value is not explicitly present or derivable from the invoice, use `null` for that value. "
    "Keys to extract: invoice_number, date, gstin (seller's GSTIN), seller_name, buyer_name, buyer_gstin, "
    "taxable_amount, cgst, sgst, igst, place_of_supply, expense_ledger, tds, hsn_sac. "
    
    "GUIDELINES FOR EXTRACTION:\n"
    "- 'invoice_number': The unique identifier of the invoice.\n"
    "- 'date': The invoice date in DD/MM/YYYY format.\n"
    "- 'taxable_amount': This is the subtotal, the amount BEFORE any taxes (CGST, SGST, IGST) are applied.\n"
    "- 'gstin': The GSTIN of the seller (the entity issuing the invoice).\n"
    "- 'buyer_gstin': The GSTIN of the buyer (the entity receiving the invoice).\n"
    "- 'hsn_sac': Crucial for Indian invoices. "
    "  - HSN (Harmonized System of Nomenclature) is for goods.\n"
    "  - SAC (Service Accounting Code) is for services.\n"
    "  - **ONLY extract the HSN/SAC code if it is explicitly mentioned on the invoice.** "
    "  - It is typically a 4, 6, or 8-digit numeric code, sometimes alphanumeric.\n"
    "  - Look for labels like 'HSN Code', 'SAC Code', 'HSN/SAC', or just the code itself near item descriptions.\n"
    "  - If multiple HSN/SAC codes are present for different line items, extract the one that appears most prominently, or the first one listed. If only one is present for the whole invoice, use that.\n"
    "  - **If HSN/SAC is NOT found or explicitly stated, the value MUST be `null`. Do NOT guess or infer it.**\n"
    
    "- 'expense_ledger': Classify the nature of expense and suggest a suitable ledger type "
    "  (e.g., 'Office Supplies', 'Professional Fees', 'Software Subscription', 'Rent').\n"
    "- 'tds': Determine TDS applicability. State 'Yes - Section [X]' if applicable with a section, 'No' if clearly not, or 'Uncertain' if unclear. Always try to identify the TDS Section (e.g., 194J, 194C, 194I) if TDS is applicable.\n"
    
    "- 'place_of_supply': Crucial for Indian invoices to determine IGST applicability. "
    "  - **PRIORITY 1:** Look for a field explicitly labeled 'Place of Supply'. Extract the value directly from there."
    "  - **PRIORITY 2:** If 'Place of Supply' is not found, look for 'Ship To:' address. Extract the state/city from this address."
    "  - **PRIORITY 3:** If 'Ship To:' is not found, look for 'Bill To:' address. Extract the state/city from this address."
    "  - **PRIORITY 4:** If neither of the above, infer from the Customer/Buyer Address. Extract the state/city from this address."
    "  - **SPECIAL CASE:** If the invoice is clearly an 'Export Invoice' or indicates foreign trade, set the value to 'Foreign'."
    "  - **FALLBACK:** If none of the above are found or inferable, the value MUST be `null`."
    
    "Return 'NOT AN INVOICE' if the document is clearly not an invoice.\n"
    "Ensure the JSON output is clean and directly parsable."
)

uploaded_files = st.file_uploader("📤 Upload scanned invoice PDFs", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    st.session_state["files_uploaded"] = True
    total_files = len(uploaded_files)
    completed_count = 0

    for idx, file in enumerate(uploaded_files):
        file_name = file.name
        # Skip if already processed
        if file_name in st.session_state["processed_results"]:
            continue

        st.markdown(f"**Processing file: {file_name} ({idx+1}/{total_files})**")
        st.session_state["processing_status"][file_name] = "⏳ Pending..."

        temp_file_path = None
        response_text = None # Initialize response_text here
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
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": main_prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                        ]
                    }
                ]

                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=chat_prompt,
                    max_tokens=2000
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
                            "HSN/SAC": "",
                            "Buyer Name": "",
                            "Buyer GSTIN": "",
                            "Expense Ledger": "", # Moved here for consistency
                            "Taxable Amount": 0.0,
                            "CGST": 0.0,
                            "SGST": 0.0,
                            "IGST": 0.0,
                            "Total Amount": 0.0,
                            "TDS Applicability": "N/A", # Not applicable if not an invoice
                            "TDS Section": None,
                            "TDS Rate": 0.0,
                            "TDS Amount": 0.0,
                            "Amount Payable": 0.0,
                            "Place of Supply": "",
                            "TDS": "",
                            "Narration": "This document was identified as not an invoice."
                        }
                    else:
                        st.warning(f"GPT returned non-JSON response for {file_name}: {response_text}")
                        raise ValueError(f"GPT returned non-JSON response or unexpected format for {file_name}.")
                else:
                    # Initialize all fields from raw_data, defaulting to empty string or 0.0 for robustness
                    invoice_number = raw_data.get("invoice_number", "")
                    date = raw_data.get("date", "")
                    seller_name = raw_data.get("seller_name", "")
                    seller_gstin = raw_data.get("gstin", "")
                    hsn_sac = raw_data.get("hsn_sac", "")
                    buyer_name = raw_data.get("buyer_name", "")
                    buyer_gstin = raw_data.get("buyer_gstin", "")
                    expense_ledger = raw_data.get("expense_ledger", "") # Moved definition here
                    taxable_amount = safe_float(raw_data.get("taxable_amount", 0.0))
                    cgst = safe_float(raw_data.get("cgst", 0.0))
                    sgst = safe_float(raw_data.get("sgst", 0.0))
                    igst = safe_float(raw_data.get("igst", 0.0))
                    place_of_supply = raw_data.get("place_of_supply", "")
                    tds_str = raw_data.get("tds", "")
                    
                    # Calculate derived fields
                    total_amount = taxable_amount + cgst + sgst + igst
                    tds_rate = determine_tds_rate(expense_ledger, tds_str)
                    
                    tds_amount = round(taxable_amount * tds_rate / 100, 2) if tds_rate > 0 else 0.0
                    
                    amount_payable = total_amount - tds_amount
                    
                    # Determine TDS Section
                    tds_section = determine_tds_section(expense_ledger)
                    
                    # Determine TDS Applicability
                    tds_applicability = "Uncertain"
                    if tds_rate > 0 or tds_amount > 0:
                        tds_applicability = "Yes"
                    elif "no" in str(tds_str).lower():
                        tds_applicability = "No"

                    # Enhanced GSTIN handling
                    if seller_gstin:
                        seller_gstin = re.sub(r'[^A-Z0-9]', '', seller_gstin.upper())
                    if not is_valid_gstin(seller_gstin):
                        fallback_gstin = extract_gstin_from_text(str(seller_name) + " " + str(seller_gstin))
                        if fallback_gstin:
                            seller_gstin = fallback_gstin
                            
                    if buyer_gstin:
                        buyer_gstin = re.sub(r'[^A-Z0-9]', '', buyer_gstin.upper())
                    if not is_valid_gstin(buyer_gstin):
                        fallback_buyer_gstin = extract_gstin_from_text(str(buyer_name) + " " + str(buyer_gstin))
                        if fallback_buyer_gstin:
                            buyer_gstin = fallback_buyer_gstin
                    
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
                        f"was issued by {seller_name} (GSTIN: {seller_gstin or 'N/A'}, HSN/SAC: {hsn_sac or 'N/A'}) "
                        f"to {buyer_name} (GSTIN: {buyer_gstin_display}), "
                        f"with a taxable amount of ₹{taxable_amount:,.2f}. "
                        f"Taxes applied - CGST: ₹{cgst:,.2f}, SGST: ₹{sgst:,.2f}, IGST: ₹{igst:,.2f}. "
                        f"Total Amount: ₹{total_amount:,.2f}. "
                        f"Place of supply: {place_of_supply or 'N/A'}. Expense: {expense_ledger or 'N/A'}. "
                        f"TDS: {tds_applicability} (Section: {tds_section or 'N/A'}) @ {tds_rate}% (₹{tds_amount:,.2f}). "
                        f"Amount Payable: ₹{amount_payable:,.2f}."
                    )
                    
                    result_row = {
                        "File Name": file_name,
                        "Invoice Number": invoice_number,
                        "Date": date,
                        "Seller Name": seller_name,
                        "Seller GSTIN": seller_gstin,
                        "HSN/SAC": hsn_sac,
                        "Buyer Name": buyer_name,
                        "Buyer GSTIN": buyer_gstin,
                        "Expense Ledger": expense_ledger, # Moved position in result_row
                        "Taxable Amount": taxable_amount,
                        "CGST": cgst,
                        "SGST": sgst,
                        "IGST": igst,
                        "Total Amount": total_amount,
                        "TDS Applicability": tds_applicability,
                        "TDS Section": tds_section,
                        "TDS Rate": tds_rate,
                        "TDS Amount": tds_amount,
                        "Amount Payable": amount_payable,
                        "Place of Supply": place_of_supply,
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
                "HSN/SAC": "",
                "Buyer Name": "",
                "Buyer GSTIN": "",
                "Expense Ledger": "", # Moved here for consistency
                "Taxable Amount": 0.0,
                "CGST": 0.0,
                "SGST": 0.0,
                "IGST": 0.0,
                "Total Amount": 0.0,
                "TDS Applicability": "Uncertain",
                "TDS Section": None,
                "TDS Rate": 0.0,
                "TDS Amount": 0.0,
                "Amount Payable": 0.0,
                "Place of Supply": "",
                "TDS": "",
                "Narration": f"Error processing file: {str(e)}. Raw response: {response_text if 'response_text' in locals() else 'No response'}"
            }
            st.session_state["processed_results"][file_name] = error_row
            st.session_state["processing_status"][file_name] = "❌ Error"
            st.error(f"❌ Error processing {file_name}: {e}")
            if response_text:
                st.text_area(f"Raw Output ({file_name})", response_text, height=200)
            else:
                st.text_area(f"Raw Output ({file_name})", "No response received.", height=100)

        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

# Get all processed results
results = list(st.session_state["processed_results"].values())

if results:
    if completed_json:
        st_lottie(completed_json, height=200, key="done_animation")

    st.markdown("<h3 style='text-align: center;'>🎉 Yippie! All invoices Processed!!! 😊</h3>", unsafe_allow_html=True)

    # Create DataFrame
    try:
        df = pd.DataFrame(results)
        
        # Define currency columns and their display names
        currency_cols_mapping = {
            "Taxable Amount": "Taxable Amount (₹)",
            "CGST": "CGST (₹)",
            "SGST": "SGST (₹)",
            "IGST": "IGST (₹)",
            "Total Amount": "Total Amount (₹)",
            "TDS Amount": "TDS Amount (₹)",
            "Amount Payable": "Amount Payable (₹)"
        }
        
        for original_col, display_col in currency_cols_mapping.items():
            if original_col in df.columns:
                df[display_col] = df[original_col].apply(format_currency)
            else:
                df[display_col] = "₹0.00"

        # Format 'TDS Rate' as percentage
        if 'TDS Rate' in df.columns:
            df['TDS Rate'] = pd.to_numeric(df['TDS Rate'], errors='coerce').fillna(0.0)
            df['TDS Rate (%)'] = df['TDS Rate'].apply(lambda x: f"{x:.1f}%" if x > 0 else "0.0%")
        else:
            df['TDS Rate (%)'] = "0.0%"

        # Reorder columns for better display
        display_cols = [
            "File Name",
            "Invoice Number",
            "Date",
            "Seller Name",
            "Seller GSTIN",
            "HSN/SAC",
            "Buyer Name",
            "Buyer GSTIN",
            "Expense Ledger", # MOVED HERE
            "Taxable Amount (₹)",
            "CGST (₹)",
            "SGST (₹)",
            "IGST (₹)",
            "Total Amount (₹)",
            "TDS Applicability", # MOVED HERE
            "TDS Section",       # MOVED HERE
            "TDS Rate (%)",      # MOVED HERE
            "TDS Amount (₹)",    # MOVED HERE
            "Amount Payable (₹)",# MOVED HERE
            "Place of Supply",
            "Narration"
            # The original 'TDS' column (tds_str from raw GPT output) is intentionally excluded from display_cols
            # if you only want the refined 'TDS Applicability', 'TDS Section', etc.
        ]
        
        # Ensure all display_cols are present in df before selecting
        actual_display_cols = [col for col in display_cols if col in df.columns]

        st.dataframe(
            df[actual_display_cols],
            column_order=actual_display_cols,
            column_config={
                "HSN/SAC": st.column_config.TextColumn(
                    "HSN/SAC",
                    help="Harmonized System of Nomenclature / Service Accounting Code",
                    default="N/A"
                ),
                "TDS Section": st.column_config.TextColumn(
                    "TDS Section",
                    help="Applicable TDS Section (e.g., 194J)",
                    default="N/A"
                ),
                "TDS Applicability": st.column_config.TextColumn(
                    "TDS Applicability",
                    help="Indicates if TDS is applicable (Yes/No/Uncertain)",
                    default="Uncertain"
                )
            },
            use_container_width=True
        )

        # Create download dataframe without status columns
        download_df = df[actual_display_cols].copy()
        
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
    st.markdown("### Debugging Information:")
    st.write("#### DataFrame Info:")
    import io
    from contextlib import redirect_stdout
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        df.info(verbose=True, show_counts=True)
    st.text(buffer.getvalue())

    st.write("#### Null Counts per Column:")
    st.dataframe(df.isnull().sum().to_frame(name='Null Count'))

    if 'TDS Applicability' in df.columns and any(df['TDS Applicability'] == "Yes"):
         st.balloons()
    elif completed_count == total_files and completed_count > 0:
        st.balloons()
else:
    st.info("Upload one or more scanned invoices to get started.")
