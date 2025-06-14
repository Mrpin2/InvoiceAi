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

# Set locale for currency formatting (e.g., for India)
# This might need to be adjusted based on the specific environment.
# On some systems, 'en_IN' might work better, on others 'en_IN.utf8'
try:
    locale.setlocale(locale.LC_ALL, 'en_IN.utf8')
except locale.Error:
    try:
        locale.setlocale(locale.LC_ALL, 'en_IN')
    except locale.Error:
        st.warning("Could not set locale 'en_IN.utf8' or 'en_IN'. Currency formatting might default to system locale.")
        locale.setlocale(locale.LC_ALL, '') # Fallback to default system locale


hello_lottie = "https://raw.githubusercontent.com/Mrpin2/InvoiceAi/refs/heads/main/Animation%20-%201749845212531.json"
completed_lottie = "https://raw.githubusercontent.com/Mrpin2/InvoiceAi/refs/heads/main/Animation%20-%201749845303699.json"

def load_lottie_json_safe(url):
    """Loads a Lottie JSON animation from a URL safely."""
    try:
        r = requests.get(url)
        r.raise_for_status() # Raise an exception for bad status codes
        return r.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error loading Lottie animation from {url}: {e}")
        return None
    except json.JSONDecodeError as e:
        st.error(f"Error decoding Lottie JSON from {url}: {e}")
        return None

hello_json = load_lottie_json_safe(hello_lottie)
completed_json = load_lottie_json_safe(completed_lottie)

# Initial Lottie animation if no files have been processed yet
if "files_uploaded" not in st.session_state:
    if hello_json:
        st_lottie(hello_json, height=200, key="hello")

st.markdown("<h2 style='text-align: center;'>📄 AI Invoice Extractor (OpenAI)</h2>", unsafe_allow_html=True)
st.markdown("Upload scanned PDF invoices and extract structured finance data using GPT-4 Vision")
st.markdown("---")

# Define the fields we want to extract
fields = [
    "invoice_number", "date", "gstin", "seller_name", "buyer_name", "buyer_gstin",
    "taxable_amount", "cgst", "sgst", "igst", "place_of_supply", "expense_ledger", 
    "tds", "hsn_sac"
]

# Initialize Streamlit session state variables
if "processed_results" not in st.session_state:
    st.session_state["processed_results"] = {}
if "processing_status" not in st.session_state:
    st.session_state["processing_status"] = {}
if "summary_rows" not in st.session_state:
    st.session_state["summary_rows"] = []
if "files_uploaded" not in st.session_state: # To control initial Lottie display
    st.session_state["files_uploaded"] = False

st.sidebar.header("🔐 AI Config")
passcode = st.sidebar.text_input("Admin Passcode", type="password")
admin_unlocked = passcode == "Essenbee" # Change "Essenbee" to your desired admin passcode

openai_api_key = None
if admin_unlocked:
    st.sidebar.success("🔓 Admin access granted. Using secrets for API key.")
    # Attempt to retrieve API key from Streamlit secrets
    openai_api_key = st.secrets.get("OPENAI_API_KEY")
    if not openai_api_key:
        st.sidebar.error("OPENAI_API_KEY missing in Streamlit secrets.")
        st.stop()
else:
    openai_api_key = st.sidebar.text_input("🔑 Enter your OpenAI API Key", type="password")
    if not openai_api_key:
        st.sidebar.warning("Please enter a valid API key to continue.")
        st.stop()

# Initialize OpenAI client only if API key is available
client = OpenAI(api_key=openai_api_key)

def convert_pdf_first_page(pdf_bytes):
    """Converts the first page of a PDF to a PIL Image."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc.load_page(0)
    pix = page.get_pixmap(dpi=300) # Increased DPI for better OCR
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    doc.close() # Close the document
    return img

def safe_float(x):
    """Converts a value to float, handling common currency symbols and commas."""
    try:
        if x is None:
            return 0.0
        cleaned = str(x).replace(",", "").replace("₹", "").replace("$", "").strip()
        # Handle cases where value might be like "-" or "N/A"
        if not cleaned or cleaned.lower() in ["-", "n/a", "null"]:
            return 0.0
        return float(cleaned)
    except ValueError:
        return 0.0 # Return 0.0 for non-convertible values

def format_currency(x):
    """Formats a float as Indian Rupees with commas and two decimal places."""
    try:
        # Use locale.currency for proper currency formatting based on locale
        # locale.currency(value, symbol=True, grouping=True)
        return locale.currency(safe_float(x), symbol=True, grouping=True)
    except Exception:
        return "₹0.00"

def is_valid_gstin(gstin):
    """Validate GSTIN format (15 alphanumeric characters)."""
    if not gstin:
        return False
        
    cleaned = re.sub(r'[^A-Z0-9]', '', str(gstin).upper())
    
    # GSTIN must be exactly 15 characters
    if len(cleaned) != 15:
        return False
        
    # Pattern: 2 digits (state code) + 10 alphanumeric (PAN) + 1 alpha/digit + 1 'Z' + 1 alpha/digit (checksum)
    # The last two characters are often 'Z' and then a single digit/alpha for checksum.
    # The regex below is a more common strict pattern, but invoices might have slight variations.
    # For robustness, we check length and general composition.
    # A stricter regex: r"^\d{2}[A-Z]{5}\d{4}[A-Z]{1}[1-9A-Z]{1}Z[0-9A-Z]{1}$"
    # For this app, the current pattern is flexible enough for common cases.
    pattern = r"^\d{2}[A-Z0-9]{10}[A-Z0-9]{1}[A-Z0-9]{1}[A-Z0-9]{1}$" # A slightly more lenient check for varied invoice data
    return bool(re.match(pattern, cleaned))

def extract_gstin_from_text(text):
    """Try to extract GSTIN from any text using pattern matching."""
    if not text:
        return ""
    # Look for GSTIN pattern in the text (e.g., "GSTIN: 22AAAAA0000A1Z5")
    # This pattern tries to be comprehensive: 2 digits, 10 alphanumeric, 1 alpha, 1 numeric/alpha, 1 Z, 1 numeric/alpha
    matches = re.findall(r'\b\d{2}[A-Z0-9]{10}[A-Z]{1}[A-Z0-9]{1}[Z]{1}[A-Z0-9]{1}\b', text.upper())
    if matches:
        return matches[0]
    return ""

def determine_tds_rate(expense_ledger, tds_str=""):
    """Determine TDS rate based on expense ledger and TDS string."""
    # First, try to extract rate from the TDS string itself if it contains a percentage
    if tds_str and isinstance(tds_str, str):
        match = re.search(r'(\d+(\.\d+)?)\s*%', tds_str)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass # Continue to next logic if conversion fails

        # Check for common TDS sections mentioned in the string
        section_rates = {
            "194j": 10.0,  # Professional/Technical services
            "194c": 2.0,   # Contracts
            "194h": 5.0,   # Commission/brokerage
            "194i": 10.0,  # Rent
            "194q": 0.1,   # Purchase of goods
            "194a": 10.0,  # Interest other than interest on securities
        }
        for section, rate in section_rates.items():
            if section in tds_str.lower():
                return rate
            
    # If no explicit rate or section in tds_str, determine by expense ledger type
    expense_ledger = expense_ledger.lower() if expense_ledger else ""
    
    if "professional" in expense_ledger or "consultancy" in expense_ledger or "legal fees" in expense_ledger or "technical service" in expense_ledger:
        return 10.0 # Section 194J
    if "contract" in expense_ledger or "work" in expense_ledger or "job work" in expense_ledger:
        return 2.0 # Section 194C (for non-individuals)
    if "commission" in expense_ledger or "brokerage" in expense_ledger:
        return 5.0 # Section 194H
    if "rent" in expense_ledger or "lease" in expense_ledger:
        return 10.0 # Section 194I
    if "advertis" in expense_ledger or "marketing" in expense_ledger or "brand promotion" in expense_ledger:
        return 1.0 # Often covered under contracts (194C) or sometimes 194A/J depending on nature. Let's assign 1.0 for advertising.
    if "goods purchase" in expense_ledger or "raw material" in expense_ledger:
        return 0.1 # Section 194Q
    if "interest" in expense_ledger and "loan" in expense_ledger:
        return 10.0 # Section 194A

    return 0.0 # Default to 0 if no applicable rate is determined

def extract_json_from_response(text):
    """Try to extract JSON from GPT response which might have extra text."""
    try:
        # Most reliable: Look for JSON code block (```json{...}```)
        matches = re.findall(r'```json\s*({.*?})\s*```', text, re.DOTALL)
        if matches:
            return json.loads(matches[0])
        
        # Second attempt: Look for plain JSON object (first '{' to last '}')
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1 and end > start:
            json_candidate = text[start:end+1]
            # Basic validation to avoid parsing non-JSON
            if json_candidate.strip().startswith('{') and json_candidate.strip().endswith('}'):
                return json.loads(json_candidate)
            
        # Last resort: Try parsing the whole text if it's purely JSON or has minimal surrounding text
        return json.loads(text.strip())
    except json.JSONDecodeError as e:
        st.error(f"Failed to decode JSON: {e}")
        st.code(text, language="json") # Show the text that failed to parse
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during JSON extraction: {e}")
        return None

# Enhanced prompt with specific GSTIN and HSN/SAC instructions
main_prompt = (
    "Extract structured invoice data as a JSON object with the following keys: "
    "invoice_number, date, gstin, seller_name, buyer_name, buyer_gstin, "
    "taxable_amount, cgst, sgst, igst, place_of_supply, expense_ledger, tds, hsn_sac. "
    "Important: 'taxable_amount' is the amount BEFORE taxes (excluding CGST, SGST, IGST). "
    "Use DD/MM/YYYY for dates. Use only values shown in the invoice. "
    "Return 'NOT AN INVOICE' if clearly not one, or if it's a blurry/unreadable image. "
    "If a key's value is not available, use null. "
    "Ensure all monetary values are numeric (floats). Do not include currency symbols in the JSON values."
    
    "SPECIAL INSTRUCTIONS FOR GSTIN: "
    "1. GSTIN is a 15-character alphanumeric code (format: 2-digits State Code, 5-chars PAN, 4-digits Entity Code, 1-char Checksum, 'Z', 1-char Checksum). "
    "   Example: '22AAAAA0000A1Z5'. "
    "2. It's usually located near the seller's name/address ('gstin', 'gst no.', 'gst number'). "
    "3. If not in a dedicated field, look in the seller details/address block. "
    "4. Prioritize the most clearly labeled GSTIN for the seller. If multiple, infer the primary seller's GSTIN."
    
    "SPECIAL INSTRUCTIONS FOR HSN/SAC: "
    "1. HSN (Harmonized System of Nomenclature) is for goods, SAC (Services Accounting Code) for services. "
    "2. Typically 4 to 8 digit codes found in item tables or tax breakdown sections. "
    "3. If multiple codes exist, use the most frequent one or the one associated with the main taxable item/service. "
    "4. If no HSN/SAC code is explicitly visible in the document, return null - DO NOT GUESS or hallucinate."
    
    "For expense_ledger, classify the nature of expense and suggest an applicable ledger type "
    "(e.g., 'Office Supplies', 'Professional Fees', 'Software Subscription', 'Advertising Expenses', 'Legal & Professional Charges'). "
    "For tds, indicate TDS applicability (e.g., 'Yes - Section 194J', 'No TDS Applicable', 'Uncertain', '0.1%', '10%')."
    "Ensure the JSON output is strictly valid and can be parsed by a Python json.loads() call."
)

uploaded_files = st.file_uploader("📤 Upload scanned invoice PDFs", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    st.session_state["files_uploaded"] = True
    total_files = len(uploaded_files)
    completed_count = 0

    for idx, file in enumerate(uploaded_files):
        file_name = file.name
        
        # Skip if already processed (useful when re-running app or uploading same files)
        if file_name in st.session_state["processed_results"] and \
           st.session_state["processing_status"].get(file_name) == "✅ Done":
            st.info(f"Skipping {file_name} as it was already processed.")
            completed_count += 1
            continue

        st.markdown(f"**Processing file: {file_name} ({idx+1}/{total_files})**")
        st.session_state["processing_status"][file_name] = "⏳ Pending..."
        
        response_text = "No response yet." # Initialize for error reporting if GPT call fails
        temp_file_path = None
        try:
            # Save PDF to a temporary file for fitz (PyMuPDF)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file.getvalue())
                temp_file_path = tmp.name

            with open(temp_file_path, "rb") as f:
                pdf_data = f.read()
                
            first_image = convert_pdf_first_page(pdf_data)

            with st.spinner(f"🧠 Extracting data from {file_name} using GPT-4o Vision..."):
                img_buf = io.BytesIO()
                first_image.save(img_buf, format="PNG")
                img_buf.seek(0)
                base64_image = base64.b64encode(img_buf.read()).decode("utf-8") # Ensure UTF-8 encoding

                chat_prompt = [
                    {"role": "system", "content": "You are a finance assistant specializing in Indian invoices. Pay special attention to GSTIN and HSN/SAC extraction. Always respond with a valid JSON object."},
                    {"role": "user", "content": [
                        {"type": "text", "text": main_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                    ]}
                ]

                response = client.chat.completions.create(
                    model="gpt-4o", # Using GPT-4o for better vision capabilities
                    messages=chat_prompt,
                    max_tokens=2000 # Increased for better extraction
                )

                response_text = response.choices[0].message.content.strip()
                st.write(f"--- Raw GPT Response for {file_name} ---")
                st.code(response_text, language="json") # DEBUG: Show raw response
                st.write(f"--- End Raw GPT Response ---")
                
                # Try to extract JSON from the response
                raw_data = extract_json_from_response(response_text)
                
                if raw_data is None:
                    # Check if the model explicitly stated it's not an invoice
                    if "not an invoice" in response_text.lower() or "blurry" in response_text.lower() or "unreadable" in response_text.lower():
                        result_row = {
                            "File Name": file_name,
                            "Invoice Number": "NOT AN INVOICE/UNREADABLE",
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
                            "HSN/SAC": "",
                            "TDS": "",
                            "Narration": "This document was identified as not an invoice or was unreadable."
                        }
                    else:
                        raise ValueError("GPT returned unparseable or unexpected response format.")
                else:
                    # Create the result row with all fields, using .get() with default values
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
                    hsn_sac = raw_data.get("hsn_sac", "") # Get HSN/SAC from raw_data
                    
                    st.write(f"--- Extracted HSN/SAC for {file_name}: '{hsn_sac}' ---") # DEBUG: Show extracted HSN/SAC

                    # Calculate derived fields
                    total_amount = taxable_amount + cgst + sgst + igst
                    tds_rate = determine_tds_rate(expense_ledger, tds_str)
                    tds_amount = round(taxable_amount * tds_rate / 100, 2)
                    amount_payable = total_amount - tds_amount
                    
                    # Enhanced GSTIN handling (validation and fallback)
                    # No need to store original_gstin unless explicitly needed for display
                    # gstin_status will indicate if it was valid, invalid, or extracted from text
                    
                    # Clean and validate Seller GSTIN
                    cleaned_seller_gstin = re.sub(r'[^A-Z0-9]', '', str(seller_gstin).upper())
                    if not is_valid_gstin(cleaned_seller_gstin):
                        fallback_gstin_seller = extract_gstin_from_text(seller_name)
                        if fallback_gstin_seller:
                            seller_gstin = fallback_gstin_seller # Use fallback if original is invalid
                            #gstin_status = "EXTRACTED_FROM_NAME (Seller)"
                        #else:
                            #gstin_status = "INVALID/MISSING (Seller)"
                    else:
                        seller_gstin = cleaned_seller_gstin # Use cleaned valid GSTIN
                        #gstin_status = "VALID (Seller)"

                    # Clean and validate Buyer GSTIN
                    cleaned_buyer_gstin = re.sub(r'[^A-Z0-9]', '', str(buyer_gstin).upper())
                    if not is_valid_gstin(cleaned_buyer_gstin):
                        fallback_gstin_buyer = extract_gstin_from_text(buyer_name)
                        if fallback_gstin_buyer:
                            buyer_gstin = fallback_gstin_buyer # Use fallback if original is invalid
                            #gstin_status += "; EXTRACTED_FROM_NAME (Buyer)"
                        #else:
                            #gstin_status += "; INVALID/MISSING (Buyer)"
                    else:
                        buyer_gstin = cleaned_buyer_gstin # Use cleaned valid GSTIN
                        #gstin_status += "; VALID (Buyer)"

                    # Parse and format date
                    try:
                        parsed_date = parser.parse(str(date), dayfirst=True)
                        date = parsed_date.strftime("%d/%m/%Y")
                    except Exception:
                        date = "" # Keep date empty if parsing fails
                        
                    # Create narration text
                    buyer_gstin_display = buyer_gstin or "N/A"
                    narration = (
                        f"Invoice {invoice_number or 'N/A'} dated {date or 'N/A'} "
                        f"was issued by {seller_name or 'N/A'} (GSTIN: {seller_gstin or 'N/A'}) "
                        f"to {buyer_name or 'N/A'} (GSTIN: {buyer_gstin_display}), "
                        f"with a taxable amount of {format_currency(taxable_amount)}. "
                        f"Taxes applied - CGST: {format_currency(cgst)}, SGST: {format_currency(sgst)}, IGST: {format_currency(igst)}. "
                        f"Total Amount: {format_currency(total_amount)}. "
                        f"Place of supply: {place_of_supply or 'N/A'}. Expense: {expense_ledger or 'N/A'}. "
                        f"HSN/SAC: {hsn_sac or 'N/A'}. "
                        f"TDS: {tds_str or 'N/A'} @ {tds_rate}% ({format_currency(tds_amount)}). "
                        f"Amount Payable: {format_currency(amount_payable)}."
                    )
                    
                    # Final result row for DataFrame
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
                        "HSN/SAC": hsn_sac, # This is where it's stored
                        "TDS": tds_str,
                        "Narration": narration
                    }

                st.session_state["processed_results"][file_name] = result_row
                st.session_state["processing_status"][file_name] = "✅ Done"
                completed_count += 1
                st.success(f"{file_name}: ✅ Done")

        except Exception as e:
            error_message = f"Error processing file: {str(e)}"
            st.session_state["processing_status"][file_name] = "❌ Error"
            st.error(f"❌ Error processing {file_name}: {error_message}")
            st.text_area(f"Raw Output (for debug, {file_name})", response_text, height=200) # Show raw response on error
            
            error_row = {
                "File Name": file_name,
                "Invoice Number": "PROCESSING ERROR",
                "Date": "", "Seller Name": "", "Seller GSTIN": "",
                "Buyer Name": "", "Buyer GSTIN": "",
                "Taxable Amount": 0.0, "CGST": 0.0, "SGST": 0.0, "IGST": 0.0,
                "Total Amount": 0.0, "TDS Rate": 0.0, "TDS Amount": 0.0,
                "Amount Payable": 0.0, "Place of Supply": "", "Expense Ledger": "",
                "HSN/SAC": "", # Ensure HSN/SAC is present even in error rows
                "TDS": "",
                "Narration": error_message
            }
            st.session_state["processed_results"][file_name] = error_row

        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                os.unlink(temp_file_path) # Clean up the temporary PDF file

# Get all processed results from session state
results = list(st.session_state["processed_results"].values())

if results:
    if completed_json:
        st_lottie(completed_json, height=200, key="done_animation")

    st.markdown("<h3 style='text-align: center;'>🎉 Yippie! All invoices processed with a smile 😊</h3>", unsafe_allow_html=True)

    try:
        df = pd.DataFrame(results)
        
        # DEBUG: Show all columns to verify HSN/SAC is present right after DataFrame creation
        st.write("--- All available columns in DataFrame (before display selection) ---")
        st.write(df.columns.tolist())
        st.write("--- End available columns ---")
        
        # Format currency columns
        currency_cols = ["Taxable Amount", "CGST", "SGST", "IGST", "Total Amount", "TDS Amount", "Amount Payable"]
        for col in currency_cols:
            if col in df.columns: # Check if column exists before formatting
                df[f"{col} (₹)"] = df[col].apply(format_currency)
            else:
                st.warning(f"Currency column '{col}' not found in DataFrame for formatting.")
                df[f"{col} (₹)"] = "N/A" # Add a placeholder if missing
                
        # Format TDS Rate as percentage
        if "TDS Rate" in df.columns: # Check if column exists before formatting
            df["TDS Rate (%)"] = df["TDS Rate"].apply(lambda x: f"{x}%")
        else:
            st.warning("Column 'TDS Rate' not found in DataFrame for percentage formatting.")
            df["TDS Rate (%)"] = "N/A" # Add a placeholder if missing

        # Ensure 'HSN/SAC' column explicitly exists in the DataFrame before selecting for display.
        # This covers cases where initial 'results' might not have it for some reason (e.g., if raw_data.get didn't provide it).
        if "HSN/SAC" not in df.columns:
            df["HSN/SAC"] = "" # Initialize with an empty string if it's truly missing from the results structure

        # Reorder and select columns for better display
        display_cols_order = [
            "File Name", "Invoice Number", "Date", "Seller Name", "Seller GSTIN", 
            "Buyer Name", "Buyer GSTIN", "Taxable Amount (₹)", 
            "CGST (₹)", "SGST (₹)", "IGST (₹)", "Total Amount (₹)", "TDS Rate (%)", 
            "TDS Amount (₹)", "Amount Payable (₹)", "Place of Supply", 
            "Expense Ledger", "HSN/SAC", "TDS", "Narration"
        ]
        
        # Filter display_cols_order to only include columns that actually exist in the DataFrame
        final_display_cols = [col for col in display_cols_order if col in df.columns]
        
        # Warn if any desired display column is missing from the actual DataFrame
        missing_display_cols = [col for col in display_cols_order if col not in df.columns]
        if missing_display_cols:
            st.warning(f"Some desired display columns were not found in the extracted data and will be skipped: {', '.join(missing_display_cols)}")

        # Display the DataFrame
        st.dataframe(df[final_display_cols], use_container_width=True)

        # Create DataFrame for download (use original numeric values for calculations in other tools)
        download_df = df[[col for col in fields if col in df.columns] + 
                         ["File Name", "Total Amount", "TDS Rate", "TDS Amount", "Amount Payable", "Narration"]].copy()

        # CSV Download
        csv_data = download_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download Results as CSV", 
            csv_data, 
            "invoice_results.csv", 
            "text/csv",
            key="download_csv"
        )

        # Excel Download
        excel_buffer = io.BytesIO()
        try:
            with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                download_df.to_excel(writer, index=False, sheet_name="Invoice Data")
            st.download_button(
                label="📥 Download Results as Excel",
                data=excel_buffer.getvalue(),
                file_name="invoice_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="download_excel"
            )
        except Exception as e:
            st.error(f"Failed to create Excel file: {str(e)}")
            
    except Exception as e:
        st.error(f"Error creating results table: {str(e)}")
        st.write("Raw results data for debugging:")
        st.json(results)

    st.markdown("---")
    # if st.session_state.summary_rows: # This state variable doesn't seem to be used/updated.
    #    st.balloons()
else:
    st.info("Upload one or more scanned invoice PDFs to get started.")
