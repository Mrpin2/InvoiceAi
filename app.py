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

# Import Gemini API client
import google.generativeai as genai 

locale.setlocale(locale.LC_ALL, '')

# Lottie animations for better UX
hello_lottie = "https://raw.githubusercontent.com/Mrpin2/InvoiceAi/refs/heads/main/Animation%20-%201749845212531.json"
completed_lottie = "https://raw.githubusercontent.com/Mrpin2/InvoiceAi/refs/heads/main/Animation%20-%201749845303699.json"

def load_lottie_json_safe(url):
    """Loads Lottie animation JSON safely from a URL."""
    try:
        r = requests.get(url)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as e:
        st.warning(f"Could not load Lottie animation from {url}: {e}")
        return None

hello_json = load_lottie_json_safe(hello_lottie)
completed_json = load_lottie_json_safe(completed_lottie)

# Display initial Lottie animation if no files have been uploaded yet
if "files_uploaded" not in st.session_state:
    st.session_state["files_uploaded"] = False
if not st.session_state["files_uploaded"]:
    if hello_json:
        st_lottie(hello_json, height=200, key="hello")

st.markdown("<h2 style='text-align: center;'>📄 AI Invoice Extractor (OpenAI + Gemini)</h2>", unsafe_allow_html=True)
st.markdown("Upload scanned PDF invoices and extract structured finance data using GPT-4 Vision and Gemini for GSTIN validation.")
st.markdown("---")

# Initialize session states for storing results and processing status
if "processed_results" not in st.session_state:
    st.session_state["processed_results"] = {}
if "processing_status" not in st.session_state:
    st.session_state["processing_status"] = {}
if "summary_rows" not in st.session_state:
    st.session_state["summary_rows"] = []
if "process_triggered" not in st.session_state:
    st.session_state["process_triggered"] = False

# This key is only for the file uploader to force reset
if "file_uploader_key" not in st.session_state:
    st.session_state["file_uploader_key"] = 0

# --- Placeholders for dynamic content, including the file uploader ---
# This placeholder will be used to completely redraw the file uploader
file_uploader_placeholder = st.empty()

# --- Admin/API Key Config ---
st.sidebar.header("🔐 AI Config")
passcode = st.sidebar.text_input("Admin Passcode", type="password")
admin_unlocked = passcode == "Rajeev"

openai_api_key = None
gemini_api_key = None # New variable for Gemini API key

if admin_unlocked:
    st.sidebar.success("🔓 Admin access granted.")
    openai_api_key = st.secrets.get("OPENAI_API_KEY")
    gemini_api_key = st.secrets.get("GEMINI_API_KEY") # Get Gemini key from secrets

    if not openai_api_key:
        st.sidebar.error("OPENAI_API_KEY missing in Streamlit secrets.")
        st.stop()
    if not gemini_api_key:
        st.sidebar.error("GEMINI_API_KEY missing in Streamlit secrets.")
        st.stop()
else:
    openai_api_key = st.sidebar.text_input("🔑 Enter your OpenAI API Key", type="password")
    gemini_api_key = st.sidebar.text_input("🔑 Enter your Gemini API Key", type="password") # User input for Gemini key

    if not openai_api_key:
        st.sidebar.warning("Please enter a valid OpenAI API key to continue.")
        st.stop()
    if not gemini_api_key:
        st.sidebar.warning("Please enter a valid Gemini API key to continue.")
        st.stop()

try:
    client = OpenAI(api_key=openai_api_key)
except Exception as e:
    st.error(f"Failed to initialize OpenAI client. Check your API key: {e}")
    st.stop()

try:
    genai.configure(api_key=gemini_api_key)
    gemini_model = genai.GenerativeModel('gemini-pro-vision' if 'vision' in genai.GenerativeModel.list_models()[0].name else 'gemini-pro') # Use vision model if available
except Exception as e:
    st.error(f"Failed to initialize Gemini client. Check your API key: {e}")
    st.stop()


# --- Functions remain the same ---
def convert_pdf_first_page(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc.load_page(0)
    pix = page.get_pixmap(dpi=300)
    img_bytes = pix.tobytes("png")
    return Image.open(io.BytesIO(img_bytes))

def safe_float(x):
    try:
        cleaned = str(x).replace(",", "").replace("₹", "").replace("$", "").strip()
        return float(cleaned) if cleaned else 0.0
    except (ValueError, TypeError):
        return 0.0

def format_currency(x):
    try:
        if isinstance(x, str) and x.startswith('₹'):
            return x
        return f"₹{safe_float(x):,.2f}"
    except:
        return "₹0.00"

def is_valid_gstin(gstin):
    if not gstin:
        return False
    cleaned = re.sub(r'[^A-Z0-9]', '', gstin.upper())
    if len(cleaned) != 15:
        return False
    pattern = r"^\d{2}[A-Z0-9]{10}[A-Z]{1}[A-Z0-9]{1}[Z]{1}[A-Z0-9]{1}$"
    return bool(re.match(pattern, cleaned))

# This function will now use Gemini API
def extract_gstin_with_gemini(image_data, text_data=""):
    """
    Extracts and validates GSTINs using the Gemini API.
    Prioritizes extraction from image if available, falls back to text.
    """
    prompt = (
        "Extract all potential GSTINs from the provided image/text. "
        "A valid Indian GSTIN is a 15-character alphanumeric string. "
        "Format the output as a JSON list of strings, e.g., ['GSTIN1', 'GSTIN2']. "
        "If no valid GSTINs are found, return an empty list []."
    )
    
    parts = [{"mime_type": "image/png", "data": image_data}]
    if text_data:
        # Optionally, you can add text as another part if you want Gemini to consider it.
        # This might be useful if the image quality is poor but text OCR is good.
        # For simplicity, we'll focus on image, but you could add:
        # parts.append({"mime_type": "text/plain", "data": text_data})
        pass

    try:
        response = gemini_model.generate_content([prompt, Image.open(io.BytesIO(image_data))])
        response_text = response.text.strip()
        
        # Attempt to parse as JSON list
        try:
            gstins = json.loads(response_text)
            if isinstance(gstins, list):
                # Filter for valid GSTINs
                return [g for g in gstins if is_valid_gstin(g)]
        except json.JSONDecodeError:
            # If not a direct JSON list, try to find patterns in the raw text
            pass

        # Fallback to regex if Gemini doesn't return a perfect JSON list
        matches = re.findall(r'\b(\d{2}[A-Z0-9]{10}[A-Z]{1}[A-Z0-9]{1}Z[A-Z0-9]{1})\b', response_text.upper())
        valid_gstins = [match for match in matches if is_valid_gstin(match)]
        return valid_gstins

    except Exception as e:
        st.warning(f"Gemini API call failed for GSTIN extraction: {e}")
        return []

def determine_tds_rate(expense_ledger, tds_str="", place_of_supply=""):
    if place_of_supply and place_of_supply.lower() == "foreign":
        return 0.0
    if tds_str and isinstance(tds_str, str):
        match = re.search(r'(\d+(\.\d+)?)%', tds_str)
        if match:
            return float(match.group(1))
        section_rates = { "194j": 10.0, "194c": 1.0, "194i": 10.0, "194h": 5.0, "194q": 0.1 }
        for section, rate in section_rates.items():
            if section in tds_str.lower():
                return rate
    expense_ledger = expense_ledger.lower() if expense_ledger else ""
    if "professional" in expense_ledger or "consultancy" in expense_ledger or "service" in expense_ledger:
        return 10.0
    if "contract" in expense_ledger or "work" in expense_ledger:
        return 1.0
    if "rent" in expense_ledger:
        return 10.0
    return 0.0

def determine_tds_section(expense_ledger, place_of_supply=""):
    if place_of_supply and place_of_supply.lower() == "foreign":
        return None
    expense_ledger = expense_ledger.lower() if expense_ledger else ""
    if "professional" in expense_ledger or "consultancy" in expense_ledger or "service" in expense_ledger:
        return "194J"
    elif "contract" in expense_ledger or "work" in expense_ledger:
        return "194C"
    elif "rent" in expense_ledger:
        return "194I"
    return None

def extract_json_from_response(text):
    try:
        matches = re.findall(r'```json\s*({.*?})\s*```', text, re.DOTALL)
        if matches:
            return json.loads(matches[0])
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1:
            return json.loads(text[start:end+1])
        return json.loads(text) # In case the JSON is the entire response
    except Exception:
        return None

main_prompt = (
    "You are an expert at extracting structured data from Indian invoices. "
    "Your task is to extract information into a JSON object with the following keys. "
    "If a key's value is not explicitly present or derivable from the invoice, use `null` for that value. "
    "Keys to extract: invoice_number, date, gstin (seller's GSTIN), seller_name, buyer_name, buyer_gstin, "
    "taxable_amount, cgst, sgst, igst, place_of_supply, expense_ledger, tds, hsn_sac. "
    
    "GUIDELINES FOR EXTRACTION:\n"
    "- 'invoice_number': The unique identifier of the invoice. Extract as is.\n"
    "- 'date': The invoice date in DD/MM/YYYY format. If year is 2-digit, assume current century (e.g., 24 -> 2024).\n"
    "- 'taxable_amount': This is the subtotal, the amount BEFORE any taxes (CGST, SGST, IGST) are applied. Must be a number.\n"
    "- 'gstin': The GSTIN of the seller (the entity issuing the invoice). Must be a 15-character alphanumeric string. Prioritize the GSTIN explicitly labeled as 'GSTIN' or associated with the seller's main details.\n"
    "- 'buyer_gstin': The GSTIN of the buyer (the entity receiving the invoice). Must be a 15-character alphanumeric string. Prioritize the GSTIN explicitly labeled as 'Buyer GSTIN' or associated with the buyer's details.\n"
    "- 'hsn_sac': Crucial for Indian invoices. "
    "  - HSN (Harmonized System of Nomenclature) is for goods."
    "  - SAC (Service Accounting Code) is for services."
    "  - **ONLY extract the HSN/SAC code if it is explicitly mentioned on the invoice.** "
    "  - It is typically a 4, 6, or 8-digit numeric code, sometimes alphanumeric."
    "  - Look for labels like 'HSN Code', 'SAC Code', 'HSN/SAC', or just the code itself near item descriptions."
    "  - If multiple HSN/SAC codes are present for different line items, extract the one that appears most prominently, or the first one listed. If only one is present for the whole invoice, use that."
    "  - **If HSN/SAC is NOT found or explicitly stated, the value MUST be `null`. Do NOT guess or infer it.**\n"
    
    "- 'expense_ledger': Classify the nature of expense and suggest a suitable ledger type. "
    "  Examples: 'Office Supplies', 'Professional Fees', 'Software Subscription', 'Rent', "
    "  'Cloud Services', 'Marketing Expenses', 'Travel Expenses'. "
    "  For invoices from cloud providers (e.g., 'Google Cloud', 'AWS', 'Microsoft Azure', 'DigitalOcean'), classify as 'Cloud Services'."
    "  If the expense is clearly related to software licenses, subscriptions, or SaaS, classify as 'Software Subscription'."
    "  Aim for a general and universal ledger type if a precise one isn't obvious from the invoice details.\n"
    
    "- 'tds': Determine TDS applicability. State 'Yes - Section [X]' if applicable with a section, 'No' if clearly not, or 'Uncertain' if unclear. Always try to identify the TDS Section (e.g., 194J, 194C, 194I) if TDS is applicable.\n"
    
    "- 'place_of_supply': Crucial for Indian invoices to determine IGST applicability. "
    "  - **PRIORITY 1:** Look for a field explicitly labeled 'Place of Supply'. Extract the exact State/City name from this field (e.g., 'Delhi', 'Maharashtra')."
    "  - **PRIORITY 2:** If 'Place of Supply' is not found, look for a 'Ship To:' address. Extract ONLY the State/City name from this address."
    "  - **PRIORITY 3:** If 'Ship To:' is not found, look for a 'Bill To:' address. Extract ONLY the State/City name from this address."
    "  - **PRIORITY 4:** If neither of the above, infer from the Customer/Buyer Address. Extract ONLY the State/City name from this address."
    "  - **SPECIAL CASE:** If the invoice text or context clearly indicates an export or foreign transaction (e.g., 'Export Invoice', mentions 'Foreign' address, non-Indian currency as primary total, or foreign recipient details), set the value to 'Foreign'."
    "  - **FALLBACK:** If none of the above are found or inferable, the value MUST be `null`."
    
    "Return 'NOT AN INVOICE' if the document is clearly not an invoice.\n"
    "Ensure the JSON output is clean and directly parsable."
)

# Render the file uploader using the placeholder
with file_uploader_placeholder.container():
    uploaded_files = st.file_uploader(
        "📤 Upload scanned invoice PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        key=f"file_uploader_{st.session_state.file_uploader_key}"
    )

    # Store uploaded files in session state to persist across reruns and allow clearing
    if uploaded_files:
        st.session_state["uploaded_files"] = uploaded_files
        st.session_state["files_uploaded"] = True
    else:
        st.session_state["uploaded_files"] = []
        st.session_state["files_uploaded"] = False

# Conditional display of buttons after file upload
if st.session_state["files_uploaded"] or st.session_state["processed_results"]:
    col_process, col_spacer, col_clear = st.columns([1, 4, 1])
    
    with col_process:
        if st.button("🚀 Process Invoices", help="Click to start extracting data from uploaded invoices."):
            st.session_state["process_triggered"] = True
            st.info("Processing initiated. Please wait...")

    with col_clear:
        if st.button("🗑️ Clear All Files & Reset", help="Click to clear all uploaded files and extracted data."):
            # Increment key BEFORE clearing session_state to ensure uploader reset
            st.session_state["file_uploader_key"] += 1
            
            # Clear all relevant session state variables explicitly
            st.session_state["files_uploaded"] = False
            st.session_state["processed_results"] = {}
            st.session_state["processing_status"] = {}
            st.session_state["summary_rows"] = []
            st.session_state["process_triggered"] = False
            st.session_state["uploaded_files"] = [] # Explicitly empty this list

            # Clear the placeholder to remove the old file uploader instance
            file_uploader_placeholder.empty()
            
            # This rerun will redraw everything, including a *new* file uploader with the incremented key
            st.rerun()

# Only proceed with processing if files are uploaded AND the "Process Invoices" button was clicked
if st.session_state["uploaded_files"] and st.session_state["process_triggered"]:
    total_files = len(st.session_state["uploaded_files"])
    progress_text = st.empty()
    progress_bar = st.progress(0)
    
    completed_count = 0

    for idx, file in enumerate(st.session_state["uploaded_files"]):
        file_name = file.name
        progress_text.text(f"Processing file: {file_name} ({idx+1}/{total_files})")
        progress_bar.progress((idx + 1) / total_files)

        if file_name in st.session_state["processed_results"]:
            completed_count += 1
            continue

        st.markdown(f"**Current File: {file_name}**")
        st.session_state["processing_status"][file_name] = "⏳ Pending..."

        temp_file_path = None
        response_text = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file.getvalue())
                temp_file_path = tmp.name

            with open(temp_file_path, "rb") as f:
                pdf_data = f.read()
                
            first_image = convert_pdf_first_page(pdf_data)

            # Convert PIL Image to bytes for Gemini
            img_buf = io.BytesIO()
            first_image.save(img_buf, format="PNG")
            img_bytes = img_buf.getvalue() # Get bytes for Gemini


            with st.spinner(f"🧠 Extracting data from {file_name} using GPT-4 Vision..."):
                base64_image = base64.b64encode(img_bytes).decode() # Use img_bytes for OpenAI

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
                    max_tokens=1500
                )

                response_text = response.choices[0].message.content.strip()
                raw_data = extract_json_from_response(response_text)
                
                if raw_data is None:
                    if "not an invoice" in response_text.lower():
                        result_row = {
                            "File Name": file_name,
                            "Invoice Number": "NOT AN INVOICE",
                            "Date": "", "Seller Name": "", "Seller GSTIN": "",
                            "HSN/SAC": "", "Buyer Name": "", "Buyer GSTIN": "",
                            "Expense Ledger": "", "Taxable Amount": 0.0,
                            "CGST": 0.0, "SGST": 0.0, "IGST": 0.0, "Total Amount": 0.0,
                            "TDS Applicability": "N/A", "TDS Section": None,
                            "TDS Rate": 0.0, "TDS Amount": 0.0, "Amount Payable": 0.0,
                            "Place of Supply": "", "TDS": "",
                            "Narration": "This document was identified as not an invoice."
                        }
                    else:
                        st.warning(f"GPT returned non-JSON or unparsable response for {file_name}. See raw output below.")
                        raise ValueError(f"GPT returned non-JSON response or unexpected format for {file_name}.")
                else:
                    invoice_number = raw_data.get("invoice_number", "")
                    date = raw_data.get("date", "")
                    seller_name = raw_data.get("seller_name", "")
                    seller_gstin_openai = raw_data.get("gstin", "") # Store OpenAI's GSTIN separately
                    hsn_sac = raw_data.get("hsn_sac", "")
                    buyer_name = raw_data.get("buyer_name", "")
                    buyer_gstin_openai = raw_data.get("buyer_gstin", "") # Store OpenAI's GSTIN separately
                    expense_ledger = raw_data.get("expense_ledger", "")
                    taxable_amount = safe_float(raw_data.get("taxable_amount", 0.0))
                    cgst = safe_float(raw_data.get("cgst", 0.0))
                    sgst = safe_float(raw_data.get("sgst", 0.0))
                    igst = safe_float(raw_data.get("igst", 0.0))
                    place_of_supply = raw_data.get("place_of_supply", "")
                    tds_str = raw_data.get("tds", "")

                    # --- Gemini API for GSTIN validation/extraction ---
                    st.info(f"🔍 Validating GSTINs for {file_name} using Gemini...")
                    extracted_gstins_gemini = extract_gstin_with_gemini(img_bytes)
                    
                    # Logic to choose the best GSTINs or combine them
                    seller_gstin = ""
                    buyer_gstin = ""

                    # Prioritize Gemini's findings if they are valid
                    if extracted_gstins_gemini:
                        # Simple heuristic: assume the first valid GSTIN is seller, second is buyer
                        # You might need more sophisticated logic here based on your prompt to Gemini
                        if len(extracted_gstins_gemini) >= 1 and is_valid_gstin(extracted_gstins_gemini[0]):
                            seller_gstin = extracted_gstins_gemini[0]
                        if len(extracted_gstins_gemini) >= 2 and is_valid_gstin(extracted_gstins_gemini[1]):
                            buyer_gstin = extracted_gstins_gemini[1]
                        elif seller_gstin and is_valid_gstin(seller_gstin_openai) and seller_gstin_openai != seller_gstin and len(extracted_gstins_gemini) == 1:
                            # If Gemini only found one, and OpenAI found another, use OpenAI's as buyer if valid
                             buyer_gstin = seller_gstin_openai
                        elif buyer_gstin_openai and not buyer_gstin and is_valid_gstin(buyer_gstin_openai):
                            # If Gemini didn't find a buyer GSTIN, use OpenAI's if it's valid
                            buyer_gstin = buyer_gstin_openai
                    
                    # If Gemini didn't find any or not enough, fallback to OpenAI's initial extraction
                    if not seller_gstin and is_valid_gstin(seller_gstin_openai):
                        seller_gstin = seller_gstin_openai
                    if not buyer_gstin and is_valid_gstin(buyer_gstin_openai):
                        buyer_gstin = buyer_gstin_openai
                    # --- End of Gemini API for GSTIN validation/extraction ---


                    total_amount = taxable_amount + cgst + sgst + igst
                    tds_rate = determine_tds_rate(expense_ledger, tds_str, place_of_supply)
                    tds_section = determine_tds_section(expense_ledger, place_of_supply)
                    tds_amount = round(taxable_amount * tds_rate / 100, 2) if tds_rate > 0 else 0.0
                    amount_payable = total_amount - tds_amount
                    
                    tds_applicability = "Uncertain"
                    if place_of_supply and place_of_supply.lower() == "foreign":
                        tds_applicability = "No"
                    elif tds_rate > 0 or tds_amount > 0:
                        tds_applicability = "Yes"
                    elif "no" in str(tds_str).lower():
                        tds_applicability = "No"

                    try:
                        parsed_date = parser.parse(str(date), dayfirst=True)
                        date = parsed_date.strftime("%d/%m/%Y")
                    except:
                        date = ""
                        
                    buyer_gstin_display = buyer_gstin or "N/A"
                    narration = (
                        f"Invoice {invoice_number or 'N/A'} dated {date or 'N/A'} "
                        f"was issued by {seller_name or 'N/A'} (GSTIN: {seller_gstin or 'N/A'}, HSN/SAC: {hsn_sac or 'N/A'}) "
                        f"to {buyer_name or 'N/A'} (GSTIN: {buyer_gstin_display}), "
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
                        "Seller GSTIN": seller_gstin, # Use the Gemini-validated/prioritized GSTIN
                        "HSN/SAC": hsn_sac,
                        "Buyer Name": buyer_name,
                        "Buyer GSTIN": buyer_gstin, # Use the Gemini-validated/prioritized GSTIN
                        "Expense Ledger": expense_ledger,
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
                "Date": "", "Seller Name": "", "Seller GSTIN": "",
                "HSN/SAC": "", "Buyer Name": "", "Buyer GSTIN": "",
                "Expense Ledger": "", "Taxable Amount": 0.0,
                "CGST": 0.0, "SGST": 0.0, "IGST": 0.0, "Total Amount": 0.0,
                "TDS Applicability": "Uncertain", "TDS Section": None,
                "TDS Rate": 0.0, "TDS Amount": 0.0, "Amount Payable": 0.0,
                "Place of Supply": "", "TDS": "",
                "Narration": f"Error processing file: {str(e)}. Raw response: {response_text if response_text else 'No response received from GPT.'}"
            }
            st.session_state["processed_results"][file_name] = error_row
            st.session_state["processing_status"][file_name] = "❌ Error"
            st.error(f"❌ Error processing {file_name}: {e}")
            if response_text:
                st.text_area(f"Raw Output ({file_name}) - Error Details", response_text, height=200)
            else:
                st.text_area(f"Raw Output ({file_name}) - Error Details", "No response received from GPT.", height=100)

        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
    
    progress_bar.empty()
    progress_text.empty()

results = list(st.session_state["processed_results"].values())

if results and st.session_state.get("process_triggered", False):
    if completed_json:
        st_lottie(completed_json, height=200, key="done_animation")

    st.markdown("<h3 style='text-align: center;'>🎉 Yippie! All invoices Processed!!! 😊</h3>", unsafe_allow_html=True)

    try:
        df = pd.DataFrame(results)
        
        currency_cols_mapping = {
            "Taxable Amount": "Taxable Amount (₹)", "CGST": "CGST (₹)", "SGST": "SGST (₹)",
            "IGST": "IGST (₹)", "Total Amount": "Total Amount (₹)", "TDS Amount": "TDS Amount (₹)",
            "Amount Payable": "Amount Payable (₹)"
        }
        for original_col, display_col in currency_cols_mapping.items():
            if original_col in df.columns:
                df[display_col] = df[original_col].apply(format_currency)
            else:
                df[display_col] = "₹0.00"

        if 'TDS Rate' in df.columns:
            df['TDS Rate'] = pd.to_numeric(df['TDS Rate'], errors='coerce').fillna(0.0)
            df['TDS Rate (%)'] = df['TDS Rate'].apply(lambda x: f"{x:.1f}%" if x > 0 else "0.0%")
        else:
            df['TDS Rate (%)'] = "0.0%"

        display_cols = [
            "File Name", "Invoice Number", "Date", "Seller Name", "Seller GSTIN", "HSN/SAC",
            "Buyer Name", "Buyer GSTIN", "Expense Ledger", "Taxable Amount (₹)", "CGST (₹)",
            "SGST (₹)", "IGST (₹)", "Total Amount (₹)", "TDS Applicability", "TDS Section",
            "TDS Rate (%)", "TDS Amount (₹)", "Amount Payable (₹)", "Place of Supply", "Narration"
        ]
        actual_display_cols = [col for col in display_cols if col in df.columns]

        st.dataframe(
            df[actual_display_cols],
            column_order=actual_display_cols,
            column_config={
                "HSN/SAC": st.column_config.TextColumn("HSN/SAC", help="Harmonized System of Nomenclature / Service Accounting Code", default="N/A"),
                "TDS Section": st.column_config.TextColumn("TDS Section", help="Applicable TDS Section (e.g., 194J)", default="N/A"),
                "TDS Applicability": st.column_config.TextColumn("TDS Applicability", help="Indicates if TDS is applicable (Yes/No/Uncertain)", default="Uncertain"),
                "Taxable Amount (₹)": st.column_config.TextColumn("Taxable Amount (₹)"),
                "CGST (₹)": st.column_config.TextColumn("CGST (₹)"),
                "SGST (₹)": st.column_config.TextColumn("SGST (₹)"),
                "IGST (₹)": st.column_config.TextColumn("IGST (₹)"),
                "Total Amount (₹)": st.column_config.TextColumn("Total Amount (₹)"),
                "TDS Amount (₹)": st.column_config.TextColumn("TDS Amount (₹)"),
                "Amount Payable (₹)": st.column_config.TextColumn("Amount Payable (₹)")
            },
            hide_index=True,
            use_container_width=True
        )

        download_df = df.copy()
        for original_col, display_col in currency_cols_mapping.items():
            if display_col in download_df.columns:
                download_df = download_df.drop(columns=[display_col])
        if 'TDS Rate (%)' in download_df.columns:
            download_df = download_df.drop(columns=['TDS Rate (%)'])
        
        download_cols_ordered = [col for col in display_cols if col not in currency_cols_mapping.values() and col != 'TDS Rate (%)']
        for col_name in ["Taxable Amount", "CGST", "SGST", "IGST", "Total Amount", "TDS Amount", "Amount Payable", "TDS Rate"]:
            if col_name in df.columns and col_name not in download_cols_ordered:
                    download_cols_ordered.append(col_name)

        csv_data = download_df[download_cols_ordered].to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Results as CSV", csv_data, "invoice_results.csv", "text/csv")

        excel_buffer = io.BytesIO()
        try:
            with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                download_df[download_cols_ordered].to_excel(writer, index=False, sheet_name="Invoice Data")
            st.download_button(
                label="📥 Download Results as Excel",
                data=excel_buffer.getvalue(),
                file_name="invoice_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        except Exception as e:
            st.error(f"Failed to create Excel file for download: {str(e)}")
            
    except Exception as e:
        st.error(f"An unexpected error occurred when trying to display or download results: {str(e)}")
        st.write("Raw results data for debugging:")
        st.json(results)

    if 'TDS Applicability' in df.columns and any(df['TDS Applicability'] == "Yes"):
        st.balloons()
    elif completed_count == total_files and completed_count > 0:
        st.balloons()

else:
    if not st.session_state.get("uploaded_files") and not st.session_state.get("process_triggered", False) and not st.session_state.get("processed_results"):
        st.info("Upload one or more scanned invoices to get started.")
    elif st.session_state.get("uploaded_files") and not st.session_state.get("process_triggered", False):
        st.info("Files uploaded. Click 'Process Invoices' to start extraction.")
