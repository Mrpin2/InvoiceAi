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

# Import Pydantic for data validation and structuring
from pydantic import BaseModel, Field, ValidationError, validator
from typing import Optional, Literal

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
st.markdown("Upload scanned PDF invoices and extract structured finance data using GPT-4 Vision, with **Gemini specifically for GSTIN extraction and validation.**")
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
    # Directly specify the model name to avoid list_models() error
    # Try gemini-1.5-flash-latest first, then fall back to gemini-pro-vision, then gemini-pro
    try:
        gemini_model = genai.GenerativeModel('gemini-1.5-flash-latest')
    except Exception:
        try:
            gemini_model = genai.GenerativeModel('gemini-pro-vision')
            st.warning("Could not load 'gemini-1.5-flash-latest'. Using 'gemini-pro-vision' for Gemini.")
        except Exception:
            gemini_model = genai.GenerativeModel('gemini-pro')
            st.warning("Could not load vision-capable Gemini models. Falling back to 'gemini-pro'. GSTIN extraction from images might be less accurate.")

except Exception as e:
    st.error(f"Failed to initialize Gemini client. Check your API key: {e}")
    st.stop()


# --- Pydantic Models for Data Validation ---

class ExtractedInvoiceData(BaseModel):
    invoice_number: Optional[str] = None
    date: Optional[str] = None
    seller_name: Optional[str] = None
    buyer_name: Optional[str] = None
    taxable_amount: Optional[float] = None
    cgst: Optional[float] = None
    sgst: Optional[float] = None
    igst: Optional[float] = None
    place_of_supply: Optional[str] = None
    expense_ledger: Optional[str] = None
    tds: Optional[str] = None
    hsn_sac: Optional[str] = None

    @validator('taxable_amount', 'cgst', 'sgst', 'igst', pre=True)
    def clean_currency_fields(cls, v):
        if v is None:
            return 0.0
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str):
            cleaned = v.replace(",", "").replace("₹", "").replace("$", "").strip()
            try:
                return float(cleaned)
            except ValueError:
                return 0.0
        return 0.0

    @validator('date', pre=True)
    def parse_date_string(cls, v):
        if v is None or not isinstance(v, str) or v.strip() == "":
            return None
        try:
            parsed = parser.parse(v, dayfirst=True)
            return parsed.strftime("%d/%m/%Y")
        except Exception:
            return None

class ExtractedGSTINs(BaseModel):
    seller_gstin: Optional[str] = None
    buyer_gstin: Optional[str] = None

    @validator('seller_gstin', 'buyer_gstin', pre=True)
    def clean_gstin(cls, v):
        if v is None:
            return None
        # Remove any non-alphanumeric characters and convert to uppercase
        cleaned = re.sub(r'[^A-Z0-9]', '', str(v).upper())
        return cleaned if is_valid_gstin(cleaned) else None # Validate against the existing function

class ProcessedInvoiceResult(BaseModel):
    file_name: str = Field(..., alias="File Name")
    invoice_number: Optional[str] = Field(None, alias="Invoice Number")
    date: Optional[str] = Field(None, alias="Date")
    seller_name: Optional[str] = Field(None, alias="Seller Name")
    seller_gstin: Optional[str] = Field(None, alias="Seller GSTIN")
    hsn_sac: Optional[str] = Field(None, alias="HSN/SAC")
    buyer_name: Optional[str] = Field(None, alias="Buyer Name")
    buyer_gstin: Optional[str] = Field(None, alias="Buyer GSTIN")
    expense_ledger: Optional[str] = Field(None, alias="Expense Ledger")
    taxable_amount: float = Field(0.0, alias="Taxable Amount")
    cgst: float = Field(0.0, alias="CGST")
    sgst: float = Field(0.0, alias="SGST")
    igst: float = Field(0.0, alias="IGST")
    total_amount: float = Field(0.0, alias="Total Amount")
    tds_applicability: Literal["Yes", "No", "Uncertain", "N/A"] = Field("Uncertain", alias="TDS Applicability")
    tds_section: Optional[str] = Field(None, alias="TDS Section")
    tds_rate: float = Field(0.0, alias="TDS Rate")
    tds_amount: float = Field(0.0, alias="TDS Amount")
    amount_payable: float = Field(0.0, alias="Amount Payable")
    place_of_supply: Optional[str] = Field(None, alias="Place of Supply")
    tds_raw: Optional[str] = Field(None, alias="TDS") # Store the raw TDS string from AI
    narration: str = Field(..., alias="Narration")

    # Calculated fields - ensure they are consistent
    @validator('total_amount', always=True)
    def calculate_total_amount(cls, v, values):
        return values.get('taxable_amount', 0.0) + values.get('cgst', 0.0) + values.get('sgst', 0.0) + values.get('igst', 0.0)

    @validator('tds_amount', always=True)
    def calculate_tds_amount(cls, v, values):
        taxable = values.get('taxable_amount', 0.0)
        rate = values.get('tds_rate', 0.0)
        return round(taxable * rate / 100, 2) if rate > 0 else 0.0

    @validator('amount_payable', always=True)
    def calculate_amount_payable(cls, v, values):
        total = values.get('total_amount', 0.0)
        tds = values.get('tds_amount', 0.0)
        return total - tds

    @validator('tds_applicability', always=True)
    def set_tds_applicability(cls, v, values):
        if values.get('place_of_supply', "").lower() == "foreign":
            return "No"
        elif values.get('tds_rate', 0.0) > 0 or values.get('tds_amount', 0.0) > 0:
            return "Yes"
        elif "no" in str(values.get('tds_raw', "")).lower():
            return "No"
        return "Uncertain"

# --- Functions remain the same ---
def convert_pdf_first_page(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc.load_page(0)
    pix = page.get_pixmap(dpi=300)
    img_bytes = pix.tobytes("png")
    return Image.open(io.BytesIO(img_bytes))

def safe_float(x):
    try:
        # Pydantic's validator will handle this, but keep for older paths or direct calls if needed
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

# This function will now ONLY use Gemini API for GSTIN extraction
def extract_gstins_with_gemini(image_data):
    """
    Extracts all potential valid GSTINs from the provided image using the Gemini API.
    Returns an ExtractedGSTINs pydantic model.
    """
    prompt = (
        "You are an expert at identifying Indian GSTINs from images of invoices. "
        "Your task is to identify the 'seller GSTIN' (the GSTIN of the entity issuing the invoice) "
        "and the 'buyer GSTIN' (the GSTIN of the entity receiving the invoice). "
        "An Indian GSTIN is a 15-character alphanumeric string, typically following the pattern: "
        "2 digits (state code) + 10 alphanumeric characters (PAN) + 1 digit (entity code) + 1 letter (checksum) + 1 Z + 1 checksum digit/letter."
        "Example: '07AAFFD8152M1Z4'."
        "If you find multiple GSTINs, clearly distinguish which one belongs to the seller and which to the buyer."
        "Return the result as a JSON object with two keys: 'seller_gstin' and 'buyer_gstin'. "
        "If a GSTIN is not found or not clearly identifiable, use `null` for that key's value."
        "Example output: `{'seller_gstin': '07AAFFD8152M1Z4', 'buyer_gstin': '07AAICE6026F1ZS'}`."
        "If no GSTINs are found at all, return `{'seller_gstin': null, 'buyer_gstin': null}`."
    )
    
    try:
        # Pydantic's Image model expects a path or bytes, here we directly pass bytes from PIL
        response = gemini_model.generate_content([prompt, Image.open(io.BytesIO(image_data))])
        response_text = response.text.strip()
        
        extracted_data_dict = extract_json_from_response(response_text)
        if extracted_data_dict:
            try:
                # Use Pydantic to validate and clean the GSTINs
                gstins = ExtractedGSTINs(**extracted_data_dict)
                return gstins
            except ValidationError as e:
                st.warning(f"Gemini output validation failed for GSTINs: {e.errors()}")
                return ExtractedGSTINs() # Return an empty model on validation failure
        return ExtractedGSTINs() # Return empty model if no JSON extracted
    except Exception as e:
        st.warning(f"Gemini API call failed for GSTIN extraction: {e}")
        return ExtractedGSTINs() # Return empty model on API error

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
    except json.JSONDecodeError:
        return None # Return None if not valid JSON
    except Exception as e:
        st.warning(f"Failed to extract JSON from response: {e}")
        return None

main_prompt = (
    "You are an expert at extracting structured data from Indian invoices. "
    "Your task is to extract information into a JSON object with the following keys. "
    "If a key's value is not explicitly present or derivable from the invoice, use `null` for that value. "
    "Keys to extract: invoice_number, date, seller_name, buyer_name, "
    "taxable_amount, cgst, sgst, igst, place_of_supply, expense_ledger, tds, hsn_sac. "
    "DO NOT attempt to extract GSTINs. GSTINs will be handled by a separate process."
    
    "GUIDELINES FOR EXTRACTION:\n"
    "- 'invoice_number': The unique identifier of the invoice. Extract as is.\n"
    "- 'date': The invoice date in DD/MM/YYYY format. If year is 2-digit, assume current century (e.g., 24 -> 2024).\n"
    "- 'taxable_amount': This is the subtotal, the amount BEFORE any taxes (CGST, SGST, IGST) are applied. Must be a number.\n"
    "- 'hsn_sac': Crucial for Indian invoices. "
    "  - HSN (Harmonized System of Nomenclature) is for goods."
    "  - SAC (Service Accounting Code) is for services."
    "  - **ONLY extract the HSN/SAC code if it is explicitly mentioned on the invoice.** "
    "  - It is typically a 4, 6, or 8-digit numeric code, sometimes alphanumeric."
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

# Define the file_uploader_placeholder here
file_uploader_placeholder = st.empty()

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

            # Convert PIL Image to bytes for Gemini and OpenAI
            img_buf = io.BytesIO()
            first_image.save(img_buf, format="PNG")
            img_bytes = img_buf.getvalue() # Get bytes for both APIs

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
                extracted_data_dict = extract_json_from_response(response_text)
                
                invoice_data = None
                if extracted_data_dict:
                    try:
                        invoice_data = ExtractedInvoiceData(**extracted_data_dict)
                    except ValidationError as e:
                        st.warning(f"Pydantic validation failed for GPT-4 Vision output for {file_name}: {e.errors()}")
                        st.text_area(f"Raw GPT-4 Output ({file_name}) - Validation Error", response_text, height=200)
                        # Proceed with partial data or mark as error later
                        invoice_data = ExtractedInvoiceData(**{k: extracted_data_dict.get(k) for k in ExtractedInvoiceData.model_fields.keys() if k in extracted_data_dict})
                
                if invoice_data is None or (invoice_data.invoice_number is None and "not an invoice" in response_text.lower()):
                    result_row_data = {
                        "File Name": file_name,
                        "Invoice Number": "NOT AN INVOICE",
                        "Narration": "This document was identified as not an invoice."
                    }
                    result_row = ProcessedInvoiceResult(**result_row_data)
                else:
                    # --- Gemini API for GSTIN extraction (sole source for GSTINs) ---
                    st.info(f"🔍 Extracting and validating GSTINs for {file_name} using Gemini...")
                    gstin_data_from_gemini: ExtractedGSTINs = extract_gstins_with_gemini(img_bytes)
                    
                    seller_gstin = gstin_data_from_gemini.seller_gstin
                    buyer_gstin = gstin_data_from_gemini.buyer_gstin
                    # --- End of Gemini API for GSTIN extraction ---

                    # Calculate derived fields for the final Pydantic model
                    tds_rate = determine_tds_rate(
                        invoice_data.expense_ledger,
                        invoice_data.tds,
                        invoice_data.place_of_supply
                    )
                    tds_section = determine_tds_section(
                        invoice_data.expense_ledger,
                        invoice_data.place_of_supply
                    )
                    
                    # Prepare dictionary for ProcessedInvoiceResult model
                    result_row_data = {
                        "File Name": file_name,
                        "Invoice Number": invoice_data.invoice_number,
                        "Date": invoice_data.date,
                        "Seller Name": invoice_data.seller_name,
                        "Seller GSTIN": seller_gstin,
                        "HSN/SAC": invoice_data.hsn_sac,
                        "Buyer Name": invoice_data.buyer_name,
                        "Buyer GSTIN": buyer_gstin,
                        "Expense Ledger": invoice_data.expense_ledger,
                        "Taxable Amount": invoice_data.taxable_amount,
                        "CGST": invoice_data.cgst,
                        "SGST": invoice_data.sgst,
                        "IGST": invoice_data.igst,
                        "Place of Supply": invoice_data.place_of_supply,
                        "TDS": invoice_data.tds, # raw TDS string
                        "TDS Rate": tds_rate,
                        "TDS Section": tds_section,
                        # total_amount, tds_amount, amount_payable, tds_applicability are calculated by Pydantic validators
                    }

                    # Create Narration
                    buyer_gstin_display = buyer_gstin or "N/A"
                    narration = (
                        f"Invoice {invoice_data.invoice_number or 'N/A'} dated {invoice_data.date or 'N/A'} "
                        f"was issued by {invoice_data.seller_name or 'N/A'} (GSTIN: {seller_gstin or 'N/A'}, HSN/SAC: {invoice_data.hsn_sac or 'N/A'}) "
                        f"to {invoice_data.buyer_name or 'N/A'} (GSTIN: {buyer_gstin_display}), "
                        f"with a taxable amount of ₹{invoice_data.taxable_amount or 0.0:,.2f}. "
                        f"Taxes applied - CGST: ₹{invoice_data.cgst or 0.0:,.2f}, SGST: ₹{invoice_data.sgst or 0.0:,.2f}, IGST: ₹{invoice_data.igst or 0.0:,.2f}. "
                        f"Place of Supply: {invoice_data.place_of_supply or 'N/A'}. Expense: {invoice_data.expense_ledger or 'N/A'}. "
                    )
                    # For narration, let's append TDS info after calculating with the Pydantic model
                    # So, we'll assign narration after the model is created and calculated fields are available
                    
                    try:
                        result_row = ProcessedInvoiceResult(**result_row_data)
                        # Now that calculated fields are populated, finalize narration
                        narration += (
                            f"Total Amount: ₹{result_row.total_amount:,.2f}. "
                            f"TDS: {result_row.tds_applicability} (Section: {result_row.tds_section or 'N/A'}) @ {result_row.tds_rate}% (₹{result_row.tds_amount:,.2f}). "
                            f"Amount Payable: ₹{result_row.amount_payable:,.2f}."
                        )
                        result_row.narration = narration

                    except ValidationError as e:
                        st.error(f"Pydantic validation failed for final result for {file_name}: {e.errors()}")
                        st.text_area(f"Final Result Data ({file_name}) - Validation Error", json.dumps(result_row_data, indent=2), height=200)
                        # Fallback to an error row if final validation fails
                        result_row_data["Invoice Number"] = "VALIDATION ERROR"
                        result_row_data["Narration"] = f"Pydantic validation error for final data: {e.errors()}"
                        result_row = ProcessedInvoiceResult(**result_row_data) # Create a partial/error model


            # Store the result by converting the Pydantic model back to a dictionary
            st.session_state["processed_results"][file_name] = result_row.model_dump(by_alias=True)
            st.session_state["processing_status"][file_name] = "✅ Done"
            completed_count += 1
            st.success(f"{file_name}: ✅ Done")

        except Exception as e:
            error_narration = f"Error processing file: {str(e)}. Raw response: {response_text if response_text else 'No response received from GPT.'}"
            error_row_data = {
                "File Name": file_name,
                "Invoice Number": "PROCESSING ERROR",
                "Narration": error_narration
            }
            error_row = ProcessedInvoiceResult(**error_row_data)
            st.session_state["processed_results"][file_name] = error_row.model_dump(by_alias=True)
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
        # Remove formatted currency columns before download, keep original numeric ones
        for original_col, display_col in currency_cols_mapping.items():
            if display_col in download_df.columns:
                download_df = download_df.drop(columns=[display_col])
        if 'TDS Rate (%)' in download_df.columns:
            download_df = download_df.drop(columns=['TDS Rate (%)'])
        
        # Ensure original numeric columns are available and reorder for download
        download_cols_ordered = [col for col in display_cols if col not in currency_cols_mapping.values() and col != 'TDS Rate (%)']
        for col_name in ["Taxable Amount", "CGST", "SGST", "IGST", "Total Amount", "TDS Amount", "Amount Payable", "TDS Rate", "TDS"]:
            if col_name in download_df.columns and col_name not in download_cols_ordered:
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

    # Check for TDS applicability in the DataFrame that's actually displayed/processed
    if 'TDS Applicability' in df.columns and any(df['TDS Applicability'] == "Yes"):
        st.balloons()
    elif completed_count == total_files and completed_count > 0:
        st.balloons()

else:
    if not st.session_state.get("uploaded_files") and not st.session_state.get("process_triggered", False) and not st.session_state.get("processed_results"):
        st.info("Upload one or more scanned invoices to get started.")
    elif st.session_state.get("uploaded_files") and not st.session_state.get("process_triggered", False):
        st.info("Files uploaded. Click 'Process Invoices' to start extraction.")
    
