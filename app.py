import streamlit as st
from pydantic import BaseModel, Field
from typing import List, Optional, Union
import pandas as pd
import os
import tempfile
import io
from datetime import datetime
import json
import base64

# --- New Imports for PDF to Image Conversion ---
try:
    import fitz # PyMuPDF
    from PIL import Image
except ImportError:
    st.warning("PyMuPDF (fitz) or Pillow not installed. PDF to image conversion will not work for OpenAI.")
    # Set these to None if not imported, so we can check for them later
    fitz = None
    Image = None

# --- New Import for Lottie Animation ---
try:
    from streamlit_lottie import st_lottie
except ImportError:
    st.warning("The 'streamlit-lottie' library is not installed. Lottie animations will not work.")
    st_lottie = None # Set to None if not imported


# --- Import Libraries for Both Models ---
try:
    from google import genai
except ImportError:
    st.warning("The 'google-generativeai' library is not installed. Gemini functionality will be unavailable.")
    genai = None # Set to None if not imported

try:
    from openai import OpenAI
except ImportError:
    st.warning("The 'openai' library is not installed. OpenAI functionality will be unavailable.")
    OpenAI = None # Set to None if not imported


# --- Pydantic Models (Common for both APIs) ---
class LineItem(BaseModel):
    description: str
    quantity: float
    gross_worth: float

class Invoice(BaseModel):
    invoice_number: str
    date: str
    gstin: str # Seller GSTIN
    seller_name: str
    buyer_name: str
    buyer_gstin: Optional[str] = None
    taxable_amount: float # New field: subtotal BEFORE taxes
    cgst: Optional[float] = None
    sgst: Optional[float] = None
    igst: Optional[float] = None
    total_amount_payable: float # This is the Gross Total (Including Tax) from the invoice, BEFORE any TDS deduction
    tds_amount: Optional[float] = None # The numerical value of TDS deducted, if explicitly stated
    tds_rate: Optional[float] = None # NEW: Added TDS Rate
    line_items: List[LineItem] # Still keep line items if they are present
    place_of_supply: Optional[str] = None
    expense_ledger: Optional[str] = None
    tds: Optional[str] = None # Applicability string (e.g., "Yes - 194J")
    hsn_sac: Optional[str] = None # Moved to invoice level based on prompt
    rcm_applicability: Optional[str] = None


# --- Utility Functions for UI/Data Formatting ---
def parse_date_safe(date_str: str) -> str:
    """Attempts to parse a date string into DD/MM/YYYY format."""
    if not date_str:
        return ""
    # Try common Indian date formats
    formats = ["%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d", "%d %b %Y", "%d %B %Y", "%Y/%m/%d"]
    for fmt in formats:
        try:
            # Handle 2-digit year by assuming current century (e.g., 24 -> 2024)
            dt_obj = datetime.strptime(date_str, fmt)
            # Simple heuristic for 2-digit years: if year is less than (current year % 100) + 10 (e.g., 2024 -> 34),
            # assume 20xx. Otherwise, assume 19xx.
            if dt_obj.year < (datetime.now().year % 100) + 10 and dt_obj.year <= 99:
                 dt_obj = dt_obj.replace(year=dt_obj.year + 2000)
            elif dt_obj.year <= 99: # For years like 98 -> 1998
                dt_obj = dt_obj.replace(year=dt_obj.year + 1900)
            return dt_obj.strftime("%d/%m/%Y")
        except ValueError:
            continue
    # If no format matches, return original string or a placeholder
    return date_str # Or "Invalid Date"

def format_currency(amount: Optional[float]) -> str:
    """Formats a float as an Indian Rupee currency string."""
    if amount is None:
        return "₹ N/A"
    return f"₹ {amount:,.2f}" # Formats with commas and 2 decimal places

def load_lottie_url(url: str):
    try:
        import requests
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception as e:
        st.error(f"Could not load Lottie animation from URL: {e}")
        return None

# --- Main Prompt for LLMs ---
main_prompt = (
    "You are an expert at extracting structured data from Indian invoices. "
    "Your task is to extract information into a JSON object with the following keys. "
    "If a key's value is not explicitly present or derivable from the invoice, use `null` for that value. "
    "Keys to extract: invoice_number, date, gstin (seller's GSTIN), seller_name, buyer_name, buyer_gstin, "
    "taxable_amount, cgst, sgst, igst, total_amount_payable, tds_amount, tds_rate, place_of_supply, expense_ledger, tds, hsn_sac, rcm_applicability. "
    "You MUST include an empty list `[]` for `line_items` if no line items are found, do not use `null` for `line_items`. "
    "For `line_items`, each item must have 'description', 'quantity', 'gross_worth'.\n\n"
    
    "GUIDELINES FOR EXTRACTION:\n"
    "- 'invoice_number': The unique identifier of the invoice. Extract as is.\n"
    "- 'date': The invoice date in DD/MM/YYYY format. If year is 2-digit, assume current century (e.g., 24 -> 2024).\n"
    "- 'taxable_amount': This is the subtotal, the amount BEFORE any taxes (CGST, SGST, IGST) are applied. Must be a number.\n"
    "- 'total_amount_payable': This is the final total amount on the invoice *including all taxes and other charges*, but *before* any TDS deduction shown on the invoice. This represents the 'Gross Total' or 'Amount before TDS' on the invoice. Must be a number.\n"
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
    "  'Cloud Services', 'Google Cloud', 'AWS', 'Microsoft Azure', 'DigitalOcean', 'Marketing Expenses', 'Travel Expenses'. "
    "  If the expense is clearly related to software licenses, subscriptions, or SaaS, classify as 'Software Subscription'."
    "  Aim for a general and universal ledger type if a precise one isn't obvious from the invoice details."
    "  **Consider common TDS sections and rates when determining expense type, e.g., 'Professional Fees' often implies TDS under Section 194J at 10%.**\n"
    
    "- 'tds': Determine TDS applicability. This field should be a string indicating applicability and section. "
    "  - If TDS is deducted, a TDS section is mentioned (e.g., 'TDS u/s 194J', 'TDS @10%'), state 'Yes - Section [X]' (e.g., 'Yes - Section 194J'). "
    "  - **If a TDS rate or amount is present but no section is explicitly mentioned, infer the most common section based on expense (e.g., 194J for professional services), otherwise state 'Yes - Section Unknown'.**"
    "  - If TDS is explicitly stated as 'Not Applicable' or no TDS details (amount, rate, section) are present and the invoice is clearly domestic, state 'No'."
    "  - If unclear or implied but no explicit section/amount, state 'Uncertain'."
    "  - **Crucially, if the buyer is explicitly stated as 'Foreign' or the 'Place of Supply' is outside India, then TDS is typically 'No'. Prefer 'No' in such cases.**\n"
    "- 'tds_amount': Extract the exact numerical value of the TDS deducted from the invoice, if explicitly stated. If not stated, set to `null`.\n"
    "- 'tds_rate': Extract the numerical percentage rate of TDS deducted (e.g., '10' for 10%), if explicitly stated. "
    "  **If a TDS Section is clearly identified but the rate is missing on the invoice, infer the standard rate for that section (e.g., 10% for 194J).** If not stated and not inferable from section, set to `null`.\n"
    
    "- 'rcm_applicability': Determine Reverse Charge Mechanism (RCM) applicability. State 'Yes' if clearly applicable, 'No' if clearly not, or 'Uncertain' if unclear.\n"

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


# --- Extraction Functions (Model-Specific) ---

def extract_from_gemini(
    client_instance: 'genai.Client',
    gemini_model_id: str,
    file_path: str,
    pydantic_schema: BaseModel,
) -> Optional[Invoice]:
    """Extracts structured data from an invoice PDF using Gemini Vision."""
    display_name = os.path.basename(file_path)
    gemini_file_resource = None

    if genai is None:
        st.error("Gemini library not initialized. Cannot use Gemini.")
        return None

    try:
        gemini_file_resource = client_instance.files.upload(
            file=file_path,
            config={'display_name': display_name.split('.')[0]}
        )
        
        prompt_content = [main_prompt, gemini_file_resource]
        
        response = client_instance.models.generate_content(
            model=gemini_model_id,
            contents=prompt_content,
            config={'response_mime_type': 'application/json', 'response_schema': pydantic_schema}
        )

        return response.parsed

    except Exception as e:
        st.error(f"Error processing '{display_name}' with Gemini: {e}")
        st.exception(e)
        return None
    finally:
        if gemini_file_resource:
            try:
                if client_instance and hasattr(client_instance, 'files') and hasattr(client_instance.files, 'delete'):
                    client_instance.files.delete(name=gemini_file_resource.name)
                else:
                    st.warning(f"Gemini: File API client does not support direct deletion or method not found. Manual cleanup may be required.")
            except Exception as e_del:
                st.warning(f"Gemini: Could not delete '{gemini_file_resource.name}': {e_del}")


def extract_from_openai(
    client_instance: 'OpenAI',
    openai_model_id: str,
    file_path: str,
    pydantic_schema: BaseModel,
) -> Optional[Invoice]:
    """Extracts structured data from an invoice PDF using OpenAI Vision by converting PDF pages to images."""
    display_name = os.path.basename(file_path)

    if OpenAI is None:
        st.error("OpenAI library not initialized. Cannot use OpenAI.")
        return None

    if fitz is None or Image is None:
        st.error("PyMuPDF (fitz) or Pillow not installed. Cannot process PDFs for OpenAI. Please check your requirements.txt.")
        return None

    try:
        doc = fitz.open(file_path)
        image_messages = []
        max_pages_to_process = 5 # Limit to prevent excessive costs/tokens for very long PDFs

        for page_num in range(min(doc.page_count, max_pages_to_process)):
            page = doc.load_page(page_num)
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2)) # Render at 2x resolution, DPI ~144-150
            
            # Using io.BytesIO to save image to memory, then base64 encode
            img_bytes_io = io.BytesIO()
            pil_image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            pil_image.save(img_bytes_io, format="PNG")
            img_bytes = img_bytes_io.getvalue()

            base64_image = base64.b64encode(img_bytes).decode('utf-8')
            image_messages.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64_image}",
                    "detail": "high" # Use 'high' detail for better OCR
                }
            })
        doc.close()

        if not image_messages:
            st.error(f"No pages could be converted to images for '{display_name}'. This might happen with corrupted PDFs or empty files.")
            return None

        # Use the main_prompt directly as system prompt
        system_prompt = main_prompt
        user_prompt_text = "Extract invoice data according to the provided schema from these document pages."

        messages_content = [{"type": "text", "text": user_prompt_text}] + image_messages

        response = client_instance.chat.completions.create(
            model=openai_model_id,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": messages_content}
            ],
            response_format={"type": "json_object"},
            max_tokens=4000
        )

        if response and response.choices and response.choices[0].message and response.choices[0].message.content:
            json_string = response.choices[0].message.content
            # Debugging: show raw JSON only if debug mode is active
            if st.session_state.get('DEBUG_MODE', False):
                st.markdown("##### Raw JSON from OpenAI:")
                st.code(json_string, language="json")
            try:
                extracted_dict = json.loads(json_string)
                extracted_invoice = pydantic_schema.parse_obj(extracted_dict)
                return extracted_invoice
            except json.JSONDecodeError as e:
                st.error(f"OpenAI: Failed to decode JSON from response for '{display_name}': {e}")
                st.error(f"Response content: {json_string[:500]}..." if json_string else "No JSON content received.")
                if st.session_state.get('DEBUG_MODE', False):
                    st.code(json_string, language="json")
                return None
            except Exception as e:
                st.error(f"OpenAI: Failed to parse extracted data into schema for '{display_name}': {e}")
                st.error(f"Response content: {json_string[:500]}..." if json_string else "No JSON content received.")
                if st.session_state.get('DEBUG_MODE', False):
                    st.code(json_string, language="json")
                return None
        else:
            st.warning(f"OpenAI: No content received or unexpected response structure for '{display_name}'.")
            if st.session_state.get('DEBUG_MODE', False):
                st.json(response.dict()) # Show full response object for debug
            return None

    except Exception as e:
        st.error(f"Error processing '{display_name}' with OpenAI: {e}")
        st.exception(e)
        return None

# --- Streamlit App UI ---
st.set_page_config(layout="wide", page_title="📄 AI Invoice Extractor")

# Custom CSS for a bit more flair and font consistency
st.markdown("""
<style>
    body {
        font-family: 'Roboto', 'Helvetica Neue', Arial, sans-serif; /* Add popular sans-serif fonts */
    }
    .stApp { 
        background-color: #f0f2f6; 
        color: #333333; 
        font-family: 'Roboto', 'Helvetica Neue', Arial, sans-serif; 
    }
    h1, h2, h3 { 
        color: #1e3a8a; 
        font-family: 'Roboto', 'Helvetica Neue', Arial, sans-serif; 
    }
    .stButton>button {
        background-color: #3b82f6; 
        color: white; 
        border-radius: 8px;
        padding: 10px 20px; 
        font-size: 16px; 
        font-weight: bold;
        transition: background-color 0.3s ease;
        font-family: 'Roboto', 'Helvetica Neue', Arial, sans-serif;
    }
    .stButton>button:hover { 
        background-color: #2563eb; 
    }
    /* General Markdown paragraph styling */
    .stMarkdown p { 
        font-size: 1.05em; 
        line-height: 1.6; 
        font-family: 'Roboto', 'Helvetica Neue', Arial, sans-serif; 
        color: #333333; /* Default dark color for general markdown text */
    }
    .stAlert { 
        border-radius: 8px; 
        font-family: 'Roboto', 'Helvetica Neue', Arial, sans-serif; 
    }
    .stAlert.info { 
        background-color: #333333; /* Darker grey background */
    }
    .stAlert.success { 
        background-color: #e8f5e9; 
        color: #2e7d32; 
    }
    .stAlert.error { 
        background-color: #ffebee; 
        color: #c62828; 
    }
    .stProgress > div > div > div > div { 
        background-color: #3b82f6 !important; 
    }
    
    /* *** CRITICAL FIX for instruction text visibility *** */
    /* Target paragraphs and list items directly inside the st.info alert */
    .stAlert.info p, 
    .stAlert.info li,
    .stAlert.info div, /* Added div as a general container */
    .stAlert.info div[data-testid="stMarkdownContainer"] li /* More specific for list items */
    { 
        color: #FFFFFF !important; /* White text for strong contrast */
        font-weight: 500 !important; /* Make it a bit bolder */
        text-shadow: none !important; /* Remove any potential text shadow */
    }
</style>
""", unsafe_allow_html=True)


st.title("📄 AI Invoice Extractor (Multi-Model Powered)")

st.sidebar.header("Configuration")

# --- Admin Panel for using secrets ---
# ADMIN_PASSWORD will be fetched from Streamlit Secrets. If not found there, it defaults to "Rajeev".
ADMIN_PASSWORD = st.secrets.get("ADMIN_PASSWORD", "Rajeev")

admin_password_input = st.sidebar.text_input("Admin Password (Optional):", type="password", key="admin_pass")

use_secrets_keys = False
if admin_password_input:
    if admin_password_input == ADMIN_PASSWORD:
        st.sidebar.success("Admin mode activated. Using API keys from secrets.")
        use_secrets_keys = True
        st.session_state.DEBUG_MODE = True # Activate debug mode when in admin mode
    else:
        st.sidebar.error("Incorrect admin password.")
        use_secrets_keys = False
        st.session_state.DEBUG_MODE = False # Deactivate debug mode if password is wrong
else:
    st.session_state.DEBUG_MODE = False # Default to no debug mode if no password entered

# Model Selection
model_choice = st.sidebar.radio(
    "Choose AI Model:",
    ("Google Gemini", "OpenAI GPT"),
    key="model_choice"
)

# API Key Inputs (Conditional based on admin mode)
selected_api_key = None
model_id_input = None

if use_secrets_keys:
    # Attempt to load from secrets
    if model_choice == "Google Gemini":
        selected_api_key = st.secrets.get("GEMINI_API_KEY")
        model_id_input = st.secrets.get("GEMINI_MODEL_ID", "gemini-1.5-flash-latest")
        if not selected_api_key:
            st.sidebar.warning("GEMINI_API_KEY not found in Streamlit Secrets. Please add it.")
        st.sidebar.text_input("Gemini Model ID:", model_id_input, key="gemini_model_id_secrets", disabled=True)
        st.sidebar.caption(f"Using model ID from secrets: `{model_id_input}`")

    elif model_choice == "OpenAI GPT":
        selected_api_key = st.secrets.get("OPENAI_API_KEY")
        model_id_input = st.secrets.get("OPENAI_MODEL_ID", "gpt-4o")
        if not selected_api_key:
            st.sidebar.warning("OPENAI_API_KEY not found in Streamlit Secrets. Please add it.")
        st.sidebar.text_input("OpenAI Model ID:", model_id_input, key="openai_model_id_secrets", disabled=True)
        st.sidebar.caption(f"Using model ID from secrets: `{model_id_input}`")

    if selected_api_key:
        st.sidebar.info(f"Using {model_choice} API Key from `Streamlit Secrets`.")
    else:
        # This fallback happens if admin mode is ON but the *specific API key* is missing from secrets.
        st.sidebar.error(f"No {model_choice} API Key loaded from `Streamlit Secrets`. "
                           "Please enter it manually below (this will override the secrets attempt).")
        if model_choice == "Google Gemini":
            selected_api_key = st.sidebar.text_input("Enter your Gemini API Key:", type="password", key="gemini_key_manual_fallback")
            if not model_id_input:
                model_id_input = st.sidebar.text_input("Gemini Model ID (Manual Fallback):", "gemini-1.5-flash-latest", key="gemini_model_id_manual_fallback")
        elif model_choice == "OpenAI GPT":
            selected_api_key = st.sidebar.text_input("Enter your OpenAI API Key:", type="password", key="openai_key_manual_fallback")
            if not model_id_input:
                model_id_input = st.sidebar.text_input("OpenAI Model ID (Manual Fallback):", "gpt-4o", key="openai_model_id_manual_fallback")

else: # Default behavior: user must enter keys manually
    if model_choice == "Google Gemini":
        selected_api_key = st.sidebar.text_input("Enter your Gemini API Key:", type="password", key="gemini_key")
        DEFAULT_GEMINI_MODEL_ID = "gemini-1.5-flash-latest"
        model_id_input = st.sidebar.text_input("Gemini Model ID:", DEFAULT_GEMINI_MODEL_ID, key="gemini_model_id")
        st.sidebar.caption(f"Default is `{DEFAULT_GEMINI_MODEL_ID}`. Ensure it supports JSON schema.")
    elif model_choice == "OpenAI GPT":
        selected_api_key = st.sidebar.text_input("Enter your OpenAI API Key:", type="password", key="openai_key")
        DEFAULT_OPENAI_MODEL_ID = "gpt-4o"
        model_id_input = st.sidebar.text_input("OpenAI Model ID:", DEFAULT_OPENAI_MODEL_ID, key="openai_model_id")
        st.sidebar.caption(f"Default is `{DEFAULT_OPENAI_MODEL_ID}`. Ensure it's a vision model and supports JSON output.")

st.info(
    "**Instructions:**\n"
    f"1. Select your preferred AI model ({model_choice}) in the sidebar.\n"
    "   💡 **Recommendation:** Use **Google Gemini** for **scanned or blurred documents**, and **OpenAI GPT** for **system-generated (clear) PDF invoices**.\n"
    "2. If you know the admin password, enter it to use pre-configured API keys from `Streamlit Secrets`.\n"
    "3. Upload one or more PDF invoice files.\n"
    "4. Click 'Process Invoices' to extract data.\n"
    "   The extracted data will be displayed in a table and available for download as Excel."
)

uploaded_files = st.file_uploader(
    "Choose PDF invoice files",
    type="pdf",
    accept_multiple_files=True
)

if 'summary_rows' not in st.session_state:
    st.session_state.summary_rows = []
if 'gemini_client' not in st.session_state:
    st.session_state.gemini_client = None
if 'openai_client' not in st.session_state:
    st.session_state.openai_client = None
if 'DEBUG_MODE' not in st.session_state:
    st.session_state.DEBUG_MODE = False # Initialize DEBUG_MODE in session state


if st.button("🚀 Process Invoices", type="primary"):
    if not selected_api_key:
        st.error(f"Please enter your {model_choice} API Key in the sidebar.")
    elif not uploaded_files:
        st.error("Please upload at least one PDF file.")
    elif not model_id_input:
        st.error(f"Please specify a {model_choice} Model ID in the sidebar.")
    else:
        client_initialized = False
        if model_choice == "Google Gemini":
            if genai: # Check if genai was imported successfully
                try:
                    st.session_state.gemini_client = genai.Client(api_key=selected_api_key)
                    st.success("Gemini client initialized successfully!")
                    client_initialized = True
                except Exception as e:
                    st.error(f"Failed to initialize Gemini client: {e}. Please check your API key.")
                    st.session_state.gemini_client = None
            else:
                st.error("Gemini library not found or failed to import. Cannot initialize Gemini client.")

        elif model_choice == "OpenAI GPT":
            if OpenAI: # Check if OpenAI was imported successfully
                try:
                    st.session_state.openai_client = OpenAI(api_key=selected_api_key)
                    st.success("OpenAI client initialized successfully!")
                    client_initialized = True
                except Exception as e:
                    st.error(f"Failed to initialize OpenAI client: {e}. Please check your API key.")
                    st.session_state.openai_client = None
            else:
                st.error("OpenAI library not found or failed to import. Cannot initialize OpenAI client.")
        
        if not client_initialized:
            st.stop()

        if (model_choice == "Google Gemini" and st.session_state.gemini_client) or \
           (model_choice == "OpenAI GPT" and st.session_state.openai_client):
            
            st.session_state.summary_rows = []
            progress_bar = st.progress(0)
            total_files = len(uploaded_files)

            for i, uploaded_file_obj in enumerate(uploaded_files):
                st.markdown(f"---")
                # Simplified progress message
                st.info(f"Processing file: **{uploaded_file_obj.name}** ({i+1}/{total_files}) using **{model_choice}**...")
                temp_file_path = None
                extracted_data = None
                try:
                    # Save uploaded file to a temporary file for PyMuPDF/OpenAI to access
                    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file_obj.name)[1]) as tmp:
                        tmp.write(uploaded_file_obj.getvalue())
                        temp_file_path = tmp.name

                    with st.spinner(f"Extracting data from {uploaded_file_obj.name} with {model_choice}..."):
                        if model_choice == "Google Gemini":
                            extracted_data = extract_from_gemini(
                                client_instance=st.session_state.gemini_client,
                                gemini_model_id=model_id_input,
                                file_path=temp_file_path,
                                pydantic_schema=Invoice
                            )
                        elif model_choice == "OpenAI GPT":
                            extracted_data = extract_from_openai(
                                client_instance=st.session_state.openai_client,
                                openai_model_id=model_id_input,
                                file_path=temp_file_path,
                                pydantic_schema=Invoice
                            )
                    
                    if extracted_data:
                        parsed_date = parse_date_safe(extracted_data.date)
                        cgst = extracted_data.cgst if extracted_data.cgst is not None else 0.0
                        sgst = extracted_data.sgst if extracted_data.sgst is not None else 0.0
                        igst = extracted_data.igst if extracted_data.igst is not None else 0.0
                        taxable_amount = extracted_data.taxable_amount if extracted_data.taxable_amount is not None else 0.0
                        gross_total_incl_tax = extracted_data.total_amount_payable if extracted_data.total_amount_payable is not None else 0.0
                        
                        tds_amount_extracted = extracted_data.tds_amount if extracted_data.tds_amount is not None else 0.0
                        tds_rate_extracted = extracted_data.tds_rate if extracted_data.tds_rate is not None else "N/A"
                        tds_display = extracted_data.tds or "N/A" # Applicability string (e.g., "Yes - Section 194J")
                        pos = extracted_data.place_of_supply or "N/A"
                        expense_ledger_display = extracted_data.expense_ledger or "N/A"

                        # --- Logic for TDS Section Extraction and Rate Inference ---
                        tds_section_display = "N/A"
                        if isinstance(tds_display, str) and "section" in tds_display.lower():
                            # Attempt to extract section number
                            parts = tds_display.split("Section ")
                            if len(parts) > 1:
                                section_part = parts[1].strip()
                                # Clean up common suffixes like brackets, periods, etc.
                                section_part = section_part.split(' ')[0].split(']')[0].split('.')[0].strip()
                                if section_part:
                                    tds_section_display = section_part
                        
                        # New: Infer TDS rate if section is known and rate is missing
                        if tds_rate_extracted == "N/A" and tds_section_display != "N/A":
                            # Simple, limited lookup for common sections
                            if tds_section_display == "194J":
                                tds_rate_extracted = 10.0 # Standard rate for professional fees
                            # Add more sections here if needed for specific inferences
                            
                        # --- End Logic for TDS Section Extraction and Rate Inference ---


                        # --- Adjust TDS if Buyer is Foreign or TDS is explicitly "No" ---
                        if pos.lower() == "foreign":
                            tds_display = "No" # Applicability
                            tds_amount_extracted = 0.0
                            tds_rate_extracted = "N/A"
                            tds_section_display = "N/A"
                            st.info(f"TDS adjusted to 'No' for **{uploaded_file_obj.name}** as Place of Supply is 'Foreign'.")
                        elif tds_display.lower() == "no": # If LLM explicitly said "No"
                            tds_amount_extracted = 0.0
                            tds_rate_extracted = "N/A"
                            tds_section_display = "N/A"
                        # --- End NEW LOGIC ---

                        total_payable_after_tds = gross_total_incl_tax - tds_amount_extracted

                        seller_name_display = extracted_data.seller_name or "N/A"
                        seller_gstin_display = extracted_data.gstin or "N/A"
                        buyer_name_display = extracted_data.buyer_name or "N/A"
                        buyer_gstin_display = extracted_data.buyer_gstin or "N/A"
                        # expense_ledger_display already assigned above
                        hsn_sac_display = extracted_data.hsn_sac or "N/A"
                        rcm_display = extracted_data.rcm_applicability or "N/A"

                        narration = (
                            f"Invoice **{extracted_data.invoice_number or 'N/A'}** dated **{parsed_date}** "
                            f"from **{seller_name_display}** (GSTIN: {seller_gstin_display}) "
                            f"to **{buyer_name_display}** (Buyer GSTIN: {buyer_gstin_display}), "
                            f"Taxable: **{format_currency(taxable_amount)}**, Gross Total (Incl Tax): **{format_currency(gross_total_incl_tax)}**, "
                            f"TDS Deducted: **{format_currency(tds_amount_extracted)}** (Rate: {tds_rate_extracted}{'%' if isinstance(tds_rate_extracted, (int, float)) else ''} Section: {tds_section_display}). Net Payable: **{format_currency(total_payable_after_tds)}**. "
                            f"Taxes: CGST {format_currency(cgst)}, SGST {format_currency(sgst)}, IGST {format_currency(igst)}. "
                            f"Place of Supply: {pos}. Expense Ledger: {expense_ledger_display}. "
                            f"RCM: {rcm_display}. HSN/SAC: {hsn_sac_display}."
                        )

                        st.session_state.summary_rows.append({
                            "File Name": uploaded_file_obj.name,
                            "Invoice Number": extracted_data.invoice_number,
                            "Date": parsed_date,
                            "Seller Name": seller_name_display,
                            "Seller GSTIN": seller_gstin_display,
                            "Buyer Name": buyer_name_display,
                            "Buyer GSTIN": buyer_gstin_display,
                            "HSN/SAC": hsn_sac_display,
                            "Place of Supply": pos,
                            "Taxable Amount": taxable_amount,
                            "Expense Ledger": expense_ledger_display,
                            "CGST": cgst,
                            "SGST": sgst,
                            "IGST": igst,
                            "TDS Rate": tds_rate_extracted,
                            "TDS Amount": tds_amount_extracted,
                            "TDS Section": tds_section_display, # Renamed key
                            "Narration": narration,
                            "Gross Total (Incl Tax)": gross_total_incl_tax, # Keep for backend calculation/Excel
                            "R. Applicability": rcm_display, # Keep for Excel
                        })

                        with st.expander(f"📋 Details for {uploaded_file_obj.name} (using {model_choice})"):
                            st.subheader("Extracted Summary (Narration):")
                            st.markdown(narration)

                            if extracted_data.line_items:
                                st.subheader("Line Items:")
                                line_item_data = [{
                                    "Description": item.description,
                                    "Quantity": item.quantity,
                                    "Gross Worth": format_currency(item.gross_worth),
                                } for item in extracted_data.line_items]
                                st.dataframe(pd.DataFrame(line_item_data), use_container_width=True)
                            else:
                                st.info("No line items extracted.")

                    else:
                        st.warning(f"Failed to extract data or no valid data returned for **{uploaded_file_obj.name}** using {model_choice}. Check error messages above.")

                except Exception as e_outer:
                    st.error(f"An unexpected error occurred while processing **{uploaded_file_obj.name}**: {e_outer}")
                    st.exception(e_outer)
                finally:
                    if temp_file_path and os.path.exists(temp_file_path):
                        os.unlink(temp_file_path) # Clean up the temporary file
                progress_bar.progress((i + 1) / total_files)

            st.markdown(f"---")
            if st.session_state.summary_rows:
                # Load and display Lottie animation for success
                lottie_success_url = "https://raw.githubusercontent.com/Mrpin2/InvoiceAi/main/Animation%20-%201749845303699.json"
                lottie_json = load_lottie_url(lottie_success_url)
                if lottie_json and st_lottie: # Ensure st_lottie is imported
                    st_lottie(lottie_json, height=200, key="success_animation")
                
                st.success("All selected invoices processed!")


if st.session_state.summary_rows:
    st.subheader("📊 Consolidated Extracted Invoice Summary")
    
    # Create the DataFrame from summary_rows
    df = pd.DataFrame(st.session_state.summary_rows)

    df_display = df.copy()

    # Apply currency formatting for display
    currency_cols = ["CGST", "SGST", "IGST", "Taxable Amount", "TDS Amount", "Gross Total (Incl Tax)"]
    for col in currency_cols:
        if col in df_display.columns:
            df_display[col] = df_display[col].apply(format_currency)

    # Format TDS Rate for display if it's a number
    if "TDS Rate" in df_display.columns:
        df_display["TDS Rate"] = df_display["TDS Rate"].apply(lambda x: f"{x:.2f}%" if isinstance(x, (int, float)) else x)


    # Define the exact display order as requested
    display_columns_order = [
        "File Name",
        "Invoice Number",
        "Date",
        "Seller Name",
        "Seller GSTIN",
        "Buyer Name",
        "Buyer GSTIN",
        "HSN/SAC",
        "Place of Supply",
        "Taxable Amount",
        "Expense Ledger",
        "CGST",
        "SGST",
        "IGST",
        "TDS Rate",
        "TDS Amount",
        "TDS Section",
        "Narration",
    ]

    # Filter df_display to only include desired columns and reorder them
    # Ensure all columns in display_columns_order exist in df_display before filtering
    actual_display_columns = [col for col in display_columns_order if col in df_display.columns]
    df_display = df_display[actual_display_columns]

    st.dataframe(df_display, use_container_width=True)

    output_excel = io.BytesIO()
    with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
        excel_columns_order = [
            "File Name",
            "Invoice Number",
            "Date",
            "Seller Name",
            "Seller GSTIN",
            "Buyer Name",
            "Buyer GSTIN",
            "HSN/SAC",
            "Place of Supply",
            "Taxable Amount",
            "Expense Ledger",
            "CGST",
            "SGST",
            "IGST",
            "TDS Rate",
            "TDS Amount",
            "TDS Section",
            "Narration",
            "Gross Total (Incl Tax)",
            "R. Applicability",
        ]
        
        # Filter and reorder for Excel export, ensuring columns exist
        actual_excel_columns = [col for col in excel_columns_order if col in df.columns]
        df_for_excel = df[actual_excel_columns]
        df_for_excel.to_excel(writer, index=False, sheet_name='InvoiceSummary')
    excel_data = output_excel.getvalue()

    st.download_button(
        label="📥 Download Consolidated Summary as Excel",
        data=excel_data,
        file_name="invoice_summary.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
elif not uploaded_files:
     st.info("Upload PDF files and click 'Process Invoices' to see results.")
