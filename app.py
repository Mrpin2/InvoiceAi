import streamlit as st
from pydantic import BaseModel, Field, create_model # create_model for dynamic Pydantic
from typing import List, Optional, Union, Dict, Any
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
    fitz = None
    Image = None

# --- New Import for Lottie Animation ---
try:
    from streamlit_lottie import st_lottie
except ImportError:
    st.warning("The 'streamlit-lottie' library is not installed. Lottie animations will not work.")
    st_lottie = None

# --- Import Libraries for Both Models ---
try:
    from google import genai
except ImportError:
    st.warning("The 'google-generativeai' library is not installed. Gemini functionality will be unavailable.")
    genai = None

try:
    from openai import OpenAI
except ImportError:
    st.warning("The 'openai' library is not installed. OpenAI functionality will be unavailable.")
    OpenAI = None


# --- Pydantic Models (Comprehensive - we'll dynamically select fields from this) ---
class LineItem(BaseModel):
    description: str
    quantity: float
    gross_worth: float

class Invoice(BaseModel):
    invoice_number: Optional[str] = Field(None, description="The unique identifier of the invoice. Extract as is.")
    date: Optional[str] = Field(None, description="The invoice date in DD/MM/YYYY format. If year is 2-digit, assume current century (e.g., 24 -> 2024).")
    gstin: Optional[str] = Field(None, description="The GSTIN of the seller (the entity issuing the invoice). Must be a 15-character alphanumeric string. Prioritize the GSTIN explicitly labeled as 'GSTIN' or associated with the seller's main details.")
    seller_name: Optional[str] = Field(None, description="The name of the seller (the entity issuing the invoice).")
    buyer_name: Optional[str] = Field(None, description="The name of the buyer (the entity receiving the invoice).")
    buyer_gstin: Optional[str] = Field(None, description="The GSTIN of the buyer (the entity receiving the invoice). Must be a 15-character alphanumeric string. Prioritize the GSTIN explicitly labeled as 'Buyer GSTIN' or associated with the buyer's details.")
    taxable_amount: Optional[float] = Field(None, description="This is the subtotal, the amount BEFORE any taxes (CGST, SGST, IGST) are applied. Must be a number.")
    cgst: Optional[float] = Field(None, description="The Central Goods and Services Tax amount. Must be a number.")
    sgst: Optional[float] = Field(None, description="The State Goods and Services Tax amount. Must be a number.")
    igst: Optional[float] = Field(None, description="The Integrated Goods and Services Tax amount. Must be a number.")
    total_amount_payable: Optional[float] = Field(None, description="This is the final total amount on the invoice *including all taxes and other charges*, but *before* any TDS deduction shown on the invoice. This represents the 'Gross Total' or 'Amount before TDS' on the invoice. Must be a number.")
    tds_amount: Optional[float] = Field(None, description="The numerical value of TDS deducted, if explicitly stated.")
    tds_rate: Optional[float] = Field(None, description="The numerical percentage rate of TDS deducted (e.g., '10' for 10%), if explicitly stated. If a TDS Section is clearly identified but the rate is missing on the invoice, infer the standard rate for that section (e.g., 10% for 194J).")
    line_items: List[LineItem] = Field(default_factory=list, description="A list of line items, each with 'description', 'quantity', and 'gross_worth'. Provide an empty list if no line items are found.")
    place_of_supply: Optional[str] = Field(None, description="The Place of Supply. Extract the exact State/City name or 'Foreign' if applicable. Prioritize 'Place of Supply' field, then 'Ship To', 'Bill To', 'Customer/Buyer Address'.")
    expense_ledger: Optional[str] = Field(None, description="Classify the nature of expense and suggest a suitable ledger type (e.g., 'Office Supplies', 'Professional Fees', 'Software Subscription'). Consider common TDS sections for inference.")
    tds: Optional[str] = Field(None, description="TDS applicability. 'Yes - Section [X]' if deducted, 'Yes - Section Unknown' if rate/amount present but no section, 'No' if not applicable or foreign transaction, 'Uncertain' if unclear.")
    hsn_sac: Optional[str] = Field(None, description="The HSN (Harmonized System of Nomenclature) or SAC (Service Accounting Code). ONLY extract if explicitly mentioned. If not found, MUST be `null`.")
    rcm_applicability: Optional[str] = Field(None, description="Reverse Charge Mechanism (RCM) applicability. State 'Yes', 'No', or 'Uncertain'.")

# A mapping from user-friendly names to Pydantic field names and their descriptions
# This dictionary will be used for both display and prompt generation
FIELD_DESCRIPTIONS = {
    "Invoice Number": ("invoice_number", "The unique identifier of the invoice."),
    "Invoice Date": ("date", "The invoice date in DD/MM/YYYY format. If year is 2-digit, assume current century (e.g., 24 -> 2024)."),
    "Seller GSTIN": ("gstin", "The GSTIN of the seller (the entity issuing the invoice). Must be a 15-character alphanumeric string."),
    "Seller Name": ("seller_name", "The name of the seller (the entity issuing the invoice)."),
    "Buyer Name": ("buyer_name", "The name of the buyer (the entity receiving the invoice)."),
    "Buyer GSTIN": ("buyer_gstin", "The GSTIN of the buyer (the entity receiving the invoice). Must be a 15-character alphanumeric string."),
    "Taxable Amount": ("taxable_amount", "The subtotal, the amount BEFORE any taxes (CGST, SGST, IGST) are applied. Must be a number."),
    "CGST": ("cgst", "The Central Goods and Services Tax amount. Must be a number."),
    "SGST": ("sgst", "The State Goods and Services Tax amount. Must be a number."),
    "IGST": ("igst", "The Integrated Goods and Services Tax amount. Must be a number."),
    "Total Amount Payable (Incl. Tax)": ("total_amount_payable", "The final total amount on the invoice *including all taxes and other charges*, but *before* any TDS deduction shown on the invoice. This represents the 'Gross Total' or 'Amount before TDS' on the invoice. Must be a number."),
    "TDS Amount": ("tds_amount", "The numerical value of TDS deducted, if explicitly stated."),
    "TDS Rate": ("tds_rate", "The numerical percentage rate of TDS deducted (e.g., '10' for 10%), if explicitly stated. Infer standard rate if section known."),
    "Line Items": ("line_items", "A list of line items, each with 'description', 'quantity', and 'gross_worth'. Provide an empty list if no line items are found."),
    "Place of Supply": ("place_of_supply", "The Place of Supply (e.g., 'Delhi', 'Maharashtra', or 'Foreign')."),
    "Expense Ledger": ("expense_ledger", "Classify the nature of expense and suggest a suitable ledger type (e.g., 'Office Supplies', 'Professional Fees')."),
    "TDS Applicability": ("tds", "TDS applicability (e.g., 'Yes - Section 194J', 'No', 'Uncertain')."),
    "HSN/SAC Code": ("hsn_sac", "The HSN (goods) or SAC (services) code. ONLY extract if explicitly mentioned. If not found, MUST be `null`."),
    "RCM Applicability": ("rcm_applicability", "Reverse Charge Mechanism (RCM) applicability. State 'Yes', 'No', or 'Uncertain'.")
}

# --- Utility Functions for UI/Data Formatting ---
def parse_date_safe(date_str: str) -> str:
    """Attempts to parse a date string into DD/MM/YYYY format."""
    if not date_str:
        return ""
    formats = ["%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d", "%d %b %Y", "%d %B %Y", "%Y/%m/%d"]
    for fmt in formats:
        try:
            dt_obj = datetime.strptime(date_str, fmt)
            # Adjust 2-digit years. Current year 2025. If year is 24, assume 2024. If 98, assume 1998.
            current_century_prefix = (datetime.now().year // 100) * 100
            if dt_obj.year < 100: # 2-digit year
                if dt_obj.year > (datetime.now().year % 100) + 10: # e.g., 90 in 2025 -> 1990
                    dt_obj = dt_obj.replace(year=dt_obj.year + 1900)
                else: # e.g., 24 in 2025 -> 2024
                    dt_obj = dt_obj.replace(year=dt_obj.year + 2000)
            return dt_obj.strftime("%d/%m/%Y")
        except ValueError:
            continue
    return date_str

def format_currency(amount: Optional[float]) -> str:
    """Formats a float as an Indian Rupee currency string."""
    if amount is None:
        return "₹ N/A"
    return f"₹ {amount:,.2f}"

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

# --- Dynamic Prompt Generation ---
def generate_dynamic_prompt(selected_fields: List[str], extraction_type: str) -> str:
    if extraction_type == "Free-form Summary":
        return (
            "You are an expert at summarizing Indian invoices. "
            "Your task is to provide a concise, natural language summary of the key details "
            "of the invoice. Include the invoice number, date, seller and buyer names, "
            "total amount, and the nature of the expense. If TDS is mentioned, also include its details. "
            "If the document is clearly not an invoice, state 'NOT AN INVOICE'."
        )

    # Structured Data Extraction
    prompt_parts = [
        "You are an expert at extracting structured data from Indian invoices. ",
        "Your task is to extract information into a JSON object with the following keys. ",
        "If a key's value is not explicitly present or derivable from the invoice, use `null` for that value. "
    ]

    required_keys = []
    field_guidelines = []
    for field_display_name in selected_fields:
        field_name, description = FIELD_DESCRIPTIONS.get(field_display_name, (None, None))
        if field_name:
            required_keys.append(field_name)
            field_guidelines.append(f"- '{field_name}': {description}")
            if field_name == "line_items":
                field_guidelines.append("  For `line_items`, each item must have 'description', 'quantity', 'gross_worth'.")
                prompt_parts.append("You MUST include an empty list `[]` for `line_items` if no line items are found, do not use `null` for `line_items`. ")


    if not required_keys:
        # Fallback if no fields are selected for structured extraction, though UI prevents this
        return "Extract common invoice details like invoice_number, date, seller_name, total_amount_payable."

    prompt_parts.append(f"Keys to extract: {', '.join(required_keys)}. ")
    prompt_parts.append("\nGUIDELINES FOR EXTRACTION:\n")
    prompt_parts.extend(field_guidelines)
    prompt_parts.append("\nReturn 'NOT AN INVOICE' if the document is clearly not an invoice.\n")
    prompt_parts.append("Ensure the JSON output is clean and directly parsable.")
    return "\n".join(prompt_parts)


# --- Dynamic Pydantic Schema Generation ---
def create_dynamic_invoice_schema(selected_fields: List[str]) -> BaseModel:
    fields = {}
    for field_display_name in selected_fields:
        field_name, _ = FIELD_DESCRIPTIONS.get(field_display_name)
        if field_name == "line_items":
            fields[field_name] = (List[LineItem], Field(default_factory=list))
        elif field_name in Invoice.model_fields: # Pydantic v2 way to check and get field info
            original_field = Invoice.model_fields.get(field_name)
            if original_field:
                fields[field_name] = (original_field.annotation, Field(None, description=original_field.description))
            else:
                fields[field_name] = (Optional[str], None) # Fallback
        else: # Fallback if field not found in original Invoice model
            fields[field_name] = (Optional[str], None)

    if not fields:
        # If no fields are selected (e.g., if user somehow bypasses validation), create a minimal schema
        fields["invoice_number"] = (Optional[str], None)
        fields["date"] = (Optional[str], None)
        fields["total_amount_payable"] = (Optional[float], None)

    return create_model('DynamicInvoice', **fields)


# --- Extraction Functions (Model-Specific) ---
def extract_from_gemini(
    client_instance: 'genai.Client',
    gemini_model_id: str,
    file_path: str,
    prompt_content_str: str, # Can be dynamic or custom
    pydantic_schema: Optional[BaseModel] = None # Optional for summary/custom non-json
) -> Optional[Union[Dict[str, Any], str]]: # Returns dict for structured, str for summary
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
        
        contents = [prompt_content_str, gemini_file_resource]
        
        generation_config = {}
        if pydantic_schema: # Only request JSON if a schema is provided for structured output
            generation_config['response_mime_type'] = 'application/json'

        response = client_instance.models.generate_content(
            model=gemini_model_id,
            contents=contents,
            generation_config=generation_config
        )

        if response and hasattr(response, 'text') and response.text:
            if st.session_state.get('DEBUG_MODE', False):
                st.markdown("##### Raw Response from Gemini (Debug Mode):")
                st.code(response.text, language="json" if pydantic_schema else "text")

            if pydantic_schema: # Attempt JSON parsing and Pydantic validation
                try:
                    extracted_dict = json.loads(response.text)
                    pydantic_schema.model_validate(extracted_dict) # This will raise if validation fails
                    return extracted_dict
                except json.JSONDecodeError as e:
                    st.error(f"Gemini: Failed to decode JSON from response for '{display_name}'. Expected JSON due to 'Structured Data Extraction' or custom prompt. Error: {e}")
                    if st.session_state.get('DEBUG_MODE', False):
                        st.error(f"Response content: {response.text[:500]}..." if response.text else "No content.")
                        st.code(response.text, language="json")
                    return None
                except Exception as e: # Catch Pydantic validation errors
                    st.error(f"Gemini: Failed to validate extracted data against schema for '{display_name}': {e}")
                    if st.session_state.get('DEBUG_MODE', False):
                        st.error(f"Response content: {response.text[:500]}..." if response.text else "No content.")
                        st.code(response.text, language="json")
                    return None
            else: # Return raw text for summary or custom non-JSON
                return response.text
        else:
            st.warning(f"Gemini: No content received or unexpected response structure for '{display_name}'.")
            if st.session_state.get('DEBUG_MODE', False):
                st.json(response.dict())
            return None

    except Exception as e:
        st.error(f"Error processing '{display_name}' with Gemini: {e}")
        if st.session_state.get('DEBUG_MODE', False):
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
    prompt_content_str: str, # Can be dynamic or custom
    pydantic_schema: Optional[BaseModel] = None # Optional for summary/custom non-json
) -> Optional[Union[Dict[str, Any], str]]: # Returns dict for structured, str for summary
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
        max_pages_to_process = 5

        for page_num in range(min(doc.page_count, max_pages_to_process)):
            page = doc.load_page(page_num)
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            
            img_bytes_io = io.BytesIO()
            pil_image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            pil_image.save(img_bytes_io, format="PNG")
            img_bytes = img_bytes_io.getvalue()

            base64_image = base64.b64encode(img_bytes).decode('utf-8')
            image_messages.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64_image}",
                    "detail": "high"
                }
            })
        doc.close()

        if not image_messages:
            st.error(f"No pages could be converted to images for '{display_name}'. This might happen with corrupted PDFs or empty files.")
            return None

        messages_content = [{"type": "text", "text": prompt_content_str}] + image_messages
        
        response_format = {"type": "json_object"} if pydantic_schema else None # Only request JSON if schema is present

        response = client_instance.chat.completions.create(
            model=openai_model_id,
            messages=[
                {"role": "user", "content": messages_content} # Use combined content
            ],
            response_format=response_format,
            max_tokens=4000
        )

        if response and response.choices and response.choices[0].message and response.choices[0].message.content:
            response_content = response.choices[0].message.content
            if st.session_state.get('DEBUG_MODE', False):
                st.markdown("##### Raw Response from OpenAI (Debug Mode):")
                st.code(response_content, language="json" if pydantic_schema else "text")

            if pydantic_schema: # Attempt JSON parsing and Pydantic validation
                try:
                    extracted_dict = json.loads(response_content)
                    pydantic_schema.model_validate(extracted_dict)
                    return extracted_dict
                except json.JSONDecodeError as e:
                    st.error(f"OpenAI: Failed to decode JSON from response for '{display_name}'. Expected JSON due to 'Structured Data Extraction' or custom prompt. Error: {e}")
                    if st.session_state.get('DEBUG_MODE', False):
                        st.error(f"Response content: {response_content[:500]}..." if response_content else "No content.")
                        st.code(response_content, language="json")
                    return None
                except Exception as e:
                    st.error(f"OpenAI: Failed to validate extracted data against schema for '{display_name}': {e}")
                    if st.session_state.get('DEBUG_MODE', False):
                        st.error(f"Response content: {response_content[:500]}..." if response_content else "No content.")
                        st.code(response_content, language="json")
                    return None
            else: # Return raw text for summary or custom non-JSON
                return response_content
        else:
            st.warning(f"OpenAI: No content received or unexpected response structure for '{display_name}'.")
            if st.session_state.get('DEBUG_MODE', False):
                st.json(response.dict())
            return None

    except Exception as e:
        st.error(f"Error processing '{display_name}' with OpenAI: {e}")
        if st.session_state.get('DEBUG_MODE', False):
            st.exception(e)
        return None

# --- Streamlit App UI ---
st.set_page_config(layout="wide", page_title="📄 AI Invoice Extractor (Dynamic)")

# Custom CSS for a bit more flair and font consistency
st.markdown("""
<style>
    body {
        font-family: 'Roboto', 'Helvetica Neue', Arial, sans-serif;
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
    .stMarkdown p { 
        font-size: 1.05em; 
        line-height: 1.6; 
        font-family: 'Roboto', 'Helvetica Neue', Arial, sans-serif; 
        color: #333333;
    }
    .stAlert { 
        border-radius: 8px; 
        font-family: 'Roboto', 'Helvetica Neue', Arial, sans-serif; 
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
    .main .block-container {
        padding-top: 3rem;
        padding-right: 3rem;
        padding-left: 3rem;
        padding-bottom: 3rem;
    }
    div[data-testid="stMarkdownContainer"]:has(p:first-child:contains("Instructions:")) {
        background-color: white;
        padding: 25px;
        border-radius: 12px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.08);
        margin-bottom: 25px;
        border: 1px solid #e0e0e0;
    }
    div[data-testid="stDataFrame"] {
        background-color: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.08);
        margin-bottom: 20px;
        border: 1px solid #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)


st.title("📄 AI Invoice Extractor (Dynamic & Multi-Model)")
st.divider()

st.sidebar.header("Configuration")

# --- Admin Panel for using secrets ---
try:
    ADMIN_PASSWORD = st.secrets.get("ADMIN_PASSWORD")
    if ADMIN_PASSWORD is None:
        st.sidebar.warning("ADMIN_PASSWORD secret not found. Admin mode is disabled.")
except AttributeError:
    st.sidebar.error("Streamlit secrets not accessible. Admin mode disabled.")
    ADMIN_PASSWORD = None

admin_password_input = st.sidebar.text_input("Admin Password (Optional):", type="password", key="admin_pass")

use_secrets_keys_for_llms = False
if admin_password_input:
    if ADMIN_PASSWORD is None:
        st.sidebar.error("Admin password is not configured in secrets. Cannot activate admin mode.")
        st.session_state.DEBUG_MODE = False
    elif admin_password_input == ADMIN_PASSWORD:
        st.sidebar.success("Admin mode activated. Now using API keys from secrets for LLMs.")
        use_secrets_keys_for_llms = True
        st.session_state.DEBUG_MODE = True
    else:
        st.sidebar.error("Incorrect admin password.")
        st.session_state.DEBUG_MODE = False
else:
    st.session_state.DEBUG_MODE = False
    use_secrets_keys_for_llms = False

# Model Selection
model_choice = st.sidebar.radio(
    "Choose AI Model:",
    ("Google Gemini", "OpenAI GPT"),
    key="model_choice"
)

# API Key and Model ID inputs
selected_api_key = None
model_id_input = None

if use_secrets_keys_for_llms:
    if model_choice == "Google Gemini":
        selected_api_key = st.secrets.get("GEMINI_API_KEY")
        model_id_input = st.secrets.get("GEMINI_MODEL_ID", "gemini-1.5-flash-latest")
        if not selected_api_key:
            st.sidebar.warning("GEMINI_API_KEY not found in Streamlit Secrets. Gemini functionality might be limited.")
        st.sidebar.text_input("Gemini Model ID (from secrets):", model_id_input, key="gemini_model_id_secrets", disabled=True)
        st.sidebar.caption(f"Using model ID from secrets: `{model_id_input}`")
        st.sidebar.info(f"Using {model_choice} API Key from `Streamlit Secrets`.")

    elif model_choice == "OpenAI GPT":
        selected_api_key = st.secrets.get("OPENAI_API_KEY")
        model_id_input = st.secrets.get("OPENAI_MODEL_ID", "gpt-4o")
        if not selected_api_key:
            st.sidebar.warning("OPENAI_API_KEY not found in Streamlit Secrets. OpenAI functionality might be limited.")
        st.sidebar.text_input("OpenAI Model ID (from secrets):", model_id_input, key="openai_model_id_secrets", disabled=True)
        st.sidebar.caption(f"Using model ID from secrets: `{model_id_input}`")
        st.sidebar.info(f"Using {model_choice} API Key from `Streamlit Secrets`.")

else:
    st.sidebar.markdown("---")
    st.sidebar.subheader("Enter Your Own API Key (Required) 👇")
    st.sidebar.markdown(f"To use the {model_choice} model, please provide your personal API key. Your key is used for processing and **not stored**.")

    if model_choice == "Google Gemini":
        selected_api_key = st.sidebar.text_input("Enter your Gemini API Key:", type="password", key="gemini_key_manual")
        DEFAULT_GEMINI_MODEL_ID = "gemini-1.5-flash-latest"
        model_id_input = st.sidebar.text_input("Gemini Model ID:", DEFAULT_GEMINI_MODEL_ID, key="gemini_model_id_manual")
        st.sidebar.caption(f"Default is `{DEFAULT_GEMINI_MODEL_ID}`. Ensure it supports JSON schema.")
    elif model_choice == "OpenAI GPT":
        selected_api_key = st.sidebar.text_input("Enter your OpenAI API Key:", type="password", key="openai_key_manual")
        DEFAULT_OPENAI_MODEL_ID = "gpt-4o"
        model_id_input = st.sidebar.text_input("OpenAI Model ID:", DEFAULT_OPENAI_MODEL_ID, key="openai_model_id_manual")
        st.sidebar.caption(f"Default is `{DEFAULT_OPENAI_MODEL_ID}`. Ensure it's a vision model and supports JSON output.")

st.markdown(
    f"""
    **Instructions:**
    - Select your preferred AI model ({model_choice}) in the sidebar.
    - **API Key Required:** To run the extraction, you'll need to provide your own API key in the sidebar. Your key is used for processing and **not stored**.
    - If you are an administrator, you can enter the admin password in the sidebar to enable debug features and use pre-configured API keys (if set in Streamlit secrets).
    - Choose an **Extraction Type**: `Structured Data Extraction` allows you to select specific fields, `Free-form Summary` provides a narrative.
    - **Custom Prompt (Optional):** If you provide a custom prompt, it will override the selected extraction type and fields.
    - Upload one or more PDF invoice files.
    - Click 'Process Invoices' to extract data.
      The extracted data will be displayed in a table and available for download as Excel.
    """
)

st.divider()

# --- New: Extraction Type Selection ---
extraction_type = st.radio(
    "Select Extraction Type:",
    ("Structured Data Extraction", "Free-form Summary"),
    key="extraction_type_selection"
)

selected_fields_for_extraction = []
if extraction_type == "Structured Data Extraction":
    st.markdown("### Choose Fields for Structured Data Extraction")
    st.info("Select the specific invoice details you wish to extract. Only these fields will be requested from the AI model.")
    # Get all user-friendly field names from our FIELD_DESCRIPTIONS
    all_available_fields = list(FIELD_DESCRIPTIONS.keys())
    
    # Pre-select common fields as a default
    default_fields = [
        "Invoice Number", "Invoice Date", "Seller Name", "Total Amount Payable (Incl. Tax)",
        "Expense Ledger", "TDS Applicability"
    ]
    # Ensure default fields exist in all_available_fields before setting as default
    default_fields = [f for f in default_fields if f in all_available_fields]

    selected_fields_for_extraction = st.multiselect(
        "Select fields to extract:",
        options=all_available_fields,
        default=default_fields,
        help="Type to search and select specific data points (columns) you want the AI to extract from the invoices. 'Line Items' will extract a nested table.",
        key="selected_fields_multiselect"
    )
    if not selected_fields_for_extraction:
        st.warning("No fields selected for structured extraction. Please select at least one field, or provide a custom prompt.")
        
# --- Custom Prompt Input ---
st.markdown("---")
st.subheader("Custom Prompt (Optional)")
custom_prompt_input = st.text_area(
    "Enter your custom prompt here (e.g., 'Extract the buyer's name and address as a JSON.').",
    height=150,
    help="If a custom prompt is provided, it will override the selected Extraction Type and fields. For structured output, ensure your prompt asks for JSON.",
    key="custom_prompt_textarea"
)
if custom_prompt_input:
    st.info("Custom prompt provided. This will supersede all other extraction settings.")


st.markdown("&nbsp;") # Small vertical space

uploaded_files = st.file_uploader(
    "Choose PDF invoice files",
    type="pdf",
    accept_multiple_files=True
)

st.markdown("&nbsp;") # Small vertical space

# Initialize session states if not present
if 'extracted_results' not in st.session_state:
    st.session_state.extracted_results = [] # Store raw extracted dicts
if 'gemini_client' not in st.session_state:
    st.session_state.gemini_client = None
if 'openai_client' not in st.session_state:
    st.session_state.openai_client = None
if 'DEBUG_MODE' not in st.session_state:
    st.session_state.DEBUG_MODE = False

if st.button("🚀 Process Invoices", type="primary"):
    if not selected_api_key:
        st.error(f"Please enter your {model_choice} API Key in the sidebar or ensure it's configured in secrets if you're an admin.")
    elif not uploaded_files:
        st.error("Please upload at least one PDF file.")
    elif not model_id_input:
        st.error(f"Please specify a {model_choice} Model ID in the sidebar.")
    elif extraction_type == "Structured Data Extraction" and not selected_fields_for_extraction and not custom_prompt_input:
        st.error("For 'Structured Data Extraction', please select at least one field to extract, or provide a custom prompt.")
    else:
        client_initialized = False
        if model_choice == "Google Gemini":
            if genai:
                try:
                    st.session_state.gemini_client = genai.Client(api_key=selected_api_key)
                    st.success("Gemini client initialized successfully!")
                    client_initialized = True
                except Exception as e:
                    st.error(f"Failed to initialize Gemini client: {e}. Please check your API key and model ID.")
                    if st.session_state.DEBUG_MODE:
                        st.exception(e)
                    st.session_state.gemini_client = None
            else:
                st.error("Google Generative AI library not found or failed to import. Cannot initialize Gemini client.")

        elif model_choice == "OpenAI GPT":
            if OpenAI:
                try:
                    st.session_state.openai_client = OpenAI(api_key=selected_api_key)
                    st.success("OpenAI client initialized successfully!")
                    client_initialized = True
                except Exception as e:
                    st.error(f"Failed to initialize OpenAI client: {e}. Please check your API key and model ID.")
                    if st.session_state.DEBUG_MODE:
                        st.exception(e)
                    st.session_state.openai_client = None
            else:
                st.error("OpenAI library not found or failed to import. Cannot initialize OpenAI client.")
        
        if not client_initialized:
            st.stop()

        if (model_choice == "Google Gemini" and st.session_state.gemini_client) or \
           (model_choice == "OpenAI GPT" and st.session_state.openai_client):
            
            st.session_state.extracted_results = []
            progress_bar = st.progress(0)
            total_files = len(uploaded_files)

            # Determine the prompt and schema to use
            prompt_to_use = custom_prompt_input if custom_prompt_input else generate_dynamic_prompt(selected_fields_for_extraction, extraction_type)
            
            dynamic_pydantic_schema = None
            # If custom prompt is used, and it's for structured, we still try to validate
            if custom_prompt_input and extraction_type == "Structured Data Extraction":
                st.warning("When using a custom prompt with 'Structured Data Extraction', ensure your prompt instructs the model to return a JSON object with keys matching your selected fields. Validation will be attempted.")
                try:
                    dynamic_pydantic_schema = create_dynamic_invoice_schema(selected_fields_for_extraction)
                except Exception as e:
                    st.error(f"Failed to create dynamic schema for custom prompt validation: {e}")
                    if st.session_state.DEBUG_MODE: st.exception(e)
                    st.stop()
            elif extraction_type == "Structured Data Extraction" and not custom_prompt_input:
                try:
                    dynamic_pydantic_schema = create_dynamic_invoice_schema(selected_fields_for_extraction)
                except Exception as e:
                    st.error(f"Failed to create dynamic schema: {e}")
                    if st.session_state.DEBUG_MODE:
                        st.exception(e)
                    st.stop()


            for i, uploaded_file_obj in enumerate(uploaded_files):
                st.markdown(f"---")
                st.info(f"Processing file: **{uploaded_file_obj.name}** ({i+1}/{total_files}) using **{model_choice}**...")
                temp_file_path = None
                extracted_output = None # This will hold either dict (structured) or str (summary)
                
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file_obj.name)[1]) as tmp:
                        tmp.write(uploaded_file_obj.getvalue())
                        temp_file_path = tmp.name

                    with st.spinner(f"Extracting data from {uploaded_file_obj.name} with {model_choice}..."):
                        if model_choice == "Google Gemini":
                            extracted_output = extract_from_gemini(
                                client_instance=st.session_state.gemini_client,
                                gemini_model_id=model_id_input,
                                file_path=temp_file_path,
                                prompt_content_str=prompt_to_use,
                                pydantic_schema=dynamic_pydantic_schema # Passed if structured or custom-structured
                            )
                        elif model_choice == "OpenAI GPT":
                            extracted_output = extract_from_openai(
                                client_instance=st.session_state.openai_client,
                                openai_model_id=model_id_input,
                                file_path=temp_file_path,
                                prompt_content_str=prompt_to_use,
                                pydantic_schema=dynamic_pydantic_schema # Passed if structured or custom-structured
                            )
                            
                    if extracted_output:
                        if isinstance(extracted_output, dict): # Structured Data Output
                            # Prepare a row for the DataFrame
                            row_data = {"File Name": uploaded_file_obj.name}
                            
                            # Determine effective fields to use for display/excel.
                            # If custom prompt, we just use whatever was returned.
                            # If not custom, we use selected_fields_for_extraction.
                            effective_fields = selected_fields_for_extraction if not custom_prompt_input else list(extracted_output.keys())
                            if custom_prompt_input and "line_items" in effective_fields:
                                st.warning("Custom prompt was used and 'line_items' was found. This will be shown in a separate expander.")

                            # Populate data based on selected fields and apply formatting
                            for field_display_name in effective_fields:
                                # Try to map display name to internal field name for existing fields
                                field_name_internal = FIELD_DESCRIPTIONS.get(field_display_name, (field_display_name, None))[0]
                                value = extracted_output.get(field_name_internal, extracted_output.get(field_display_name, None)) # Try both keys

                                if field_name_internal == "date":
                                    row_data[field_display_name] = parse_date_safe(value or "")
                                elif field_name_internal in ["taxable_amount", "cgst", "sgst", "igst", "total_amount_payable", "tds_amount"]:
                                    row_data[field_display_name] = value if value is not None else 0.0
                                elif field_name_internal == "tds_rate":
                                    # New: Infer TDS rate if section is known and rate is missing from LLM response
                                    tds_section_val = extracted_output.get("tds") # Raw TDS applicability string
                                    tds_section_display = "N/A"
                                    if isinstance(tds_section_val, str) and "section" in tds_section_val.lower():
                                        parts = tds_section_val.split("Section ")
                                        if len(parts) > 1:
                                            section_part = parts[1].strip()
                                            section_part = section_part.split(' ')[0].split(']')[0].split('.')[0].strip()
                                            if section_part:
                                                tds_section_display = section_part
                                    
                                    inferred_tds_rate = value if value is not None else "N/A"
                                    if inferred_tds_rate == "N/A" and tds_section_display != "N/A":
                                        if tds_section_display == "194J":
                                            inferred_tds_rate = 10.0 # Standard rate for professional fees
                                    # Store the (potentially inferred) rate
                                    row_data[field_display_name] = inferred_tds_rate

                                elif field_name_internal == "tds":
                                    # Adjust TDS if Place of Supply is 'Foreign' or LLM explicitly says 'No'
                                    pos_val = extracted_output.get("place_of_supply")
                                    if pos_val and pos_val.lower() == "foreign":
                                        row_data[field_display_name] = "No"
                                        # Also ensure TDS amount/rate are 0/N/A if foreign
                                        if "tds_amount" in extracted_output:
                                            row_data[FIELD_DESCRIPTIONS.get("TDS Amount")[0]] = 0.0
                                        if "tds_rate" in extracted_output:
                                            row_data[FIELD_DESCRIPTIONS.get("TDS Rate")[0]] = "N/A"
                                        if "TDS Section (Derived)" in extracted_output: # Needs a mapping for this
                                            row_data["TDS Section (Derived)"] = "N/A"
                                        st.info(f"TDS adjusted to 'No' for **{uploaded_file_obj.name}** as Place of Supply is 'Foreign'.")
                                    elif value and value.lower() == "no":
                                        row_data[field_display_name] = "No"
                                        if "tds_amount" in extracted_output:
                                            row_data[FIELD_DESCRIPTIONS.get("TDS Amount")[0]] = 0.0
                                        if "tds_rate" in extracted_output:
                                            row_data[FIELD_DESCRIPTIONS.get("TDS Rate")[0]] = "N/A"
                                        if "TDS Section (Derived)" in extracted_output:
                                            row_data["TDS Section (Derived)"] = "N/A"
                                    else:
                                        row_data[field_display_name] = value or "N/A"

                                elif field_name_internal == "line_items":
                                    # Line items will be processed separately for display in expander
                                    pass # Don't add directly to main row_data
                                else:
                                    row_data[field_display_name] = value or "N/A" # Default for other fields

                            # Special handling for "TDS Section (Derived)" if it's not a direct field but derived
                            if "TDS Applicability" in (selected_fields_for_extraction if not custom_prompt_input else extracted_output.keys()):
                                tds_section_display_val = "N/A"
                                if "tds" in extracted_output and isinstance(extracted_output["tds"], str) and "section" in extracted_output["tds"].lower():
                                    parts = extracted_output["tds"].split("Section ")
                                    if len(parts) > 1:
                                        section_part = parts[1].strip()
                                        section_part = section_part.split(' ')[0].split(']')[0].split('.')[0].strip()
                                        if section_part:
                                            tds_section_display_val = section_part
                                row_data["TDS Section (Derived)"] = tds_section_display_val # Use a distinct name for derived field


                            st.session_state.extracted_results.append({
                                "file_name": uploaded_file_obj.name,
                                "extraction_type": "structured", # Indicate type for later processing
                                "extracted_data": row_data,
                                "line_items": extracted_output.get("line_items", []) # Store line items separately
                            })

                            with st.expander(f"📋 Details for {uploaded_file_obj.name} (using {model_choice})"):
                                st.subheader("Extracted Raw JSON Data:")
                                st.json(extracted_output) # Show raw extracted JSON for verification

                                if "line_items" in extracted_output and extracted_output["line_items"]:
                                    st.subheader("Line Items:")
                                    line_item_data = [{
                                        "Description": item.get("description"),
                                        "Quantity": item.get("quantity"),
                                        "Gross Worth": format_currency(item.get("gross_worth")),
                                    } for item in extracted_output["line_items"]]
                                    st.dataframe(pd.DataFrame(line_item_data), use_container_width=True)
                                elif "Line Items" in (selected_fields_for_extraction if not custom_prompt_input else extracted_output.keys()):
                                    st.info("No line items extracted.")
                                else:
                                    st.info("Line items were not selected for extraction or not found in custom prompt output.")
                        
                        elif isinstance(extracted_output, str): # Free-form Summary Output
                            st.session_state.extracted_results.append({
                                "file_name": uploaded_file_obj.name,
                                "extraction_type": "summary", # Indicate type
                                "summary_text": extracted_output
                            })
                            st.success(f"Summary for {uploaded_file_obj.name} generated.")
                            st.markdown(f"**Summary for {uploaded_file_obj.name}:**")
                            st.write(extracted_output)
                        
                    else:
                        st.warning(f"Failed to extract data or no valid data returned for **{uploaded_file_obj.name}** using {model_choice}. Check error messages above.")

                except Exception as e_outer:
                    st.error(f"An unexpected error occurred while processing **{uploaded_file_obj.name}**: {e_outer}")
                    if st.session_state.DEBUG_MODE:
                        st.exception(e_outer)
                finally:
                    if temp_file_path and os.path.exists(temp_file_path):
                        os.unlink(temp_file_path)
                progress_bar.progress((i + 1) / total_files)

            st.markdown(f"---")
            if st.session_state.extracted_results:
                lottie_success_url = "https://raw.githubusercontent.com/Mrpin2/InvoiceAi/main/Animation%20-%201749845303699.json"
                lottie_json = load_lottie_url(lottie_success_url)
                if lottie_json and st_lottie:
                    st_lottie(lottie_json, height=200, key="success_animation")
                
                st.success("All selected invoices processed!")


if st.session_state.extracted_results:
    st.divider()
    st.subheader("📊 Consolidated Extracted Invoice Summary")
    
    # Separate structured vs. summary results
    structured_results = [r for r in st.session_state.extracted_results if r["extraction_type"] == "structured"]
    summary_results = [r for r in st.session_state.extracted_results if r["extraction_type"] == "summary"]

    if structured_results:
        st.markdown("#### Structured Data Results")
        summary_rows_for_display = []
        for result in structured_results:
            file_name = result["file_name"]
            extracted_data_dict = result["extracted_data"]
            
            # Create a display row, including formatting
            display_row = {"File Name": file_name}
            
            # Determine which columns to display for this specific file,
            # especially if a custom prompt was used
            columns_to_display = list(extracted_data_dict.keys())
            if "File Name" in columns_to_display:
                columns_to_display.remove("File Name") # Remove as it's added separately
            
            for col_key in columns_to_display:
                value = extracted_data_dict.get(col_key, "N/A")
                
                # Apply formatting based on known field types if possible
                if col_key in [FIELD_DESCRIPTIONS["Taxable Amount"][0], FIELD_DESCRIPTIONS["CGST"][0],
                               FIELD_DESCRIPTIONS["SGST"][0], FIELD_DESCRIPTIONS["IGST"][0],
                               FIELD_DESCRIPTIONS["TDS Amount"][0], FIELD_DESCRIPTIONS["Total Amount Payable (Incl. Tax)"][0]]:
                    display_row[col_key] = format_currency(value if value != "N/A" else None)
                elif col_key == FIELD_DESCRIPTIONS["TDS Rate"][0]:
                    display_row[col_key] = f"{value:.2f}%" if isinstance(value, (int, float)) else value
                else:
                    display_row[col_key] = value
            
            summary_rows_for_display.append(display_row)

        df_display = pd.DataFrame(summary_rows_for_display)

        # Reorder columns for display if original selection was structured and no custom prompt
        if not custom_prompt_input and extraction_type == "Structured Data Extraction":
            desired_display_order = ["File Name"] + selected_fields_for_extraction
            if "TDS Applicability" in selected_fields_for_extraction:
                desired_display_order.append("TDS Section (Derived)")
            desired_display_order = [col for col in desired_display_order if col != "Line Items"]
            
            # Filter df_display to only include desired columns and reorder them
            actual_display_columns_present = [col for col in desired_display_order if col in df_display.columns]
            df_display = df_display[actual_display_columns_present]

        st.dataframe(df_display, use_container_width=True)

        # --- Excel Download for Structured Data ---
        output_excel = io.BytesIO()
        with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
            # Prepare a DataFrame for Excel export using raw values (not formatted strings)
            excel_data_rows = []
            for result in structured_results:
                row = {"File Name": result["file_name"]}
                # Add all extracted_data fields to the row for Excel
                for k, v in result["extracted_data"].items():
                    row[k] = v
                excel_data_rows.append(row)
            
            df_for_excel = pd.DataFrame(excel_data_rows)

            # Drop 'Line Items' column from the main sheet if it exists
            if "Line Items" in df_for_excel.columns:
                df_for_excel = df_for_excel.drop(columns=["Line Items"])

            # Sort columns based on FIELD_DESCRIPTIONS order if not custom prompt, else alphabetical
            if not custom_prompt_input and extraction_type == "Structured Data Extraction":
                # Create a comprehensive list of all possible columns in order of FIELD_DESCRIPTIONS
                all_possible_ordered_columns = ["File Name"] + list(FIELD_DESCRIPTIONS.keys())
                if "TDS Applicability" in selected_fields_for_extraction: # Add derived field if applicable
                    all_possible_ordered_columns.append("TDS Section (Derived)")
                
                # Filter to only columns actually present in df_for_excel and maintain order
                final_excel_columns = [col for col in all_possible_ordered_columns if col in df_for_excel.columns]
                df_for_excel = df_for_excel[final_excel_columns]
            else:
                # If custom prompt or other type, just use current columns (will be somewhat arbitrary order)
                pass # No special reordering

            df_for_excel.to_excel(writer, index=False, sheet_name='InvoiceSummary')

            # Add Line Items to a separate sheet if selected/extracted
            if any(item.get("line_items") for item in structured_results) and \
               ("Line Items" in selected_fields_for_extraction or (custom_prompt_input and any("line_items" in r["extracted_data"] for r in structured_results))):
                all_line_items = []
                for result in structured_results:
                    file_name = result["file_name"]
                    for li in result.get("line_items", []): # Use .get with default empty list
                        all_line_items.append({
                            "File Name": file_name,
                            "Description": li.get("description"),
                            "Quantity": li.get("quantity"),
                            "Gross Worth": li.get("gross_worth")
                        })
                if all_line_items:
                    df_line_items = pd.DataFrame(all_line_items)
                    df_line_items.to_excel(writer
