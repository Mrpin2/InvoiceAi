import streamlit as st
from pydantic import BaseModel, Field, create_model
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
    # This comprehensive model serves as a reference for common fields and their types
    # It also holds detailed descriptions for prompt generation.
    invoice_number: Optional[str] = Field(None, description="The unique identifier of the invoice.")
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
# This will be used to look up detailed descriptions for prompt generation.
FIELD_METADATA = {
    "Invoice Number": {"field_name": "invoice_number", "description": "The unique identifier of the invoice."},
    "Invoice Date": {"field_name": "date", "description": "The invoice date in DD/MM/YYYY format. If year is 2-digit, assume current century (e.g., 24 -> 2024)."},
    "Seller GSTIN": {"field_name": "gstin", "description": "The GSTIN of the seller (the entity issuing the invoice). Must be a 15-character alphanumeric string. Prioritize the GSTIN explicitly labeled as 'GSTIN' or associated with the seller's main details."},
    "Seller Name": {"field_name": "seller_name", "description": "The name of the seller (the entity issuing the invoice)."},
    "Buyer Name": {"field_name": "buyer_name", "description": "The name of the buyer (the entity receiving the invoice)."},
    "Buyer GSTIN": {"field_name": "buyer_gstin", "description": "The GSTIN of the buyer (the entity receiving the invoice). Must be a 15-character alphanumeric string. Prioritize the GSTIN explicitly labeled as 'Buyer GSTIN' or associated with the buyer's details."},
    "Taxable Amount": {"field_name": "taxable_amount", "description": "This is the subtotal, the amount BEFORE any taxes (CGST, SGST, IGST) are applied. Must be a number."},
    "CGST": {"field_name": "cgst", "description": "The Central Goods and Services Tax amount. Must be a number."},
    "SGST": {"field_name": "sgst", "description": "The State Goods and Services Tax amount. Must be a number."},
    "IGST": {"field_name": "igst", "description": "The Integrated Goods and Services Tax amount. Must be a number."},
    "Total Amount Payable (Incl. Tax)": {"field_name": "total_amount_payable", "description": "This is the final total amount on the invoice *including all taxes and other charges*, but *before* any TDS deduction shown on the invoice. This represents the 'Gross Total' or 'Amount before TDS' on the invoice. Must be a number."},
    "TDS Amount": {"field_name": "tds_amount", "description": "The numerical value of TDS deducted, if explicitly stated."},
    "TDS Rate": {"field_name": "tds_rate", "description": "The numerical percentage rate of TDS deducted (e.g., '10' for 10%), if explicitly stated. If a TDS Section is clearly identified but the rate is missing on the invoice, infer the standard rate for that section (e.g., 10% for 194J)."},
    "Line Items": {"field_name": "line_items", "description": "A list of line items, each with 'description', 'quantity', and 'gross_worth'. Provide an empty list if no line items are found."},
    "Place of Supply": {"field_name": "place_of_supply", "description": "The Place of Supply. Extract the exact State/City name or 'Foreign' if applicable. Prioritize 'Place of Supply' field, then 'Ship To', 'Bill To', 'Customer/Buyer Address'."},
    "Expense Ledger": {"field_name": "expense_ledger", "description": "Classify the nature of expense and suggest a suitable ledger type (e.g., 'Office Supplies', 'Professional Fees', 'Software Subscription'). Consider common TDS sections for inference."},
    "TDS Applicability": {"field_name": "tds", "description": "TDS applicability. 'Yes - Section [X]' if deducted, 'Yes - Section Unknown' if rate/amount present but no section, 'No' if not applicable or foreign transaction, 'Uncertain' if unclear."},
    "HSN/SAC Code": {"field_name": "hsn_sac", "description": "The HSN (Harmonized System of Nomenclature) or SAC (Service Accounting Code). ONLY extract if explicitly mentioned. If not found, MUST be `null`."},
    "RCM Applicability": {"field_name": "rcm_applicability", "description": "Reverse Charge Mechanism (RCM) applicability. State 'Yes', 'No', or 'Uncertain'."}
}

# Define common fields that can be quickly added
COMMON_FIELD_NAMES = [
    "Invoice Number", "Invoice Date", "Seller Name", "Total Amount Payable (Incl. Tax)",
    "Expense Ledger", "TDS Applicability", "Seller GSTIN", "Buyer Name", "Buyer GSTIN",
    "Taxable Amount", "CGST", "SGST", "IGST", "TDS Amount", "TDS Rate", "HSN/SAC Code",
    "Place of Supply", "RCM Applicability", "Line Items"
]


# --- Utility Functions for UI/Data Formatting ---
def parse_date_safe(date_str: str) -> str:
    """Attempts to parse a date string into DD/MM/YYYY format."""
    if not date_str:
        return ""
    formats = ["%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d", "%d %b %Y", "%d %B %Y", "%Y/%m/%d"]
    for fmt in formats:
        try:
            dt_obj = datetime.strptime(date_str, fmt)
            # Adjust 2-digit years. Current year (e.g., 2025). If year is 24, assume 2024. If 98, assume 1998.
            if dt_obj.year < 100: # 2-digit year
                current_year_last_two_digits = datetime.now().year % 100
                if dt_obj.year > current_year_last_two_digits + 10: # e.g., 90 in 2025 -> 1990 (heuristics)
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
def generate_dynamic_prompt(selected_fields_display_names: List[str], extraction_type: str) -> str:
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

    field_guidelines = []
    
    # Prioritize fields from FIELD_METADATA for detailed guidelines
    for field_display_name in selected_fields_display_names:
        metadata = FIELD_METADATA.get(field_display_name)
        if metadata:
            field_name_internal = metadata["field_name"]
            description = metadata["description"]
            field_guidelines.append(f"- '{field_name_internal}': {description}")
            if field_name_internal == "line_items":
                prompt_parts.append("You MUST include an empty list `[]` for `line_items` if no line items are found, do not use `null` for `line_items`. ")
        else:
            # For custom, user-defined fields, provide a generic guideline
            field_guidelines.append(f"- '{field_display_name}': Extract the value as it appears in the invoice.")

    # Ensure all selected fields are listed as required keys for the LLM
    required_keys_for_llm = [FIELD_METADATA.get(f, {}).get("field_name", f) for f in selected_fields_display_names]

    if not required_keys_for_llm:
        return "Extract common invoice details like invoice_number, date, seller_name, total_amount_payable."

    prompt_parts.append(f"Keys to extract: {', '.join(required_keys_for_llm)}. ")
    prompt_parts.append("\nGUIDELINES FOR EXTRACTION:\n")
    prompt_parts.extend(field_guidelines)
    prompt_parts.append("\nReturn 'NOT AN INVOICE' if the document is clearly not an invoice.\n")
    prompt_parts.append("Ensure the JSON output is clean and directly parsable.")
    return "\n".join(prompt_parts)


# --- Dynamic Pydantic Schema Generation ---
def create_dynamic_invoice_schema(selected_fields_display_names: List[str]) -> BaseModel:
    fields = {}
    for field_display_name in selected_fields_display_names:
        metadata = FIELD_METADATA.get(field_display_name)
        if metadata:
            field_name_internal = metadata["field_name"]
            if field_name_internal == "line_items":
                fields[field_name_internal] = (List[LineItem], Field(default_factory=list))
            elif field_name_internal in Invoice.model_fields:
                original_field = Invoice.model_fields.get(field_name_internal)
                if original_field:
                    fields[field_name_internal] = (original_field.annotation, Field(None, description=original_field.description))
                else:
                    fields[field_name_internal] = (Optional[str], None) # Fallback
            else:
                fields[field_name_internal] = (Optional[str], None) # Fallback for unknown internal names
        else:
            # For truly custom fields, default to Optional[str]
            fields[field_display_name] = (Optional[str], None)

    if not fields:
        # If no fields are selected, create a minimal schema for basic extraction
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
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&family=Inter:wght@300;400;500;600;700&display=swap');

    :root {
        --primary-color: #4A90E2; /* A vibrant blue */
        --secondary-color: #50E3C2; /* A light teal for accents */
        --text-color: #333333; /* Dark gray for general text */
        --heading-color: #1A237E; /* Dark blue for headings */
        --background-color: #F0F2F6; /* Light grey main background */
        --card-background: white; /* White background for cards/sections */
        --sidebar-background: #F8F8F8; /* Very light gray for sidebar */
        --border-color: #E0E0E0;
        --shadow-color: rgba(0, 0, 0, 0.08);
        --success-bg: #E8F5E9;
        --success-text: #2E7D32;
        --info-bg: #E3F2FD;
        --info-text: #1976D2;
        --warning-bg: #FFFDE7;
        --warning-text: #FBC02D;
        --error-bg: #FFEBE9;
        --error-text: #C62828;
    }

    body {
        font-family: 'Inter', sans-serif;
        color: var(--text-color);
    }
    .stApp { 
        background-color: var(--background-color); 
        color: var(--text-color); 
        font-family: 'Inter', sans-serif; 
    }
    h1, h2, h3, h4, h5, h6 { 
        color: var(--heading-color); 
        font-family: 'Poppins', sans-serif; 
        font-weight: 700;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    h1 { font-size: 2.5rem; }
    h2 { font-size: 2rem; }
    h3 { font-size: 1.5rem; }

    /* Sidebar Styling */
    .st-emotion-cache-h5rpjc { /* This targets the sidebar container */
        background-color: var(--sidebar-background); /* Light background for sidebar */
        color: var(--text-color); /* Dark text in sidebar */
        border-right: 1px solid var(--border-color);
    }
    /* Ensure all text within sidebar is dark */
    .st-emotion-cache-h5rpjc h1, .st-emotion-cache-h5rpjc h2, 
    .st-emotion-cache-h5rpjc h3, .st-emotion-cache-h5rpjc h4,
    .st-emotion-cache-h5rpjc .stRadio > label > div > div > p,
    .st-emotion-cache-h5rpjc .stTextInput > label > div > p,
    .st-emotion-cache-h5rpjc .stTextInput > div > div > input,
    .st-emotion-cache-h5rpjc .stTextInput > div > div > textarea,
    .st-emotion-cache-h5rpjc .stMarkdown p,
    .st-emotion-cache-h5rpjc .stCheckbox span { /* Explicitly target checkbox span in sidebar */
        color: var(--text-color) !important; 
    }
    /* Adjust input backgrounds in sidebar */
    .st-emotion-cache-h5rpjc .stTextInput > div > div > input,
    .st-emotion-cache-h5rpjc .stTextInput > div > div > textarea {
        background-color: var(--card-background) !important; /* White input background */
        border: 1px solid var(--border-color) !important;
    }
    /* Radio button specific styling */
    .st-emotion-cache-h5rpjc .stRadio > label > div > div > div { /* For radio button circle */
        border-color: var(--text-color) !important; /* Dark border for unchecked */
    }
    .st-emotion-cache-h5rpjc .stRadio > label > div > div > div[data-checked="true"] {
        background-color: var(--primary-color) !important;
        border-color: var(--primary-color) !important;
    }
    /* Checkbox specific styling in sidebar */
    .st-emotion-cache-h5rpjc .stCheckbox > label > div:first-child > div:first-child { /* The checkbox box */
        border-color: var(--text-color) !important;
    }
    .st-emotion-cache-h5rpjc .stCheckbox > label > div:first-child > div:first-child[data-checked="true"] {
        background-color: var(--primary-color) !important;
        border-color: var(--primary-color) !important;
    }


    /* Main Content Buttons */
    .stButton>button {
        background-color: var(--primary-color);
        color: white; /* Keep white for primary buttons for contrast on blue */
        border-radius: 12px;
        padding: 12px 25px;
        font-size: 17px;
        font-weight: 600;
        transition: background-color 0.3s ease, transform 0.2s ease;
        border: none;
        box-shadow: 0 4px 8px var(--shadow-color);
    }
    .stButton>button:hover { 
        background-color: #357AE8;
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }

    /* Secondary Buttons (e.g., Remove 'x' button) */
    button[data-testid^="stButton-secondary"] { /* Target native secondary buttons */
        background-color: #D1D1D1; /* Light grey for secondary actions */
        color: var(--text-color); /* Dark text for secondary buttons */
        border-radius: 8px;
        padding: 5px 10px;
        font-size: 14px;
        font-weight: 500;
        margin-left: 10px;
        box-shadow: none;
        transition: background-color 0.3s ease;
    }
    button[data-testid^="stButton-secondary"]:hover {
        background-color: #B0B0B0;
        transform: none;
        box-shadow: none;
    }

    /* Specific style for the remove 'x' buttons in selected fields */
    /* This overrides the general secondary button for the 'x' button specifically */
    button[data-testid^="stButton-secondary"][kind="secondaryFormSubmit"] {
        background-color: #ef5350 !important; /* Red for remove */
        color: white !important; /* Keep white on red for contrast */
        font-weight: bold;
    }
    button[data-testid^="stButton-secondary"][kind="secondaryFormSubmit"]:hover {
        background-color: #d32f2f !important; /* Darker red */
    }


    /* Text and Markdown */
    .stMarkdown p { 
        font-size: 1.05em; 
        line-height: 1.6; 
        font-family: 'Inter', sans-serif; 
        color: var(--text-color);
    }
    .stAlert { 
        border-radius: 8px; 
        font-family: 'Inter', sans-serif; 
        padding: 15px 20px;
        line-height: 1.5;
        margin-bottom: 1rem; /* Space below alerts */
    }
    .stAlert.success { background-color: var(--success-bg); color: var(--success-text); border: 1px solid #A5D6A7; }
    .stAlert.error { background-color: var(--error-bg); color: var(--error-text); border: 1px solid #EF9A9A; }
    .stAlert.info { background-color: var(--info-bg); color: var(--info-text); border: 1px solid #90CAF9; }
    .stAlert.warning { background-color: var(--warning-bg); color: var(--warning-text); border: 1px solid #FFE082; }

    /* Progress Bar */
    .stProgress > div > div > div > div { 
        background-color: var(--primary-color) !important; 
        border-radius: 8px;
    }

    /* Card-like Sections */
    .main .block-container {
        padding-top: 2rem;
        padding-right: 3rem;
        padding-left: 3rem;
        padding-bottom: 3rem;
    }
    /* Targets st.container for card styling */
    div.stContainer {
        background-color: var(--card-background);
        padding: 25px;
        border-radius: 12px;
        box-shadow: 0 4px 15px var(--shadow-color);
        margin-bottom: 2rem;
        border: 1px solid var(--border-color);
    }

    /* Expander Headers as Cards */
    .streamlit-expanderHeader {
        background-color: var(--card-background) !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 15px var(--shadow-color);
        margin-bottom: 1rem;
        border: 1px solid var(--border-color);
        padding: 15px 20px !important;
        color: var(--heading-color) !important;
        font-weight: 600 !important;
        font-family: 'Poppins', sans-serif !important;
        transition: all 0.2s ease-in-out;
    }
    .streamlit-expanderHeader:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.12);
    }
    /* Content inside expander */
    .streamlit-expanderContent {
        background-color: var(--card-background);
        border-bottom-left-radius: 12px;
        border-bottom-right-radius: 12px;
        box-shadow: 0 4px 15px var(--shadow-color);
        padding: 20px;
        border: 1px solid var(--border-color);
        border-top: none;
        margin-bottom: 2rem;
    }
    /* Dataframes within cards */
    div[data-testid="stDataFrame"] {
        background-color: #fcfcfc;
        border-radius: 8px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        border: 1px solid #eee;
    }

    /* Input Field Styling */
    .stTextInput>div>div>input {
        border-radius: 8px;
        border: 1px solid var(--border-color);
        padding: 10px 15px;
        font-size: 1em;
        color: var(--text-color);
        background-color: #fcfcfc;
        box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.06);
    }
    .stTextInput>div>div>textarea { /* For text_area */
        border-radius: 8px;
        border: 1px solid var(--border-color);
        padding: 10px 15px;
        font-size: 1em;
        color: var(--text-color);
        background-color: #fcfcfc;
        box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.06);
    }
    .stTextInput>label>div>p, .stSelectbox>label>div>p, .stRadio>label>div>p {
        font-weight: 600; /* Labels bold */
        color: var(--heading-color);
        margin-bottom: 0.5rem;
    }

    /* Selected Field Row */
    .selected-field-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 8px 12px;
        margin-bottom: 8px; /* More space between items */
        background-color: #E6EEF7; /* Lighter blueish background */
        border-radius: 8px;
        border: 1px solid #CBDCEB;
        font-weight: 500;
        color: var(--heading-color);
    }
    .selected-field-row strong {
        color: var(--heading-color); /* Ensure strong text is dark blue */
    }
</style>
""", unsafe_allow_html=True)


st.title("📄 AI Invoice Extractor (Dynamic & Multi-Model)")
st.markdown("""
    <p style='font-size:1.2em; color:#666;'>
        Empower your financial operations with intelligent invoice data extraction. 
        Select your desired fields or provide a custom prompt, and let AI do the heavy lifting!
    </p>
""", unsafe_allow_html=True)
st.markdown("---") # Visually separates title and intro

st.sidebar.header("⚙️ Configuration")

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
st.sidebar.markdown("---")
model_choice = st.sidebar.radio(
    "✨ Choose AI Model:",
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
        st.sidebar.info(f"Using *{model_choice}* API Key from `Streamlit Secrets`.")

    elif model_choice == "OpenAI GPT":
        selected_api_key = st.secrets.get("OPENAI_API_KEY")
        model_id_input = st.secrets.get("OPENAI_MODEL_ID", "gpt-4o")
        if not selected_api_key:
            st.sidebar.warning("OPENAI_API_KEY not found in Streamlit Secrets. OpenAI functionality might be limited.")
        st.sidebar.text_input("OpenAI Model ID (from secrets):", model_id_input, key="openai_model_id_secrets", disabled=True)
        st.sidebar.caption(f"Using model ID from secrets: `{model_id_input}`")
        st.sidebar.info(f"Using *{model_choice}* API Key from `Streamlit Secrets`.")

else:
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔑 Enter Your Own API Key (Required)")
    st.sidebar.markdown(f"To use the *{model_choice}* model, please provide your personal API key. Your key is used for processing and **not stored**.")

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

st.markdown("---") # Separator

# Initialize session state for custom fields if not present
if 'custom_extracted_fields' not in st.session_state:
    st.session_state.custom_extracted_fields = []
# Initialize the value for the text input for adding new fields
if 'new_custom_field_input_value' not in st.session_state:
    st.session_state.new_custom_field_input_value = ""


# --- Extraction Type Selection ---
extraction_type = st.radio(
    "📊 **Select Extraction Type:**",
    ("Structured Data Extraction", "Free-form Summary"),
    key="extraction_type_selection"
)

if extraction_type == "Structured Data Extraction":
    with st.expander("🛠️ Define Your Desired Columns (Structured Data)", expanded=True):
        st.markdown("Here, you can build your custom list of fields you want to extract.")

        col1_add, col2_add = st.columns([0.7, 0.3])
        with col1_add:
            # Use the session state variable to control the input's value
            new_field_input = st.text_input(
                "✍️ **Add Custom Field:**",
                placeholder="e.g., 'Company Address', 'Shipping Date'",
                key="new_custom_field_input", # This key is for the widget itself
                value=st.session_state.new_custom_field_input_value # Control its value
            )
        with col2_add:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True) # Spacer
            if st.button("➕ Add Field", key="add_field_button"):
                if new_field_input and new_field_input not in st.session_state.custom_extracted_fields:
                    st.session_state.custom_extracted_fields.append(new_field_input)
                    st.toast(f"Added '{new_field_input}' to your fields!")
                    st.session_state.new_custom_field_input_value = "" # Clear the input value in session state
                    st.rerun() # Rerun to update the display and clear input
                else:
                    st.warning("Please enter a unique field name.")
        
        st.markdown("---")
        st.subheader("🚀 Quick Add Common Fields")
        st.markdown("Check the boxes below to quickly add common invoice data points to your list.")
        
        # Display common fields in columns
        num_cols_common = 4
        cols = st.columns(num_cols_common)
        
        # Track which common fields are currently checked/unchecked
        common_fields_state = {field: False for field in COMMON_FIELD_NAMES}
        for field in COMMON_FIELD_NAMES:
            if field in st.session_state.custom_extracted_fields:
                common_fields_state[field] = True

        rerun_needed_for_checkbox = False
        for i, field in enumerate(COMMON_FIELD_NAMES):
            with cols[i % num_cols_common]:
                current_checked_state = st.checkbox(field, value=common_fields_state[field], key=f"common_field_checkbox_{field}")
                if current_checked_state and field not in st.session_state.custom_extracted_fields:
                    st.session_state.custom_extracted_fields.append(field)
                    rerun_needed_for_checkbox = True
                elif not current_checked_state and field in st.session_state.custom_extracted_fields:
                    st.session_state.custom_extracted_fields.remove(field)
                    rerun_needed_for_checkbox = True
        
        if rerun_needed_for_checkbox:
            st.rerun()


        st.markdown("---")
        st.subheader("📋 Your Selected Fields:")
        if st.session_state.custom_extracted_fields:
            # Display selected fields with remove buttons
            with st.container(border=True): # Use a container for the list of fields
                for i, field in enumerate(st.session_state.custom_extracted_fields):
                    col_name, col_remove = st.columns([0.8, 0.2])
                    with col_name:
                        st.markdown(f"<div class='selected-field-row'><strong>{field}</strong></div>", unsafe_allow_html=True)
                    with col_remove:
                        if st.button("x", key=f"remove_field_btn_{field}_{i}", help=f"Remove '{field}'", type="secondary"):
                            st.session_state.custom_extracted_fields.remove(field)
                            st.toast(f"Removed '{field}'.")
                            st.rerun()
            
            selected_fields_for_extraction = st.session_state.custom_extracted_fields
        else:
            selected_fields_for_extraction = []
            st.info("No fields defined for structured extraction. Add some above using the input box or quick-add common fields.")
            
# --- Custom Prompt Input ---
st.markdown("---")
with st.expander("✍️ Custom Prompt (Advanced)", expanded=False):
    st.info("Enter your own prompt to completely control the AI's output. This overrides the 'Extraction Type' and 'Defined Columns'.")
    st.markdown("_For structured output with a custom prompt, ensure your prompt explicitly asks for JSON data!_")
    custom_prompt_input = st.text_area(
        "Your Custom Prompt:",
        height=150,
        placeholder="e.g., 'Extract the sender's full legal name and their registered office address as a JSON object with keys `legal_name` and `address`.'",
        key="custom_prompt_textarea"
    )
    if custom_prompt_input:
        st.warning("⚠️ Custom prompt provided. This will supersede all other extraction settings!")


st.markdown("---") # Separator before file uploader

uploaded_files = st.file_uploader(
    "📂 **Choose PDF Invoice Files**",
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
        st.error("For 'Structured Data Extraction', please define at least one field, or provide a custom prompt.")
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

            # Determine the prompt and schema to use based on custom prompt precedence
            prompt_to_use = custom_prompt_input if custom_prompt_input else generate_dynamic_prompt(selected_fields_for_extraction, extraction_type)
            
            dynamic_pydantic_schema = None
            # If custom prompt is used AND structured extraction is implicitly or explicitly targeted, try to create a schema
            if custom_prompt_input and extraction_type == "Structured Data Extraction": # Assume custom prompt for structured implies user wants validation
                # With a custom prompt, we create a schema based on selected fields IF they were provided.
                # Otherwise, it's a completely open-ended custom prompt, and we won't validate with Pydantic for strictness.
                if selected_fields_for_extraction:
                    try:
                        dynamic_pydantic_schema = create_dynamic_invoice_schema(selected_fields_for_extraction)
                    except Exception as e:
                        st.error(f"Failed to create dynamic schema for custom prompt validation: {e}")
                        if st.session_state.DEBUG_MODE: st.exception(e)
                        st.stop()
            elif extraction_type == "Structured Data Extraction" and not custom_prompt_input:
                # Standard structured extraction path
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
                            # If custom prompt was used, we display ALL keys returned by LLM
                            # If not custom prompt, we use the fields selected by the user
                            effective_fields_for_this_row = selected_fields_for_extraction if not custom_prompt_input else list(extracted_output.keys())
                            
                            if custom_prompt_input and "line_items" in effective_fields_for_this_row:
                                st.info("Custom prompt was used and 'line_items' was found. This will be shown in a separate expander.")

                            # Populate data based on effective fields and apply formatting
                            for field_display_name in effective_fields_for_this_row:
                                # First, try to get the internal field name from metadata, otherwise use display name directly
                                metadata = FIELD_METADATA.get(field_display_name)
                                field_name_internal = metadata["field_name"] if metadata else field_display_name
                                
                                # Get the value from the extracted dict using the internal name or display name
                                value = extracted_output.get(field_name_internal, extracted_output.get(field_display_name, None))

                                if field_name_internal == "date":
                                    row_data[field_display_name] = parse_date_safe(value or "")
                                elif field_name_internal in ["taxable_amount", "cgst", "sgst", "igst", "total_amount_payable", "tds_amount"]:
                                    row_data[field_display_name] = value if value is not None else 0.0
                                elif field_name_internal == "tds_rate":
                                    # Logic to infer TDS rate if TDS Applicability was also extracted
                                    tds_section_val = extracted_output.get(FIELD_METADATA["TDS Applicability"]["field_name"]) if "TDS Applicability" in selected_fields_for_extraction else None
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
                                    row_data[field_display_name] = inferred_tds_rate

                                elif field_name_internal == "tds": # This is "TDS Applicability"
                                    pos_val = extracted_output.get(FIELD_METADATA["Place of Supply"]["field_name"]) if "Place of Supply" in selected_fields_for_extraction else None
                                    if pos_val and pos_val.lower() == "foreign":
                                        row_data[field_display_name] = "No"
                                        # Adjust related fields if applicable
                                        if FIELD_METADATA["TDS Amount"]["field_name"] in extracted_output:
                                            row_data[FIELD_METADATA["TDS Amount"]["field_name"]] = 0.0
                                        if FIELD_METADATA["TDS Rate"]["field_name"] in extracted_output:
                                            row_data[FIELD_METADATA["TDS Rate"]["field_name"]] = "N/A"
                                        # Note: TDS Section (Derived) is added below, so it will be affected by this
                                        st.info(f"TDS adjusted to 'No' for **{uploaded_file_obj.name}** as Place of Supply is 'Foreign'.")
                                    elif value and value.lower() == "no":
                                        row_data[field_display_name] = "No"
                                        if FIELD_METADATA["TDS Amount"]["field_name"] in extracted_output:
                                            row_data[FIELD_METADATA["TDS Amount"]["field_name"]] = 0.0
                                        if FIELD_METADATA["TDS Rate"]["field_name"] in extracted_output:
                                            row_data[FIELD_METADATA["TDS Rate"]["field_name"]] = "N/A"
                                    else:
                                        row_data[field_display_name] = value or "N/A"

                                elif field_name_internal == "line_items":
                                    pass # Handled separately

                                else:
                                    row_data[field_display_name] = value or "N/A"

                            # Always add TDS Section (Derived) if TDS Applicability was chosen or explicitly extracted by custom prompt
                            if ("TDS Applicability" in effective_fields_for_this_row or "tds" in extracted_output):
                                tds_section_display_val = "N/A"
                                if "tds" in extracted_output and isinstance(extracted_output["tds"], str) and "section" in extracted_output["tds"].lower():
                                    parts = extracted_output["tds"].split("Section ")
                                    if len(parts) > 1:
                                        section_part = parts[1].strip()
                                        section_part = section_part.split(' ')[0].split(']')[0].split('.')[0].strip()
                                        if section_part:
                                            tds_section_display_val = section_part
                                row_data["TDS Section (Derived)"] = tds_section_display_val

                            st.session_state.extracted_results.append({
                                "file_name": uploaded_file_obj.name,
                                "extraction_type": "structured", # Indicate type for later processing
                                "extracted_data": row_data,
                                "raw_extracted_dict": extracted_output, # Store raw dict for debug/line items
                                "selected_fields_at_extraction": effective_fields_for_this_row # Store what was actually requested/returned
                            })

                            with st.expander(f"📋 Details for {uploaded_file_obj.name} (using {model_choice})"):
                                st.subheader("Extracted Raw JSON Data:")
                                st.json(extracted_output) # Show raw extracted JSON for verification

                                # Display line items if they were extracted and are present
                                if "line_items" in extracted_output and extracted_output["line_items"]:
                                    st.subheader("Line Items:")
                                    line_item_data = [{
                                        "Description": item.get("description"),
                                        "Quantity": item.get("quantity"),
                                        "Gross Worth": format_currency(item.get("gross_worth")),
                                    } for item in extracted_output["line_items"]]
                                    st.dataframe(pd.DataFrame(line_item_data), use_container_width=True)
                                elif "Line Items" in effective_fields_for_this_row:
                                    st.info("No line items extracted.")
                                else:
                                    st.info("Line items were not requested for extraction.")
                        
                        elif isinstance(extracted_output, str): # Free-form Summary Output
                            st.session_state.extracted_results.append({
                                "file_name": uploaded_file_obj.name,
                                "extraction_type": "summary",
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
    st.header("📊 Consolidated Extracted Invoice Summary")
    
    # Separate structured vs. summary results
    structured_results = [r for r in st.session_state.extracted_results if r["extraction_type"] == "structured"]
    summary_results = [r for r in st.session_state.extracted_results if r["extraction_type"] == "summary"]

    if structured_results:
        st.subheader("📋 Structured Data Results")
        
        # Collect all unique column names that appeared in any structured result
        all_unique_structured_cols = set()
        for result in structured_results:
            all_unique_structured_cols.update(result["extracted_data"].keys())
        
        # Sort these column names for consistent display, with "File Name" first
        display_cols_order = ["File Name"] + sorted([col for col in list(all_unique_structured_cols) if col != "File Name"])

        summary_rows_for_display = []
        for result in structured_results:
            row_to_add = {"File Name": result["file_name"]}
            for col_key in display_cols_order:
                if col_key == "File Name": continue # Already added
                value = result["extracted_data"].get(col_key, "N/A") # Get value, default to N/A

                # Apply formatting
                # We need to map back to original field names to check types if it's a "common field"
                is_common_field_numeric = False
                for display_name, metadata in FIELD_METADATA.items():
                    # Check if the display name (key in FIELD_METADATA) or its internal field_name matches col_key
                    if metadata["field_name"] == col_key or display_name == col_key:
                        if metadata["field_name"] in ["taxable_amount", "cgst", "sgst", "igst", "tds_amount", "total_amount_payable"]:
                            is_common_field_numeric = True
                            break
                
                if is_common_field_numeric:
                    row_to_add[col_key] = format_currency(value if value != "N/A" else None)
                elif col_key == FIELD_METADATA["TDS Rate"]["field_name"] or col_key == "TDS Rate": # Explicitly handle TDS Rate
                    row_to_add[col_key] = f"{value:.2f}%" if isinstance(value, (int, float)) else value
                else:
                    row_to_add[col_key] = value
            summary_rows_for_display.append(row_to_add)

        df_display = pd.DataFrame(summary_rows_for_display)
        # Ensure the display order is respected
        df_display = df_display[[col for col in display_cols_order if col in df_display.columns]]
        st.dataframe(df_display, use_container_width=True)

        # --- Excel Download for Structured Data ---
        output_excel = io.BytesIO()
        with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
            excel_data_rows = []
            for result in structured_results:
                row = {"File Name": result["file_name"]}
                # For Excel, we take all raw extracted data, not just formatted display ones
                # We use raw_extracted_dict because it contains the keys exactly as returned by LLM,
                # which could include custom fields not in FIELD_METADATA.
                raw_data_from_llm = result.get("raw_extracted_dict", {})
                
                for k, v in raw_data_from_llm.items():
                    # Attempt to map internal field names to user-friendly display names for Excel headers
                    # If not found in FIELD_METADATA, use the raw key as the column name
                    display_name_for_excel = next((d_name for d_name, meta in FIELD_METADATA.items() if meta["field_name"] == k), k)
                    row[display_name_for_excel] = v
                
                # Also add the derived TDS Section explicitly for Excel if it was present
                if "TDS Section (Derived)" in result["extracted_data"]:
                    row["TDS Section (Derived)"] = result["extracted_data"]["TDS Section (Derived)"]
                
                excel_data_rows.append(row)
            
            df_for_excel = pd.DataFrame(excel_data_rows)

            # Drop 'Line Items' column from the main sheet if it exists
            # This handles both standard 'Line Items' and potential 'line_items' as raw key
            if "Line Items" in df_for_excel.columns:
                df_for_excel = df_for_excel.drop(columns=["Line Items"])
            if "line_items" in df_for_excel.columns: # Check for internal name too
                 df_for_excel = df_for_excel.drop(columns=["line_items"])

            # Sort Excel columns for consistency
            # Get all current columns in the DataFrame, then sort them but keep "File Name" first
            all_excel_cols_present = [col for col in df_for_excel.columns if col != "File Name"]
            sorted_excel_cols = ["File Name"] + sorted(all_excel_cols_present)
            
            df_for_excel = df_for_excel[[col for col in sorted_excel_cols if col in df_for_excel.columns]]
            
            df_for_excel.to_excel(writer, index=False, sheet_name='InvoiceSummary')

            # Add Line Items to a separate sheet if they were extracted
            any_line_items_extracted = any(item.get("raw_extracted_dict", {}).get("line_items") for item in structured_results)
            if any_line_items_extracted:
                all_line_items = []
                for result in structured_results:
                    file_name = result["file_name"]
                    # Access line_items from raw_extracted_dict for fidelity
                    raw_line_items = result.get("raw_extracted_dict", {}).get("line_items", [])
                    for li in raw_line_items:
                        all_line_items.append({
                            "File Name": file_name,
                            "Description": li.get("description"),
                            "Quantity": li.get("quantity"),
                            "Gross Worth": li.get("gross_worth")
                        })
                if all_line_items:
                    df_line_items = pd.DataFrame(all_line_items)
                    df_line_items.to_excel(writer, index=False, sheet_name='LineItems')

        excel_data = output_excel.getvalue()

        st.download_button(
            label="📥 Download Consolidated Structured Data as Excel",
            data=excel_data,
            file_name="invoice_structured_summary.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    
    if summary_results:
        st.subheader("📚 Free-form Summaries")
        summary_df_data = []
        for result in summary_results:
            summary_df_data.append({
                "File Name": result["file_name"],
                "Summary": result["summary_text"]
            })
        df_summary = pd.DataFrame(summary_df_data)
        st.dataframe(df_summary, use_container_width=True)

        output_excel_summary = io.BytesIO()
        with pd.ExcelWriter(output_excel_summary, engine='openpyxl') as writer:
            df_summary.to_excel(writer, index=False, sheet_name='InvoiceSummaries')
        excel_data_summary = output_excel_summary.getvalue()

        st.download_button(
            label="📥 Download Summaries as Excel",
            data=excel_data_summary,
            file_name="invoice_summaries.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

elif not uploaded_files:
    st.info("Upload PDF files and click 'Process Invoices' to see results.")
