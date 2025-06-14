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
    hsn_sac: Optional[str] = None # Added HSN/SAC

class Invoice(BaseModel):
    invoice_number: str
    date: str
    gstin: str # Seller GSTIN
    seller_name: str
    buyer_name: str
    buyer_gstin: Optional[str] = None
    line_items: List[LineItem]
    total_gross_worth: float
    cgst: Optional[float] = None
    sgst: Optional[float] = None
    igst: Optional[float] = None
    place_of_supply: Optional[str] = None
    expense_ledger: Optional[str] = None
    tds: Optional[str] = None
    rcm_applicability: Optional[str] = None # Added RCM Applicability


# --- Utility Functions for UI/Data Formatting ---
def parse_date_safe(date_str: str) -> str:
    """Attempts to parse a date string into DD/MM/YYYY format."""
    if not date_str:
        return ""
    # Try common Indian date formats
    formats = ["%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d", "%d %b %Y", "%d %B %Y", "%Y/%m/%d"]
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt).strftime("%d/%m/%Y")
        except ValueError:
            continue
    # If no format matches, return original string or a placeholder
    return date_str # Or "Invalid Date"

def format_currency(amount: Optional[float]) -> str:
    """Formats a float as an Indian Rupee currency string."""
    if amount is None:
        return "₹ N/A"
    return f"₹ {amount:,.2f}" # Formats with commas and 2 decimal places

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
        st.info(f"Gemini: Uploading '{display_name}' to Gemini File API...")
        gemini_file_resource = client_instance.files.upload(
            file=file_path,
            config={'display_name': display_name.split('.')[0]}
        )
        st.success(f"Gemini: '{display_name}' uploaded. Gemini file name: {gemini_file_resource.name}")

        prompt = (
            "Extract all relevant and clear information from the invoice, adhering to Indian standards "
            "for dates (DD/MM/YYYY or DD-MM-YYYY) and codes (like GSTIN, HSN/SAC). "
            "Accurately identify the total amount payable. "
            "For each line item, extract 'description', 'quantity', 'gross_worth', and 'hsn_sac' (if available). "
            "Classify the nature of expense and suggest an "
            "applicable ledger type (e.g., 'Office Supplies', 'Professional Fees', 'Software Subscription'). "
            "Determine TDS applicability (e.g., 'Yes - Section 194J', 'No', 'Uncertain'). "
            "Determine reverse charge GST (RCM) applicability (e.g., 'Yes', 'No', 'Uncertain'). "
            "Handle missing data appropriately by setting fields to null or an empty string where "
            "Optional, and raise an issue if critical data is missing for required fields. "
            "Do not make assumptions or perform calculations beyond what's explicitly stated in the invoice text. "
            "If a value is clearly zero, represent it as 0.0 for floats. For dates, prefer DD/MM/YYYY."
        )
        st.info(f"Gemini: Sending '{display_name}' to model '{gemini_model_id}' for extraction...")
        response = client_instance.models.generate_content(
            model=gemini_model_id,
            contents=[prompt, gemini_file_resource],
            config={'response_mime_type': 'application/json', 'response_schema': pydantic_schema}
        )

        st.success(f"Gemini: Data extracted for '{display_name}'.")
        return response.parsed

    except Exception as e:
        st.error(f"Error processing '{display_name}' with Gemini: {e}")
        st.exception(e)
        return None
    finally:
        if gemini_file_resource:
            try:
                st.info(f"Gemini: Attempting to delete '{gemini_file_resource.name}' from File API...")
                if client_instance and hasattr(client_instance, 'files') and hasattr(client_instance.files, 'delete'):
                    client_instance.files.delete(name=gemini_file_resource.name)
                    st.success(f"Gemini: Successfully deleted '{gemini_file_resource.name}'.")
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
        # OpenAI Vision can take up to 20 images per request.
        # For invoices, one or two pages are usually enough.
        # Adjust `max_pages_to_process` if your invoices are multi-page.
        max_pages_to_process = 5 # Limit to prevent excessive costs/tokens for very long PDFs

        for page_num in range(min(doc.page_count, max_pages_to_process)):
            page = doc.load_page(page_num)
            # Render at 2x resolution (matrix=fitz.Matrix(2, 2)) for better OCR, DPI ~144-150
            # Higher resolution means larger images, more tokens, more cost. Balance is key.
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            img_bytes_io = io.BytesIO()
            # Convert pixmap to PIL Image, then save to bytes in PNG format
            pil_image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            pil_image.save(img_bytes_io, format="PNG")
            img_bytes = img_bytes_io.getvalue()

            base64_image = base64.b64encode(img_bytes).decode('utf-8')
            image_messages.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64_image}",
                    "detail": "high" # Use 'high' detail for better OCR, 'low' for faster/cheaper
                }
            })
        doc.close()

        if not image_messages:
            st.error(f"No pages could be converted to images for '{display_name}'. This might happen with corrupted PDFs or empty files.")
            return None

        system_prompt = (
            "You are an expert invoice data extractor. Extract all specified details from the provided invoice document(s). "
            "Output the data strictly as a JSON object conforming to the following Pydantic schema structure. "
            "Ensure dates are in DD/MM/YYYY format. "
            "If a field is not found or is optional, set it to `null` or an empty string as appropriate. "
            "Here is the JSON schema you must adhere to:\n"
            f"```json\n{pydantic_schema.schema_json(indent=2)}\n```\n"
            "Do not include any conversational text or explanations outside the JSON. "
            "Be very precise with amounts and details, including all line items if present. "
            "If the invoice has multiple pages, consolidate information from all pages."
        )

        user_prompt_text = "Extract invoice data according to the provided schema from these document pages."

        st.info(f"OpenAI: Sending {len(image_messages)} image(s) from '{display_name}' to model '{openai_model_id}' for extraction...")

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
                st.success(f"OpenAI: Data extracted and validated for '{display_name}'.")
                return extracted_invoice
            except json.JSONDecodeError as e:
                st.error(f"OpenAI: Failed to decode JSON from response for '{display_name}': {e}")
                st.error(f"Response content: {json_string[:500]}...") # Show beginning of problematic JSON
                if st.session_state.get('DEBUG_MODE', False):
                    st.code(json_string, language="json")
                return None
            except Exception as e:
                st.error(f"OpenAI: Failed to parse extracted data into schema for '{display_name}': {e}")
                st.error(f"Response content: {json_string[:500]}...") # Show beginning of problematic JSON
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

# Custom CSS for a bit more flair
st.markdown("""
<style>
    .stApp { background-color: #f0f2f6; color: #333333; }
    h1, h2, h3 { color: #1e3a8a; }
    .stButton>button {
        background-color: #3b82f6; color: white; border-radius: 8px;
        padding: 10px 20px; font-size: 16px; font-weight: bold;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover { background-color: #2563eb; }
    .stMarkdown p { font-size: 1.05em; line-height: 1.6; }
    .stAlert { border-radius: 8px; }
    .stAlert.info { background-color: #e0f2f7; color: #0288d1; }
    .stAlert.success { background-color: #e8f5e9; color: #2e7d32; }
    .stAlert.error { background-color: #ffebee; color: #c62828; }
    .stProgress > div > div > div > div { background-color: #3b82f6 !important; }
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
    "2. If you know the admin password, enter it to use pre-configured API keys from `Streamlit Secrets`.\n"
    "   Otherwise, enter your own API keys manually.\n"
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
    st.session_state.DEBUG_MODE = False


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
                        st.success(f"Successfully extracted data from **{uploaded_file_obj.name}** 🎉")

                        parsed_date = parse_date_safe(extracted_data.date)
                        cgst = extracted_data.cgst if extracted_data.cgst is not None else 0.0
                        sgst = extracted_data.sgst if extracted_data.sgst is not None else 0.0
                        igst = extracted_data.igst if extracted_data.igst is not None else 0.0
                        total_gross_worth = extracted_data.total_gross_worth if extracted_data.total_gross_worth is not None else 0.0
                        pos = extracted_data.place_of_supply or "N/A"
                        seller_name_display = extracted_data.seller_name or "N/A"
                        seller_gstin_display = extracted_data.gstin or "N/A"
                        buyer_name_display = extracted_data.buyer_name or "N/A"
                        buyer_gstin_display = extracted_data.buyer_gstin or "N/A"
                        expense_ledger_display = extracted_data.expense_ledger or "N/A"
                        tds_display = extracted_data.tds or "N/A"
                        rcm_display = extracted_data.rcm_applicability or "N/A"

                        narration = (
                            f"Invoice **{extracted_data.invoice_number or 'N/A'}** dated **{parsed_date}** "
                            f"from **{seller_name_display}** (GSTIN: {seller_gstin_display}) "
                            f"to **{buyer_name_display}** (Buyer GSTIN: {buyer_gstin_display}), "
                            f"totaling **{format_currency(total_gross_worth)}**. "
                            f"Taxes: CGST {format_currency(cgst)}, SGST {format_currency(sgst)}, IGST {format_currency(igst)}. "
                            f"Place of Supply: {pos}. Expense Ledger: {expense_ledger_display}. "
                            f"TDS: {tds_display}. RCM: {rcm_display}."
                        )

                        st.session_state.summary_rows.append({
                            "File Name": uploaded_file_obj.name,
                            "Invoice Number": extracted_data.invoice_number,
                            "Date": parsed_date,
                            "Seller Name": seller_name_display,
                            "Seller GSTIN": seller_gstin_display,
                            "Buyer Name": buyer_name_display,
                            "Buyer GSTIN": buyer_gstin_display,
                            "Total Gross Worth": total_gross_worth,
                            "CGST": cgst,
                            "SGST": sgst,
                            "IGST": igst,
                            "Place of Supply": pos,
                            "Expense Ledger": expense_ledger_display,
                            "TDS": tds_display,
                            "RCM Applicability": rcm_display,
                            "Narration": narration,
                        })

                        with st.expander(f"📋 Details for {uploaded_file_obj.name} (using {model_choice})"):
                            st.subheader("Raw Extracted Data (JSON):")
                            st.json(extracted_data.dict())
                            st.subheader("Extracted Summary (Narration):")
                            st.markdown(narration)

                            if extracted_data.line_items:
                                st.subheader("Line Items:")
                                line_item_data = [{
                                    "Description": item.description,
                                    "Quantity": item.quantity,
                                    "Gross Worth": format_currency(item.gross_worth),
                                    "HSN/SAC": item.hsn_sac or "N/A"
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
                st.balloons()
                st.success("All selected invoices processed!")


if st.session_state.summary_rows:
    st.subheader("📊 Consolidated Extracted Invoice Summary")
    df = pd.DataFrame(st.session_state.summary_rows)

    df_display = df.copy()
    # Format currency columns for display in the DataFrame
    for col in ["Total Gross Worth", "CGST", "SGST", "IGST"]:
        df_display[col] = df_display[col].apply(format_currency)

    st.dataframe(df_display, use_container_width=True)

    output_excel = io.BytesIO()
    with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
        # Write the unformatted DataFrame to Excel to keep numbers as numbers
        df.to_excel(writer, index=False, sheet_name='InvoiceSummary')
    excel_data = output_excel.getvalue()

    st.download_button(
        label="📥 Download Consolidated Summary as Excel",
        data=excel_data,
        file_name="invoice_summary.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
elif not uploaded_files:
     st.info("Upload PDF files and click 'Process Invoices' to see results.")
