import streamlit as st
from pydantic import BaseModel, Field
from typing import List, Optional, Union
import pandas as pd
import os
import tempfile
import io
from datetime import datetime
import json # Import for JSON parsing from OpenAI response
import base64 # Import for base64 encoding for OpenAI Vision


# --- Import Libraries for Both Models ---
try:
    from google import genai
except ImportError:
    st.warning("The 'google-generativeai' library is not installed. Gemini functionality will be unavailable.")

try:
    from openai import OpenAI
except ImportError:
    st.warning("The 'openai' library is not installed. OpenAI functionality will be unavailable.")


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
    client_instance: 'genai.Client', # Type hint as string literal for forward reference if needed
    gemini_model_id: str,
    file_path: str,
    pydantic_schema: BaseModel,
) -> Optional[Invoice]:
    """Extracts structured data from an invoice PDF using Gemini Vision."""
    display_name = os.path.basename(file_path)
    gemini_file_resource = None

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
                # client_instance.files.delete is part of the newer genai library versions
                if hasattr(client_instance, 'files') and hasattr(client_instance.files, 'delete'):
                    client_instance.files.delete(name=gemini_file_resource.name)
                    st.success(f"Gemini: Successfully deleted '{gemini_file_resource.name}'.")
                else:
                    st.warning(f"Gemini: File API client does not support direct deletion or method not found. Manual cleanup may be required.")
            except Exception as e_del:
                st.warning(f"Gemini: Could not delete '{gemini_file_resource.name}': {e_del}")


def extract_from_openai(
    client_instance: 'OpenAI', # Type hint as string literal
    openai_model_id: str,
    file_path: str,
    pydantic_schema: BaseModel,
) -> Optional[Invoice]:
    """Extracts structured data from an invoice PDF/image using OpenAI Vision."""
    display_name = os.path.basename(file_path)

    try:
        # Read the file bytes and base64 encode for OpenAI Vision
        with open(file_path, "rb") as f:
            file_content_bytes = f.read()
        base64_file = base64.b64encode(file_content_bytes).decode('utf-8')

        system_prompt = (
            "You are an expert invoice data extractor. Extract all specified details from the provided invoice document. "
            "Output the data strictly as a JSON object conforming to the following Pydantic schema structure. "
            "Ensure dates are in DD/MM/YYYY format. "
            "If a field is not found or is optional, set it to `null` or an empty string as appropriate. "
            "Here is the JSON schema you must adhere to:\n"
            f"```json\n{pydantic_schema.schema_json(indent=2)}\n```\n"
            "Do not include any conversational text or explanations outside the JSON. "
            "Be very precise with amounts and details, including all line items if present."
        )

        user_prompt_text = "Extract invoice data according to the provided schema from this document."

        st.info(f"OpenAI: Sending '{display_name}' to model '{openai_model_id}' for extraction...")

        response = client_instance.chat.completions.create(
            model=openai_model_id,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "text", "text": user_prompt_text},
                    {"type": "image_url", "image_url": {
                        # OpenAI's vision models support PDF data URLs for some models (e.g., gpt-4o)
                        "url": f"data:application/pdf;base64,{base64_file}"
                    }}
                ]}
            ],
            response_format={"type": "json_object"}, # Instruct to return JSON
            max_tokens=4000 # Increase max tokens to ensure full JSON can be returned
        )

        if response and response.choices and response.choices[0].message and response.choices[0].message.content:
            json_string = response.choices[0].message.content
            st.markdown("##### Raw JSON from OpenAI:")
            st.code(json_string, language="json") # Show the raw JSON from OpenAI for debugging
            try:
                extracted_dict = json.loads(json_string)
                # Validate with Pydantic model
                extracted_invoice = pydantic_schema.parse_obj(extracted_dict)
                st.success(f"OpenAI: Data extracted and validated for '{display_name}'.")
                return extracted_invoice
            except json.JSONDecodeError as e:
                st.error(f"OpenAI: Failed to decode JSON from response for '{display_name}': {e}")
                st.code(json_string, language="json") # Show malformed JSON
                return None
            except Exception as e:
                st.error(f"OpenAI: Failed to parse extracted data into schema for '{display_name}': {e}")
                st.code(json_string, language="json") # Show the JSON it tried to parse
                return None
        else:
            st.warning(f"OpenAI: No content received or unexpected response structure for '{display_name}'.")
            st.json(response.dict()) # Show the full response object
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
    /* General app styling */
    .stApp {
        background-color: #f0f2f6; /* Light gray background */
        color: #333333;
    }
    h1, h2, h3 {
        color: #1e3a8a; /* Dark blue for headers */
    }
    .stButton>button {
        background-color: #3b82f6; /* Blue button */
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 16px;
        font-weight: bold;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #2563eb; /* Darker blue on hover */
    }
    .stMarkdown p {
        font-size: 1.05em;
        line-height: 1.6;
    }
    /* Info/Success/Error boxes */
    .stAlert {
        border-radius: 8px;
    }
    .stAlert.info {
        background-color: #e0f2f7; /* Light blue */
        color: #0288d1;
    }
    .stAlert.success {
        background-color: #e8f5e9; /* Light green */
        color: #2e7d32;
    }
    .stAlert.error {
        background-color: #ffebee; /* Light red */
        color: #c62828;
    }
    .stProgress > div > div > div > div {
        background-color: #3b82f6 !important; /* Blue progress bar */
    }
</style>
""", unsafe_allow_html=True)


st.title("📄 AI Invoice Extractor (Multi-Model Powered)")

st.sidebar.header("Configuration")

# Model Selection
model_choice = st.sidebar.radio(
    "Choose AI Model:",
    ("Google Gemini", "OpenAI GPT"),
    key="model_choice"
)

# API Key Inputs (Conditional)
# Using st.empty() to control rendering ensures only one set of inputs is shown
api_key_placeholder = st.sidebar.empty()
model_id_placeholder = st.sidebar.empty()
model_caption_placeholder = st.sidebar.empty()

selected_api_key = None
model_id_input = None

if model_choice == "Google Gemini":
    selected_api_key = api_key_placeholder.text_input("Enter your Gemini API Key:", type="password", key="gemini_key")
    DEFAULT_GEMINI_MODEL_ID = "gemini-1.5-flash-latest"
    model_id_input = model_id_placeholder.text_input("Gemini Model ID:", DEFAULT_GEMINI_MODEL_ID, key="gemini_model_id")
    model_caption_placeholder.caption(f"Default is `{DEFAULT_GEMINI_MODEL_ID}`. Ensure it supports JSON schema.")
elif model_choice == "OpenAI GPT":
    selected_api_key = api_key_placeholder.text_input("Enter your OpenAI API Key:", type="password", key="openai_key")
    DEFAULT_OPENAI_MODEL_ID = "gpt-4o" # GPT-4o is excellent for vision + JSON
    model_id_input = model_id_placeholder.text_input("OpenAI Model ID:", DEFAULT_OPENAI_MODEL_ID, key="openai_model_id")
    model_caption_placeholder.caption(f"Default is `{DEFAULT_OPENAI_MODEL_ID}`. Ensure it's a vision model and supports JSON output.")

st.info(
    "**Instructions:**\n"
    f"1. Select your preferred AI model ({model_choice}) in the sidebar.\n"
    "2. Enter the corresponding API Key and Model ID in the sidebar.\n"
    "3. Upload one or more PDF invoice files.\n"
    "4. Click 'Process Invoices' to extract data.\n"
    "   The extracted data will be displayed in a table and available for download as Excel."
)

uploaded_files = st.file_uploader(
    "Choose PDF invoice files",
    type="pdf",
    accept_multiple_files=True
)

# Initialize session state variables if they don't exist
if 'summary_rows' not in st.session_state:
    st.session_state.summary_rows = []
if 'gemini_client' not in st.session_state:
    st.session_state.gemini_client = None
if 'openai_client' not in st.session_state:
    st.session_state.openai_client = None


if st.button("🚀 Process Invoices", type="primary"):
    if not selected_api_key:
        st.error(f"Please enter your {model_choice} API Key in the sidebar.")
    elif not uploaded_files:
        st.error("Please upload at least one PDF file.")
    elif not model_id_input:
        st.error(f"Please specify a {model_choice} Model ID in the sidebar.")
    else:
        # Initialize the correct client based on choice
        client_initialized = False
        if model_choice == "Google Gemini":
            if 'genai' in globals(): # Check if genai was successfully imported
                try:
                    st.session_state.gemini_client = genai.Client(api_key=selected_api_key)
                    st.success("Gemini client initialized successfully!")
                    client_initialized = True
                except Exception as e:
                    st.error(f"Failed to initialize Gemini client: {e}. Please check your API key.")
                    st.session_state.gemini_client = None
            else:
                st.error("Gemini library not found. Cannot initialize Gemini client.")

        elif model_choice == "OpenAI GPT":
            if 'OpenAI' in globals(): # Check if OpenAI was successfully imported
                try:
                    st.session_state.openai_client = OpenAI(api_key=selected_api_key)
                    st.success("OpenAI client initialized successfully!")
                    client_initialized = True
                except Exception as e:
                    st.error(f"Failed to initialize OpenAI client: {e}. Please check your API key.")
                    st.session_state.openai_client = None
            else:
                st.error("OpenAI library not found. Cannot initialize OpenAI client.")
        
        if not client_initialized:
            st.stop() # Stop execution if client failed to initialize or library missing

        # Proceed only if the client was successfully initialized
        if (model_choice == "Google Gemini" and st.session_state.gemini_client) or \
           (model_choice == "OpenAI GPT" and st.session_state.openai_client):
            
            st.session_state.summary_rows = [] # Clear previous results
            progress_bar = st.progress(0)
            total_files = len(uploaded_files)

            for i, uploaded_file_obj in enumerate(uploaded_files):
                st.markdown(f"---") # Separator between files
                st.info(f"Processing file: **{uploaded_file_obj.name}** ({i+1}/{total_files}) using **{model_choice}**...")
                temp_file_path = None
                extracted_data = None
                try:
                    # Save UploadedFile to a temporary file to get a file_path
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

                        # Apply formatting and handle None values for consistent display
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

                        # Store data for summary table
                        st.session_state.summary_rows.append({
                            "File Name": uploaded_file_obj.name,
                            "Invoice Number": extracted_data.invoice_number,
                            "Date": parsed_date,
                            "Seller Name": seller_name_display,
                            "Seller GSTIN": seller_gstin_display,
                            "Buyer Name": buyer_name_display,
                            "Buyer GSTIN": buyer_gstin_display,
                            "Total Gross Worth": total_gross_worth, # Keep raw for Excel
                            "CGST": cgst,
                            "SGST": sgst,
                            "IGST": igst,
                            "Place of Supply": pos,
                            "Expense Ledger": expense_ledger_display,
                            "TDS": tds_display,
                            "RCM Applicability": rcm_display,
                            "Narration": narration,
                        })

                        # Display detailed extraction for the current file using an expander
                        with st.expander(f"📋 Details for {uploaded_file_obj.name} (using {model_choice})"):
                            # This will show the Pydantic object's dictionary representation
                            st.subheader("Raw Extracted Data (JSON):")
                            st.json(extracted_data.dict())
                            st.subheader("Extracted Summary (Narration):")
                            st.markdown(narration)

                            if extracted_data.line_items:
                                st.subheader("Line Items:")
                                line_item_data = [{
                                    "Description": item.description,
                                    "Quantity": item.quantity,
                                    "Gross Worth": format_currency(item.gross_worth), # Format for display
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
                    # Clean up: Delete the temporary local file
                    if temp_file_path and os.path.exists(temp_file_path):
                        os.unlink(temp_file_path)
                        # st.write(f"Cleaned up temporary local file: `{temp_file_path}`") # Can be noisy, uncomment for debug
                progress_bar.progress((i + 1) / total_files)

            st.markdown(f"---")
            if st.session_state.summary_rows:
                st.balloons() # Celebrate successful batch processing!
                st.success("All selected invoices processed!")


if st.session_state.summary_rows:
    st.subheader("📊 Consolidated Extracted Invoice Summary")
    df = pd.DataFrame(st.session_state.summary_rows)

    # Make a copy for display to apply string formatting without affecting raw numbers for Excel
    df_display = df.copy()
    for col in ["Total Gross Worth", "CGST", "SGST", "IGST"]:
        df_display[col] = df_display[col].apply(format_currency)

    st.dataframe(df_display, use_container_width=True)

    # Provide download link for Excel
    output_excel = io.BytesIO()
    # Use the original df which has numerical values for excel export
    with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
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
