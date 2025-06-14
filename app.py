import streamlit as st
from pydantic import BaseModel, Field
from typing import List, Optional
import pandas as pd
import os
import tempfile # For temporary file handling
import io # For Excel download
import re # For GSTIN validation
import json # For handling JSON response

# Try to import google.generativeai, show error if not found
try:
    from google import generativeai as genai
except ImportError:
    st.error("The 'google-generativeai' library is not installed. Please install it by running: pip install google-generativeai")
    st.stop()

# --- Pydantic Models (as provided) ---
class LineItem(BaseModel):
    description: str
    quantity: float
    gross_worth: float

class Invoice(BaseModel):
    invoice_number: str
    date: str
    gstin: str # This is seller GSTIN
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

# --- GSTIN Validation Function ---
def is_valid_gstin(gstin):
    if not gstin:
        return False
    # Clean non-alphanumeric characters and convert to uppercase
    cleaned = re.sub(r'[^A-Z0-9]', '', gstin.upper())
    if len(cleaned) != 15:
        return False
    # Standard Indian GSTIN pattern
    # Format: 2-digit State Code + 10-char PAN + 1-char Entity Code + 1-char Checksum + 1-char 'Z' + 1-char Checksum
    pattern = r"^\d{2}[A-Z0-9]{10}[A-Z]{1}[A-Z0-9]{1}[Z]{1}[A-Z0-9]{1}$"
    return bool(re.match(pattern, cleaned))

# --- Gemini API Interaction Function ---
def extract_structured_data(
    client_instance, # The genai.Client object
    gemini_model_id: str,
    file_path: str,
    pydantic_schema: BaseModel,
):
    display_name = os.path.basename(file_path)
    gemini_file_resource = None # To store the Gemini File API object for deletion

    try:
        # 1. Upload the file to the File API
        st.write(f"Uploading '{display_name}' to Gemini File API...")
        gemini_file_resource = client_instance.files.upload(
            file=file_path,
            config={'display_name': display_name.split('.')[0]}
        )
        st.write(f"'{display_name}' uploaded. Gemini file name: {gemini_file_resource.name}")

        # 2. Generate a structured response using the Gemini API
        prompt = (
            "Extract all relevant and clear information from the invoice, adhering to Indian standards "
            "for dates (DD/MM/YYYY or DD-MM-YYYY) and codes (like GSTIN, HSN/SAC). "
            "Accurately identify the total amount payable. Classify the nature of expense and suggest an "
            "applicable ledger type (e.g., 'Office Supplies', 'Professional Fees', 'Software Subscription'). "
            "Determine TDS applicability (e.g., 'Yes - Section 194J', 'No', 'Uncertain'). "
            "Determine reverse charge GST (RCM) applicability (e.g., 'Yes', 'No', 'Uncertain'). "
            "Handle missing data appropriately by setting fields to null or an empty string where "
            "Optional, and raise an issue if critical data is missing for required fields. "
            "Do not make assumptions or perform calculations beyond what's explicitly stated in the invoice text. "
            "If a value is clearly zero, represent it as 0.0 for floats. For dates, prefer DD/MM/YYYY."
        )
        st.write(f"Sending '{display_name}' to Gemini model '{gemini_model_id}' for extraction...")
        
        # Call the model with response_schema
        response = client_instance.models.generate_content(
            model=gemini_model_id,
            contents=[prompt, gemini_file_resource],
            tool_config={"response_schema": pydantic_schema} # Correct parameter for response schema
        )

        st.write(f"Data extracted for '{display_name}'.")
        # Attempt to parse the structured response from the Gemini model output
        if response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'function_call') and part.function_call and hasattr(part.function_call, 'args'):
                    # For structured output using response_schema, the data is typically in function_call.args
                    try:
                        return pydantic_schema.model_validate(part.function_call.args)
                    except Exception as validation_e:
                        st.error(f"Pydantic validation error for {display_name}: {validation_e}")
                        return None
        st.error(f"Could not find structured content in Gemini response for {display_name}. Full response: {response.text}")
        return None

    except Exception as e:
        st.error(f"Error processing '{display_name}' with Gemini: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None
    finally:
        # 3. Clean up: Delete the file from Gemini File API if it was uploaded
        if gemini_file_resource and hasattr(client_instance, 'files') and hasattr(client_instance.files, 'delete'):
            try:
                st.write(f"Attempting to delete '{gemini_file_resource.name}' from Gemini File API...")
                client_instance.files.delete(name=gemini_file_resource.name) # Assumes this method exists
                st.write(f"Successfully deleted '{gemini_file_resource.name}' from Gemini.")
            except Exception as e_del:
                st.warning(f"Could not delete '{gemini_file_resource.name}' from Gemini File API: {e_del}")
        elif gemini_file_resource:
            st.warning(f"Could not determine how to delete Gemini file '{gemini_file_resource.name}'. "
                        "Manual cleanup may be required in your Gemini project console.")


# --- Streamlit App UI ---
st.set_page_config(layout="wide")
st.title("📄 PDF Invoice Extractor (Gemini AI)")

st.sidebar.header("Configuration")
api_key_input = st.sidebar.text_input("Enter your Gemini API Key:", type="password")

# Use a known good model, but allow override if user is sure about "gemini-2.0-flash"
DEFAULT_GEMINI_MODEL_ID = "gemini-1.5-flash-latest" # Changed to a publicly available model
gemini_model_id_input = st.sidebar.text_input("Gemini Model ID for Extraction:", DEFAULT_GEMINI_MODEL_ID)
st.sidebar.caption(f"Default is `{DEFAULT_GEMINI_MODEL_ID}`. Ensure the model ID is correct and supports schema-based JSON output.")


st.info(
    "**Instructions:**\n"
    "1. Enter your Gemini API Key in the sidebar.\n"
    "2. Optionally, change the Gemini Model ID if needed.\n"
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
if 'client' not in st.session_state:
    st.session_state.client = None


if st.button("🚀 Process Invoices", type="primary"):
    if not api_key_input:
        st.error("Please enter your Gemini API Key in the sidebar.")
    elif not uploaded_files:
        st.error("Please upload at least one PDF file.")
    elif not gemini_model_id_input:
        st.error("Please specify a Gemini Model ID in the sidebar.")
    else:
        try:
            # Initialize client here, as API key might change or processing is requested
            st.session_state.client = genai.Client(api_key=api_key_input)
            st.success("Gemini client initialized successfully with the provided API Key.")
        except Exception as e:
            st.error(f"Failed to initialize Gemini client: {e}")
            st.session_state.client = None # Reset client on failure

        if st.session_state.client:
            st.session_state.summary_rows = [] # Clear previous results
            progress_bar = st.progress(0)
            total_files = len(uploaded_files)

            for i, uploaded_file_obj in enumerate(uploaded_files):
                st.markdown(f"---")
                st.info(f"Processing file: {uploaded_file_obj.name} ({i+1}/{total_files})")
                temp_file_path = None
                try:
                    # Save UploadedFile to a temporary file to get a file_path
                    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file_obj.name)[1]) as tmp:
                        tmp.write(uploaded_file_obj.getvalue())
                        temp_file_path = tmp.name

                    with st.spinner(f"Extracting data from {uploaded_file_obj.name}..."):
                        extracted_data = extract_structured_data(
                            client_instance=st.session_state.client,
                            gemini_model_id=gemini_model_id_input,
                            file_path=temp_file_path,
                            pydantic_schema=Invoice # The Pydantic model for structuring output
                        )

                    if extracted_data:
                        # --- GSTIN Validation Logic ---
                        seller_gstin_raw = extracted_data.gstin
                        buyer_gstin_raw = extracted_data.buyer_gstin
                        
                        seller_gstin_validated = None
                        buyer_gstin_validated = None
                        
                        gstin_narration_additions = []

                        if seller_gstin_raw and is_valid_gstin(seller_gstin_raw):
                            seller_gstin_validated = seller_gstin_raw
                        else:
                            seller_gstin_validated = None # Set to None if extracted but invalid
                            if seller_gstin_raw:
                                gstin_narration_additions.append(f"Seller GSTIN '{seller_gstin_raw}' found but is invalid.")
                            else:
                                gstin_narration_additions.append("Seller GSTIN not found or not extracted.")

                        if buyer_gstin_raw and is_valid_gstin(buyer_gstin_raw):
                            buyer_gstin_validated = buyer_gstin_raw
                        else:
                            buyer_gstin_validated = None # Set to None if extracted but invalid
                            if buyer_gstin_raw:
                                gstin_narration_additions.append(f"Buyer GSTIN '{buyer_gstin_raw}' found but is invalid.")
                            # No addition to narration if buyer GSTIN is genuinely missing (as it's Optional)

                        st.success(f"Successfully extracted data from {uploaded_file_obj.name}")
                        cgst = extracted_data.cgst if extracted_data.cgst is not None else 0.0
                        sgst = extracted_data.sgst if extracted_data.sgst is not None else 0.0
                        igst = extracted_data.igst if extracted_data.igst is not None else 0.0
                        pos = extracted_data.place_of_supply if extracted_data.place_of_supply else "N/A"
                        
                        # Use validated GSTINs for display and narration
                        seller_gstin_display = seller_gstin_validated if seller_gstin_validated else "N/A (Invalid/Missing)"
                        buyer_gstin_display = buyer_gstin_validated if buyer_gstin_validated else "N/A (Invalid/Missing)"

                        narration = (
                            f"Invoice {extracted_data.invoice_number} dated {extracted_data.date} "
                            f"was issued by {extracted_data.seller_name} (GSTIN: {seller_gstin_display}) "
                            f"to {extracted_data.buyer_name} (GSTIN: {buyer_gstin_display}), "
                            f"with a total value of ₹{extracted_data.total_gross_worth:.2f}. "
                            f"Taxes applied - CGST: ₹{cgst:.2f}, SGST: ₹{sgst:.2f}, IGST: ₹{igst:.2f}. "
                            f"Place of supply: {pos}. Expense: {extracted_data.expense_ledger or 'N/A'}. "
                            f"TDS: {extracted_data.tds or 'N/A'}."
                        )
                        if gstin_narration_additions:
                            narration += " " + " ".join(gstin_narration_additions)

                        st.session_state.summary_rows.append({
                            "File Name": uploaded_file_obj.name,
                            "Invoice Number": extracted_data.invoice_number,
                            "Date": extracted_data.date,
                            "Seller Name": extracted_data.seller_name,
                            "Seller GSTIN": seller_gstin_validated, # Store validated GSTIN
                            "Buyer Name": extracted_data.buyer_name,
                            "Buyer GSTIN": buyer_gstin_validated, # Store validated GSTIN
                            "Total Gross Worth": extracted_data.total_gross_worth,
                            "CGST": cgst,
                            "SGST": sgst,
                            "IGST": igst,
                            "Place of Supply": pos,
                            "Expense Ledger": extracted_data.expense_ledger,
                            "TDS": extracted_data.tds,
                            "Narration": narration,
                        })
                    else:
                        st.warning(f"Failed to extract data or no data returned for {uploaded_file_obj.name}")

                except Exception as e_outer:
                    st.error(f"An unexpected error occurred while processing {uploaded_file_obj.name}: {e_outer}")
                finally:
                    # Clean up: Delete the temporary local file
                    if temp_file_path and os.path.exists(temp_file_path):
                        os.unlink(temp_file_path)
                        st.write(f"Deleted temporary local file: {temp_file_path}")
                progress_bar.progress((i + 1) / total_files)

            st.markdown(f"---")
            if st.session_state.summary_rows:
                st.balloons()


if st.session_state.summary_rows:
    st.subheader("📊 Extracted Invoice Summary")
    df = pd.DataFrame(st.session_state.summary_rows)
    st.dataframe(df)

    # Provide download link for Excel
    output_excel = io.BytesIO()
    with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='InvoiceSummary')
    excel_data = output_excel.getvalue()

    st.download_button(
        label="📥 Download Summary as Excel",
        data=excel_data,
        file_name="invoice_summary.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
elif not uploaded_files:
     st.info("Upload PDF files and click 'Process Invoices' to see results.")
