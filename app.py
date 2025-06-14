import streamlit as st
from openai import OpenAI
import json
import re
import pandas as pd
import io

# Initialize OpenAI Client
# Ensure you have your OpenAI API key set as an environment variable (OPENAI_API_KEY)
# or replace os.getenv("OPENAI_API_KEY") with your actual key (not recommended for production)
client = OpenAI()

# --- Utility Functions ---

def is_valid_gstin(gstin):
    """
    Validates an Indian GSTIN with a more commonly accepted regex,
    allowing for variations in the 13th character if it's alphanumeric.
    """
    if not gstin:
        return False
    # Remove any spaces, dashes, or other non-alphanumeric chars
    cleaned = re.sub(r'[^A-Z0-9]', '', gstin.upper())

    # Pattern: DD + AAAAA + DDDD + A + [1-9A-Z] + Z + [0-9A-Z]
    # This specifically looks for the PAN structure within the GSTIN.
    # The 13th character is usually 1-9 for regular registrants.
    # We allow A-Z as well to be slightly more robust to OCR errors or less common cases.
    pattern = r"^\d{2}[A-Z]{5}\d{4}[A-Z]{1}[1-9A-Z]{1}Z[0-9A-Z]{1}$"

    return bool(re.match(pattern, cleaned))

def extract_gstin_from_text(text):
    """
    Extracts and validates GSTINs from a given text.
    Prioritizes the first valid 15-character GSTIN found.
    """
    if not text:
        return ""
    # Normalize the text by removing common separators like spaces, hyphens, and slashes
    cleaned_text = re.sub(r'[\s\-/]', '', text.upper())

    # Look for any 15-character alphanumeric sequence that might be a GSTIN.
    # This pattern is intentionally broad to *capture* potential GSTINs,
    # which will then be *strictly validated* by is_valid_gstin.
    matches = re.findall(r'[A-Z0-9]{15}', cleaned_text)

    for match in matches:
        if is_valid_gstin(match):
            return match
    return ""

def extract_json_from_response(response_text):
    """
    Extracts a JSON object from a given text, looking for content within triple backticks.
    """
    try:
        # Find content within triple backticks (```json ... ```)
        match = re.search(r"```json\n(.*?)```", response_text, re.DOTALL)
        if match:
            json_str = match.group(1)
            return json.loads(json_str)
        else:
            # If no triple backticks, try to parse the whole string as JSON
            return json.loads(response_text)
    except json.JSONDecodeError as e:
        st.error(f"Failed to decode JSON from response: {e}")
        st.text(f"Problematic response text: {response_text}")
        return None

# --- Main Prompt for GPT-4o ---
main_prompt = (
    "You are an expert at extracting structured data from Indian invoices. "
    "Your primary goal is to extract information into a JSON object with the specified keys. "
    "If a key's value is not explicitly present or derivable from the invoice, use `null` for that value. "
    "**ABSOLUTELY CRITICAL: All extracted GSTINs (`gstin` and `buyer_gstin`) MUST strictly adhere to the 15-character Indian GSTIN format.** "
    "**If any extracted GSTIN does not precisely match the described format, you MUST set its value to `null`. Do NOT infer, guess, or provide partial/incorrect GSTINs.**\n"
    
    "Keys to extract: `invoice_number`, `date`, `gstin` (seller's GSTIN), `seller_name`, `buyer_name`, `buyer_gstin`, "
    "`taxable_amount`, `cgst`, `sgst`, `igst`, `place_of_supply`, `expense_ledger`, `tds`, `hsn_sac`. "
    
    "GUIDELINES FOR EXTRACTION:\n"
    "- 'invoice_number': The unique identifier of the invoice. Extract as is. If multiple are present, prioritize the most prominent one, usually near 'Invoice No.' or 'Bill No.'.\n"
    "- 'date': The invoice date in **DD/MM/YYYY** format. If year is 2-digit, assume current century (e.g., 24 -> 2024). Always try to extract a full date.\n"
    "- 'taxable_amount': This is the subtotal, the amount BEFORE any taxes (CGST, SGST, IGST) are applied. Must be a number (float). Look for terms like 'Sub Total', 'Taxable Value', 'Net Amount'.\n"
    
    "- 'gstin': **SELLER'S GSTIN.** This is the Goods and Services Tax Identification Number of the entity ISSUING the invoice. "
    "  It MUST be a **15-character alphanumeric string** with the following structure:\n"
    "  - First 2 digits: State Code (e.g., 27 for Maharashtra, 07 for Delhi).\n"
    "  - Next 10 characters: PAN (Permanent Account Number) of the entity, typically 5 letters, 4 digits, 1 letter (e.g., ABCDE1234F).\n"
    "  - 13th character: Entity registration number for the same PAN (usually 1-9 or A-Z).\n"
    "  - 14th character: Fixed 'Z'.\n"
    "  - 15th character: Checksum character (digit or letter).\n"
    "  **Example VALID GSTINs: '27AAAAA0000A1Z1', '07BBBBB0000B1Z2'.**\n"
    "  Search for labels like 'GSTIN', 'GST No.', 'GST Registration No.' near the seller's name or address. "
    "  **IF THE EXTRACTED STRING DOES NOT EXACTLY MATCH THIS 15-CHARACTER VALID FORMAT (after removing spaces/hyphens), SET `gstin` to `null`.**\n"
    
    "- 'buyer_gstin': **BUYER'S GSTIN.** This is the GSTIN of the entity RECEIVING the invoice. "
    "  It MUST also be a **15-character alphanumeric string** following the exact same validation rules as the seller's GSTIN. "
    "  Look for labels like 'Buyer GSTIN', 'Recipient GSTIN', 'GSTIN of Buyer', usually near the buyer's name or 'Bill To'/'Ship To' address. "
    "  **IF THE EXTRACTED STRING DOES NOT EXACTLY MATCH THIS 15-CHARACTER VALID FORMAT, SET `buyer_gstin` to `null`.**\n"
    
    "- 'hsn_sac': Crucial for Indian invoices. "
    "  - HSN (Harmonized System of Nomenclature) is for goods. SAC (Service Accounting Code) is for services."
    "  - **ONLY extract the HSN/SAC code if it is EXPLICITLY mentioned on the invoice.** "
    "  - It is typically a 4, 6, or 8-digit numeric code, sometimes alphanumeric (e.g., '998313', '8471')."
    "  - Look for labels like 'HSN Code', 'SAC Code', 'HSN/SAC', or just the code itself near item descriptions."
    "  - If multiple HSN/SAC codes are present for different line items, extract the one that appears most prominently, or the first one listed. If only one is present for the whole invoice, use that."
    "  - **If HSN/SAC is NOT found or explicitly stated, the value MUST be `null`. Do NOT guess or infer it.**\n"
    
    "- 'expense_ledger': Classify the nature of expense and suggest a suitable ledger type. "
    "  Examples: 'Office Supplies', 'Professional Fees', 'Software Subscription', 'Rent', "
    "  'Cloud Services', 'Marketing Expenses', 'Travel Expenses'. "
    "  For invoices from cloud providers (e.g., 'Google Cloud', 'AWS', 'Microsoft Azure', 'DigitalOcean'), classify as 'Cloud Services'."
    "  If the expense is clearly related to software licenses, subscriptions, or SaaS, classify as 'Software Subscription'."
    "  Aim for a general and universal ledger type if a precise one isn't obvious from the invoice details.\n"
    
    "- 'tds': Determine TDS applicability. State 'Yes - Section [X]' if applicable with a section (e.g., 'Yes - Section 194J', 'Yes - Section 194C', 'Yes - Section 194I'), 'No' if clearly not, or 'Uncertain' if unclear. Always try to identify the TDS Section (e.g., 194J, 194C, 194I) if TDS is applicable.\n"
    
    "- 'place_of_supply': Crucial for Indian invoices to determine IGST applicability. "
    "  - **PRIORITY 1:** Look for a field explicitly labeled 'Place of Supply'. Extract the exact State/City name from this field (e.g., 'Delhi', 'Maharashtra')."
    "  - **PRIORITY 2:** If 'Place of Supply' is not found, look for a 'Ship To:' address. Extract ONLY the State/City name from this address."
    "  - **PRIORITY 3:** If 'Ship To:' is not found, look for a 'Bill To:' address. Extract ONLY the State/City name from this address."
    "  - **PRIORITY 4:** If neither of the above, infer from the Customer/Buyer Address. Extract ONLY the State/City name from this address."
    "  - **SPECIAL CASE:** If the invoice text or context clearly indicates an export or foreign transaction (e.g., 'Export Invoice', mentions 'Foreign' address, non-Indian currency as primary total, or foreign recipient details), set the value to 'Foreign'."
    "  - **FALLBACK:** If none of the above are found or inferable, the value MUST be `null`."
    
    "Return 'NOT AN INVOICE' if the document is clearly not an invoice."
    "The output MUST be a JSON object, clearly formatted, and parsable. Wrap the JSON in triple backticks (```json...```)."
    "Example of desired output structure with valid and null GSTINs:\n"
    "```json\n"
    "{\n"
    '  "invoice_number": "INV-2024-001",\n'
    '  "date": "15/05/2024",\n'
    '  "gstin": "27AAAAA0000A1Z1",\n'
    '  "seller_name": "Tech Solutions Pvt Ltd",\n'
    '  "buyer_name": "Acme Corp",\n'
    '  "buyer_gstin": "07BBBBB0000B1Z2",\n'
    '  "taxable_amount": 1000.00,\n'
    '  "cgst": 90.00,\n'
    '  "sgst": 90.00,\n'
    '  "igst": null,\n'
    '  "place_of_supply": "Delhi",\n'
    '  "expense_ledger": "Software Subscription",\n'
    '  "tds": "Yes - Section 194J",\n'
    '  "hsn_sac": "998313"\n'
    "}\n"
    "```\n"
    "**IMPORTANT:** If a GSTIN is present on the invoice but appears malformed or not in the exact 15-character format, you must still output `null` for that specific GSTIN field. Prioritize the strict format validation over imperfect OCR results. If the invoice mentions 'GSTIN: Not Applicable' or equivalent, set the GSTIN field to `null`."
    "Ensure all monetary values are floats with two decimal places if applicable."
)

# --- Streamlit UI ---
st.set_page_config(layout="wide", page_title="Invoice Data Extractor (Indian Invoices)")

st.title("📄 Indian Invoice Data Extractor")
st.markdown(
    """
    Upload your Indian invoices (PDFs or images) to automatically extract key financial data.
    The tool uses advanced AI to identify and structure information like GSTINs, invoice numbers, amounts, and more.
    """
)

uploaded_files = st.file_uploader(
    "Upload PDF or Image Invoices (Multiple files allowed)",
    type=["pdf", "jpg", "jpeg", "png"],
    accept_multiple_files=True
)

extracted_data_list = []

if uploaded_files:
    progress_bar = st.progress(0)
    for i, uploaded_file in enumerate(uploaded_files):
        file_name = uploaded_file.name
        file_type = uploaded_file.type

        st.subheader(f"Processing: {file_name}")

        try:
            # Prepare the file for OpenAI Vision API
            if "pdf" in file_type:
                # OpenAI Vision API can directly handle PDF bytes
                file_content = uploaded_file.read()
                mime_type = "application/pdf"
            elif "image" in file_type:
                file_content = uploaded_file.read()
                mime_type = file_type
            else:
                st.warning(f"Unsupported file type for {file_name}: {file_type}. Skipping.")
                continue

            # Construct the chat prompt for OpenAI Vision
            chat_prompt = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": main_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{base64.b64encode(file_content).decode('utf-8')}"
                            },
                        },
                    ],
                }
            ]

            # Make the API call
            with st.spinner(f"Extracting data from {file_name}..."):
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=chat_prompt,
                    max_tokens=1500
                )

                response_text = response.choices[0].message.content.strip()

                # --- DEBUGGING OUTPUT ---
                st.text_area(f"Raw GPT-4o Response for {file_name}", response_text, height=300)
                # --- END DEBUGGING OUTPUT ---

                raw_data = extract_json_from_response(response_text)

                if raw_data is None or raw_data == "NOT AN INVOICE":
                    st.warning(f"Could not extract structured data or '{file_name}' is not an invoice.")
                    # Add a placeholder for files that couldn't be processed
                    extracted_data_list.append({
                        "File Name": file_name,
                        "Invoice Number": "N/A",
                        "Date": "N/A",
                        "Seller GSTIN": "N/A",
                        "Seller Name": "N/A",
                        "Buyer Name": "N/A",
                        "Buyer GSTIN": "N/A",
                        "Taxable Amount": "N/A",
                        "CGST": "N/A",
                        "SGST": "N/A",
                        "IGST": "N/A",
                        "Place of Supply": "N/A",
                        "Expense Ledger": "N/A",
                        "TDS": "N/A",
                        "HSN/SAC": "N/A",
                        "Status": "Failed/Not Invoice"
                    })
                else:
                    # Extract and validate fields
                    invoice_number = raw_data.get("invoice_number", "")
                    date = raw_data.get("date", "")
                    seller_name = raw_data.get("seller_name", "")

                    extracted_seller_gstin = raw_data.get("gstin", "")
                    seller_gstin = extracted_seller_gstin if is_valid_gstin(extracted_seller_gstin) else "" # Validate

                    buyer_name = raw_data.get("buyer_name", "")
                    extracted_buyer_gstin = raw_data.get("buyer_gstin", "")
                    buyer_gstin = extracted_buyer_gstin if is_valid_gstin(extracted_buyer_gstin) else "" # Validate

                    taxable_amount = raw_data.get("taxable_amount", None)
                    cgst = raw_data.get("cgst", None)
                    sgst = raw_data.get("sgst", None)
                    igst = raw_data.get("igst", None)
                    place_of_supply = raw_data.get("place_of_supply", "")
                    expense_ledger = raw_data.get("expense_ledger", "")
                    tds = raw_data.get("tds", "")
                    hsn_sac = raw_data.get("hsn_sac", "")

                    extracted_data_list.append({
                        "File Name": file_name,
                        "Invoice Number": invoice_number,
                        "Date": date,
                        "Seller GSTIN": seller_gstin,
                        "Seller Name": seller_name,
                        "Buyer Name": buyer_name,
                        "Buyer GSTIN": buyer_gstin,
                        "Taxable Amount": taxable_amount,
                        "CGST": cgst,
                        "SGST": sgst,
                        "IGST": igst,
                        "Place of Supply": place_of_supply,
                        "Expense Ledger": expense_ledger,
                        "TDS": tds,
                        "HSN/SAC": hsn_sac,
                        "Status": "Success"
                    })
            progress_bar.progress((i + 1) / len(uploaded_files))

        except Exception as e:
            st.error(f"An error occurred while processing {file_name}: {e}")
            extracted_data_list.append({
                "File Name": file_name,
                "Invoice Number": "N/A",
                "Date": "N/A",
                "Seller GSTIN": "N/A",
                "Seller Name": "N/A",
                "Buyer Name": "N/A",
                "Buyer GSTIN": "N/A",
                "Taxable Amount": "N/A",
                "CGST": "N/A",
                "SGST": "N/A",
                "IGST": "N/A",
                "Place of Supply": "N/A",
                "Expense Ledger": "N/A",
                "TDS": "N/A",
                "HSN/SAC": "N/A",
                "Status": f"Error: {e}"
            })
            progress_bar.progress((i + 1) / len(uploaded_files))

    if extracted_data_list:
        st.success("All files processed!")
        df = pd.DataFrame(extracted_data_list)
        st.dataframe(df)

        # Option to download as CSV
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="Download Data as CSV",
            data=csv_buffer.getvalue(),
            file_name="extracted_invoice_data.csv",
            mime="text/csv",
        )

st.markdown(
    """
    ---
    **Note on API Key:** This application requires an OpenAI API key.
    It's recommended to set it as an environment variable (`OPENAI_API_KEY`) for security.
    """
)
