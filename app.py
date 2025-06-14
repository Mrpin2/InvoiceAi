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

locale.setlocale(locale.LC_ALL, '')

# Lottie animation URLs
hello_lottie = "https://raw.githubusercontent.com/Mrpin2/InvoiceAi/refs/heads/main/Animation%20-%201749845212531.json"
completed_lottie = "https://raw.githubusercontent.com/Mrpin2/InvoiceAi/refs/heads/main/Animation%20-%201749845303699.json"

def load_lottie_json_safe(url):
    """Safely load a Lottie JSON from a URL, returning None on failure."""
    try:
        r = requests.get(url)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None

# Load animations
hello_json = load_lottie_json_safe(hello_lottie)
completed_json = load_lottie_json_safe(completed_lottie)

# Display initial animation if no files are uploaded yet
if "files_uploaded" not in st.session_state:
    if hello_json:
        st_lottie(hello_json, height=200, key="hello")

# --- App Title ---
st.markdown("<h2 style='text-align: center;'>📄 AI Invoice Extractor (OpenAI)</h2>", unsafe_allow_html=True)
st.markdown("Upload scanned PDF invoices and extract structured finance data using GPT-4o.")
st.markdown("---")


# --- Session State Initialization ---
if "processed_results" not in st.session_state:
    st.session_state["processed_results"] = {}
if "processing_status" not in st.session_state:
    st.session_state["processing_status"] = {}

# --- Sidebar for API Key Configuration ---
st.sidebar.header("🔐 AI Config")
passcode = st.sidebar.text_input("Admin Passcode", type="password")
admin_unlocked = passcode == "Rajeev" # Replace "Rajeev" with your desired passcode

openai_api_key = None
if admin_unlocked:
    st.sidebar.success("🔓 Admin access granted.")
    openai_api_key = st.secrets.get("OPENAI_API_KEY")
    if not openai_api_key:
        st.sidebar.error("OPENAI_API_KEY is missing from Streamlit secrets.")
        st.stop()
else:
    openai_api_key = st.sidebar.text_input("🔑 Enter your OpenAI API Key", type="password")
    if not openai_api_key:
        st.sidebar.warning("Please enter your API key to proceed.")
        st.stop()

# Initialize OpenAI client
client = OpenAI(api_key=openai_api_key)

# --- Core Functions ---

def convert_pdf_first_page(pdf_bytes):
    """Convert the first page of a PDF to a high-resolution PNG image."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc.load_page(0)
    pix = page.get_pixmap(dpi=300)
    return Image.open(io.BytesIO(pix.tobytes("png")))

def safe_float(x):
    """Safely convert a value to float, cleaning currency symbols and commas."""
    try:
        cleaned = str(x).replace(",", "").replace("₹", "").replace("$", "").strip()
        return float(cleaned) if cleaned else 0.0
    except (ValueError, TypeError):
        return 0.0

def format_currency(x):
    """Format a number as Indian Rupees (₹)."""
    try:
        return f"₹{safe_float(x):,.2f}"
    except Exception:
        return "₹0.00"

def is_valid_gstin(gstin):
    """Validate a 15-character GSTIN using a regex pattern."""
    if not isinstance(gstin, str) or len(gstin) != 15:
        return False
    pattern = r"^\d{2}[A-Z]{5}\d{4}[A-Z]{1}[1-9A-Z]{1}Z[0-9A-Z]{1}$"
    return bool(re.match(pattern, gstin))

def extract_gstin_from_text(text):
    """Extract the first valid GSTIN pattern found in a block of text."""
    matches = re.findall(r'\b\d{2}[A-Z]{5}\d{4}[A-Z]{1}[1-9A-Z]{1}Z[0-9A-Z]{1}\b', text.upper())
    return matches[0] if matches else ""

def determine_tds_rate(expense_ledger, tds_str=""):
    """Determine the TDS rate based on keywords in the expense ledger or a TDS string."""
    if tds_str and isinstance(tds_str, str):
        match = re.search(r'(\d+(\.\d+)?)%', tds_str)
        if match:
            return float(match.group(1))
        
        section_rates = {"194j": 10.0, "194c": 2.0, "194i": 10.0, "194h": 5.0, "194q": 0.1}
        for section, rate in section_rates.items():
            if section in tds_str.lower():
                return rate

    expense_ledger = (expense_ledger or "").lower()
    if any(keyword in expense_ledger for keyword in ["professional", "consultancy", "service"]):
        return 10.0  # 194J
    if "contract" in expense_ledger:
        return 1.0   # 194C
    if "rent" in expense_ledger:
        return 10.0  # 194I
    return 0.0

def determine_tds_section(expense_ledger):
    """Determine the TDS section based on the expense ledger."""
    expense_ledger = (expense_ledger or "").lower()
    if any(keyword in expense_ledger for keyword in ["professional", "consultancy", "service"]):
        return "194J"
    if "contract" in expense_ledger:
        return "194C"
    if "rent" in expense_ledger:
        return "194I"
    return None

def extract_json_from_response(text):
    """Extract a JSON object from a string, even if it's embedded in text or code blocks."""
    if not text: return None
    try:
        match = re.search(r'```json\s*({.*?})\s*```', text, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1:
            return json.loads(text[start:end+1])
        
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None

# --- Main AI Prompt ---
# This is the updated prompt with enhanced logic for "Place of Supply" including exports
main_prompt = (
    "You are an expert at extracting structured data from Indian invoices. "
    "Your task is to analyze the invoice image and return a clean, parsable JSON object. "
    "If a value is not found, use `null`. "
    "Keys to extract: invoice_number, date, gstin (seller's), seller_name, buyer_name, buyer_gstin, "
    "taxable_amount, cgst, sgst, igst, place_of_supply, expense_ledger, tds, hsn_sac.\n\n"
    
    "GUIDELINES:\n"
    "- 'date': Extract the invoice date in DD/MM/YYYY format.\n"
    "- 'taxable_amount': The subtotal amount *before* any taxes (CGST, SGST, IGST).\n"
    "- 'gstin': The seller's GSTIN. For export invoices, the buyer will not have a GSTIN.\n"
    "- 'buyer_gstin': The buyer's GSTIN. This will likely be `null` for an export invoice.\n"
    "- 'hsn_sac': Extract the HSN or SAC code *only* if explicitly mentioned. It's usually a 4-8 digit code. If not found, it *must* be `null`.\n"
    "- 'expense_ledger': Classify the expense (e.g., 'Professional Fees', 'Software Subscription', 'Goods Exported').\n"
    "- 'tds': State 'Yes - Section [X]' if applicable (e.g., 'Yes - Section 194J'), 'No' if not, or 'Uncertain'. TDS is generally not applicable on export invoices.\n"
    "- **'place_of_supply'**: This is crucial. First, look for a field explicitly labeled 'Place of Supply'. "
    "  If not present, infer it from the buyer's address (under 'Bill to:', 'Buyer:', etc.). "
    "  **If the buyer's address is clearly outside of India, this is an export invoice. In this case, set the place of supply to the buyer's country (e.g., 'USA', 'Germany'). If the specific country is not clear but the invoice is for export, use the term 'Foreign'.**"
)

# --- File Uploader and Processing Logic ---
uploaded_files = st.file_uploader(
    "📤 Upload scanned invoice PDFs",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    st.session_state["files_uploaded"] = True
    total_files = len(uploaded_files)

    for idx, file in enumerate(uploaded_files):
        file_name = file.name
        if file_name in st.session_state["processed_results"]:
            continue

        st.markdown(f"**Processing file: {file_name} ({idx+1}/{total_files})**")
        st.session_state["processing_status"][file_name] = "⏳ Pending..."
        
        response_text = None
        try:
            # Convert PDF to image for processing
            pdf_bytes = file.getvalue()
            first_image = convert_pdf_first_page(pdf_bytes)

            with st.spinner("🧠 Extracting data using GPT-4o Vision..."):
                img_buf = io.BytesIO()
                first_image.save(img_buf, format="PNG")
                base64_image = base64.b64encode(img_buf.getvalue()).decode()

                chat_prompt = [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": main_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                    ]
                }]

                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=chat_prompt,
                    max_tokens=2000,
                    response_format={"type": "json_object"} # Use JSON mode
                )
                
                response_text = response.choices[0].message.content.strip()
                raw_data = extract_json_from_response(response_text)

                if raw_data is None:
                    raise ValueError(f"Failed to parse JSON from GPT's response.")
                
                # --- Data Cleaning and Structuring ---
                taxable_amount = safe_float(raw_data.get("taxable_amount"))
                cgst = safe_float(raw_data.get("cgst"))
                sgst = safe_float(raw_data.get("sgst"))
                igst = safe_float(raw_data.get("igst"))
                expense_ledger = raw_data.get("expense_ledger")
                tds_str = raw_data.get("tds", "")
                
                seller_gstin = (raw_data.get("gstin") or "").upper().strip()
                if not is_valid_gstin(seller_gstin):
                    seller_gstin = extract_gstin_from_text(str(raw_data.get("seller_name", "")) + " " + seller_gstin)

                buyer_gstin = (raw_data.get("buyer_gstin") or "").upper().strip()
                if not is_valid_gstin(buyer_gstin):
                    buyer_gstin = extract_gstin_from_text(str(raw_data.get("buyer_name", "")) + " " + buyer_gstin)

                try:
                    date = parser.parse(str(raw_data.get("date", "")), dayfirst=True).strftime("%d/%m/%Y")
                except (parser.ParserError, TypeError):
                    date = ""

                # --- Derived Field Calculations ---
                total_amount = taxable_amount + cgst + sgst + igst
                tds_rate = determine_tds_rate(expense_ledger, tds_str)
                tds_section = determine_tds_section(expense_ledger)
                tds_amount = round(taxable_amount * tds_rate / 100, 2)
                amount_payable = total_amount - tds_amount
                
                tds_applicability = "No"
                if tds_rate > 0:
                    tds_applicability = "Yes"
                elif "yes" in str(tds_str).lower():
                     tds_applicability = "Yes"
                elif "uncertain" in str(tds_str).lower():
                    tds_applicability = "Uncertain"
                
                # Create a narrative summary
                narration = (
                    f"Invoice {raw_data.get('invoice_number', 'N/A')} dated {date or 'N/A'} "
                    f"from {raw_data.get('seller_name', 'N/A')} (GSTIN: {seller_gstin or 'N/A'}, HSN/SAC: {raw_data.get('hsn_sac') or 'N/A'}) "
                    f"to {raw_data.get('buyer_name', 'N/A')} (GSTIN: {buyer_gstin or 'N/A'}). "
                    f"Expense: {expense_ledger or 'N/A'}. Place of Supply: {raw_data.get('place_of_supply') or 'N/A'}. "
                    f"Taxable: {format_currency(taxable_amount)}, Total: {format_currency(total_amount)}. "
                    f"TDS: {tds_applicability} (Section: {tds_section or 'N/A'}) @ {tds_rate}% = {format_currency(tds_amount)}. "
                    f"Payable: {format_currency(amount_payable)}."
                )

                # Store final structured row
                st.session_state["processed_results"][file_name] = {
                    "File Name": file_name,
                    "Invoice Number": raw_data.get("invoice_number"),
                    "Date": date,
                    "Seller Name": raw_data.get("seller_name"),
                    "Seller GSTIN": seller_gstin,
                    "HSN/SAC": raw_data.get("hsn_sac"),
                    "Buyer Name": raw_data.get("buyer_name"),
                    "Buyer GSTIN": buyer_gstin,
                    "Expense Ledger": expense_ledger,
                    "Taxable Amount": taxable_amount,
                    "CGST": cgst, "SGST": sgst, "IGST": igst,
                    "Total Amount": total_amount,
                    "TDS Applicability": tds_applicability,
                    "TDS Section": tds_section,
                    "TDS Rate": tds_rate,
                    "TDS Amount": tds_amount,
                    "Amount Payable": amount_payable,
                    "Place of Supply": raw_data.get("place_of_supply"),
                    "Narration": narration
                }
                st.session_state["processing_status"][file_name] = "✅ Done"
                st.success(f"✅ Successfully processed {file_name}")

        except Exception as e:
            st.session_state["processing_status"][file_name] = "❌ Error"
            st.error(f"❌ Error processing {file_name}: {e}")
            # Log error details in the results table for debugging
            st.session_state["processed_results"][file_name] = {
                "File Name": file_name, "Invoice Number": "PROCESSING ERROR",
                "Narration": f"Error: {str(e)}. Raw response: {response_text or 'No response received.'}"
            }
            if response_text:
                st.text_area(f"Raw Output for {file_name}", response_text, height=150)


# --- Display Results ---
results = list(st.session_state["processed_results"].values())

if results:
    if completed_json:
        st_lottie(completed_json, height=180, key="done_animation")
    st.markdown("<h3 style='text-align: center;'>🎉 Yippie! All Invoices Processed! 😊</h3>", unsafe_allow_html=True)

    try:
        df = pd.DataFrame(results).fillna("")
        
        # --- Format DataFrame for Display ---
        currency_cols = ["Taxable Amount", "CGST", "SGST", "IGST", "Total Amount", "TDS Amount", "Amount Payable"]
        for col in currency_cols:
            df[f"{col} (₹)"] = df[col].apply(format_currency)

        df['TDS Rate (%)'] = pd.to_numeric(df['TDS Rate'], errors='coerce').fillna(0.0).apply(lambda x: f"{x:.1f}%")

        display_cols = [
            "File Name", "Invoice Number", "Date", "Seller Name", "Seller GSTIN",
            "HSN/SAC", "Buyer Name", "Buyer GSTIN", "Expense Ledger", "Place of Supply",
            "Taxable Amount (₹)", "CGST (₹)", "SGST (₹)", "IGST (₹)", "Total Amount (₹)",
            "TDS Applicability", "TDS Section", "TDS Rate (%)", "TDS Amount (₹)", "Amount Payable (₹)",
            "Narration"
        ]
        
        # Ensure all display columns exist before showing
        actual_display_cols = [col for col in display_cols if col in df.columns]
        
        st.dataframe(
            df[actual_display_cols],
            column_config={
                "HSN/SAC": st.column_config.TextColumn("HSN/SAC", help="Harmonized System of Nomenclature / Service Accounting Code", default="N/A"),
                "TDS Section": st.column_config.TextColumn("TDS Section", help="Applicable TDS Section (e.g., 194J)", default="N/A"),
            },
            use_container_width=True
        )

        # --- Download Buttons ---
        @st.cache_data
        def to_csv(df_to_download):
            return df_to_download.to_csv(index=False).encode("utf-8")

        @st.cache_data
        def to_excel(df_to_download):
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                df_to_download.to_excel(writer, index=False, sheet_name="Invoice Data")
            return excel_buffer.getvalue()

        download_df = df[actual_display_cols].copy()
        
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "📥 Download as CSV",
                to_csv(download_df),
                "invoice_results.csv",
                "text/csv",
                use_container_width=True
            )
        with col2:
            st.download_button(
                "📥 Download as Excel",
                to_excel(download_df),
                "invoice_results.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

        if any(df.get('TDS Applicability', pd.Series(dtype=str)) == "Yes"):
            st.balloons()

    except Exception as e:
        st.error(f"Error creating results table: {e}")
        st.write("Raw results data:")
        st.json(results)

else:
    st.info("Upload one or more scanned invoices to get started.")
