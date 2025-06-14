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

# --- Core Functions (from your working script) ---

def convert_pdf_first_page(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc.load_page(0)
    pix = page.get_pixmap(dpi=300)
    return Image.open(io.BytesIO(pix.tobytes("png")))

def safe_float(x):
    try:
        cleaned = str(x).replace(",", "").replace("₹", "").replace("$", "").strip()
        return float(cleaned) if cleaned else 0.0
    except:
        return 0.0

def format_currency(x):
    try:
        if isinstance(x, str) and x.startswith('₹'):
            return x
        return f"₹{safe_float(x):,.2f}"
    except:
        return "₹0.00"

def is_valid_gstin(gstin):
    """Validate GSTIN format with more flexibility (from your script)."""
    if not gstin:
        return False
    cleaned = re.sub(r'[^A-Z0-9]', '', gstin.upper())
    if len(cleaned) != 15:
        return False
    pattern = r"^\d{2}[A-Z0-9]{10}[A-Z]{1}[A-Z0-9]{1}[Z]{1}[A-Z0-9]{1}$"
    return bool(re.match(pattern, cleaned))

def extract_gstin_from_text(text):
    """Try to extract GSTIN from any text using pattern matching (from your script)."""
    matches = re.findall(r'\b\d{2}[A-Z0-9]{10}[A-Z]{1}[A-Z0-9]{1}Z[A-Z0-9]{1}\b', text.upper())
    if matches:
        return matches[0]
    return ""

def determine_tds_rate(expense_ledger, tds_str=""):
    if tds_str and isinstance(tds_str, str):
        match = re.search(r'(\d+(\.\d+)?)%', tds_str)
        if match:
            return float(match.group(1))
        
        section_rates = {"194j": 10.0, "194c": 2.0, "194i": 10.0, "194h": 5.0, "194q": 0.1}
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

def determine_tds_section(expense_ledger):
    expense_ledger = expense_ledger.lower() if expense_ledger else ""
    if "professional" in expense_ledger or "consultancy" in expense_ledger or "service" in expense_ledger:
        return "194J"
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
        return json.loads(text)
    except Exception:
        return None

# --- Main AI Prompt (ENHANCED) ---
main_prompt = (
    "You are an expert at extracting structured data from Indian invoices. "
    "Your task is to analyze the invoice image and return a clean, parsable JSON object. "
    "If a value is not found, use `null`. "
    "Keys to extract: invoice_number, date, gstin (seller's), seller_name, buyer_name, buyer_gstin, "
    "taxable_amount, cgst, sgst, igst, place_of_supply, expense_ledger, tds, hsn_sac.\n\n"
    
    "GUIDELINES:\n"
    "- 'invoice_number': The unique identifier of the invoice.\n"
    "- 'date': The invoice date in DD/MM/YYYY format.\n"
    "- 'taxable_amount': The subtotal amount *before* any taxes (CGST, SGST, IGST).\n"
    "- 'gstin': The GSTIN of the seller (the entity issuing the invoice).\n"
    "- 'buyer_gstin': The GSTIN of the buyer (the entity receiving the invoice). This will be `null` for an export invoice.\n"
    "- 'hsn_sac': Extract the HSN or SAC code *only* if explicitly mentioned. If not found, it *must* be `null`.\n"
    "- 'expense_ledger': Classify the expense (e.g., 'Professional Fees', 'Software Subscription', 'Goods Exported').\n"
    "- 'tds': State 'Yes - Section [X]' if applicable (e.g., 'Yes - Section 194J'), 'No' if not, or 'Uncertain'.\n"
    "- **'place_of_supply'**: This is crucial. First, look for a field explicitly labeled 'Place of Supply'. "
    "  If not present, infer it from the buyer's address (under 'Bill to:', 'Buyer:', etc.). "
    "  **If the buyer's address is clearly outside of India, this is an export invoice. Set the place of supply to the buyer's country (e.g., 'USA', 'Germany'). If the country is not clear but it's an export, use 'Foreign'.**"
)

# --- File Uploader and Processing Logic ---
uploaded_files = st.file_uploader("📤 Upload scanned invoice PDFs", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    st.session_state["files_uploaded"] = True
    total_files = len(uploaded_files)
    completed_count = 0

    for idx, file in enumerate(uploaded_files):
        file_name = file.name
        if file_name in st.session_state["processed_results"]:
            continue

        st.markdown(f"**Processing file: {file_name} ({idx+1}/{total_files})**")
        st.session_state["processing_status"][file_name] = "⏳ Pending..."
        
        response_text = None
        try:
            pdf_data = file.getvalue()
            first_image = convert_pdf_first_page(pdf_data)

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

                # API Call with JSON mode for reliability
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=chat_prompt,
                    max_tokens=2000,
                    response_format={"type": "json_object"} 
                )
                
                response_text = response.choices[0].message.content.strip()
                raw_data = extract_json_from_response(response_text)

                if raw_data is None:
                    raise ValueError(f"Failed to parse JSON from GPT's response.")

                # --- Data Cleaning and Structuring (using your proven logic) ---
                invoice_number = raw_data.get("invoice_number", "")
                date = raw_data.get("date", "")
                seller_name = raw_data.get("seller_name", "")
                seller_gstin = raw_data.get("gstin", "")
                hsn_sac = raw_data.get("hsn_sac", "")
                buyer_name = raw_data.get("buyer_name", "")
                buyer_gstin = raw_data.get("buyer_gstin", "")
                expense_ledger = raw_data.get("expense_ledger", "")
                taxable_amount = safe_float(raw_data.get("taxable_amount", 0.0))
                cgst = safe_float(raw_data.get("cgst", 0.0))
                sgst = safe_float(raw_data.get("sgst", 0.0))
                igst = safe_float(raw_data.get("igst", 0.0))
                place_of_supply = raw_data.get("place_of_supply", "")
                tds_str = raw_data.get("tds", "")
                
                # Derived fields
                total_amount = taxable_amount + cgst + sgst + igst
                tds_rate = determine_tds_rate(expense_ledger, tds_str)
                tds_amount = round(taxable_amount * tds_rate / 100, 2) if tds_rate > 0 else 0.0
                amount_payable = total_amount - tds_amount
                tds_section = determine_tds_section(expense_ledger)
                
                tds_applicability = "Uncertain"
                if tds_rate > 0 or tds_amount > 0:
                    tds_applicability = "Yes"
                elif "no" in str(tds_str).lower():
                    tds_applicability = "No"

                # Enhanced GSTIN handling (from your script)
                if seller_gstin:
                    seller_gstin = re.sub(r'[^A-Z0-9]', '', seller_gstin.upper())
                if not is_valid_gstin(seller_gstin):
                    fallback_gstin = extract_gstin_from_text(str(seller_name) + " " + str(seller_gstin))
                    if fallback_gstin:
                        seller_gstin = fallback_gstin
                
                if buyer_gstin:
                    buyer_gstin = re.sub(r'[^A-Z0-9]', '', buyer_gstin.upper())
                if not is_valid_gstin(buyer_gstin):
                    fallback_buyer_gstin = extract_gstin_from_text(str(buyer_name) + " " + str(buyer_gstin))
                    if fallback_buyer_gstin:
                        buyer_gstin = fallback_buyer_gstin
                
                # Parse and format date
                try:
                    parsed_date = parser.parse(str(date), dayfirst=True)
                    date = parsed_date.strftime("%d/%m/%Y")
                except:
                    date = ""
                
                # Narration
                narration = (
                    f"Invoice {invoice_number} dated {date} from {seller_name} (GSTIN: {seller_gstin or 'N/A'}) "
                    f"to {buyer_name} (GSTIN: {buyer_gstin or 'N/A'}). "
                    f"Place of Supply: {place_of_supply or 'N/A'}. "
                    f"Total: ₹{total_amount:,.2f}. Payable: ₹{amount_payable:,.2f}."
                )
                
                result_row = {
                    "File Name": file_name, "Invoice Number": invoice_number, "Date": date,
                    "Seller Name": seller_name, "Seller GSTIN": seller_gstin, "HSN/SAC": hsn_sac,
                    "Buyer Name": buyer_name, "Buyer GSTIN": buyer_gstin, "Expense Ledger": expense_ledger,
                    "Taxable Amount": taxable_amount, "CGST": cgst, "SGST": sgst, "IGST": igst,
                    "Total Amount": total_amount, "TDS Applicability": tds_applicability,
                    "TDS Section": tds_section, "TDS Rate": tds_rate, "TDS Amount": tds_amount,
                    "Amount Payable": amount_payable, "Place of Supply": place_of_supply, "Narration": narration
                }

                st.session_state["processed_results"][file_name] = result_row
                st.session_state["processing_status"][file_name] = "✅ Done"
                completed_count += 1
                st.success(f"✅ Successfully processed {file_name}")

        except Exception as e:
            st.session_state["processing_status"][file_name] = "❌ Error"
            st.error(f"❌ Error processing {file_name}: {e}")
            st.session_state["processed_results"][file_name] = {
                "File Name": file_name, "Invoice Number": "PROCESSING ERROR",
                "Narration": f"Error: {str(e)}. Raw response: {response_text or 'No response.'}"
            }
            if response_text:
                st.text_area(f"Raw Output ({file_name})", response_text, height=150)

# --- Display Results ---
results = list(st.session_state["processed_results"].values())

if results:
    if completed_json:
        st_lottie(completed_json, height=180, key="done_animation")
    st.markdown("<h3 style='text-align: center;'>🎉 Yippie! All Invoices Processed! 😊</h3>", unsafe_allow_html=True)

    try:
        df = pd.DataFrame(results).fillna("")

        currency_cols_mapping = {
            "Taxable Amount": "Taxable Amount (₹)", "CGST": "CGST (₹)", "SGST": "SGST (₹)",
            "IGST": "IGST (₹)", "Total Amount": "Total Amount (₹)", "TDS Amount": "TDS Amount (₹)",
            "Amount Payable": "Amount Payable (₹)"
        }
        for original_col, display_col in currency_cols_mapping.items():
            if original_col in df.columns:
                df[display_col] = df[original_col].apply(format_currency)

        if 'TDS Rate' in df.columns:
            df['TDS Rate (%)'] = pd.to_numeric(df['TDS Rate'], errors='coerce').fillna(0.0).apply(lambda x: f"{x:.1f}%")

        display_cols = [
            "File Name", "Invoice Number", "Date", "Seller Name", "Seller GSTIN", "HSN/SAC",
            "Buyer Name", "Buyer GSTIN", "Expense Ledger", "Place of Supply",
            "Taxable Amount (₹)", "CGST (₹)", "SGST (₹)", "IGST (₹)", "Total Amount (₹)",
            "TDS Applicability", "TDS Section", "TDS Rate (%)", "TDS Amount (₹)", "Amount Payable (₹)",
            "Narration"
        ]
        
        actual_display_cols = [col for col in display_cols if col in df.columns]
        
        st.dataframe(df[actual_display_cols], use_container_width=True)

        download_df = df[actual_display_cols].copy()
        
        # CSV Download
        csv_data = download_df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download as CSV", csv_data, "invoice_results.csv", "text/csv")

        # Excel Download
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
            download_df.to_excel(writer, index=False, sheet_name="Invoice Data")
        st.download_button(
            label="📥 Download as Excel",
            data=excel_buffer.getvalue(),
            file_name="invoice_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
        if completed_count == total_files and completed_count > 0:
            st.balloons()

    except Exception as e:
        st.error(f"Error creating results table: {e}")
        st.json(results)

else:
    st.info("Upload one or more scanned invoices to get started.")
