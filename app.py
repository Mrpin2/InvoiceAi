import streamlit as st
st.set_page_config(layout="wide")

from PIL import Image
import fitz # PyMuPDF
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
    try:
        r = requests.get(url)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None

hello_json = load_lottie_json_safe(hello_lottie)
completed_json = load_lottie_json_safe(completed_lottie)

if "files_uploaded" not in st.session_state:
    if hello_json:
        st_lottie(hello_json, height=200, key="hello")

st.markdown("<h2 style='text-align: center;'>📄 AI Invoice Extractor (OpenAI)</h2>", unsafe_allow_html=True)
st.markdown("Upload scanned PDF invoices and extract structured finance data using GPT-4o.")
st.markdown("---")

# --- Session State Initialization (from your script) ---
if "processed_results" not in st.session_state:
    st.session_state["processed_results"] = {}
if "processing_status" not in st.session_state:
    st.session_state["processing_status"] = {}
if "summary_rows" not in st.session_state:
    st.session_state["summary_rows"] = []

# --- Sidebar for API Key Configuration ---
st.sidebar.header("🔐 AI Config")
passcode = st.sidebar.text_input("Admin Passcode", type="password")
admin_unlocked = passcode == "Rajeev"

openai_api_key = None
if admin_unlocked:
    st.sidebar.success("🔓 Admin access granted.")
    openai_api_key = st.secrets.get("OPENAI_API_KEY")
    if not openai_api_key:
        st.sidebar.error("OPENAI_API_KEY missing in Streamlit secrets.")
        st.stop()
else:
    openai_api_key = st.sidebar.text_input("🔑 Enter your OpenAI API Key", type="password")
    if not openai_api_key:
        st.sidebar.warning("Please enter a valid API key to continue.")
        st.stop()

# Initialize OpenAI client
client = OpenAI(api_key=openai_api_key)

# --- Core Helper Functions ---

def convert_pdf_to_image_and_text(pdf_bytes):
    """Opens a PDF, extracts the first page as an image, and extracts all text from it."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc.load_page(0)
    
    # Get image for AI Vision
    pix = page.get_pixmap(dpi=300)
    image = Image.open(io.BytesIO(pix.tobytes("png")))
    
    # Get all text for fallback search
    text = page.get_text("text")
    
    doc.close()
    return image, text

def safe_float(x):
    try:
        cleaned = str(x).replace(",", "").replace("₹", "").replace("$", "").strip()
        return float(cleaned) if cleaned else 0.0
    except (ValueError, TypeError):
        return 0.0

def format_currency(x):
    try:
        if isinstance(x, str) and x.startswith('₹'):
            return x
        return f"₹{safe_float(x):,.2f}"
    except Exception:
        return "₹0.00"

def is_valid_gstin(gstin):
    """Validates if a string is a 15-character GSTIN. Case-insensitive."""
    if not gstin or not isinstance(gstin, str):
        return False
    # A valid GSTIN must be exactly 15 characters long.
    return len(gstin) == 15 and bool(re.match(r"^\d{2}[A-Z]{5}\d{4}[A-Z]{1}[A-Z1-9]{1}Z[A-Z0-9]{1}$", gstin.upper()))

def extract_gstin_from_text(text):
    """Finds all potential 15-character GSTIN patterns in a block of text."""
    if not text:
        return []
    # This regex finds any 15-character alphanumeric string that starts with 2 digits.
    # It's a broad but effective net for finding candidates in raw text.
    return re.findall(r'\b(\d{2}[A-Z0-9]{13})\b', text.upper())

def determine_tds_rate(expense_ledger, tds_str=""):
    # This function is kept as it was in your script.
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
        return 10.0
    if "contract" in expense_ledger:
        return 1.0
    if "rent" in expense_ledger:
        return 10.0
    return 0.0

def determine_tds_section(expense_ledger):
    expense_ledger = (expense_ledger or "").lower()
    if any(keyword in expense_ledger for keyword in ["professional", "consultancy", "service"]):
        return "194J"
    return None

def extract_json_from_response(text):
    if not text: return None
    try:
        match = re.search(r'```json\s*({.*?})\s*```', text, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        return json.loads(text)
    except (json.JSONDecodeError, IndexError):
        # Fallback for plain JSON without code blocks
        try:
            start = text.find('{')
            end = text.rfind('}')
            if start != -1 and end != -1:
                return json.loads(text[start:end+1])
        except json.JSONDecodeError:
            return None
    return None

# --- Main AI Prompt (The most effective version) ---
main_prompt = (
    "You are an expert at extracting structured data from Indian invoices. "
    "Analyze the invoice image and return a clean, parsable JSON object with the specified keys. "
    "If a value is not found, use `null`. Do not make up data.\n"
    "Keys: invoice_number, date, gstin, seller_name, buyer_name, buyer_gstin, "
    "taxable_amount, cgst, sgst, igst, place_of_supply, expense_ledger, tds, hsn_sac.\n\n"
    "GUIDELINES:\n"
    "- 'gstin': The 15-digit GSTIN of the seller.\n"
    "- 'buyer_gstin': The 15-digit GSTIN of the buyer. Will be `null` for export or B2C invoices.\n"
    "- 'hsn_sac': Extract the code ONLY if explicitly mentioned. Otherwise, `null`.\n"
    "- 'place_of_supply': Crucial field. First, look for an explicit 'Place of Supply' label. "
    "  If not found, infer it from the buyer's address ('Bill to', 'Customer', etc.). "
    "  If the buyer's address is outside India, this is an export. Use the buyer's country (e.g., 'USA'). "
    "  If the country is unclear but it's an export, use 'Foreign'."
)

# --- File Uploader and Main Processing Loop ---
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
            pdf_bytes = file.getvalue()
            # **NEW**: Get both image for AI and full text for fallback
            invoice_image, full_page_text = convert_pdf_to_image_and_text(pdf_bytes)

            with st.spinner("🧠 Extracting data using GPT-4o Vision..."):
                img_buf = io.BytesIO()
                invoice_image.save(img_buf, format="PNG")
                base64_image = base64.b64encode(img_buf.getvalue()).decode()

                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": main_prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                        ]
                    }],
                    max_tokens=2000,
                    response_format={"type": "json_object"}
                )
                
                response_text = response.choices[0].message.content
                raw_data = extract_json_from_response(response_text)

                if raw_data is None:
                    raise ValueError("Failed to parse JSON from AI response.")

                # --- Data Cleaning and Structuring ---
                seller_gstin = raw_data.get("gstin")
                buyer_gstin = raw_data.get("buyer_gstin")

                # **NEW ROBUST GSTIN LOGIC**
                # Clean the initial AI response
                cleaned_seller_gstin = re.sub(r'[^A-Z0-9]', '', str(seller_gstin)).upper()
                if not is_valid_gstin(cleaned_seller_gstin):
                    # If invalid, search the ENTIRE page text for any valid GSTIN
                    all_found_gstins = extract_gstin_from_text(full_page_text)
                    if all_found_gstins:
                        # For now, we assume the first one found is the seller's.
                        # This can be made more sophisticated later if needed.
                        cleaned_seller_gstin = all_found_gstins[0]
                
                cleaned_buyer_gstin = re.sub(r'[^A-Z0-9]', '', str(buyer_gstin)).upper()
                if not is_valid_gstin(cleaned_buyer_gstin):
                    all_found_gstins = extract_gstin_from_text(full_page_text)
                    # Simple logic: if we find two distinct GSTINs, the second one is likely the buyer's
                    if len(set(all_found_gstins)) > 1:
                        cleaned_buyer_gstin = [g for g in all_found_gstins if g != cleaned_seller_gstin][0]
                    else:
                        cleaned_buyer_gstin = "" # No valid buyer GSTIN found

                # --- Process other fields ---
                taxable_amount = safe_float(raw_data.get("taxable_amount"))
                total_amount = taxable_amount + safe_float(raw_data.get("cgst")) + safe_float(raw_data.get("sgst")) + safe_float(raw_data.get("igst"))
                tds_rate = determine_tds_rate(raw_data.get("expense_ledger"), raw_data.get("tds"))
                tds_amount = round(taxable_amount * tds_rate / 100, 2)
                
                try:
                    date_str = parser.parse(str(raw_data.get("date")), dayfirst=True).strftime("%d/%m/%Y")
                except (parser.ParserError, TypeError):
                    date_str = ""

                # --- Store Results ---
                result_row = {
                    "File Name": file_name,
                    "Invoice Number": raw_data.get("invoice_number", "N/A"),
                    "Date": date_str,
                    "Seller Name": raw_data.get("seller_name"),
                    "Seller GSTIN": cleaned_seller_gstin,
                    "HSN/SAC": raw_data.get("hsn_sac"),
                    "Buyer Name": raw_data.get("buyer_name"),
                    "Buyer GSTIN": cleaned_buyer_gstin,
                    "Expense Ledger": raw_data.get("expense_ledger"),
                    "Taxable Amount": taxable_amount,
                    "CGST": safe_float(raw_data.get("cgst")),
                    "SGST": safe_float(raw_data.get("sgst")),
                    "IGST": safe_float(raw_data.get("igst")),
                    "Total Amount": total_amount,
                    "TDS Applicability": "Yes" if tds_rate > 0 else "No",
                    "TDS Section": determine_tds_section(raw_data.get("expense_ledger")),
                    "TDS Rate": tds_rate,
                    "TDS Amount": tds_amount,
                    "Amount Payable": total_amount - tds_amount,
                    "Place of Supply": raw_data.get("place_of_supply"),
                    "Narration": f"Invoice from {raw_data.get('seller_name', 'N/A')} to {raw_data.get('buyer_name', 'N/A')} for {format_currency(total_amount)}."
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
                "Narration": f"Error: {traceback.format_exc()}"
            }
            if response_text:
                st.text_area(f"Raw AI Output for {file_name}", response_text, height=150)

# --- Display Results Table (Structure from your script) ---
results = list(st.session_state["processed_results"].values())

if results:
    if completed_json:
        st_lottie(completed_json, height=200, key="done_animation")
    st.markdown("<h3 style='text-align: center;'>🎉 Yippie! All invoices Processed!!! 😊</h3>", unsafe_allow_html=True)

    try:
        df = pd.DataFrame(results).fillna("")
        
        currency_cols_mapping = {
            "Taxable Amount": "Taxable Amount (₹)", "CGST": "CGST (₹)", "SGST": "SGST (₹)",
            "IGST": "IGST (₹)", "Total Amount": "Total Amount (₹)", "TDS Amount": "TDS Amount (₹)",
            "Amount Payable": "Amount Payable (₹)"
        }
        for col, new_name in currency_cols_mapping.items():
            if col in df.columns:
                df[new_name] = df[col].apply(format_currency)

        if 'TDS Rate' in df.columns:
            df['TDS Rate (%)'] = pd.to_numeric(df['TDS Rate'], errors='coerce').fillna(0.0).apply(lambda x: f"{x:.1f}%")

        display_cols = [
            "File Name", "Invoice Number", "Date", "Seller Name", "Seller GSTIN", "HSN/SAC",
            "Buyer Name", "Buyer GSTIN", "Expense Ledger", "Taxable Amount (₹)", "CGST (₹)", "SGST (₹)",
            "IGST (₹)", "Total Amount (₹)", "TDS Applicability", "TDS Section", "TDS Rate (%)",
            "TDS Amount (₹)", "Amount Payable (₹)", "Place of Supply", "Narration"
        ]
        
        actual_display_cols = [col for col in display_cols if col in df.columns]
        st.dataframe(df[actual_display_cols], use_container_width=True)

        # --- Download Buttons ---
        download_df = df[actual_display_cols].copy()
        csv_data = download_df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Results as CSV", csv_data, "invoice_results.csv", "text/csv")

        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
            download_df.to_excel(writer, index=False, sheet_name="Invoice Data")
        st.download_button(
            label="📥 Download Results as Excel", data=excel_buffer.getvalue(),
            file_name="invoice_results.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
        # --- Debugging Section (Restored from your script) ---
        st.markdown("---")
        st.markdown("### Debugging Information:")
        with st.expander("Click to see DataFrame Info and Null Counts"):
            st.write("#### DataFrame Info:")
            buffer = io.StringIO()
            df.info(buf=buffer, verbose=True)
            st.text(buffer.getvalue())
            
            st.write("#### Null Counts per Column:")
            st.dataframe(df.isnull().sum().to_frame(name='Null Count'))
        
        if completed_count == total_files and completed_count > 0:
            st.balloons()

    except Exception as e:
        st.error(f"Error creating results table: {e}")
        st.json(results)
else:
    st.info("Upload one or more scanned invoices to get started.")
