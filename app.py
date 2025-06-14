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

# Set locale for currency formatting
locale.setlocale(locale.LC_ALL, '')

# URLs for Lottie animations
hello_lottie = (
    "https://raw.githubusercontent.com/Mrpin2/InvoiceAi/refs/heads/"
    "main/Animation%20-%201749845212531.json"
)
completed_lottie = (
    "https://raw.githubusercontent.com/Mrpin2/InvoiceAi/refs/heads/"
    "main/Animation%20-%201749845303699.json"
)

def load_lottie_json_safe(url):
    """Load Lottie JSON from URL, return None on failure."""
    try:
        r = requests.get(url)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None

hello_json = load_lottie_json_safe(hello_lottie)
completed_json = load_lottie_json_safe(completed_lottie)

# Initial animation on first load
if "files_uploaded" not in st.session_state and hello_json:
    st_lottie(hello_json, height=200, key="hello")

# Page header
st.markdown(
    "<h2 style='text-align: center;'>📄 AI Invoice Extractor (OpenAI)</h2>",
    unsafe_allow_html=True
)
st.markdown("Upload scanned PDF invoices and extract structured finance data using GPT-4 Vision")
st.markdown("---")

# Initialize session state for results
if "processed_results" not in st.session_state:
    st.session_state["processed_results"] = {}
if "processing_status" not in st.session_state:
    st.session_state["processing_status"] = {}
if "summary_rows" not in st.session_state:
    st.session_state["summary_rows"] = []

# Sidebar: API key entry
st.sidebar.header("🔐 AI Config")
passcode = st.sidebar.text_input("Admin Passcode", type="password")
admin_unlocked = passcode == "Essenbee"
if admin_unlocked:
    st.sidebar.success("🔓 Admin access granted.")
    openai_api_key = st.secrets.get("OPENAI_API_KEY")
    if not openai_api_key:
        st.sidebar.error("OPENAI_API_KEY missing in secrets.")
        st.stop()
else:
    openai_api_key = st.sidebar.text_input(
        "🔑 Enter your OpenAI API Key", type="password"
    )
    if not openai_api_key:
        st.sidebar.warning("Please enter a valid API key to continue.")
        st.stop()

# Instantiate OpenAI client
client = OpenAI(api_key=openai_api_key)

# Utility functions

def convert_pdf_first_page(pdf_bytes):
    """Convert first page of PDF bytes to PIL Image."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc.load_page(0)
    pix = page.get_pixmap(dpi=300)
    return Image.open(io.BytesIO(pix.tobytes("png")))


def safe_float(x):
    """Convert input to float after cleaning punctuation and currency symbols."""
    try:
        cleaned = str(x).replace(",", "").replace("₹", "").replace("$", "").strip()
        return float(cleaned) if cleaned else 0.0
    except:
        return 0.0


def format_currency(x):
    """Format numeric value as currency string '₹#,###.##'."""
    try:
        return f"₹{safe_float(x):,.2f}"
    except:
        return "₹0.00"

# GSTIN validation and extraction

def is_valid_gstin(gstin):
    """Return True if gstin matches 15-character GSTIN pattern."""
    if not gstin or not isinstance(gstin, str):
        return False
    cleaned = re.sub(r'[^A-Z0-9]', '', gstin.upper())
    if len(cleaned) != 15:
        return False
    pattern = r"^\d{2}[A-Z0-9]{10}[A-Z]{1}[A-Z0-9]{1}Z[A-Z0-9]{1}$"
    return bool(re.match(pattern, cleaned))


def extract_gstin_from_text(text):
    """Extract first GSTIN-like substring from text."""
    matches = re.findall(
        r'\b\d{2}[A-Z0-9]{10}[A-Z]{1}[A-Z0-9]{1}Z[A-Z0-9]{1}\b',
        (text or "").upper()
    )
    return matches[0] if matches else ""


def handle_gstin(raw_gstin, seller_name, response_text):
    """
    Clean and validate raw_gstin; fallback to seller_name or full response if needed.
    Returns a valid GSTIN or empty string.
    """
    # Try raw field first
    gst = extract_gstin_from_text(raw_gstin)
    if is_valid_gstin(gst):
        return gst

    # Fallback: seller name
    gst = extract_gstin_from_text(seller_name)
    if is_valid_gstin(gst):
        return gst

    # Fallback: entire GPT response text
    gst = extract_gstin_from_text(response_text)
    return gst if is_valid_gstin(gst) else ""

# Determine TDS rate

def determine_tds_rate(expense_ledger, tds_str=""):
    """
    Determine TDS percentage based on explicit tds_str or expense ledger keywords.
    """
    if tds_str and isinstance(tds_str, str):
        match = re.search(r'(\d+(\.\d+)?)%', tds_str)
        if match:
            return float(match.group(1))
        # Common sections
        sections = {
            "194j": 10.0,
            "194c": 2.0,
            "194h": 5.0,
            "194i": 10.0,
            "194q": 1.0
        }
        for sec, rate in sections.items():
            if sec in tds_str.lower():
                return rate

    exp = (expense_ledger or "").lower()
    if any(k in exp for k in ["professional", "service"]):
        return 10.0
    if any(k in exp for k in ["contract", "work"]):
        return 2.0
    if "commission" in exp:
        return 5.0
    if "rent" in exp:
        return 10.0
    if any(k in exp for k in ["advertis", "marketing"]):
        return 1.0
    return 0.0

# Extract JSON from GPT response

def extract_json_from_response(text):
    """
    Attempt to parse JSON from text, stripping markdown fences or extra content.
    """
    try:
        # Look for ```json blocks
        matches = re.findall(r'```json\s*({.*?})\s*```', text, re.DOTALL)
        if matches:
            return json.loads(matches[0])

        # Fallback: find first { ... }
        start, end = text.find('{'), text.rfind('}')
        if start != -1 and end != -1:
            return json.loads(text[start:end+1])

        # As is
        return json.loads(text)
    except Exception:
        return None

# Main prompt for GPT
main_prompt = (
    "Extract structured invoice data as a JSON object with the following keys: "
    "invoice_number, date, gstin, seller_name, buyer_name, buyer_gstin, "
    "taxable_amount, cgst, sgst, igst, place_of_supply, expense_ledger, tds. "
    "Important: 'taxable_amount' is the amount BEFORE taxes. Use DD/MM/YYYY for dates. "
    "Return 'NOT AN INVOICE' if clearly not one. Use null if missing. "
    "SPECIAL: Look for GSTIN in seller details; label may vary."
)

# File uploader widget
uploaded_files = st.file_uploader(
    "📤 Upload scanned invoice PDFs", type=["pdf"], accept_multiple_files=True
)

# Processing loop
if uploaded_files:
    st.session_state["files_uploaded"] = True
    total = len(uploaded_files)
    for idx, file in enumerate(uploaded_files, start=1):
        name = file.name
        if name in st.session_state["processed_results"]:
            continue
        st.markdown(f"**Processing file: {name} ({idx}/{total})**")
        st.session_state["processing_status"][name] = "⏳ Pending..."
        temp_path = None
        try:
            # Write PDF to temp
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file.getvalue())
                temp_path = tmp.name

            # Read and convert first page
            pdf_bytes = open(temp_path, "rb").read()
            img = convert_pdf_first_page(pdf_bytes)

            # Encode image to base64
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            buf.seek(0)
            b64 = base64.b64encode(buf.read()).decode()

            # Construct chat prompt
            messages = [
                {"role": "system", "content": "You are a finance assistant specializing in Indian invoices."},
                {"role": "user", "content": [
                    {"type": "text", "text": main_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
                ]}
            ]

            # Call ChatGPT Vision
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=1500
            )
            resp_text = response.choices[0].message.content.strip()
            raw_data = extract_json_from_response(resp_text)

            # Initialize result_row
            if raw_data is None:
                if "not an invoice" in resp_text.lower():
                    # Not an invoice case
                    row = {
                        "File Name": name,
                        "Invoice Number": "NOT AN INVOICE",
                        **{field: None for field in [
                            "Date","Seller Name","Seller GSTIN","Buyer Name",
                            "Buyer GSTIN","Taxable Amount","CGST","SGST","IGST",
                            "Total Amount","TDS Rate","TDS Amount","Amount Payable",
                            "Place of Supply","Expense Ledger","TDS","Narration"
                        ]}
                    }
                else:
                    raise ValueError("GPT returned non-JSON response")
            else:
                # Extract fields from JSON
                inv_no = raw_data.get("invoice_number", "")
                date_raw = raw_data.get("date", "")
                seller_name = raw_data.get("seller_name", "")
                raw_gstin = raw_data.get("gstin", "")
                buyer_name = raw_data.get("buyer_name", "")
                buyer_gstin = raw_data.get("buyer_gstin", "")
                ta = safe_float(raw_data.get("taxable_amount", 0.0))
                cg = safe_float(raw_data.get("cgst", 0.0))
                sg = safe_float(raw_data.get("sgst", 0.0))
                ig = safe_float(raw_data.get("igst", 0.0))
                place = raw_data.get("place_of_supply", "")
                ledger = raw_data.get("expense_ledger", "")
                tds_str = raw_data.get("tds", "")

                # Derived calculations
                total_amt = ta + cg + sg + ig
                tds_rate = determine_tds_rate(ledger, tds_str)
                tds_amt = round(ta * tds_rate / 100, 2)
                payable = total_amt - tds_amt

                # Clean/validate GSTIN silently
                seller_gstin = handle_gstin(raw_gstin, seller_name, resp_text)

                # Format date
                try:
                    parsed = parser.parse(date_raw, dayfirst=True)
                    date_str = parsed.strftime("%d/%m/%Y")
                except:
                    date_str = date_raw

                # Build narration
                narration = (
                    f"Invoice {inv_no} dated {date_str} was issued by {seller_name} "
                    f"(GSTIN: {seller_gstin or 'N/A'}) to {buyer_name} "
                    f"(GSTIN: {buyer_gstin or 'N/A'}), taxable amount ₹{ta:,.2f}, "
                    f"CGST ₹{cg:,.2f}, SGST ₹{sg:,.2f}, IGST ₹{ig:,.2f}, total ₹{total_amt:,.2f}, "
                    f"TDS {tds_str or 'N/A'} @{tds_rate}% (₹{tds_amt:,.2f}), payable ₹{payable:,.2f}."
                )

                # Compile result row
                row = {
                    "File Name": name,
                    "Invoice Number": inv_no,
                    "Date": date_str,
                    "Seller Name": seller_name,
                    "Seller GSTIN": seller_gstin,
                    "Buyer Name": buyer_name,
                    "Buyer GSTIN": buyer_gstin,
                    "Taxable Amount": ta,
                    "CGST": cg,
                    "SGST": sg,
                    "IGST": ig,
                    "Total Amount": total_amt,
                    "TDS Rate": tds_rate,
                    "TDS Amount": tds_amt,
                    "Amount Payable": payable,
                    "Place of Supply": place,
                    "Expense Ledger": ledger,
                    "TDS": tds_str,
                    "Narration": narration
                }

            # Store results in session state
            st.session_state["processed_results"][name] = row
            st.session_state["processing_status"][name] = "✅ Done"
            st.success(f"{name}: ✅ Done")

        except Exception as e:
            # Handle errors
            err_row = {"File Name": name, "Error": str(e)}
            st.session_state["processed_results"][name] = err_row
            st.session_state["processing_status"][name] = "❌ Error"
            st.error(f"Error processing {name}: {e}")

        finally:
            # Clean up temp file
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)

# Display final results
results = list(st.session_state.get("processed_results", {}).values())
if results:
    if completed_json:
        st_lottie(completed_json, height=200, key="done_animation")
    st.markdown(
        "<h3 style='text-align: center;'>🎉 All invoices processed successfully!</h3>",
        unsafe_allow_html=True
    )

    # Create DataFrame and format
    df = pd.DataFrame(results)
    currency_cols = [
        "Taxable Amount", "CGST", "SGST", "IGST",
        "Total Amount", "TDS Amount", "Amount Payable"
    ]
    for c in currency_cols:
        df[f"{c} (₹)"] = df[c].apply(format_currency)
    df["TDS Rate (%)"] = df["TDS Rate"].apply(lambda x: f"{x}%")

    # Define display order
    display_cols = [
        "File Name", "Invoice Number", "Date", "Seller Name", "Seller GSTIN",
        "Buyer Name", "Buyer GSTIN",
        "Taxable Amount (₹)", "CGST (₹)", "SGST (₹)", "IGST (₹)",
        "Total Amount (₹)", "TDS Rate (%)", "TDS Amount (₹)",
        "Amount Payable (₹)", "Place of Supply", "Expense Ledger", "TDS", "Narration"
    ]

    st.dataframe(df[display_cols])

    # CSV download button
    csv_data = df[display_cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Download Results as CSV", csv_data, "invoice_results.csv", "text/csv"
    )

    # Excel download button
    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
        df[display_cols].to_excel(writer, index=False, sheet_name="Invoice Data")
    st.download_button(
        "📥 Download Results as Excel",
        excel_buffer.getvalue(),
        "invoice_results.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    st.markdown("---")
else:
    st.info("Upload one or more scanned invoices to get started.")
