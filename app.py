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
    try:
        r = requests.get(url)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None

hello_json = load_lottie_json_safe(hello_lottie)
completed_json = load_lottie_json_safe(completed_lottie)

# Initial animation
if "files_uploaded" not in st.session_state:
    if hello_json:
        st_lottie(hello_json, height=200, key="hello")

# Page header
st.markdown("<h2 style='text-align: center;'>📄 AI Invoice Extractor (OpenAI)</h2>", unsafe_allow_html=True)
st.markdown("Upload scanned PDF invoices and extract structured finance data using GPT-4 Vision")
st.markdown("---")

# Fields to extract (for reference)
fields = [
    "invoice_number", "date", "gstin", "seller_name", "buyer_name", "buyer_gstin",
    "taxable_amount", "cgst", "sgst", "igst", "place_of_supply", "expense_ledger", "tds"
]

# Session state for results
if "processed_results" not in st.session_state:
    st.session_state["processed_results"] = {}
if "processing_status" not in st.session_state:
    st.session_state["processing_status"] = {}
if "summary_rows" not in st.session_state:
    st.session_state["summary_rows"] = []

# Sidebar config
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
    openai_api_key = st.sidebar.text_input("🔑 Enter your OpenAI API Key", type="password")
    if not openai_api_key:
        st.sidebar.warning("Please enter a valid API key to continue.")
        st.stop()

client = OpenAI(api_key=openai_api_key)

# Utility functions

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
        return f"₹{safe_float(x):,.2f}"
    except:
        return "₹0.00"

# GSTIN functions

def is_valid_gstin(gstin):
    if not gstin or not isinstance(gstin, str):
        return False
    cleaned = re.sub(r'[^A-Z0-9]', '', gstin.upper())
    if len(cleaned) != 15:
        return False
    pattern = r"^\d{2}[A-Z0-9]{10}[A-Z]{1}[A-Z0-9]{1}Z[A-Z0-9]{1}$"
    return bool(re.match(pattern, cleaned))

def extract_gstin_from_text(text):
    matches = re.findall(r'\b\d{2}[A-Z0-9]{10}[A-Z]{1}[A-Z0-9]{1}Z[A-Z0-9]{1}\b', text.upper())
    return matches[0] if matches else ""

def handle_gstin(raw_gstin, seller_name, response_text):
    # Try raw field first
    gst = extract_gstin_from_text(raw_gstin)
    if is_valid_gstin(gst):
        return gst
    # Fallback: seller_name
    gst = extract_gstin_from_text(seller_name)
    if is_valid_gstin(gst):
        return gst
    # Fallback: entire response
    gst = extract_gstin_from_text(response_text)
    return gst if is_valid_gstin(gst) else ""

# TDS determination (unchanged)

def determine_tds_rate(expense_ledger, tds_str=""):
    if tds_str and isinstance(tds_str, str):
        match = re.search(r'(\d+(\.\d+)?)%', tds_str)
        if match:
            return float(match.group(1))
        section_rates = {
            "194j": 10.0,
            "194c": 2.0,
            "194h": 5.0,
            "194i": 10.0,
            "194q": 1.0
        }
        for section, rate in section_rates.items():
            if section in tds_str.lower():
                return rate
    exp = expense_ledger.lower() if expense_ledger else ""
    if "professional" in exp or "service" in exp:
        return 10.0
    if "contract" in exp or "work" in exp:
        return 2.0
    if "commission" in exp:
        return 5.0
    if "rent" in exp:
        return 10.0
    if "advertis" in exp or "marketing" in exp:
        return 1.0
    return 0.0

# JSON extraction (unchanged)

def extract_json_from_response(text):
    try:
        matches = re.findall(r'```json\s*({.*?})\s*```', text, re.DOTALL)
        if matches:
            return json.loads(matches[0])
        start, end = text.find('{'), text.rfind('}')
        if start != -1 and end != -1:
            return json.loads(text[start:end+1])
        return json.loads(text)
    except:
        return None

# Main GPT prompt
main_prompt = (
    "Extract structured invoice data as a JSON object with the following keys: "
    "invoice_number, date, gstin, seller_name, buyer_name, buyer_gstin, "
    "taxable_amount, cgst, sgst, igst, place_of_supply, expense_ledger, tds. "
    "Important: 'taxable_amount' is the amount BEFORE taxes. "
    "Use DD/MM/YYYY for dates. Use only values shown in the invoice. "
    "Return 'NOT AN INVOICE' if clearly not one. If a value is not available, use null. "
    "SPECIAL GSTIN INSTRUCTIONS: Look near seller details; label may vary."
)

# File uploader
uploaded_files = st.file_uploader("📤 Upload scanned invoice PDFs", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    st.session_state["files_uploaded"] = True
    total_files = len(uploaded_files)
    for idx, file in enumerate(uploaded_files):
        file_name = file.name
        if file_name in st.session_state["processed_results"]:
            continue
        st.markdown(f"**Processing file: {file_name} ({idx+1}/{total_files})**")
        st.session_state["processing_status"][file_name] = "⏳ Pending..."
        temp_path = None
        try:
            # Save temp PDF
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file.getvalue())
                temp_path = tmp.name
            pdf_bytes = open(temp_path, "rb").read()
            first_image = convert_pdf_first_page(pdf_bytes)
            # Prepare image for GPT
            buf = io.BytesIO()
            first_image.save(buf, format="PNG")
            buf.seek(0)
            b64 = base64.b64encode(buf.read()).decode()
            # Chat completion
            messages=[
                {"role":"system","content":"You are a finance assistant specializing in Indian invoices."},
                {"role":"user","content":[{"type":"text","text":main_prompt},{"type":"image_url","image_url":{"url":f"data:image/png;base64,{b64}"}}]}
            ]
            resp = client.chat.completions.create(model="gpt-4o", messages=messages, max_tokens=1500)
            resp_text = resp.choices[0].message.content.strip()
            data = extract_json_from_response(resp_text)
            if data is None:
                if "not an invoice" in resp_text.lower():
                    row = {"File Name":file_name, "Invoice Number":"NOT AN INVOICE", **{f:n for f in ["Date","Seller Name","Seller GSTIN","Buyer Name","Buyer GSTIN","Taxable Amount","CGST","SGST","IGST","Total Amount","TDS Rate","TDS Amount","Amount Payable","Place of Supply","Expense Ledger","TDS","Narration"] for f in []}}
                else:
                    raise ValueError("Unexpected GPT response format")
            else:
                # Extract fields
                inv_no = data.get("invoice_number","")
                date = data.get("date","")
                seller_name = data.get("seller_name","")
                raw_gstin = data.get("gstin","")
                buyer_name = data.get("buyer_name","")
                buyer_gstin = data.get("buyer_gstin","")
                ta = safe_float(data.get("taxable_amount",0.0))
                cg = safe_float(data.get("cgst",0.0))
                sg = safe_float(data.get("sgst",0.0))
                ig = safe_float(data.get("igst",0.0))
                pos = data.get("place_of_supply","")
                exp_led = data.get("expense_ledger","")
                tds_str = data.get("tds","")
                # Compute derived
                total_amt = ta+cg+sg+ig
                tds_rate = determine_tds_rate(exp_led, tds_str)
                tds_amt = round(ta*tds_rate/100,2)
                payable = total_amt - tds_amt
                # Enhanced GSTIN handling
                seller_gstin = handle_gstin(raw_gstin, seller_name, resp_text)
                # Format date
                try:
                    parsed = parser.parse(date, dayfirst=True)
                    date = parsed.strftime("%d/%m/%Y")
                except:
                    date = date
                # Narration
                narration = (
                    f"Invoice {inv_no} dated {date} was issued by {seller_name} (GSTIN: {seller_gstin or 'N/A'}) "
                    f"to {buyer_name} (GSTIN: {buyer_gstin or 'N/A'}), taxable amount ₹{ta:,.2f}, "
                    f"CGST ₹{cg:,.2f}, SGST ₹{sg:,.2f}, IGST ₹{ig:,.2f}, total ₹{total_amt:,.2f}, "
                    f"TDS {tds_str or 'N/A'} @{tds_rate}% (₹{tds_amt:,.2f}), payable ₹{payable:,.2f}."
                )
                row = {
                    "File Name": file_name,
                    "Invoice Number": inv_no,
                    "Date": date,
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
                    "Place of Supply": pos,
                    "Expense Ledger": exp_led,
                    "TDS": tds_str,
                    "Narration": narration
                }
            st.session_state["processed_results"][file_name] = row
            st.session_state["processing_status"][file_name] = "✅ Done"
            st.success(f"{file_name}: ✅ Done")
        except Exception as e:
            err = {"File Name": file_name, "Error": str(e)}
            st.session_state["processed_results"][file_name] = err
            st.session_state["processing_status"][file_name] = "❌ Error"
            st.error(f"Error processing {file_name}: {e}")
        finally:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)

# Display results
results = list(st.session_state.get("processed_results", {}).values())
if results:
    if completed_json:
        st_lottie(completed_json, height=200, key="done_anim")
    st.markdown("<h3 style='text-align: center;'>🎉 All invoices processed!</h3>", unsafe_allow_html=True)
    df = pd.DataFrame(results)
    # Format columns
    for col in ["Taxable Amount","CGST","SGST","IGST","Total Amount","TDS Amount","Amount Payable"]:
        df[f"{col} (₹)"] = df[col].apply(format_currency)
    df["TDS Rate (%)"] = df["TDS Rate"].apply(lambda x: f"{x}%")
    cols = ["File Name","Invoice Number","Date","Seller Name","Seller GSTIN","Buyer Name","Buyer GSTIN",
            "Taxable Amount (₹)","CGST (₹)","SGST (₹)","IGST (₹)","Total Amount (₹)","TDS Rate (%)",
            "TDS Amount (₹)","Amount Payable (₹)","Place of Supply","Expense Ledger","TDS","Narration"]
    st.dataframe(df[cols])
    # Downloads
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download CSV", csv, "invoice_results.csv", "text/csv")
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name="Invoice Data")
    st.download_button("📥 Download Excel", buf.getvalue(), "invoice_results.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    st.markdown("---")
else:
    st.info("Upload one or more scanned invoices to get started.")
