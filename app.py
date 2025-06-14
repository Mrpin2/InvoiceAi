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

# Lottie animations
hello_lottie = "https://raw.githubusercontent.com/Mrpin2/InvoiceAi/refs/heads/main/Animation%20-%201749845212531.json"
completed_lottie = "https://raw.githubusercontent.com/Mrpin2/InvoiceAi/refs/heads/main/Animation%20-%201749845303699.json"

def load_lottie_json_safe(url):
    try:
        r = requests.get(url)
        r.raise_for_status()
        return r.json()
    except:
        return None

hello_json = load_lottie_json_safe(hello_lottie)
completed_json = load_lottie_json_safe(completed_lottie)

if "files_uploaded" not in st.session_state and hello_json:
    st_lottie(hello_json, height=200, key="hello")

st.markdown("<h2 style='text-align: center;'>📄 AI Invoice Extractor (OpenAI)</h2>", unsafe_allow_html=True)
st.markdown("Upload scanned PDF invoices and extract structured finance data using GPT-4 Vision")
st.markdown("---")

# Define extraction fields
fields = [
    "invoice_number", "date", "gstin", "seller_name", "buyer_name", "buyer_gstin",
    "taxable_amount", "cgst", "sgst", "igst", "place_of_supply", "expense_ledger", "tds"
]

# Session state initialization
for key in ["processed_results", "processing_status", "summary_rows"]:
    if key not in st.session_state:
        st.session_state[key] = {} if key != "summary_rows" else []

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

def convert_pdf_first_page(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc.load_page(0)
    pix = page.get_pixmap(dpi=300)
    return Image.open(io.BytesIO(pix.tobytes("png"))), doc

# Helpers

def safe_float(x):
    try:
        return float(str(x).replace(",", "").replace("₹", "").replace("$", "").strip())
    except:
        return 0.0


def format_currency(x):
    return f"₹{safe_float(x):,.2f}"


def is_valid_gstin(gstin):
    return bool(gstin and re.match(r"^[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[1-9A-Z]{1}[Z]{1}[0-9A-Z]{1}$", gstin))


def determine_tds_rate(expense_ledger, tds_str=""):
    # Look for explicit rate
    match = re.search(r"(\d+(?:\.\d+)?)%", str(tds_str))
    if match:
        return float(match.group(1))
    # Section-based
    for sec, rate in {"194j":10.0, "194c":2.0, "194h":5.0, "194i":10.0, "194q":1.0}.items():
        if sec in str(tds_str).lower():
            return rate
    # Ledger-based
    ld = (expense_ledger or "").lower()
    if any(x in ld for x in ["professional","consult","service"]): return 10.0
    if any(x in ld for x in ["contract","work"]): return 2.0
    if any(x in ld for x in ["commission","brokerage"]): return 5.0
    if "rent" in ld: return 10.0
    if any(x in ld for x in ["advertis","market"]): return 1.0
    return 0.0


def extract_json_from_response(text):
    try:
        # code-block extraction
        m = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
        if m: return json.loads(m.group(1))
        # plain JSON
        s, e = text.find('{'), text.rfind('}')
        if s>=0 and e>=0: return json.loads(text[s:e+1])
    except:
        pass
    return None

# Prompt setup
main_prompt = (
    "Extract structured invoice data as JSON with keys: " + ",".join(fields) + ". "
    "Use DD/MM/YYYY. Return 'NOT AN INVOICE' if not one."
)

# File upload
uploads = st.file_uploader("📤 Upload scanned invoice PDFs", type=["pdf"], accept_multiple_files=True)

if uploads:
    for i, f in enumerate(uploads,1):
        name = f.name
        if name in st.session_state.processed_results: continue
        st.markdown(f"**Processing ({i}/{len(uploads)}): {name}**")
        st.session_state.processing_status[name] = "⏳ Pending..."
        try:
            data = f.read()
            img, doc = convert_pdf_first_page(data)
            with st.spinner("🧠 Analyzing..."):
                buf = io.BytesIO(); img.save(buf, "PNG"); buf.seek(0)
                b64 = base64.b64encode(buf.read()).decode()
                msgs = [
                    {"role":"system","content":"You are a finance assistant."},
                    {"role":"user","content":[{"type":"text","text":main_prompt},{"type":"image_url","image_url":{"url":f"data:image/png;base64,{b64}"}}]}
                ]
                resp = client.chat.completions.create(model="gpt-4o", messages=msgs, max_tokens=1200)
                txt = resp.choices[0].message.content

            raw = extract_json_from_response(txt) or {}
            invoice_no = raw.get("invoice_number","")
            date = raw.get("date","")
            seller = raw.get("seller_name","")
            gstin = raw.get("gstin","")
            # fallback OCR regex
            if not is_valid_gstin(gstin):
                text_page = "".join(p.get_text("text") for p in doc)
                m = re.search(r"[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z][0-9A-Z]Z[0-9A-Z]", text_page)
                if m: gstin = m.group(0)
            gstin = gstin if is_valid_gstin(gstin) else "MISSING"

            buyer = raw.get("buyer_name","")
            buyer_gstin = raw.get("buyer_gstin","")
            ta = safe_float(raw.get("taxable_amount",0))
            cg = safe_float(raw.get("cgst",0))
            sg = safe_float(raw.get("sgst",0))
            ig = safe_float(raw.get("igst",0))
            pos = raw.get("place_of_supply","")
            exp = raw.get("expense_ledger","")
            tds_info = raw.get("tds","")

            total = ta+cg+sg+ig
            rate = determine_tds_rate(exp, tds_info)
            amt_tds = round(ta * rate/100,2)
            pay = total - amt_tds
            # TDS label
            tds_rate_label = "Not Applicable" if gstin=="MISSING" else (f"{rate}%" if rate>0 else "0%")

            row = {
                "File":name, "Invoice No":invoice_no, "Date":date,
                "Seller":seller, "GSTIN":gstin,
                "Buyer":buyer, "Buyer GSTIN":buyer_gstin,
                "Taxable":ta, "CGST":cg, "SGST":sg, "IGST":ig,
                "Total":total, "TDS Rate":tds_rate_label,
                "TDS Amt":amt_tds, "Payable":pay,
                "Place":pos, "Expense":exp, "Narration":"" }
            st.session_state.processed_results[name] = row
            st.session_state.processing_status[name] = "✅ Done"
        except Exception as e:
            st.session_state.processed_results[name] = {**{k:" " for k in ["File","Invoice No"]}, "Narration":f"Error: {e}"}
            st.session_state.processing_status[name] = "❌ Error"
            st.error(f"Error {name}: {e}")

# Display results
res = list(st.session_state.processed_results.values())
if res:
    if completed_json: st_lottie(completed_json, height=200, key="done")
    st.markdown("<h3 style='text-align:center;'>🎉 Done!</h3>",unsafe_allow_html=True)
    df = pd.DataFrame(res)
    # formatting
    for c in ["Taxable","CGST","SGST","IGST","Total","TDS Amt","Payable"]:
        df[f"{c} (₹)"] = df[c].apply(format_currency)
    # reorder
    disp = ["File","Invoice No","Date","Seller","GSTIN","Buyer","Taxable (₹)","CGST (₹)","SGST (₹)","IGST (₹)","Total (₹)","TDS Rate","TDS Amt (₹)","Payable (₹)","Place","Expense"]
    st.dataframe(df[disp])
    # Download
    st.download_button("📥 CSV", df.to_csv(index=False),"res.csv")
    buf=io.BytesIO(); df.to_excel(buf,index=False); st.download_button("📥 XLSX",buf.getvalue(),"res.xlsx")
else:
    st.info("Upload PDFs to start.")
