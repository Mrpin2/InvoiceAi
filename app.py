import streamlit as st
st.set_page_config(layout="wide")

from PIL import Image
import fitz
import io
import pandas as pd
import base64
import requests
from streamlit_lottie import st_lottie
from openai import OpenAI
import tempfile
import os
import locale
from dateutil import parser
import json

# Import shared utilities
from utils.general_utils import (
    load_lottie_json_safe,
    convert_pdf_first_page,
    safe_float,
    format_currency,
    is_valid_gstin,
    extract_gstin_from_text,
    handle_gstin,
    determine_tds_rate,
    extract_json_from_response
)

# Set locale for currency formatting
locale.setlocale(locale.LC_ALL, '')

# Lottie animation URLs
HELLO_LOTTIE_URL = (
    "https://raw.githubusercontent.com/Mrpin2/InvoiceAi/refs/heads/"
    "main/Animation%20-%201749845212531.json"
)
COMPLETED_LOTTIE_URL = (
    "https://raw.githubusercontent.com/Mrpin2/InvoiceAi/refs/heads/"
    "main/Animation%20-%201749845303699.json"
)

# Load animations
hello_json = load_lottie_json_safe(HELLO_LOTTIE_URL)
completed_json = load_lottie_json_safe(COMPLETED_LOTTIE_URL)

# Show initial animation once
if "files_uploaded" not in st.session_state and hello_json:
    st_lottie(hello_json, height=200, key="hello")

# Page header
st.markdown(
    "<h2 style='text-align: center;'>📄 AI Invoice Extractor (OpenAI)</h2>",
    unsafe_allow_html=True
)
st.markdown("Upload scanned PDF invoices and extract structured finance data using GPT-4 Vision")
st.markdown("---")

# Initialize session state
if "processed_results" not in st.session_state:
    st.session_state["processed_results"] = {}
if "processing_status" not in st.session_state:
    st.session_state["processing_status"] = {}
if "summary_rows" not in st.session_state:
    st.session_state["summary_rows"] = []

# Sidebar: API key
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

client = OpenAI(api_key=openai_api_key)

# Main GPT prompt
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
            # Save PDF to temp
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file.getvalue())
                temp_path = tmp.name

            # Convert first page to image
            pdf_bytes = open(temp_path, "rb").read()
            img = convert_pdf_first_page(pdf_bytes)

            # Encode image
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            buf.seek(0)
            b64 = base64.b64encode(buf.read()).decode()

            # Build messages
            messages = [
                {"role": "system", "content": "You are a finance assistant specializing in Indian invoices."},
                {"role": "user", "content": [
                    {"type": "text", "text": main_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
                ]}
            ]

            # GPT call
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=1500
            )
            resp_text = response.choices[0].message.content.strip()
            raw_data = extract_json_from_response(resp_text)

            # Prepare result_row
            if raw_data is None:
                if "not an invoice" in resp_text.lower():
                    row = {"File Name": name, "Invoice Number": "NOT AN INVOICE"}
                    for col in [
                        "Date","Seller Name","Seller GSTIN","Buyer Name","Buyer GSTIN",
                        "Taxable Amount","CGST","SGST","IGST","Total Amount",
                        "TDS Rate","TDS Amount","Amount Payable","Place of Supply",
                        "Expense Ledger","TDS","Narration"
                    ]:
                        row[col] = None
                else:
                    raise ValueError("GPT returned non-JSON response")
            else:
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
                tds_s = raw_data.get("tds", "")

                total_amt = ta + cg + sg + ig
                tds_rate = determine_tds_rate(ledger, tds_s)
                tds_amt = round(ta * tds_rate / 100, 2)
                payable = total_amt - tds_amt

                seller_gstin = handle_gstin(raw_gstin, seller_name, resp_text)

                try:
                    parsed = parser.parse(date_raw, dayfirst=True)
                    date_str = parsed.strftime("%d/%m/%Y")
                except:
                    date_str = date_raw

                narration = (
                    f"Invoice {inv_no} dated {date_str} was issued by {seller_name} "
                    f"(GSTIN: {seller_gstin or 'N/A'}) to {buyer_name} "
                    f"(GSTIN: {buyer_gstin or 'N/A'}), taxable amount ₹{ta:,.2f}, "
                    f"CGST ₹{cg:,.2f}, SGST ₹{sg:,.2f}, IGST ₹{ig:,.2f}, total ₹{total_amt:,.2f}, "
                    f"TDS {tds_s or 'N/A'} @{tds_rate}% (₹{tds_amt:,.2f}), payable ₹{payable:,.2f}."
                )

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
                    "TDS": tds_s,
                    "Narration": narration
                }

            st.session_state["processed_results"][name] = row
            st.session_state["processing_status"][name] = "✅ Done"
            st.success(f"{name}: ✅ Done")

        except Exception as e:
            err_row = {"File Name": name, "Error": str(e)}
            st.session_state["processed_results"][name] = err_row
            st.session_state["processing_status"][name] = "❌ Error"
            st.error(f"Error processing {name}: {e}")

        finally:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)

# Display final DataFrame if any results
results = list(st.session_state["processed_results"].values())
if results:
    if completed_json:
        st_lottie(completed_json, height=200, key="done")
    st.markdown(
        "<h3 style='text-align: center;'>🎉 All invoices processed successfully!</h3>",
        unsafe_allow_html=True
    )
    df = pd.DataFrame(results)
    for col in ["Taxable Amount","CGST","SGST","IGST","Total Amount","TDS Amount","Amount Payable"]:
        df[f"{col} (₹)"] = df[col].apply(format_currency)
    df["TDS Rate (%)"] = df["TDS Rate"].apply(lambda x: f"{x}%")
    display_cols = [
        "File Name","Invoice Number","Date","Seller Name","Seller GSTIN",
        "Buyer Name","Buyer GSTIN","Taxable Amount (₹)","CGST (₹)","SGST (₹)",
        "IGST (₹)","Total Amount (₹)","TDS Rate (%)","TDS Amount (₹)",
        "Amount Payable (₹)","Place of Supply","Expense Ledger","TDS","Narration"
    ]
    st.dataframe(df[display_cols])
    csv_data = df[display_cols].to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download CSV", csv_data, "invoice_results.csv", "text/csv")
    excel_buf = io.BytesIO()
    with pd.ExcelWriter(excel_buf, engine='xlsxwriter') as writer:
        df[display_cols].to_excel(writer, index=False, sheet_name="Invoice Data")
    st.download_button(
        "📥 Download Excel", excel_buf.getvalue(), "invoice_results.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    st.markdown("---")
else:
    st.info("Upload one or more scanned invoices to get started.")
