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
st.markdown("Upload scanned PDF invoices and extract structured finance data using GPT-4 Vision")
st.markdown("---")

columns = [
    "Vendor Name", "Invoice No", "GSTIN", "HSN/SAC", "Buyer Name", "Invoice Date",
    "Basic Amount", "CGST", "SGST", "IGST", "Total Payable", "Gross Amount", "TDS Rate", "TDS Amount", "Net Payable",
    "GST Type", "Tax Rate", "Place of Supply", "Expense Ledger", "Narration", "GST Input Eligible", "TDS Applicable", "Currency"
]

if "processed_results" not in st.session_state:
    st.session_state["processed_results"] = {}
if "processing_status" not in st.session_state:
    st.session_state["processing_status"] = {}
if "summary_rows" not in st.session_state:
    st.session_state["summary_rows"] = []

st.sidebar.header("🔐 AI Config")
passcode = st.sidebar.text_input("Admin Passcode", type="password")
admin_unlocked = passcode == "Essenbee"

openai_api_key = None
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
    return Image.open(io.BytesIO(pix.tobytes("png")))

def safe_float(x):
    try:
        return float(str(x).replace(",", "").replace("₹", "").strip())
    except:
        return 0.0

def format_currency(x):
    try:
        return f"₹{safe_float(x):,.2f}"
    except:
        return "₹0.00"

def is_valid_gstin(gstin):
    return bool(re.match(r"^[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[1-9A-Z]{1}[Z]{1}[0-9A-Z]{1}$", gstin))

def smart_tds_rate(section):
    if "194j" in section.lower(): return 10
    if "194c" in section.lower(): return 2
    if "194h" in section.lower(): return 5
    return 0

main_prompt = (
    "Extract structured invoice data as a JSON object with keys: Vendor Name, Invoice No, GSTIN, HSN/SAC, Buyer Name, Place of Supply, "
    "Invoice Date, Expense Ledger, GST Type, Tax Rate, Basic Amount, CGST, SGST, IGST, Total Payable, Narration, GST Input Eligible, TDS Applicable, TDS Rate. "
    "Use DD/MM/YYYY for dates. Use only values shown in the invoice. Return 'NOT AN INVOICE' if clearly not one."
)

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

        temp_file_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file.read())
                temp_file_path = tmp.name

            pdf_data = open(temp_file_path, "rb").read()
            first_image = convert_pdf_first_page(pdf_data)

            with st.spinner("🧠 Extracting data using GPT-4 Vision..."):
                img_buf = io.BytesIO()
                first_image.save(img_buf, format="PNG")
                img_buf.seek(0)
                base64_image = base64.b64encode(img_buf.read()).decode()

                chat_prompt = [
                    {"role": "system", "content": "You are a finance assistant."},
                    {"role": "user", "content": [
                        {"type": "text", "text": main_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                    ]}
                ]

                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=chat_prompt,
                    max_tokens=1000
                )

                response_text = response.choices[0].message.content.strip()
                if not response_text:
                    raise ValueError("Empty response from OpenAI")

                try:
                    raw_data = json.loads(response_text)
                    row = [raw_data.get(field, "") for field in columns]
                    row += ["INR"]

                    if not is_valid_gstin(row[2]):
                        row[2] = "MISSING"

                    try:
                        parsed_date = parser.parse(row[6], dayfirst=False)
                        row[6] = parsed_date.strftime("%d/%m/%Y")
                    except:
                        row[6] = ""

                    if len(row[7].strip()) < 3 or re.match(r"^\d{2}/\d{2}/\d{4}$", row[7]):
                        row[7] = "MISSING"

                    for i in [10, 11, 12, 13, 14]:
                        if not re.match(r"^\d+(\.\d+)?$", str(row[i]).replace(",", "").replace("₹", "")):
                            row[i] = "0"

                    if "%" in str(row[9]):
                        row[9] = str(row[9]).replace("%", "")

                    narration_text = (
                        f"Invoice {row[1]} dated {row[6]} was issued by {row[0]} (GSTIN: {row[2]}) "
                        f"to {row[4]} (GSTIN: {row[4]}), with a total value of ₹{row[14]}. "
                        f"Taxes applied - CGST: ₹{row[11] or '0.00'}, SGST: ₹{row[12] or '0.00'}, "
                        f"IGST: ₹{row[13] or '0.00'}. " + (f"Place of supply: {row[5]}. " if row[5] else "") +
                        f"Expense: {row[7]}. TDS: {row[17]}."
                    )
                    row[15] = narration_text
                    result_row = row

                except json.JSONDecodeError:
                    if "not an invoice" in response_text.lower():
                        result_row = ["NOT AN INVOICE"] + ["-"] * (len(columns) - 1)
                    else:
                        st.text_area("❌ Raw GPT Response", value=response_text, height=200)
                        raise ValueError("Unable to decode JSON from GPT response.")

                st.session_state["processed_results"][file_name] = result_row
                st.session_state["processing_status"][file_name] = "✅ Done"
                completed_count += 1
                st.success(f"{file_name}: ✅ Done")

        except Exception as e:
            st.session_state["processed_results"][file_name] = ["NOT AN INVOICE"] + ["-"] * (len(columns) - 1)
            st.session_state["processing_status"][file_name] = "❌ Error"
            st.error(f"❌ Error processing {file_name}: {e}")
            st.text_area(f"Raw Output ({file_name})", traceback.format_exc())

        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

results = list(st.session_state["processed_results"].values())
if results:
    if completed_json:
        st_lottie(completed_json, height=200, key="done_animation")

    st.markdown("<h3 style='text-align: center;'>🎉 Yippie! All invoices processed with a smile 😊</h3>", unsafe_allow_html=True)

    df = pd.DataFrame(results, columns=columns)
    df.insert(0, "S. No", range(1, len(df) + 1))

    df["TDS Amount"] = df.apply(lambda row: round(safe_float(row["Basic Amount"]) * smart_tds_rate(str(row["TDS Applicable"])) / 100, 2), axis=1)
    df["Gross Amount"] = df.apply(lambda row: sum([
        safe_float(row["Basic Amount"]),
        safe_float(row["CGST"]),
        safe_float(row["SGST"]),
        safe_float(row["IGST"])
    ]), axis=1)
    df["Net Payable"] = df["Gross Amount"] - df["TDS Amount"]

    for col in ["Basic Amount", "CGST", "SGST", "IGST", "Total Payable", "Gross Amount", "TDS Amount", "Net Payable"]:
        df[f"{col} (₹)"] = df[col].apply(format_currency)

    st.dataframe(df)

    csv_data = df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Results as CSV", csv_data, "invoice_results.csv", "text/csv")

    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name="Invoice Data")
    st.download_button(
        label="📥 Download Results as Excel",
        data=excel_buffer.getvalue(),
        file_name="invoice_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    st.markdown("---")
    if st.session_state.summary_rows:
        st.balloons()
else:
    st.info("Upload one or more scanned invoices to get started.")
