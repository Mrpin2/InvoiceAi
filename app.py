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

# Lottie Animations
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
    "Vendor Name", "Invoice No", "GSTIN", "HSN/SAC", "Buyer Name", "Place of Supply", "Invoice Date", "Expense Ledger",
    "GST Type", "Tax Rate", "Basic Amount", "CGST", "SGST", "IGST",
    "Total Payable", "Narration", "GST Input Eligible", "TDS Applicable", "TDS Rate"
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

main_prompt = (
    "Extract all relevant and clear information from the invoice, adhering to Indian standards "
    "for dates (DD/MM/YYYY or DD-MM-YYYY) and codes (like GSTIN, HSN/SAC). "
    "Accurately identify the total amount payable. Classify the nature of expense and suggest an "
    "applicable ledger type (e.g., 'Office Supplies', 'Professional Fees', 'Software Subscription'). "
    "Determine TDS applicability (e.g., 'Yes - Section 194J', 'No', 'Uncertain'). "
    "Determine reverse charge GST (RCM) applicability (e.g., 'Yes', 'No', 'Uncertain'). "
    "Handle missing data appropriately by setting fields to null or an empty string where "
    "Optional, and raise an issue if critical data is missing for required fields. "
    "Do not make assumptions or perform calculations beyond what's explicitly stated in the invoice text. "
    "If a value is clearly zero, represent it as 0.0 for floats. For dates, prefer DD/MM/YYYY.\n"
    "Return the following fields strictly as a comma-separated line in this order:"
    "Vendor Name, Invoice No, GSTIN, HSN/SAC, Buyer Name, Place of Supply, Invoice Date, Expense Ledger, "
    "GST Type, Tax Rate, Basic Amount, CGST, SGST, IGST, Total Payable, Narration, GST Input Eligible, "
    "TDS Applicable, TDS Rate"
)

def convert_pdf_first_page(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc.load_page(0)
    pix = page.get_pixmap(dpi=300)
    return Image.open(io.BytesIO(pix.tobytes("png")))

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
                csv_line = response.choices[0].message.content.strip()
                if csv_line.upper().startswith("NOT AN INVOICE"):
                    result_row = ["NOT AN INVOICE"] + ["-"] * (len(columns) - 1)
                else:
                    row = [x.strip().strip('"') for x in csv_line.split(",")]
                    result_row = row[:len(columns)] if len(row) >= len(columns) else row + ["-"] * (len(columns) - len(row))

                    # Smart narration
                    try:
                        narration_text = (
                            f"Invoice {result_row[1]} dated {result_row[6]} was issued by {result_row[0]} (GSTIN: {result_row[2]}) "
                            f"to {result_row[4]} (GSTIN: {result_row[4]}), with a total value of ₹{result_row[14]}. "
                            f"Taxes applied - CGST: ₹{result_row[11] or '0.00'}, SGST: ₹{result_row[12] or '0.00'}, "
                            f"IGST: ₹{result_row[13] or '0.00'}. Place of supply: {result_row[5]}. "
                            f"Expense: {result_row[7]}. TDS: {result_row[17]}."
                        )
                        result_row[15] = narration_text
                    except Exception:
                        pass

                st.session_state["processed_results"][file_name] = result_row
                st.session_state["processing_status"][file_name] = "✅ Done"
                completed_count += 1
                st.success(f"{file_name}: ✅ Done")
                st.info(f"🤖 {completed_count} out of {total_files} files processed")

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

    # Add calculated columns
    df["TDS Amount"] = df.apply(lambda row: round(float(row["Basic Amount"]) * 0.10, 2) if "194j" in str(row["TDS Applicable"]).lower() else 0.0, axis=1)
    df["Gross Amount"] = df.apply(lambda row: sum([
        float(row["Basic Amount"] or 0),
        float(row["CGST"] or 0),
        float(row["SGST"] or 0),
        float(row["IGST"] or 0)
    ]), axis=1)
    df["Net Payable"] = df["Gross Amount"] - df["TDS Amount"]

    st.dataframe(df)

    csv_data = df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Results as CSV", csv_data, "invoice_results.csv", "text/csv")

    st.markdown("---")
    if st.session_state.summary_rows:
        st.balloons()
else:
    st.info("Upload one or more scanned invoices to get started.")
