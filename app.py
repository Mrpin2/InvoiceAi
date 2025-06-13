import streamlit as st
from PIL import Image
import openai
import fitz  # PyMuPDF
import io
import pandas as pd
import base64
import requests
import traceback
from streamlit_lottie import st_lottie

# ---------- Load Lottie Animation from URL ----------
def load_lottie_url(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_url = "https://assets2.lottiefiles.com/packages/lf20_3vbOcw.json"
lottie_json = load_lottie_url(lottie_url)

# ---------- UI CONFIGURATION ----------
st.set_page_config(layout="wide")
if lottie_json:
    st_lottie(lottie_json, height=200, key="animation")

st.title("📄 AI Invoice Extractor (ChatGPT)")
st.markdown("Upload scanned PDF invoices and extract clean finance data using ChatGPT")
st.markdown("---")

# ---------- Table Columns ----------
columns = [
    "File Name", "Vendor Name", "Invoice No", "Invoice Date", "Expense Ledger",
    "GST Type", "Tax Rate", "Basic Amount", "CGST", "SGST", "IGST",
    "Total Payable", "Narration", "GST Input Eligible", "TDS Applicable", "TDS Rate"
]

# ---------- Session State Init ----------
if "processed_results" not in st.session_state:
    st.session_state["processed_results"] = {}

# ---------- Auth ----------
openai_api_key = st.secrets["OPENAI_API_KEY"]
openai.api_key = openai_api_key
openai_model = "gpt-4o-mini"

# ---------- Prompts ----------
prompt = """
You are a finance assistant. Extract the following details from the invoice image:
Vendor Name, Invoice No, Invoice Date, Expense Ledger (Office Supplies, Legal, Travel, etc.),
GST Type (IGST/CGST+SGST/NA), Tax Rate %, Basic Amount, CGST, SGST, IGST,
Total Payable, Narration (short reason for bill), GST Input Eligible (Yes/No),
TDS Applicable (Yes/No), TDS Rate %.

⚠️ Return a single comma-separated line without headers or explanation. If it's not a valid invoice, just return:
NOT AN INVOICE
"""

# ---------- PDF to Image ----------
def convert_pdf_first_page(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc.load_page(0)
    pix = page.get_pixmap(dpi=200)
    return Image.open(io.BytesIO(pix.tobytes("png")))

# ---------- Upload ----------
uploaded_files = st.file_uploader("📤 Upload scanned invoice PDFs", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    for file in uploaded_files:
        file_name = file.name
        if file_name in st.session_state["processed_results"]:
            continue

        st.subheader(f"📄 Processing: {file_name}")
        try:
            pdf_data = file.read()
            first_image = convert_pdf_first_page(pdf_data)
        except Exception as e:
            st.error(f"❌ Error reading PDF: {e}")
            st.session_state["processed_results"][file_name] = [file_name] + ["NOT AN INVOICE"] + ["-"] * (len(columns) - 2)
            continue

        with st.spinner("🧠 Extracting with ChatGPT..."):
            try:
                img_buf = io.BytesIO()
                first_image.save(img_buf, format="PNG")
                img_buf.seek(0)
                b64_image = base64.b64encode(img_buf.read()).decode()

                messages = [
                    {"role": "system", "content": "You are a finance assistant."},
                    {"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image}"}}
                    ]}
                ]

                response = openai.ChatCompletion.create(
                    model=openai_model,
                    messages=messages,
                    max_tokens=1000
                )

                csv_line = response.choices[0].message.content.strip()

                if csv_line.upper().startswith("NOT AN INVOICE"):
                    row = [file_name] + ["NOT AN INVOICE"] + ["-"] * (len(columns) - 2)
                else:
                    row_data = [x.strip().strip('"') for x in csv_line.split(",")]
                    if len(row_data) >= len(columns) - 1:
                        row = [file_name] + row_data[:len(columns) - 1]
                    else:
                        row = [file_name] + ["NOT AN INVOICE"] + ["-"] * (len(columns) - 2)

                st.session_state["processed_results"][file_name] = row

            except Exception as e:
                st.error(f"❌ Error during AI processing: {e}")
                st.text_area("Debug Info", traceback.format_exc())
                st.session_state["processed_results"][file_name] = [file_name] + ["NOT AN INVOICE"] + ["-"] * (len(columns) - 2)

# ---------- Show Results ----------
results = list(st.session_state["processed_results"].values())
if results:
    df = pd.DataFrame(results, columns=columns)
    df.index += 1
    df.reset_index(inplace=True)
    df.rename(columns={"index": "S. No"}, inplace=True)

    st.success("✅ All invoices processed!")
    st.dataframe(df)

    csv_data = df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Results as CSV", csv_data, "invoice_results.csv", "text/csv")
    st.balloons()
else:
    st.info("Upload scanned invoices to get started.")
