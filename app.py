import streamlit as st
from PIL import Image
import fitz  # PyMuPDF
import io
import pandas as pd
import base64
import requests
import traceback
from streamlit_lottie import st_lottie
from openai import OpenAI
import tempfile
import os

# ---------- Lottie Animations ----------
hello_lottie = "https://assets1.lottiefiles.com/packages/lf20_touohxv0.json"
rocket_lottie = "https://assets10.lottiefiles.com/packages/lf20_dyflv0zz.json"
pop_lottie = "https://assets10.lottiefiles.com/packages/lf20_qp1q7mct.json"
balloons_lottie = "https://assets4.lottiefiles.com/private_files/lf30_m3pibtur.json"
processing_gif_url = "https://raw.githubusercontent.com/Mrpin2/InvoiceAi/main/processing.gif"

def load_lottie_json_safe(url):
    try:
        r = requests.get(url)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as req_err:
        st.warning(f"⚠️ Request error for {url}: {req_err}")
    except ValueError:
        st.warning(f"⚠️ Not a valid Lottie JSON at {url}")
    except Exception as e:
        st.warning(f"⚠️ Unexpected error for {url}: {e}")
    return None

hello_json = load_lottie_json_safe(hello_lottie)
processing_json = load_lottie_json_safe(rocket_lottie)
pop_json = load_lottie_json_safe(pop_lottie)
balloon_json = load_lottie_json_safe(balloons_lottie)

# ---------- UI CONFIGURATION ----------
st.set_page_config(layout="wide")

if "files_uploaded" not in st.session_state:
    if hello_json:
        st_lottie(hello_json, height=200, key="hello")

st.markdown("<h2 style='text-align: center;'>📄 AI Invoice Extractor (ChatGPT)</h2>", unsafe_allow_html=True)
st.markdown("Upload scanned PDF invoices and extract clean finance data using ChatGPT Vision")
st.markdown("---")

# ---------- Table Columns ----------
columns = [
    "File Name", "Vendor Name", "Invoice No", "GSTIN", "HSN/SAC", "Buyer Name", "Place of Supply", "Invoice Date", "Expense Ledger",
    "GST Type", "Tax Rate", "Basic Amount", "CGST", "SGST", "IGST",
    "Total Payable", "Narration", "GST Input Eligible", "TDS Applicable", "TDS Rate"
]

# ---------- Session State ----------
if "processed_results" not in st.session_state:
    st.session_state["processed_results"] = {}
if "processing_status" not in st.session_state:
    st.session_state["processing_status"] = {}

# ---------- Sidebar Auth ----------
st.sidebar.header("🔐 AI Config")
passcode = st.sidebar.text_input("Admin Passcode", type="password")
admin_unlocked = passcode == "Essenbee"

# ---------- API Key ----------
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

# ---------- Prompt ----------
main_prompt = """
You are a professional assistant. Read this scanned document and extract the following:

Vendor Name, Invoice No, GSTIN, HSN/SAC, Buyer Name, Place of Supply, Invoice Date, Expense Ledger,
GST Type, Tax Rate, Basic Amount, CGST, SGST, IGST, Total Payable, Narration, GST Input Eligible, TDS Applicable, TDS Rate

If it's a valid invoice, respond with a single comma-separated line in that exact order (no labels, no newlines, no extra words).
If the file is clearly NOT a GST invoice (e.g. bank statement), only then say:
NOT AN INVOICE
"""

def is_placeholder_row(text):
    placeholder_keywords = ["Vendor Name", "Invoice No", "Invoice Date", "Expense Ledger"]
    return all(x.lower() in text.lower() for x in placeholder_keywords)

# ---------- Convert PDF to Image ----------
def convert_pdf_first_page(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc.load_page(0)
    pix = page.get_pixmap(dpi=300)
    return Image.open(io.BytesIO(pix.tobytes("png")))

# ---------- Upload PDFs ----------
uploaded_files = st.file_uploader("📤 Upload scanned invoice PDFs", type=["pdf"], accept_multiple_files=True)
if uploaded_files:
    st.session_state["files_uploaded"] = True
    st.markdown(f"<div style='text-align:center;'><img src='{processing_gif_url}' height='180px'><br><b>Processing! Hold On...</b></div>", unsafe_allow_html=True)

    total_files = len(uploaded_files)
    completed_count = 0

    for idx, file in enumerate(uploaded_files):
        file_name = file.name

        if file_name in st.session_state["processed_results"]:
            continue

        st.markdown(f"**Processing file: {file_name} ({idx+1}/{total_files})**")
        st.session_state["processing_status"][file_name] = "⏳ Pending..."
        st.info(f"{file_name}: ⏳ Pending...")

        temp_file_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file.read())
                temp_file_path = tmp.name

            pdf_data = open(temp_file_path, "rb").read()
            first_image = convert_pdf_first_page(pdf_data)

            with st.spinner("🧠 Extracting data using ChatGPT..."):
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

                if csv_line.upper().startswith("NOT AN INVOICE") or is_placeholder_row(csv_line):
                    result_row = [file_name] + ["NOT AN INVOICE"] + ["-"] * (len(columns) - 2)
                else:
                    matched = False
                    for line in csv_line.strip().split("\n"):
                        try:
                            row = [x.strip().strip('"') for x in line.split(",")]
                            if len(row) >= len(columns) - 1:
                                result_row = [file_name] + row[:len(columns) - 1]
                                matched = True
                                break
                        except Exception:
                            pass
                    if not matched:
                        result_row = [file_name] + ["NOT AN INVOICE"] + ["-"] * (len(columns) - 2)

                st.session_state["processed_results"][file_name] = result_row
                st.session_state["processing_status"][file_name] = "✅ Done"
                completed_count += 1
                st.success(f"{file_name}: ✅ Done")
                st.info(f"🤖 {completed_count} out of {total_files} files processed")

        except Exception as e:
            st.session_state["processed_results"][file_name] = [file_name] + ["NOT AN INVOICE"] + ["-"] * (len(columns) - 2)
            st.session_state["processing_status"][file_name] = "❌ Error"
            st.error(f"❌ Error processing {file_name}: {e}")
            st.text_area(f"Raw Output ({file_name})", traceback.format_exc())

        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

# ---------- Display Results ----------
results = list(st.session_state["processed_results"].values())
if results:
    if processing_json:
        st_lottie(processing_json, height=180, key="complete")
    if pop_json:
        st_lottie(pop_json, height=160, key="popcelebrate")
    if balloon_json:
        st_lottie(balloon_json, height=120, key="balloons")

    st.markdown("<h3 style='text-align: center;'>🎉 Yippie! All invoices processed with a smile 😊</h3>", unsafe_allow_html=True)

    df = pd.DataFrame(results, columns=columns)
    df.insert(0, "S. No", range(1, len(df) + 1))
    st.dataframe(df)

    csv_data = df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Results as CSV", csv_data, "invoice_results.csv", "text/csv")
else:
    st.info("Upload one or more scanned invoices to get started.")
