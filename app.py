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

# Animations
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

st.markdown("<h2 style='text-align: center;'>📄 AI Invoice Extractor (ChatGPT)</h2>", unsafe_allow_html=True)
st.markdown("Upload scanned PDF invoices and extract clean finance data using ChatGPT Vision")
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

main_prompt = """<same as previous long prompt>"""

def is_placeholder_row(text):
    return all(x.lower() in text.lower() for x in ["Vendor Name", "Invoice No", "Invoice Date", "Expense Ledger"])

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
                    result_row = ["NOT AN INVOICE"] + ["-"] * (len(columns) - 1)
                else:
                    matched = False
                    for line in csv_line.strip().split("\n"):
                        try:
                            row = [x.strip().strip('"') for x in line.split(",")]
                            if len(row) >= len(columns):
                                row = row[:len(columns)]
                                narration = row[15].lower()
                                full_text = narration + ' ' + ' '.join(row).lower()

                                if row[7].strip().lower() in ["missing", "", "consulting"]:
                                    if any(word in full_text for word in ["professional", "consulting", "advisory", "legal"]):
                                        row[7] = "Professional Fees"
                                    elif any(word in full_text for word in ["software", "subscription", "license"]):
                                        row[7] = "Software Subscription"
                                    elif any(word in full_text for word in ["marketing", "branding", "ads"]):
                                        row[7] = "Marketing"
                                    elif any(word in full_text for word in ["travel", "flight", "hotel"]):
                                        row[7] = "Travel"

                                if "194j" in full_text or "professional" in full_text:
                                    row[17] = "Yes - Section 194J"
                                    if not row[18].strip() or row[18].lower() == "missing":
                                        row[18] = "10%"
                                elif "194c" in full_text:
                                    row[17] = "Yes - Section 194C"
                                    if not row[18].strip() or row[18].lower() == "missing":
                                        row[18] = "2%"
                                elif "194h" in full_text:
                                    row[17] = "Yes - Section 194H"
                                    if not row[18].strip() or row[18].lower() == "missing":
                                        row[18] = "5%"

                                result_row = row
                                matched = True
                                break
                        except Exception:
                            pass
                    if not matched:
                        result_row = ["NOT AN INVOICE"] + ["-"] * (len(columns) - 1)

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

    sanitized_results = []
    for r in results:
        if len(r) == len(columns):
            sanitized_results.append(r)
        elif len(r) == len(columns) + 1:
            sanitized_results.append(r[1:])
        elif len(r) < len(columns):
            padded = r + ["-"] * (len(columns) - len(r))
            sanitized_results.append(padded)
        else:
            sanitized_results.append(r[:len(columns)])

    df = pd.DataFrame(sanitized_results, columns=columns)
    df.insert(0, "S. No", range(1, len(df) + 1))

    display_columns = ["S. No"] + columns
    st.dataframe(df[display_columns])

    csv_data = df[display_columns].to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Results as CSV", csv_data, "invoice_results.csv", "text/csv")

    st.markdown("---")
    if st.session_state.summary_rows:
        st.balloons()
else:
    st.info("Upload one or more scanned invoices to get started.")
