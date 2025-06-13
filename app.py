import streamlit as st
from PIL import Image
import google.generativeai as genai
import openai
import fitz  # PyMuPDF
import io
import pandas as pd
import base64
import requests
import tempfile
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
st_lottie(lottie_json, height=200, key="animation")
st.markdown("""<h2 style='text-align: center;'>📄 AI Invoice Extractor</h2>""", unsafe_allow_html=True)
st.markdown("Upload scanned PDF invoices and extract clean finance data using Gemini or ChatGPT")
st.markdown("---")

# ---------- Table Columns ----------
columns = [
    "File Name", "Vendor Name", "Invoice No", "Invoice Date", "Expense Ledger",
    "GST Type", "Tax Rate", "Basic Amount", "CGST", "SGST", "IGST",
    "Total Payable", "Narration", "GST Input Eligible", "TDS Applicable", "TDS Rate"
]

# ---------- Sidebar Auth ----------
st.sidebar.header("🔐 AI Config")
passcode = st.sidebar.text_input("Admin Passcode (optional)", type="password")
admin_unlocked = passcode == "Essenbee"

# ---------- Model Selection ----------
ai_model = None
model_choice = None
gemini_api_key = None
openai_api_key = None
openai_model = "gpt-4-vision-preview"

if admin_unlocked:
    st.sidebar.success("🔓 Admin access granted.")
    model_choice = st.sidebar.radio("Use which AI?", ["Gemini", "ChatGPT"])

    if model_choice == "Gemini":
        gemini_api_key = "AIzaSyA5Jnd7arMlbZ1x_ZpiE-AezrmsaXams7Y"
        genai.configure(api_key=gemini_api_key)
        ai_model = genai.GenerativeModel("gemini-1.5-flash-latest")

    elif model_choice == "ChatGPT":
        openai_api_key = "sk-admin-openai-key-here"  # Replace with real admin key
        openai.api_key = openai_api_key

else:
    model_choice = st.sidebar.radio("Choose AI Model", ["Gemini", "ChatGPT"])
    if model_choice == "Gemini":
        gemini_api_key = st.sidebar.text_input("🔑 Enter your Gemini API Key", type="password")
        if gemini_api_key:
            genai.configure(api_key=gemini_api_key)
            ai_model = genai.GenerativeModel("gemini-1.5-flash-latest")
    elif model_choice == "ChatGPT":
        openai_api_key = st.sidebar.text_input("🔑 Enter your OpenAI API Key", type="password")
        if openai_api_key:
            openai.api_key = openai_api_key

# ---------- PDF to Image ----------
def convert_pdf_first_page(file_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    doc = fitz.open(tmp_path)
    page = doc.load_page(0)
    pix = page.get_pixmap(dpi=200)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    return img

# ---------- PDF UPLOAD ----------
uploaded_files = st.file_uploader("📤 Upload scanned invoice PDFs", type=["pdf"], accept_multiple_files=True)

results = []
if uploaded_files:
    for file in uploaded_files:
        st.subheader(f"📄 Processing: {file.name}")
        try:
            file_bytes = file.read()
            first_image = convert_pdf_first_page(file_bytes)
        except Exception as e:
            st.error(f"❌ Error reading PDF: {e}")
            continue

        with st.spinner("🧠 Extracting data using AI..."):
            prompt = """
            You are a professional finance assistant. Only process invoices. Ignore bank statements or non-invoice documents.
            Extract the following fields from the invoice image:
            Vendor Name, Invoice No, Invoice Date, Expense Ledger (like Office Supplies, Travel, Legal Fees, etc.),
            GST Type (IGST or CGST+SGST or NA), Tax Rate (%, single value), Basic Amount,
            CGST, SGST, IGST, Total Payable, Narration (short sentence),
            GST Input Eligible (Yes/No — No if travel, food, hotel, etc.),
            TDS Applicable (Yes/No), TDS Rate (in % if applicable).
            Respond with CSV-style values in this exact order:
            Vendor Name, Invoice No, Invoice Date, Expense Ledger,
            GST Type, Tax Rate, Basic Amount, CGST, SGST, IGST,
            Total Payable, Narration, GST Input Eligible, TDS Applicable, TDS Rate.
            If the document is not an invoice, respond with 'NOT AN INVOICE'.
            """
            try:
                if model_choice == "Gemini" and gemini_api_key:
                    response = ai_model.generate_content([first_image, prompt])
                    csv_line = response.text.strip()

                elif model_choice == "ChatGPT" and openai_api_key:
                    img_buf = io.BytesIO()
                    first_image.save(img_buf, format="PNG")
                    img_buf.seek(0)
                    base64_image = base64.b64encode(img_buf.read()).decode()
                    chat_prompt = [
                        {"role": "system", "content": "You are a finance assistant."},
                        {"role": "user", "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                        ]}
                    ]
                    response = openai.ChatCompletion.create(
                        model=openai_model,
                        messages=chat_prompt,
                        max_tokens=1000
                    )
                    csv_line = response.choices[0].message.content.strip()
                else:
                    raise Exception("❌ No valid API key provided.")

                if csv_line.strip().lower().startswith("not an invoice"):
                    st.warning(f"Skipped {file.name} — AI determined it's not an invoice.")
                    results.append([file.name] + ["NOT AN INVOICE"] + ["-"] * (len(columns) - 2))
                    continue

                lines = [line for line in csv_line.strip().splitlines() if line.strip()]
                matched = False
                for line in lines:
                    row = [x.strip() for x in line.split(",")]
                    if len(row) >= len(columns) - 1:
                        row = row[:len(columns) - 1]
                        results.append([file.name] + row)
                        matched = True
                        break

                if not matched:
                    st.warning(f"Likely not invoice or could not parse {file.name}.")
                    st.text_area(f"Raw Output ({file.name})", csv_line)
                    results.append([file.name] + ["NOT AN INVOICE"] + ["-"] * (len(columns) - 2))

            except Exception as e:
                st.error(f"❌ Error processing {file.name}: {e}")
                st.text_area(f"Raw Output ({file.name})", traceback.format_exc())
                results.append([file.name] + ["NOT AN INVOICE"] + ["-"] * (len(columns) - 2))

# ---------- DISPLAY RESULTS ----------
if results:
    df = pd.DataFrame(results, columns=columns)
    st.success("✅ All invoices processed!")
    st.dataframe(df)

    csv = df.to_csv(index=False).encode()
    st.download_button("📥 Download Extracted Data", csv, "invoice_data.csv", "text/csv")
    st.balloons()
else:
    st.info("Upload one or more scanned invoices to get started.")
