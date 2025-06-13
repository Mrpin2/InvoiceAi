import streamlit as st
from PIL import Image
import google.generativeai as genai
import openai
import fitz  # PyMuPDF
import io
import pandas as pd
import base64
import tempfile
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
st_lottie(lottie_json, height=200, key="animation")
st.markdown("<h2 style='text-align: center;'>\ud83d\udcc4 AI Invoice Extractor</h2>", unsafe_allow_html=True)
st.markdown("Upload scanned PDF invoices and extract clean finance data using Gemini or ChatGPT")
st.markdown("---")

# ---------- Table Columns ----------
columns = [
    "Vendor Name", "Invoice No", "Invoice Date", "Expense Ledger",
    "GST Type", "Tax Rate", "Basic Amount", "CGST", "SGST", "IGST",
    "Total Payable", "Narration", "GST Input Eligible", "TDS Applicable", "TDS Rate"
]

# ---------- Sidebar Auth ----------
st.sidebar.header("\ud83d\udd10 AI Config")
passcode = st.sidebar.text_input("Admin Passcode (optional)", type="password")
admin_unlocked = passcode == "Essenbee"

# ---------- Model Selection ----------
ai_model = None
model_choice = None
gemini_api_key = None
openai_api_key = None
openai_model = "gpt-4-vision-preview"

if admin_unlocked:
    st.sidebar.success("\ud83d\udd13 Admin access granted.")
    model_choice = st.sidebar.radio("Use which AI?", ["Gemini", "ChatGPT"])

    if model_choice == "Gemini":
        gemini_api_key = "AIzaSyA5Jnd7arMlbZ1x_ZpiE-AezrmsaXams7Y"
        genai.configure(api_key=gemini_api_key)
        ai_model = genai.GenerativeModel("gemini-1.5-flash-latest")

    elif model_choice == "ChatGPT":
        openai_api_key = "sk-admin-openai-key-here"  # <-- Replace with real key
        openai.api_key = openai_api_key

else:
    model_choice = st.sidebar.radio("Choose AI Model", ["Gemini", "ChatGPT"])
    if model_choice == "Gemini":
        gemini_api_key = st.sidebar.text_input("\ud83d\udd11 Enter your Gemini API Key", type="password")
        if gemini_api_key:
            genai.configure(api_key=gemini_api_key)
            ai_model = genai.GenerativeModel("gemini-1.5-flash-latest")
    elif model_choice == "ChatGPT":
        openai_api_key = st.sidebar.text_input("\ud83d\udd11 Enter your OpenAI API Key", type="password")
        if openai_api_key:
            openai.api_key = openai_api_key

# ---------- PDF UPLOAD ----------
uploaded_files = st.file_uploader("\ud83d\udcc4 Upload scanned invoice PDFs", type=["pdf"], accept_multiple_files=True)
results = []

if uploaded_files:
    for file in uploaded_files:
        st.subheader(f"\ud83d\udcc4 Processing: {file.name}")
        try:
            # Save to temp file (Gemini needs path)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(file.read())
                temp_path = tmp_file.name

            with st.spinner("\ud83e\uddec Extracting data using AI..."):
                prompt = """
                You are a professional finance assistant. Extract the following fields from the invoice:
                Vendor Name, Invoice No, Invoice Date, Expense Ledger (Office Supplies, Travel, etc.),
                GST Type (IGST, CGST+SGST, NA), Tax Rate (%, single value), Basic Amount,
                CGST, SGST, IGST, Total Payable, Narration (1 line),
                GST Input Eligible (Yes/No), TDS Applicable (Yes/No), TDS Rate (%).
                Respond in this exact CSV order:
                Vendor Name, Invoice No, Invoice Date, Expense Ledger,
                GST Type, Tax Rate, Basic Amount, CGST, SGST, IGST,
                Total Payable, Narration, GST Input Eligible, TDS Applicable, TDS Rate.
                """

                if model_choice == "Gemini" and gemini_api_key:
                    file_resource = genai.upload_file(path=temp_path)
                    response = ai_model.generate_content(
                        contents=[prompt, file_resource],
                        generation_config={"response_mime_type": "text/plain"}
                    )
                    csv_line = response.text.strip()

                elif model_choice == "ChatGPT" and openai_api_key:
                    doc = fitz.open(temp_path)
                    page = doc.load_page(0)
                    pix = page.get_pixmap(dpi=200)
                    image = Image.open(io.BytesIO(pix.tobytes("png")))
                    img_buf = io.BytesIO()
                    image.save(img_buf, format="PNG")
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
                    raise Exception("\u274c No valid API key provided.")

                row = [x.strip() for x in csv_line.split(",")]
                if len(row) != len(columns):
                    raise ValueError("Mismatch in extracted field count.")
                results.append(row)

        except Exception as e:
            st.error(f"\u274c Error processing {file.name}: {e}")
            st.text_area(f"Raw Output ({file.name})", traceback.format_exc())

# ---------- DISPLAY RESULTS ----------
if results:
    df = pd.DataFrame(results, columns=columns)
    st.success("\u2705 All invoices processed!")
    st.dataframe(df)

    csv = df.to_csv(index=False).encode()
    st.download_button("\ud83d\udcc5 Download Extracted Data", csv, "invoice_data.csv", "text/csv")
    st.balloons()
else:
    st.info("Upload one or more scanned invoices to get started.")
