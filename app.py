import streamlit as st
from PIL import Image
import google.generativeai as genai
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
st_lottie(lottie_json, height=200, key="animation")
st.markdown("<h2 style='text-align: center;'>📄 AI Invoice Extractor</h2>", unsafe_allow_html=True)
st.markdown("Upload scanned PDF invoices and extract clean finance data using Gemini or ChatGPT")
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
        gemini_api_key = st.secrets.get("GEMINI_API_KEY")
        if gemini_api_key:
            genai.configure(api_key=gemini_api_key)
        else:
            st.sidebar.error("🔑 GEMINI_API_KEY not found in Streamlit secrets.")

    elif model_choice == "ChatGPT":
        openai_api_key = "sk-admin-openai-key-here"
        openai.api_key = openai_api_key

else:
    model_choice = st.sidebar.radio("Choose AI Model", ["Gemini", "ChatGPT"])
    if model_choice == "Gemini":
        gemini_api_key = st.sidebar.text_input("🔑 Enter your Gemini API Key", type="password")
        if gemini_api_key:
            genai.configure(api_key=gemini_api_key)
    elif model_choice == "ChatGPT":
        openai_api_key = st.sidebar.text_input("🔑 Enter your OpenAI API Key", type="password")
        if openai_api_key:
            openai.api_key = openai_api_key

# ---------- Prompts ----------
strict_prompt = """
You are a professional finance assistant. If the uploaded document is NOT a proper GST invoice
(e.g., if it's a bank statement, email, quote, or missing required fields), respond with exactly:
NOT AN INVOICE

Otherwise, extract the following values from the invoice:

Vendor Name, Invoice No, Invoice Date, Expense Ledger (like Office Supplies, Travel, Legal Fees, etc.),
GST Type (IGST or CGST+SGST or NA), Tax Rate (%, only the rate like 5, 12, 18), Basic Amount (before tax),
CGST, SGST, IGST, Total Payable (after tax), Narration (short meaningful line about the expense),
GST Input Eligible (Yes/No — mark No if food, hotel, travel), TDS Applicable (Yes/No), TDS Rate (%)

⚠️ Output a single comma-separated line of values (no headers, no multi-line, no bullets, no quotes).
⚠️ Do NOT echo the field names or table headings. Just the values.
⚠️ If key values are missing, write: NOT AN INVOICE
"""

def is_placeholder_row(text):
    """Check if the row contains field names instead of values"""
    if not text:
        return False
    
    # List of field name keywords to check
    field_keywords = ["vendor name", "invoice no", "invoice date", "expense ledger", "gst type"]
    
    # Convert to lowercase for comparison
    text_lower = text.lower()
    
    # Count how many field keywords appear
    matches = sum(1 for keyword in field_keywords if keyword in text_lower)
    
    # If 2 or more field names are found, it's likely a placeholder row
    return matches >= 2

# ---------- PDF to Image ----------
def convert_pdf_first_page(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc.load_page(0)
    pix = page.get_pixmap(dpi=200)
    return Image.open(io.BytesIO(pix.tobytes("png")))

# ---------- PDF UPLOAD ----------
uploaded_files = st.file_uploader("📤 Upload scanned invoice PDFs", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    for file in uploaded_files:
        file_name = file.name

        # Skip if already processed
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

        with st.spinner("🧠 Extracting data using AI..."):
            csv_line = ""
            try:
                if model_choice == "Gemini" and gemini_api_key:
                    temp_model = genai.GenerativeModel("gemini-1.5-flash-latest")
                    response = temp_model.generate_content([first_image, strict_prompt])
                    csv_line = response.text.strip()
                    
                    # Debug: Show what AI returned
                    with st.expander(f"Debug: Raw AI Response for {file_name}"):
                        st.text(csv_line)

                elif model_choice == "ChatGPT" and openai_api_key:
                    img_buf = io.BytesIO()
                    first_image.save(img_buf, format="PNG")
                    img_buf.seek(0)
                    base64_image = base64.b64encode(img_buf.read()).decode()
                    chat_prompt = [
                        {"role": "system", "content": "You are a finance assistant."},
                        {"role": "user", "content": [
                            {"type": "text", "text": strict_prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                        ]}
                    ]
                    response = openai.ChatCompletion.create(
                        model=openai_model,
                        messages=chat_prompt,
                        max_tokens=1000
                    )
                    csv_line = response.choices[0].message.content.strip()
                    
                    # Debug: Show what AI returned
                    with st.expander(f"Debug: Raw AI Response for {file_name}"):
                        st.text(csv_line)
                else:
                    raise Exception("❌ No valid API key provided.")

                # Process the response
                if csv_line.upper().startswith("NOT AN INVOICE"):
                    result_row = [file_name] + ["NOT AN INVOICE"] + ["-"] * (len(columns) - 2)
                elif is_placeholder_row(csv_line):
                    # Try to find a valid data line
                    lines = csv_line.strip().split('\n')
                    valid_line_found = False
                    
                    for line in lines:
                        if line.strip() and not is_placeholder_row(line):
                            # This might be a valid data line
                            try:
                                values = [x.strip().strip('"') for x in line.split(",")]
                                if len(values) >= 5:  # At least have basic fields
                                    # Pad or truncate to match column count
                                    if len(values) < len(columns) - 1:
                                        values.extend(["-"] * (len(columns) - 1 - len(values)))
                                    else:
                                        values = values[:len(columns) - 1]
                                    
                                    result_row = [file_
