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
import re

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

# ---------- Improved Prompts ----------
strict_prompt = """
You are a professional finance assistant. Extract data from this GST invoice image.

IMPORTANT RULES:
1. If this is NOT a proper GST invoice, respond with exactly: NOT AN INVOICE
2. Extract ONLY the actual values, NOT the field names
3. Return ONLY ONE LINE of comma-separated values
4. Do NOT include field names like "Vendor Name", "Invoice No" etc.
5. Replace any missing values with a hyphen (-)

Extract these fields in this exact order:
Vendor Name, Invoice No, Invoice Date, Expense Ledger, GST Type, Tax Rate, Basic Amount,
CGST, SGST, IGST, Total Payable, Narration, GST Input Eligible, TDS Applicable, TDS Rate

Example good output: Essenbee Advisors LLP,INV-000257,16 May 2025,Consulting Fees,IGST,18,5000,0,0,900,5900,Professional consulting services,Yes,Yes,10

DO NOT output field names. Output ONLY the values.
"""

soft_prompt = """
Extract invoice data and return ONLY values (not field names) in this order:
Vendor Name, Invoice No, Invoice Date, Expense Ledger, GST Type, Tax Rate, Basic Amount,
CGST, SGST, IGST, Total Payable, Narration, GST Input Eligible, TDS Applicable, TDS Rate

Use hyphen (-) for missing values. One line only, comma-separated.
"""

def is_placeholder_row(text):
    """Enhanced function to detect if the row contains field names instead of values"""
    if not text:
        return False
    
    # List of field name keywords to check
    field_keywords = [
        "vendor name", "invoice no", "invoice date", "expense ledger",
        "gst type", "tax rate", "basic amount", "cgst", "sgst", "igst",
        "total payable", "narration", "gst input eligible", "tds applicable", "tds rate"
    ]
    
    # Convert to lowercase for comparison
    text_lower = text.lower()
    
    # Check if any field name appears in the text
    field_name_count = sum(1 for keyword in field_keywords if keyword in text_lower)
    
    # If more than 2 field names are found, it's likely a placeholder row
    return field_name_count >= 2

def clean_csv_line(csv_line):
    """Clean and validate the CSV line"""
    if not csv_line:
        return None
    
    # Remove any lines that contain field names
    lines = csv_line.strip().split('\n')
    valid_lines = []
    
    for line in lines:
        if not is_placeholder_row(line) and line.strip():
            valid_lines.append(line)
    
    # Return the first valid line
    return valid_lines[0] if valid_lines else None

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
            retry_count = 0
            max_retries = 2
            
            while retry_count < max_retries:
                try:
                    if model_choice == "Gemini" and gemini_api_key:
                        temp_model = genai.GenerativeModel("gemini-1.5-flash-latest")
                        
                        # Use strict prompt first
                        current_prompt = strict_prompt if retry_count == 0 else soft_prompt
                        response = temp_model.generate_content([first_image, current_prompt])
                        csv_line = response.text.strip()
                        
                        # Clean the response
                        cleaned_line = clean_csv_line(csv_line)
                        
                        # If we got a valid response, break
                        if cleaned_line and not is_placeholder_row(cleaned_line):
                            csv_line = cleaned_line
                            break
                        
                    elif model_choice == "ChatGPT" and openai_api_key:
                        img_buf = io.BytesIO()
                        first_image.save(img_buf, format="PNG")
                        img_buf.seek(0)
                        base64_image = base64.b64encode(img_buf.read()).decode()
                        
                        current_prompt = strict_prompt if retry_count == 0 else soft_prompt
                        chat_prompt = [
                            {"role": "system", "content": "You are a finance assistant. Extract only values, not field names."},
                            {"role": "user", "content": [
                                {"type": "text", "text": current_prompt},
                                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                            ]}
                        ]
                        response = openai.ChatCompletion.create(
                            model=openai_model,
                            messages=chat_prompt,
                            max_tokens=1000
                        )
                        csv_line = response.choices[0].message.content.strip()
                        
                        # Clean the response
                        cleaned_line = clean_csv_line(csv_line)
                        
                        # If we got a valid response, break
                        if cleaned_line and not is_placeholder_row(cleaned_line):
                            csv_line = cleaned_line
                            break
                    else:
                        raise Exception("❌ No valid API key provided.")
                    
                    retry_count += 1
                    
                except Exception as e:
                    st.error(f"❌ Error on attempt {retry_count + 1}: {e}")
                    retry_count += 1

            # Process the final result
            if csv_line.upper().startswith("NOT AN INVOICE"):
                result_row = [file_name] + ["NOT AN INVOICE"] + ["-"] * (len(columns) - 2)
            else:
                try:
                    # Split by comma and clean each value
                    values = [x.strip().strip('"') for x in csv_line.split(",")]
                    
                    # Ensure we have the right number of values
                    if len(values) < len(columns) - 1:
                        # Pad with hyphens if needed
                        values.extend(["-"] * (len(columns) - 1 - len(values)))
                    elif len(values) > len(columns) - 1:
                        # Truncate if too many values
                        values = values[:len(columns) - 1]
                    
                    result_row = [file_name] + values
                    
                except Exception as e:
                    st.error(f"❌ Error parsing CSV: {e}")
                    result_row = [file_name] + ["PARSE ERROR"] + ["-"] * (len(columns) - 2)

            st.session_state["processed_results"][file_name] = result_row

# ---------- DISPLAY RESULTS ----------
results = list(st.session_state["processed_results"].values())
if results:
    df = pd.DataFrame(results, columns=columns)
    df.index = df.index + 1
    df.reset_index(inplace=True)
    df.rename(columns={"index": "S. No"}, inplace=True
