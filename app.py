import streamlit as st
from pdf2image import convert_from_bytes
from PIL import Image
import google.generativeai as genai
import openai
import io
import pandas as pd
import base64
import json
from streamlit_lottie import st_lottie

# ---------- LOTTIE ANIMATION ----------
lottie_json = {
  "v": "5.5.9", "fr": 30, "ip": 0, "op": 180, "w": 600, "h": 600, "nm": "Invoice",
  "ddd": 0, "assets": [], "layers": [{
    "ddd": 0, "ind": 1, "ty": 4, "nm": "Layer 1", "sr": 1,
    "ks": {"o": {"a": 0, "k": 100}, "r": {"a": 0, "k": 0},
           "p": {"a": 0, "k": [300, 300, 0]}, "a": {"a": 0, "k": [0, 0, 0]},
           "s": {"a": 0, "k": [100, 100, 100]}},
    "shapes": [{"ty": "rc", "d": 1, "s": {"a": 0, "k": [150, 80]}, "p": {"a": 0, "k": [0, 0]}, "r": {"a": 0, "k": 20},
                "nm": "rect", "hd": False}],
    "ip": 0, "op": 180, "st": 0, "bm": 0
  }]
}

# ---------- UI CONFIGURATION ----------
st.set_page_config(layout="wide")
st_lottie(lottie_json, height=200, key="animation")
st.markdown("<h2 style='text-align: center;'>📄 AI Invoice Extractor</h2>", unsafe_allow_html=True)
st.markdown("Upload scanned PDF invoices and extract clean finance data using Gemini or ChatGPT")
st.markdown("---")

# ---------- Table Columns ----------
columns = [
    "Vendor Name", "Invoice No", "Invoice Date", "Expense Ledger",
    "GST Type", "Tax Rate", "Basic Amount", "CGST", "SGST", "IGST",
    "Total Payable", "Narration", "GST Input Eligible", "TDS Applicable", "TDS Rate"
]

# ---------- SIDEBAR AUTH & MODEL SELECT ----------
st.sidebar.header("🔐 AI Config")

passcode = st.sidebar.text_input("Admin Passcode (optional)", type="password")
admin_unlocked = passcode == "Essenbee"

model_choice = st.sidebar.radio("Choose AI Model", ["Gemini", "ChatGPT"])

if model_choice == "Gemini":
    if admin_unlocked:
        gemini_api_key = st.sidebar.text_input("🔑 Gemini API Key", type="password")
    else:
        st.sidebar.caption("Using default embedded Gemini key.")
        gemini_api_key = "AIzaSyA5Jnd7arMlbZ1x_ZpiE-AezrmsaXams7Y"
    genai.configure(api_key=gemini_api_key)
    gemini_model_id = "gemini-1.5-flash-latest"
    ai_model = genai.GenerativeModel(gemini_model_id)

elif model_choice == "ChatGPT":
    openai_api_key = st.sidebar.text_input("🔑 OpenAI API Key", type="password")
    openai.api_key = openai_api_key
    openai_model = "gpt-4-vision-preview"

# ---------- PDF UPLOAD ----------
uploaded_files = st.file_uploader("📤 Upload scanned invoice PDFs", type=["pdf"], accept_multiple_files=True)

results = []
if uploaded_files:
    for file in uploaded_files:
        st.subheader(f"📄 Processing: {file.name}")
        try:
            images = convert_from_bytes(file.read(), dpi=200)
            first_image = images[0]
            st.image(first_image, caption=f"{file.name}", use_container_width=True)
        except Exception as e:
            st.error(f"❌ Error reading PDF: {e}")
            continue

        with st.spinner("🧠 Extracting data using AI..."):
            prompt = """
            You are a professional finance assistant. Extract the following fields from the invoice image:
            Vendor Name, Invoice No, Invoice Date, Expense Ledger (like Office Supplies, Travel, Legal Fees, etc.),
            GST Type (IGST or CGST+SGST or NA), Tax Rate (%, single value), Basic Amount,
            CGST, SGST, IGST, Total Payable, Narration (short sentence),
            GST Input Eligible (Yes/No — No if travel, food, hotel, etc.),
            TDS Applicable (Yes/No), TDS Rate (in % if applicable).
            Respond with CSV-style values in this exact order:
            Vendor Name, Invoice No, Invoice Date, Expense Ledger,
            GST Type, Tax Rate, Basic Amount, CGST, SGST, IGST,
            Total Payable, Narration, GST Input Eligible, TDS Applicable, TDS Rate.
            """
            try:
                if model_choice == "Gemini":
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
                    raise Exception("No API key configured.")

                row = [x.strip() for x in csv_line.split(",")]
                if len(row) != len(columns):
                    raise ValueError("Mismatch in extracted field count.")
                results.append(row)

            except Exception as e:
                st.error(f"❌ Error processing {file.name}: {e}")
                st.text_area("Raw AI Output", csv_line if 'csv_line' in locals() else "N/A")

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
