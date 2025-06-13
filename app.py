import streamlit as st
import fitz  # PyMuPDF
import google.generativeai as genai
from pathlib import Path
from pydantic import BaseModel, ValidationError
import json

# Load API key
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# Load prompt
PROMPT = Path("prompt_template.txt").read_text()

# Streamlit UI
st.set_page_config(page_title="Invoice Extractor", layout="wide")
st.title("🧾 Gemini Invoice Extractor")

uploaded_file = st.file_uploader("Upload PDF invoice or bank statement", type=["pdf"])

if uploaded_file:
    with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
        text = "\n\n".join(page.get_text() for page in doc)

    full_prompt = f"{PROMPT.strip()}\n\n---\n{text.strip()}\n---"

    with st.spinner("Extracting data using Gemini..."):
        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content(full_prompt)

    try:
        data = json.loads(response.text)
        st.success("✅ Extraction successful!")

        st.subheader("📋 Extracted Data")
        st.json(data)

        st.download_button("Download JSON", json.dumps(data, indent=2), file_name="invoice_data.json")
    except Exception as e:
        st.error("❌ Could not parse Gemini output.")
        st.text_area("Raw Output", response.text)
        st.exception(e)
