
import streamlit as st
import fitz  # PyMuPDF
from PIL import Image
import google.generativeai as genai
import io
import pandas as pd
import base64

# Set Streamlit UI
st.set_page_config(layout="wide")
st.title("📄 AI Invoice Extractor (Gemini Only - Streamlit Ready)")

# Define output columns
columns = [
    "Vendor Name", "Invoice No", "Invoice Date", "Expense Ledger",
    "GST Type", "Tax Rate", "Basic Amount", "CGST", "SGST", "IGST",
    "Total Payable", "Narration", "GST Input Eligible", "TDS Applicable", "TDS Rate"
]

# --- Security: App Access Control ---
st.sidebar.header("🔐 Admin Access")
passcode = st.sidebar.text_input("Enter Passcode to Use the App:", type="password")
admin_verified = passcode == "Essenbee"

if not admin_verified:
    st.sidebar.warning("Enter correct passcode to use the app.")
    st.stop()

# Gemini API Setup
GEMINI_API_KEY = "AIzaSyA5Jnd7arMlbZ1x_ZpiE-AezrmsaXams7Y"
GEMINI_MODEL_ID = "gemini-1.5-flash-latest"
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel(GEMINI_MODEL_ID)

# Upload multiple PDFs
uploaded_files = st.file_uploader("Upload scanned invoice PDFs", type=["pdf"], accept_multiple_files=True)

results = []
if uploaded_files:
    for file in uploaded_files:
        st.subheader(f"📄 Processing: {file.name}")
        doc = fitz.open(stream=file.read(), filetype="pdf")
        page = doc.load_page(0)
        pix = page.get_pixmap()
        first_image = Image.open(io.BytesIO(pix.tobytes("png")))
        st.image(first_image, caption=f"Preview of {file.name}", use_column_width=True)

        with st.spinner("Extracting data using Gemini..."):
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
                response = model.generate_content([first_image, prompt])
                csv_line = response.text.strip()
                row = [x.strip() for x in csv_line.split(",")]
                if len(row) != len(columns):
                    raise ValueError("Mismatch in extracted field count.")
                results.append(row)
            except Exception as e:
                st.error(f"❌ Error processing {file.name}: {e}")
                st.text_area("Raw AI Output", csv_line if 'csv_line' in locals() else "N/A")

    if results:
        df = pd.DataFrame(results, columns=columns)
        st.success("✅ All invoices processed!")
        st.dataframe(df)

        csv = df.to_csv(index=False).encode()
        st.download_button("📥 Download Extracted Data", csv, "invoice_data.csv", "text/csv")
else:
    st.info("Upload one or more scanned invoices to get started.")
