import streamlit as st
from prompt import main_prompt
from utils.general_utils import (
    load_lottie_json_safe, convert_pdf_first_page,
    safe_float, format_currency, handle_gstin,
    determine_tds_rate, extract_json_from_response
)
from openai import OpenAI
import io, base64, tempfile, os, locale, pandas as pd
from PIL import Image
import fitz
import requests
from streamlit_lottie import st_lottie
from dateutil import parser
import json

# Setup
st.set_page_config(layout="wide")
locale.setlocale(locale.LC_ALL, '')

# OpenAI client
api_key = st.sidebar.text_input("🔑 OpenAI API Key", type="password")
if not api_key:
    st.sidebar.warning("Enter API key")
    st.stop()
client = OpenAI(api_key=api_key)

# Animations
hello = load_lottie_json_safe(
    "https://raw.githubusercontent.com/Mrpin2/InvoiceAi/main/Animation%20-%201749845212531.json"
)
done  = load_lottie_json_safe(
    "https://raw.githubusercontent.com/Mrpin2/InvoiceAi/main/Animation%20-%201749845303699.json"
)
if "started" not in st.session_state and hello:
    st_lottie(hello, height=200)

# Header
st.markdown("<h2 align='center'>📄 AI Invoice Extractor</h2>", unsafe_allow_html=True)
st.markdown("Upload PDF invoices to extract data via GPT-4 Vision")
st.markdown("---")

# File uploader
files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
if files:
    rows=[]
    for f in files:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(f.read()); path=tmp.name
        img = convert_pdf_first_page(open(path,'rb').read())
        os.unlink(path)
        buf=io.BytesIO(); img.save(buf,'PNG'); b64=base64.b64encode(buf.getvalue()).decode()
        messages=[
            {"role":"system","content":"You are a finance assistant."},
            {"role":"user","content":[{"type":"text","text":main_prompt},
             {"type":"image_url","image_url":{"url":f"data:image/png;base64,{b64}"}}]}
        ]
        resp=client.chat.completions.create(model="gpt-4o", messages=messages, max_tokens=1500)
        data=extract_json_from_response(resp.choices[0].message.content)
        if not data or not data.get('invoice_number'):
            rows.append({'File Name':f.name,'Invoice Number':'NOT AN INVOICE'})
            continue
        ta, cg, sg, ig = [safe_float(data.get(x,0)) for x in ('taxable_amount','cgst','sgst','igst')]
        total=ta+cg+sg+ig; rate=determine_tds_rate(data.get('expense_ledger',''), data.get('tds',''))
        tds_amt=round(ta*rate/100,2); payable=total-tds_amt
        gst=handle_gstin(data.get('gstin',''), data.get('seller_name',''), resp.choices[0].message.content)
        rows.append({
            'File Name':f.name,
            'Invoice Number':data['invoice_number'],
            'Date':data.get('date',''),
            'Seller Name':data.get('seller_name',''),
            'Seller GSTIN':gst,
            'Buyer Name':data.get('buyer_name',''),
            'Buyer GSTIN':data.get('buyer_gstin',''),
            'Taxable Amount':ta,'CGST':cg,'SGST':sg,'IGST':ig,
            'Total Amount':total,'TDS Rate':rate,'TDS Amount':tds_amt,'Amount Payable':payable
        })
    df=pd.DataFrame(rows)
    for c in ['Taxable Amount','CGST','SGST','IGST','Total Amount','TDS Amount','Amount Payable']:
        df[f"{c} (₹)"]=df[c].apply(format_currency)
    df['TDS Rate (%)']=df['TDS Rate'].astype(str)+'%'
    st.dataframe(df)
    st.download_button('📥 CSV',df.to_csv(index=False), 'results.csv')
    if done: st_lottie(done, height=200)
else:
    st.info('Upload a PDF to start.')
