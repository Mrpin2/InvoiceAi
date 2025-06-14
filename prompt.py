# prompt.py
# Edit this file to change the GPT extraction prompt without touching app.py

main_prompt = (
    "Extract structured invoice data as a JSON object with the following keys: "
    "invoice_number, date, gstin, seller_name, buyer_name, buyer_gstin, "
    "taxable_amount, cgst, sgst, igst, place_of_supply, expense_ledger, tds. "
    "Important: 'taxable_amount' is the amount BEFORE taxes. Use DD/MM/YYYY for dates. "
    "Return 'NOT AN INVOICE' if clearly not one. Use null if missing. "
    "SPECIAL: Look for GSTIN in seller details; label may vary."
)
