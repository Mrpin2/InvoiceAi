import json
import re

MAIN_PROMPT = (
    "Extract structured invoice data as a JSON object with the following keys: "
    "invoice_number, date, gstin, seller_name, buyer_name, buyer_gstin, "
    "taxable_amount, cgst, sgst, igst, place_of_supply, expense_ledger, tds. "
    "Important: 'taxable_amount' is the amount BEFORE taxes. "
    "Use DD/MM/YYYY for dates. Use only values shown in the invoice. "
    "Return 'NOT AN INVOICE' if clearly not one. "
    "If a value is not available, use null. "
    
    "SPECIAL INSTRUCTIONS FOR GSTIN: "
    "1. GSTIN is a 15-digit alphanumeric code (format: 22AAAAA0000A1Z5) "
    "2. It's usually located near the seller's name or address "
    "3. If you can't find GSTIN in the dedicated field, look in the seller details section "
    "4. GSTIN might be labeled as 'GSTIN', 'GST No.', or 'GST Number' "
    
    "For expense_ledger, classify the nature of expense and suggest an applicable ledger type "
    "(e.g., 'Office Supplies', 'Professional Fees', 'Software Subscription'). "
    "For tds, determine TDS applicability (e.g., 'Yes - Section 194J', 'No', 'Uncertain')."
)

def extract_json_from_response(text):
    """Try to extract JSON from GPT response which might have extra text"""
    try:
        # Look for JSON code block
        matches = re.findall(r'```json\s*({.*?})\s*```', text, re.DOTALL)
        if matches:
            return json.loads(matches[0])
        
        # Look for plain JSON
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1:
            return json.loads(text[start:end+1])
        
        # Try parsing the whole text
        return json.loads(text)
    except Exception:
        return None
