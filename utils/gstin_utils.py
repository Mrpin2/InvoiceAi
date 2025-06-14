import re

def is_valid_gstin(gstin):
    """Validate GSTIN format with more flexibility"""
    if not gstin:
        return False
        
    # Clean the GSTIN: remove spaces, special characters, convert to uppercase
    cleaned = re.sub(r'[^A-Z0-9]', '', gstin.upper())
    
    # GSTIN must be exactly 15 characters
    if len(cleaned) != 15:
        return False
        
    # Validate pattern: 2 digits + 10 alphanumeric + 1 letter + 1 alphanumeric + 1 letter
    pattern = r"^\d{2}[A-Z0-9]{10}[A-Z]{1}[A-Z0-9]{1}[Z]{1}[A-Z0-9]{1}$"
    return bool(re.match(pattern, cleaned))

def extract_gstin_from_text(text):
    """Try to extract GSTIN from any text using pattern matching"""
    # Look for GSTIN pattern in the text
    matches = re.findall(r'\b\d{2}[A-Z0-9]{10}[A-Z]{1}[A-Z0-9]{1}Z[A-Z0-9]{1}\b', text.upper())
    return matches[0] if matches else ""
