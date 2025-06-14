import requests
import fitz
import io
import re
import json
from PIL import Image
from dateutil import parser


def load_lottie_json_safe(url: str):
    """
    Load a Lottie JSON from the given URL. Returns the parsed JSON or None on failure.
    """
    try:
        r = requests.get(url)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def convert_pdf_first_page(pdf_bytes: bytes) -> Image.Image:
    """
    Convert the first page of a PDF (given as bytes) to a PIL Image at 300 dpi.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc.load_page(0)
    pix = page.get_pixmap(dpi=300)
    return Image.open(io.BytesIO(pix.tobytes("png")))


def safe_float(x) -> float:
    """
    Convert a value to float, stripping commas, currency symbols, and handling failures.
    """
    try:
        cleaned = str(x).replace(",", "").replace("₹", "").replace("$", "").strip()
        return float(cleaned) if cleaned else 0.0
    except Exception:
        return 0.0


def format_currency(x) -> str:
    """
    Format a numeric value as Indian rupee currency, e.g. '₹1,234.56'.
    """
    try:
        return f"₹{safe_float(x):,.2f}"
    except Exception:
        return "₹0.00"


def is_valid_gstin(gstin: str) -> bool:
    """
    Validate if the input string is a valid 15-character GSTIN.
    """
    if not gstin or not isinstance(gstin, str):
        return False
    cleaned = re.sub(r'[^A-Z0-9]', '', gstin.upper())
    if len(cleaned) != 15:
        return False
    pattern = r"^\d{2}[A-Z0-9]{10}[A-Z]{1}[A-Z0-9]{1}Z[A-Z0-9]{1}$"
    return bool(re.match(pattern, cleaned))


def extract_gstin_from_text(text: str) -> str:
    """
    Extract the first matching GSTIN-like substring from text, or empty string if none.
    """
    if not text or not isinstance(text, str):
        return ""
    matches = re.findall(
        r"\b\d{2}[A-Z0-9]{10}[A-Z]{1}[A-Z0-9]{1}Z[A-Z0-9]{1}\b",
        text.upper()
    )
    return matches[0] if matches else ""


def handle_gstin(raw_gstin: str, seller_name: str, response_text: str) -> str:
    """
    Clean and validate the raw GSTIN. Fallback to extracting from seller_name or the full GPT response.
    Returns a valid GSTIN or empty string.
    """
    gst = extract_gstin_from_text(raw_gstin)
    if is_valid_gstin(gst):
        return gst

    gst = extract_gstin_from_text(seller_name)
    if is_valid_gstin(gst):
        return gst

    gst = extract_gstin_from_text(response_text)
    return gst if is_valid_gstin(gst) else ""


def determine_tds_rate(expense_ledger: str, tds_str: str = "") -> float:
    """
    Determine the applicable TDS rate based on explicit TDS annotations or ledger keywords.
    """
    if tds_str and isinstance(tds_str, str):
        match = re.search(r"(\d+(\.\d+)?)%", tds_str)
        if match:
            return float(match.group(1))
        sections = {
            "194j": 10.0,
            "194c": 2.0,
            "194h": 5.0,
            "194i": 10.0,
            "194q": 1.0
        }
        for sec, rate in sections.items():
            if sec in tds_str.lower():
                return rate

    exp = (expense_ledger or "").lower()
    if any(k in exp for k in ["professional", "consultancy", "service"]):
        return 10.0
    if any(k in exp for k in ["contract", "work"]):
        return 2.0
    if "commission" in exp:
        return 5.0
    if "rent" in exp:
        return 10.0
    if any(k in exp for k in ["advertis", "marketing"]):
        return 1.0
    return 0.0


def extract_json_from_response(text: str) -> dict:
    """
    Extract a JSON object from GPT response text, handling code fences or extra text.
    Returns parsed dict or None on failure.
    """
    try:
        matches = re.findall(r'```json\s*({.*?})\s*```', text, re.DOTALL)
        if matches:
            return json.loads(matches[0])

        start, end = text.find('{'), text.rfind('}')
        if start != -1 and end != -1:
            return json.loads(text[start:end+1])

        return json.loads(text)
    except Exception:
        return None
