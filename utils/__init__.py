# utils/__init__.py

# Expose shared utility functions from general_utils
from .general_utils import (
    load_lottie_json_safe,
    convert_pdf_first_page,
    safe_float,
    format_currency,
    is_valid_gstin,
    extract_gstin_from_text,
    handle_gstin,
    determine_tds_rate,
    extract_json_from_response
)
