# utils/__init__.py
# Make the utils directory a package and expose shared utility functions

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

# Optionally expose other utility modules
# from .gstin_utils import *\# if needed
# from .openai_utils import *
# from .pdf_utils import *
# from .tds_utils import *
