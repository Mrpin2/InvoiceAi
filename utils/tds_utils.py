import re

def determine_tds_rate(expense_ledger, tds_str=""):
    """Determine TDS rate based on expense ledger and TDS string"""
    # First check if TDS string contains a specific rate
    if tds_str and isinstance(tds_str, str):
        # Look for percentage in the TDS string
        match = re.search(r'(\d+(\.\d+)?)%', tds_str)
        if match:
            return float(match.group(1))
        
        # Check for TDS sections
        section_rates = {
            "194j": 10.0,  # Professional services
            "194c": 2.0,   # Contracts
            "194h": 5.0,   # Commission/brokerage
            "194i": 10.0,  # Rent
            "194q": 1.0    # Advertising
        }
        for section, rate in section_rates.items():
            if section in tds_str.lower():
                return rate
    
    # If no TDS string info, determine by expense ledger
    expense_ledger = expense_ledger.lower() if expense_ledger else ""
    
    # Professional services - 10%
    if "professional" in expense_ledger or "consultancy" in expense_ledger or "service" in expense_ledger:
        return 10.0
    
    # Contract work - 2%
    if "contract" in expense_ledger or "sub-contract" in expense_ledger or "work" in expense_ledger:
        return 2.0
    
    # Commission, brokerage - 5%
    if "commission" in expense_ledger or "brokerage" in expense_ledger:
        return 5.0
    
    # Rent - 10%
    if "rent" in expense_ledger:
        return 10.0
    
    # Advertisement - 1%
    if "advertis" in expense_ledger or "marketing" in expense_ledger:
        return 1.0
    
    # Default to 0 if not applicable
    return 0.0
