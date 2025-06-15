import re

def simple_interest(P, R, T):
    return (P * R * T) / 100

def compound_interest(P, R, T):
    return P * ((1 + R/100) ** T) - P

def parse_simple_interest_input(text):
    match = re.search(r"\$?([\d\.]+).*?at\s+([\d\.]+)%.*?for\s+([\d\.]+)\s*year", text.lower())
    if match:
        return map(float, match.groups())
    return None, None, None

def parse_compound_interest_input(text):
    match = re.search(r"\$?([\d\.]+).*?at\s+([\d\.]+)%.*?for\s+([\d\.]+)\s*year.*compound", text.lower())
    if match:
        return map(float, match.groups())
    return None, None, None
