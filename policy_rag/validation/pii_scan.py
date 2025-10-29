import re

EMAIL_RE = r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
PHONE_RE = r"(\+?\d[\d\s\-]{7,}\d)"
NI_RE = r"\b([A-CEGHJ-PR-TW-Z]{2}\d{6}[A-D])\b"  # UK-ish National Insurance number pattern
ADDRESS_HINT_RE = r"\b(Street|St\.|Road|Rd\.|Avenue|Ave\.|Postcode|ZIP)\b"

def scan_pii(pages):
    """
    Returns a PIIReport dict:
    {
      "matches":[{"type","text","page"},...],
      "severity":"low|medium|high",
      "action":"ok|redact|block_ingestion"
    }
    """
    matches = []
    for p in pages:
        t = p["text"]

        for m in re.findall(EMAIL_RE, t):
            matches.append({"type":"email", "text":m, "page":p["page_num"]})

        for m in re.findall(PHONE_RE, t):
            matches.append({"type":"phone", "text":m.strip(), "page":p["page_num"]})

        for m in re.findall(NI_RE, t):
            matches.append({"type":"ni_number", "text":m, "page":p["page_num"]})

        if re.search(ADDRESS_HINT_RE, t, flags=re.I):
            matches.append({"type":"address_hint","text":"possible address content","page":p["page_num"]})

    # naive severity logic
    if any(m["type"] == "ni_number" for m in matches):
        severity = "high"
        action = "block_ingestion"
    elif any(m["type"] in ["email","phone","address_hint"] for m in matches):
        severity = "medium"
        action = "redact"
    else:
        severity = "low"
        action = "ok"

    return {
        "matches": matches,
        "severity": severity,
        "action": action
    }

def redact_pii_text(text: str):
    text = re.sub(EMAIL_RE, "<REDACTED_EMAIL>", text)
    text = re.sub(PHONE_RE, "<REDACTED_PHONE>", text)
    text = re.sub(NI_RE, "<REDACTED_NI>", text)
    return text
