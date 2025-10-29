import re

REQUIRED_SECTIONS = [
    r"\bPurpose\b",
    r"\bScope\b",
    r"\bResponsibilities?\b|Roles and Responsibilities",
    r"\bRetention\b|\bDisciplinary\b|\bIncident Reporting\b",
    r"\bRevision\b|\bReview Date\b|\bEffective Date\b"
]

def check_policy_format(full_text: str):
    found = {}
    for pat in REQUIRED_SECTIONS:
        m = re.search(pat, full_text, flags=re.IGNORECASE)
        found[pat] = bool(m)

    missing_sections = [
        pat for pat, ok in found.items() if not ok
    ]

    # Hard fail if missing "Purpose" or "Scope"
    is_valid = True
    for pat in REQUIRED_SECTIONS[:2]:
        if not found[pat]:
            is_valid = False
            break

    return {
        "has_purpose": found[REQUIRED_SECTIONS[0]],
        "has_scope": found[REQUIRED_SECTIONS[1]],
        "has_responsibilities": found[REQUIRED_SECTIONS[2]],
        "has_reporting_or_enforcement": found[REQUIRED_SECTIONS[3]],
        "has_review_or_revision": found[REQUIRED_SECTIONS[4]],
        "missing_sections": missing_sections,
        "is_valid": is_valid
    }
