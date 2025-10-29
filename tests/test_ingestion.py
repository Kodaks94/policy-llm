from policy_rag.validation.policy_format_check import check_policy_format
from policy_rag.validation.pii_scan import scan_pii

def test_format_check(loaded_doc):
    text = "\n".join([p["text"] for p in loaded_doc["pages"]])
    result = check_policy_format(text)
    assert isinstance(result["is_valid"], bool)

def test_pii_scan(loaded_doc):
    report = scan_pii(loaded_doc["pages"])
    assert "severity" in report
    assert report["severity"] in ["low", "medium", "high"]
