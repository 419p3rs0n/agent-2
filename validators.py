import json
import re

def verify_structural_accuracy(category, draft_text):
    """Programmatic zero-cost sanity check on the local output."""
    if not draft_text or len(draft_text.strip()) == 0:
        return False
        
    category_lower = category.lower()

    # 1. Sentiment Classification Strict Constraint
    if "sentiment" in category_lower:
        # Ensures the local model output maps exactly to the expected token options
        return draft_text.strip() in ["Positive", "Negative", "Neutral"]

    # 2. Named Entity Recognition / Structured Schema Constraint
    if "ner" in category_lower or "json" in category_lower:
        try:
            # Enforces that structured data tasks output cleanly parseable JSON data
            json.loads(draft_text)
            return True
        except (ValueError, json.JSONDecodeError):
            return False

    # 3. Code Structural Check
    if "code" in category_lower:
        # Rejects the answer if the model output contains conversational filler
        filler_patterns = ["Sure!", "Here is the code", "```", "I'll help", "I can help"]
        if any(pattern in draft_text for pattern in filler_patterns):
            return False

    return True
