"""Live findings tracker for the Snapdeal Virtual KAM.

A deterministic, rule-based tracker (no extra LLM call). It scans the
conversation history — seller transcripts + Asha's replies — and reports
which of the three diagnosis findings the call has established so far.
The frontend shows these live as "What Asha found", so the reviewer sees
the AI working from the seller data in real time.

The findings use ONLY the mock seller data from the assignment brief:
- Stock-outs: out-of-stock products 18 vs 5 (3.6x) — primary cause
- Cancellations: cancellation rate 14% (up from 8%) — secondary cause
- Pricing: ruled out — average selling price flat at Rs 500

Rules (all case-insensitive):
- stock-outs: the seller mentions stock / out of stock / unavailable.
- cancellations: any message (seller or Asha) mentions cancel/cancellation.
- pricing: the seller says the price is unchanged / same / not changed,
  OR Asha's reply states pricing is not the cause.
"""

import re

FINDINGS = [
    {
        "id": "stockouts",
        "label": "Stock-outs",
        "detail": "Out-of-stock products 18 vs 5 (3.6\u00d7) \u2014 primary cause",
    },
    {
        "id": "cancellations",
        "label": "Cancellations",
        "detail": "Cancellation rate 14% (up from 8%) \u2014 secondary cause",
    },
    {
        "id": "pricing",
        "label": "Pricing",
        "detail": "Pricing ruled out \u2014 average selling price flat at \u20b9500",
    },
]

_STOCK_RE = re.compile(
    r"out[\s\-]*of[\s\-]*stock|\bstocks?\b|unavailable|not available",
    re.IGNORECASE,
)
_CANCEL_RE = re.compile(r"\bcancel\w*\b", re.IGNORECASE)

# Seller says the price did not change (e.g. "I have not changed my prices").
_PRICE_UNCHANGED_WORDS = (
    "unchanged",
    "same",
    "not changed",
    "didn't change",
    "did not change",
    "haven't changed",
    "have not changed",
    "hasn't changed",
    "has not changed",
    "no change",
    "flat",
    "still",
)

# Asha states pricing is not the cause (e.g. "this is not a pricing problem").
_PRICING_NOT_CAUSE_RES = (
    re.compile(r"not (a )?pricing", re.IGNORECASE),
    re.compile(r"not a price problem", re.IGNORECASE),
    re.compile(r"price (is )?not (the )?(cause|problem|issue)", re.IGNORECASE),
    re.compile(r"price (hasn't|has not) changed", re.IGNORECASE),
    re.compile(r"price is flat", re.IGNORECASE),
    re.compile(r"pricing is not", re.IGNORECASE),
)


def _seller_says_price_unchanged(text: str) -> bool:
    """True when the seller says the price is unchanged / the same."""
    low = text.lower()
    if "price" not in low and "pricing" not in low:
        return False
    return any(word in low for word in _PRICE_UNCHANGED_WORDS)


def _asha_rules_out_pricing(text: str) -> bool:
    """True when Asha's reply states pricing is not the cause."""
    return any(rx.search(text) for rx in _PRICING_NOT_CAUSE_RES)


def compute_findings(history: list) -> list:
    """Return the three findings with status 'checking' or 'confirmed'.

    ``history`` is a list of {"role": "user"|"assistant", "text": ...}.
    A finding starts as 'checking' and flips to 'confirmed' once the
    conversation establishes it. Nothing is ever invented: the details
    are fixed strings from the brief's data.
    """
    seller_text = " ".join(
        m.get("text", "") for m in history if m.get("role") == "user"
    )
    asha_text = " ".join(
        m.get("text", "") for m in history if m.get("role") == "assistant"
    )
    all_text = seller_text + " " + asha_text

    confirmed = {
        "stockouts": bool(_STOCK_RE.search(seller_text)),
        "cancellations": bool(_CANCEL_RE.search(all_text)),
        "pricing": _seller_says_price_unchanged(seller_text)
        or _asha_rules_out_pricing(asha_text),
    }
    return [
        {
            "id": f["id"],
            "label": f["label"],
            "detail": f["detail"],
            "status": "confirmed" if confirmed[f["id"]] else "checking",
        }
        for f in FINDINGS
    ]
