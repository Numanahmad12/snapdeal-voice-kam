"""Groq generation backend for the Snapdeal Virtual KAM voice bot.

Follows the BolInterview ``llm.py`` pattern: the API key comes ONLY from
the ``GROQ_API_KEY`` environment variable — it is never hardcoded, logged,
or shipped. When no key is set, or the Groq call fails for any reason,
generation returns "" and callers degrade gracefully — a reply is never
invented as if it came from the model.

Model: Groq openai/gpt-oss-120b, with ``reasoning_effort: "low"`` — without
it, this reasoning model's internal reasoning can eat the whole
max_tokens budget and the reply comes back with empty content
(verified live in BolInterview, 8 Nov 2026).

Fallback: when Groq is unavailable, ``kam_reply`` / ``kam_summary``
return a clearly-flagged, deterministic fallback built ONLY from the
mock seller data in the assignment brief (no invented facts). The caller
sees ``fallback: True`` and the frontend shows a small note, so a demo
without a key is honest about what is happening.

No live Groq call is ever made in tests — monkeypatch ``_get_groq_session``
(or ``requests``) with a fake session.
"""

import json
import logging
import os
import re

try:
    import requests
except Exception:  # pragma: no cover
    requests = None

logger = logging.getLogger(__name__)

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

_groq_session = None


def _get_groq_api_key() -> str:
    """Read the Groq API key from the environment (never hardcoded)."""
    return os.getenv("GROQ_API_KEY", "").strip()


def llm_available() -> bool:
    """True when a Groq API key is configured."""
    return bool(_get_groq_api_key())


def _get_groq_session():
    """Return a reusable requests.Session for Groq calls."""
    global _groq_session
    if _groq_session is None and requests is not None:
        _groq_session = requests.Session()
        _groq_session.headers.update({"Content-Type": "application/json"})
    return _groq_session


def _chat_completion(system_prompt: str, messages: list, max_tokens: int) -> str:
    """One Groq chat-completions call with full history. Returns reply or ""."""
    api_key = _get_groq_api_key()
    if not api_key:
        return ""
    session = _get_groq_session()
    if session is None:
        return ""

    try:
        response = session.post(
            GROQ_API_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": GROQ_MODEL,
                # gpt-oss-120b is a reasoning model: without this, its internal
                # reasoning can eat the whole max_tokens budget and the reply
                # comes back with empty content (verified live 8 Nov 2026).
                "reasoning_effort": "low",
                "messages": [{"role": "system", "content": system_prompt}] + messages,
                "max_tokens": max_tokens,
                "temperature": 0.3,
            },
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()
        choices = payload.get("choices") or []
        if not choices:
            return ""
        message = choices[0].get("message") or {}
        return (message.get("content") or "").strip()
    except Exception as exc:
        # Never raise from generation — callers degrade gracefully.
        logger.warning("Groq call failed (%s).", exc)
        return ""


def _parse_json_object(raw: str):
    """Parse a JSON object out of model output (tolerant of surrounding text)."""
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        pass
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Deterministic fallback (no LLM). Built ONLY from the mock seller data in
# the assignment brief — nothing invented. Turn-aware so the conversation
# still walks the diagnosis even without a Groq key. Always flagged
# ``fallback: True`` by the caller.
# ---------------------------------------------------------------------------

FALLBACK_REPLIES = {
    "en-IN": [
        # turn 1 — after the seller's first message
        "I understand — a drop like that is worrying, and I'll help you get to the bottom of it. "
        "Before we dig in, tell me: which of your products have gone out of stock recently — are they your usual bestsellers?",
        # turn 2
        "Thank you, that's very useful. One more thing — your cancellation rate has gone up from 8% to 14%. "
        "Have you noticed dispatch delays, or buyers cancelling because the product wasn't available when they ordered?",
        # turn 3 and later — diagnosis + actions
        "Thank you — I now have a clear picture. Your selling price hasn't changed at all, it is still 500 rupees, "
        "so this is not a pricing problem. The drop is coming from two things: 18 products are out of stock, up from just 5 last month, "
        "and cancellations have jumped to 14%. When bestsellers go out of stock, orders fall and cancellations rise. "
        "Here is what I recommend. First, restock your out-of-stock bestsellers on priority — start with the ones that sold the most last month. "
        "Second, fix the cancellations: dispatch faster, keep your stock counts accurate, and message buyers proactively if there is any delay. "
        "Third, push visibility on the products you do have in stock, so your store keeps selling while the restock is underway. "
        "Shall I wrap this up into a summary for you?",
    ],
    "hi-IN": [
        "Samajh gaya — aisi girawat chinta ki baat hai, aur main aapki madad karunga. "
        "Pehle ye batayein: haal mein aapke kaun se products out of stock hue hain — kya woh aapke bestsellers hain?",
        "Shukriya, ye bahut kaam ki jaankari hai. Ek aur baat — aapka cancellation rate 8% se badh kar 14% ho gaya hai. "
        "Kya aapne dispatch mein deri dekhi hai, ya buyers isliye cancel kar rahe hain kyunki order ke time product available nahi tha?",
        "Shukriya — ab tasveer saaf hai. Aapka selling price bilkul nahi badla, abhi bhi 500 rupay hai, "
        "toh ye pricing ki samasya nahi hai. Girawat ki do wajah hain: 18 products out of stock hain, pichle mahine sirf 5 the, "
        "aur cancellations 14% tak badh gaye hain. Jab bestsellers out of stock hote hain, toh orders girte hain aur cancellations badhte hain. "
        "Meri salah ye hai. Pehle, apne out-of-stock bestsellers ko turant restock karein — unse shuru karein jo pichle mahine sabse zyada bike. "
        "Doosra, cancellations theek karein: tez dispatch karein, stock count sahi rakhein, aur deri ho toh buyer ko pehle se message karein. "
        "Teesra, jo products stock mein hain unki visibility badhayein, taaki restock hote hue bhi aapki dukaan chalti rahe. "
        "Kya main iska ek summary bana doon?",
    ],
}

FALLBACK_SUMMARY = {
    "en-IN": {
        "issues": [
            "Sales fell 32.5% — from Rs 2,00,000 last month to Rs 1,35,000 this month — and orders fell the same way, from 400 to 270.",
            "Out-of-stock products jumped from 5 to 18, a 3.6x rise — the key driver of the drop.",
            "Cancellation rate rose sharply from 8% to 14%, likely linked to the same stock and dispatch problems.",
            "Average selling price is flat at Rs 500, so this is a volume and fulfilment problem, not a pricing problem.",
        ],
        "recommendations": [
            "Urgently restock the out-of-stock bestsellers first — prioritise the products that sold most last month.",
            "Fix the cancellations: dispatch faster, keep stock counts accurate, and message buyers proactively about any delay.",
            "Push visibility on in-stock products so the store keeps selling while the restock is underway.",
        ],
        "next_steps": [
            "Share the list of the 18 out-of-stock products so the bestsellers can be prioritised for restock.",
            "Check dispatch times for the last two weeks to find where cancellations are coming from.",
            "Review sales again after one week of restocking to see if orders recover.",
        ],
    },
    "hi-IN": {
        "issues": [
            "Sales mein 32.5% ki girawat — pichle mahine Rs 2,00,000 se is mahine Rs 1,35,000 — aur orders bhi waise hi 400 se 270 ho gaye.",
            "Out-of-stock products 5 se badh kar 18 ho gaye, 3.6 guna zyada — girawat ki sabse badi wajah.",
            "Cancellation rate 8% se badh kar 14% ho gaya, jo shayad stock aur dispatch ki samasya se juda hai.",
            "Average selling price Rs 500 par stable hai, toh ye pricing ki nahi, volume aur fulfilment ki samasya hai.",
        ],
        "recommendations": [
            "Out-of-stock bestsellers ko turant restock karein — pehle un products ko jinhe pichle mahine sabse zyada bikri mili.",
            "Cancellations theek karein: tez dispatch, sahi stock count, aur deri par buyer ko pehle se message.",
            "Jo products stock mein hain unki visibility badhayein, taaki restock ke dauraan bhi bikri chalti rahe.",
        ],
        "next_steps": [
            "18 out-of-stock products ki list share karein taaki bestsellers ko restock mein pehle rakha ja sake.",
            "Pichle do hafton ke dispatch time check karein taaki pata chale cancellations kahan se aa rahe hain.",
            "Restock ke ek hafte baad sales dobara review karein aur dekhein ki orders wapas badh rahe hain ya nahi.",
        ],
    },
}


def kam_reply(system_prompt: str, history: list, language: str) -> dict:
    """Generate the KAM's next spoken reply from the conversation history.

    ``history`` is a list of {"role": "user"|"assistant", "text": ...}.
    Returns {"text": str, "fallback": bool}. When Groq is unavailable the
    text is a deterministic, data-only fallback and fallback is True.
    """
    messages = [
        {"role": m["role"], "content": m["text"]}
        for m in history
        if m.get("text", "").strip()
    ]
    raw = _chat_completion(system_prompt, messages, max_tokens=600)
    if raw:
        return {"text": raw, "fallback": False}
    # Deterministic fallback, turn-aware (user turns so far, excluding greeting).
    user_turns = sum(1 for m in history if m.get("role") == "user")
    replies = FALLBACK_REPLIES.get(language, FALLBACK_REPLIES["en-IN"])
    index = min(max(user_turns - 1, 0), len(replies) - 1)
    return {"text": replies[index], "fallback": True}


def kam_summary(system_prompt: str, history: list, language: str) -> dict:
    """Summarise the call: issues, recommendations, next steps + spoken text.

    Returns {"issues": [...], "recommendations": [...], "next_steps": [...],
    "summary_text": str, "fallback": bool}. The LLM path returns parsed JSON;
    on any failure it falls back to the deterministic data-only summary.
    """
    transcript_lines = []
    for m in history:
        who = "Seller" if m.get("role") == "user" else "KAM"
        transcript_lines.append(f"{who}: {m.get('text', '')}")
    transcript = "\n".join(transcript_lines)
    lang_name = "Hindi (Devanagari script)" if language == "hi-IN" else "English"
    summary_system = (
        system_prompt + "\n\n"
        "Now summarise this call. Reply with ONLY a JSON object like "
        '{"issues": ["...", "..."], "recommendations": ["...", "..."], '
        f'"next_steps": ["...", "..."], "summary_text": "..."}}. All text in {lang_name}. '
        "summary_text is a short spoken closing (3-4 sentences) that states the main issue, "
        "the recommendations and the next steps. Base everything on the seller data and "
        "what the seller actually said — do not invent numbers or facts."
    )
    raw = _chat_completion(
        summary_system,
        [{"role": "user", "content": f"Conversation transcript:\n{transcript}\n\nSummarise the call."}],
        max_tokens=800,
    )
    parsed = _parse_json_object(raw) if raw else None
    if parsed is not None:
        issues = [s for s in (parsed.get("issues") or []) if isinstance(s, str) and s.strip()][:4]
        recs = [s for s in (parsed.get("recommendations") or []) if isinstance(s, str) and s.strip()][:3]
        steps = [s for s in (parsed.get("next_steps") or []) if isinstance(s, str) and s.strip()][:3]
        summary_text = (parsed.get("summary_text") or "").strip()
        if issues and recs and steps and summary_text:
            return {
                "issues": issues,
                "recommendations": recs,
                "next_steps": steps,
                "summary_text": summary_text,
                "fallback": False,
            }
    fb = FALLBACK_SUMMARY.get(language, FALLBACK_SUMMARY["en-IN"])
    if language == "hi-IN":
        spoken = (
            "Call ka summary ye raha. Sabse badi samasya ye hai ki 18 products out of stock hain aur "
            "cancellations 14% tak badh gaye hain, jabki price mein koi badlav nahi hai. "
            "Sabse pehle apne bestsellers ko restock karein, dispatch tez karein aur stock count sahi rakhein, "
            "aur stock mein maujood products ki visibility badhayein. Agle hafte hum sales dobara check karenge. "
            "Shukriya, aur koi madad chahiye toh main yahin hoon."
        )
    else:
        spoken = (
            "Here is the summary of our call. The main problem is that 18 products are out of stock and "
            "cancellations have risen to 14%, while your price has not changed at all. "
            "First, restock your bestsellers, speed up dispatch and keep your stock counts accurate, "
            "and push visibility on the products you do have in stock. We will review sales again next week. "
            "Thank you — and I'm here if you need any more help."
        )
    return {
        "issues": list(fb["issues"]),
        "recommendations": list(fb["recommendations"]),
        "next_steps": list(fb["next_steps"]),
        "summary_text": spoken,
        "fallback": True,
    }
