"""Snapdeal Virtual KAM engine: conversation sessions + seller-data prompt.

A session is an open-ended voice conversation (not a fixed question bank).
The bot — "Asha", a Snapdeal Virtual Key Account Manager — greets the
seller, asks targeted follow-up questions, uses the mock seller data to
diagnose the sales drop, recommends 2-3 practical actions, and ends with
a structured summary (issues / recommendations / next steps).

Sessions live in an in-memory dict keyed by uuid4. All voice/LLM calls go
through the ``voice`` and ``llm`` modules so tests can monkeypatch them —
no live calls in tests.

The seller data below is the mock data from the assignment brief. Nothing
else is claimed about the seller: the bot must not invent products, order
numbers, or metrics beyond this table, and must label any assumption as
an assumption.
"""

import base64
import uuid

from . import findings, llm, voice

MAX_VOICE_AUDIO_BYTES = 5 * 1024 * 1024
AUDIO_MIME = "audio/mpeg"

LANGUAGES = [
    {"id": "hi-IN", "name": "Hindi"},
    {"id": "en-IN", "name": "English"},
]

SELLER_DATA = (
    "Mock seller account data (use ONLY these numbers; never invent others):\n"
    "- Sales: Rs 2,00,000 last month -> Rs 1,35,000 this month (down 32.5%)\n"
    "- Orders: 400 -> 270 (down 32.5%)\n"
    "- Cancellation rate: 8% -> 14% (up sharply)\n"
    "- Average selling price: Rs 500 -> Rs 500 (flat — price is NOT the issue)\n"
    "- Out-of-stock products: 5 -> 18 (up 3.6x)"
)

KAM_SYSTEM_PROMPT = (
    "You are Asha, a warm and professional Virtual Key Account Manager (KAM) "
    "for Snapdeal, speaking with a seller whose sales have dropped this month. "
    "Speak naturally, like a helpful account manager on a phone call — short, "
    "clear sentences, 2-4 sentences per reply, never a long lecture. The seller "
    "is Indian; natural Hinglish (Hindi mixed with English) is welcome when the "
    "conversation is in Hindi, and simple, warm English otherwise. Take ownership "
    "of the problem: you are here to fix this with the seller, not just answer "
    "questions.\n\n"
    + SELLER_DATA + "\n\n"
    "Your diagnosis, from this data: the drop is driven by stock-outs (18 products "
    "unavailable, up from 5 — likely including bestsellers) plus the cancellation "
    "spike (possibly dispatch delays or listing/stock problems from the same stock "
    "issue). Average selling price is flat at Rs 500, so this is a volume and "
    "fulfilment problem, NOT a pricing problem.\n\n"
    "How the conversation should flow:\n"
    "1. The seller has just told you sales dropped. Empathise, then ask ONE "
    "targeted follow-up question at a time — for example: which products are out "
    "of stock, and are they your usual bestsellers? When did cancellations start "
    "rising? Have there been dispatch delays?\n"
    "2. After 2-3 answers, share the diagnosis in plain words, using the numbers "
    "above: sales and orders both down about a third, cancellations up to 14%, "
    "stock-outs up 3.6x, price unchanged.\n"
    "3. Recommend 2-3 practical actions: (a) urgently restock the out-of-stock "
    "bestsellers first; (b) fix the cancellations — faster dispatch, accurate "
    "stock counts, proactive messages to buyers about any delay; (c) push "
    "visibility on the products that are in stock while the restock is underway.\n"
    "4. When the seller has nothing more to add, offer to wrap up with a summary "
    "of the issues, recommendations and next steps.\n\n"
    "Rules: never invent products, order numbers, dates or metrics beyond the "
    "table above. If you make an assumption, say so clearly ('I am assuming...'). "
    "Never claim to have placed an order, changed a listing, or contacted anyone — "
    "you can only advise. Keep every reply short enough to speak comfortably."
)

GREETINGS = {
    "en-IN": (
        "Hello, I'm Asha, your Snapdeal Virtual Key Account Manager. "
        "I can see your sales have dropped this month, and I'm here to help you fix that. "
        "Tell me — what's worrying you the most right now?"
    ),
    "hi-IN": (
        "Namaste, main Asha hoon, aapki Snapdeal Virtual Key Account Manager. "
        "Maine dekha hai ki is mahine aapki sales mein girawat aayi hai, aur main aapki madad ke liye yahaan hoon. "
        "Batayein — aapko sabse zyada kis baat ki chinta hai?"
    ),
}

_sessions: dict = {}


class KamError(Exception):
    """An error with an HTTP status, raised by the engine for main.py to map."""

    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def _audio_b64(data: bytes):
    """Base64-encode mp3 bytes for JSON, or None when there is no audio."""
    if not data:
        return None
    return base64.b64encode(data).decode("ascii")


def _language_name(language: str) -> str:
    for lang in LANGUAGES:
        if lang["id"] == language:
            return lang["name"]
    return language


def start_session(language: str) -> dict:
    """Create a session and speak the greeting.

    Raises KamError: 400 unknown language, 503 voice not configured.
    """
    if language not in ("hi-IN", "en-IN"):
        raise KamError(400, "Unknown language. Pick hi-IN or en-IN.")
    if not voice.voice_available():
        raise KamError(
            503, "Voice is not configured — set GNANI_API_KEY on the backend."
        )
    session_id = str(uuid.uuid4())
    greeting = GREETINGS[language]
    _sessions[session_id] = {
        "language": language,
        "history": [{"role": "assistant", "text": greeting}],
        "done": False,
        "summary": None,
    }
    audio = voice.synthesize_speech(greeting, language=language)
    return {
        "session_id": session_id,
        "language": language,
        "greeting_text": greeting,
        "greeting_audio_base64": _audio_b64(audio),
        "audio_mime": AUDIO_MIME,
        "findings": findings.compute_findings(_sessions[session_id]["history"]),
    }


def talk(session_id: str, audio_bytes: bytes, filename: str) -> dict:
    """Transcribe one spoken seller message and reply as the KAM.

    Appends both turns to the session history. Raises KamError: 404 unknown
    session, 400 already finished / empty or oversized audio / empty
    transcript, 503 voice not configured.
    """
    session = _sessions.get(session_id)
    if session is None:
        raise KamError(404, "Unknown session_id.")
    if session["done"]:
        raise KamError(400, "This call is already finished — start a new call.")
    if not audio_bytes:
        raise KamError(400, "No audio received — please record a message.")
    if len(audio_bytes) > MAX_VOICE_AUDIO_BYTES:
        raise KamError(400, "That recording is too large (max 5 MB, about 30 seconds of speech).")
    if not voice.voice_available():
        raise KamError(
            503, "Voice is not configured — set GNANI_API_KEY on the backend."
        )

    language = session["language"]
    transcript = voice.transcribe_audio(audio_bytes, filename=filename, language_code=language)
    if not transcript or not transcript.strip():
        raise KamError(400, "No speech detected — please try again.")

    session["history"].append({"role": "user", "text": transcript})
    reply = llm.kam_reply(KAM_SYSTEM_PROMPT, session["history"], language)
    reply_text = reply.get("text") or ""
    session["history"].append({"role": "assistant", "text": reply_text})
    reply_audio = voice.synthesize_speech(reply_text, language=language)
    return {
        "session_id": session_id,
        "transcript": transcript,
        "reply_text": reply_text,
        "reply_audio_base64": _audio_b64(reply_audio),
        "audio_mime": AUDIO_MIME,
        "llm_available": llm.llm_available(),
        "fallback": bool(reply.get("fallback")),
        "turns": len(session["history"]),
        "history": list(session["history"]),
        "findings": findings.compute_findings(session["history"]),
    }


def end_call(session_id: str) -> dict:
    """End the call and return the spoken + structured summary.

    Raises KamError: 404 unknown session, 400 already finished.
    """
    session = _sessions.get(session_id)
    if session is None:
        raise KamError(404, "Unknown session_id.")
    if session["done"]:
        raise KamError(400, "This call is already finished.")
    language = session["language"]
    summary = llm.kam_summary(KAM_SYSTEM_PROMPT, session["history"], language)
    summary_text = summary.get("summary_text") or ""
    session["done"] = True
    session["summary"] = summary
    session["history"].append({"role": "assistant", "text": summary_text})
    summary_audio = voice.synthesize_speech(summary_text, language=language)
    return {
        "session_id": session_id,
        "issues": summary.get("issues") or [],
        "recommendations": summary.get("recommendations") or [],
        "next_steps": summary.get("next_steps") or [],
        "summary_text": summary_text,
        "summary_audio_base64": _audio_b64(summary_audio),
        "audio_mime": AUDIO_MIME,
        "llm_available": llm.llm_available(),
        "fallback": bool(summary.get("fallback")),
        "history": list(session["history"]),
    }


def get_history(session_id: str) -> dict:
    """Return the conversation history for a session."""
    session = _sessions.get(session_id)
    if session is None:
        raise KamError(404, "Unknown session_id.")
    return {
        "session_id": session_id,
        "language": session["language"],
        "done": session["done"],
        "history": list(session["history"]),
    }


def clear_sessions() -> None:
    """Test helper: drop all sessions."""
    _sessions.clear()
