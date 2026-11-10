"""Voice KAM tests — all Gnani + Groq calls mocked (zero live calls).

Covers the start/talk/summary flow, the deterministic fallback when the
LLM is unavailable (flagged, never presented as an AI reply), validation
errors, and language selection (Hindi vs English params passed to the
voice layer).
"""

import base64

import pytest
from fastapi.testclient import TestClient

from app import kam, llm, voice
from app.main import app

client = TestClient(app)

FAKE_MP3 = b"FAKE-MP3-BYTES"
FAKE_AUDIO = b"FAKE-RECORDED-AUDIO"


@pytest.fixture(autouse=True)
def _clean_sessions():
    kam.clear_sessions()
    yield
    kam.clear_sessions()


@pytest.fixture()
def voice_on(monkeypatch):
    """Voice configured; STT/TTS replaced with fakes that record their args."""
    calls = {"transcribe": [], "synthesize": []}

    def fake_transcribe(audio_bytes, filename="clip.m4a", language_code="en-IN"):
        calls["transcribe"].append(
            {"bytes": audio_bytes, "filename": filename, "language_code": language_code}
        )
        return "My sales have dropped this month, I don't know why."

    def fake_synthesize(text, language="en-IN"):
        calls["synthesize"].append({"text": text, "language": language})
        return FAKE_MP3

    monkeypatch.setattr(voice, "voice_available", lambda: True)
    monkeypatch.setattr(voice, "transcribe_audio", fake_transcribe)
    monkeypatch.setattr(voice, "synthesize_speech", fake_synthesize)
    return calls


@pytest.fixture()
def llm_on(monkeypatch):
    """Groq replaced with a fake KAM that asks one follow-up, then diagnoses."""

    def fake_reply(system_prompt, history, language):
        user_turns = sum(1 for m in history if m.get("role") == "user")
        if user_turns <= 1:
            return {"text": "Which of your products have gone out of stock — are they your usual bestsellers?", "fallback": False}
        return {
            "text": "Thank you. Your price is flat at Rs 500, so this is not pricing — it is stock-outs and cancellations. Restock your bestsellers first.",
            "fallback": False,
        }

    def fake_summary(system_prompt, history, language):
        return {
            "issues": ["Sales fell 32.5% and 18 products are out of stock."],
            "recommendations": ["Restock the out-of-stock bestsellers first."],
            "next_steps": ["Share the list of the 18 out-of-stock products."],
            "summary_text": "Sales fell because of stock-outs. Restock your bestsellers first.",
            "fallback": False,
        }

    monkeypatch.setattr(llm, "kam_reply", fake_reply)
    monkeypatch.setattr(llm, "kam_summary", fake_summary)
    monkeypatch.setattr(llm, "llm_available", lambda: True)


def _start(language="en-IN"):
    res = client.post("/api/kam/start", json={"language": language})
    assert res.status_code == 200, res.text
    return res.json()


def _talk(session_id, audio=FAKE_AUDIO, filename="clip.m4a"):
    return client.post(
        "/api/kam/talk",
        params={"session_id": session_id},
        files={"audio": (filename, audio, "audio/mp4")},
    )


def test_health():
    res = client.get("/api/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert "gnani_configured" in body
    assert "groq_configured" in body


def test_start_returns_greeting(voice_on):
    body = _start("en-IN")
    assert body["session_id"]
    assert body["greeting_text"] == kam.GREETINGS["en-IN"]
    assert base64.b64decode(body["greeting_audio_base64"]) == FAKE_MP3
    assert body["audio_mime"] == "audio/mpeg"
    assert voice_on["synthesize"][0]["language"] == "en-IN"


def test_start_invalid_language(voice_on):
    assert client.post("/api/kam/start", json={"language": "fr-FR"}).status_code == 400


def test_start_missing_key_503(monkeypatch):
    monkeypatch.setattr(voice, "voice_available", lambda: False)
    res = client.post("/api/kam/start", json={"language": "en-IN"})
    assert res.status_code == 503
    assert "GNANI_API_KEY" in res.json()["detail"]


def test_talk_flow_to_summary(voice_on, llm_on):
    body = _start("en-IN")
    session_id = body["session_id"]

    res = _talk(session_id)
    assert res.status_code == 200, res.text
    data = res.json()
    assert data["transcript"] == "My sales have dropped this month, I don't know why."
    assert "out of stock" in data["reply_text"]
    assert data["fallback"] is False
    assert data["llm_available"] is True
    assert base64.b64decode(data["reply_audio_base64"]) == FAKE_MP3
    assert len(data["history"]) == 3  # greeting + user + reply

    res = _talk(session_id)
    assert res.status_code == 200, res.text
    data = res.json()
    assert "not pricing" in data["reply_text"]
    assert len(data["history"]) == 5

    res = client.post(f"/api/kam/summary/{session_id}")
    assert res.status_code == 200, res.text
    summary = res.json()
    assert len(summary["issues"]) == 1
    assert len(summary["recommendations"]) == 1
    assert len(summary["next_steps"]) == 1
    assert summary["summary_text"]
    assert base64.b64decode(summary["summary_audio_base64"]) == FAKE_MP3
    assert summary["fallback"] is False

    # History shows the finished call.
    res = client.get(f"/api/kam/history/{session_id}")
    assert res.status_code == 200
    assert res.json()["done"] is True
    assert len(res.json()["history"]) == 6  # + summary turn


def test_fallback_when_llm_unavailable(voice_on, monkeypatch):
    """No Groq key: replies come from the data-only fallback, flagged True."""
    monkeypatch.setattr(llm, "llm_available", lambda: False)
    body = _start("en-IN")
    session_id = body["session_id"]
    res = _talk(session_id)
    assert res.status_code == 200, res.text
    data = res.json()
    assert data["fallback"] is True
    assert data["llm_available"] is False
    # Fallback turn 1 asks about out-of-stock products, using only mock data.
    assert "out of stock" in data["reply_text"]
    res = client.post(f"/api/kam/summary/{session_id}")
    assert res.status_code == 200, res.text
    summary = res.json()
    assert summary["fallback"] is True
    assert len(summary["issues"]) == 4
    assert len(summary["recommendations"]) == 3
    assert len(summary["next_steps"]) == 3
    assert "32.5%" in summary["issues"][0]


def test_talk_unknown_session_404(voice_on):
    assert _talk("no-such-session").status_code == 404


def test_talk_empty_audio_400(voice_on):
    body = _start()
    assert _talk(body["session_id"], audio=b"").status_code == 400


def test_talk_oversized_audio_400(voice_on):
    body = _start()
    assert _talk(body["session_id"], audio=b"x" * (5 * 1024 * 1024 + 1)).status_code == 400


def test_talk_empty_transcript_400(voice_on, monkeypatch):
    monkeypatch.setattr(voice, "transcribe_audio", lambda *a, **k: "")
    body = _start()
    res = _talk(body["session_id"])
    assert res.status_code == 400
    assert "No speech detected" in res.json()["detail"]


def test_talk_after_done_400(voice_on, llm_on):
    body = _start()
    session_id = body["session_id"]
    assert _talk(session_id).status_code == 200
    assert client.post(f"/api/kam/summary/{session_id}").status_code == 200
    assert _talk(session_id).status_code == 400
    assert client.post(f"/api/kam/summary/{session_id}").status_code == 400


def test_language_selection_changes_voice_params(voice_on, llm_on):
    body = _start("hi-IN")
    assert body["greeting_text"] == kam.GREETINGS["hi-IN"]
    assert voice_on["synthesize"][0]["language"] == "hi-IN"
    res = _talk(body["session_id"])
    assert res.status_code == 200
    assert voice_on["transcribe"][0]["language_code"] == "hi-IN"


def test_voice_for_language_mapping():
    assert voice.voice_for_language("en-IN") == "Kaveri"
    assert voice.voice_for_language("hi-IN") == "Nalini"
    assert voice.voice_for_language("xx-YY") == "Kaveri"  # default


def test_system_prompt_contains_seller_data():
    for fragment in ("Rs 2,00,000", "Rs 1,35,000", "400 -> 270", "8% -> 14%",
                     "Rs 500", "5 -> 18"):
        assert fragment in kam.KAM_SYSTEM_PROMPT
    assert "not a pricing problem" in kam.KAM_SYSTEM_PROMPT.lower() or \
        "NOT a pricing problem" in kam.KAM_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Live findings tracker (app/findings.py) — deterministic, no LLM call.
# ---------------------------------------------------------------------------

from app import findings as findings_mod


def _statuses(history):
    return {f["id"]: f["status"] for f in findings_mod.compute_findings(history)}


def test_findings_all_checking_on_greeting_only():
    history = [{"role": "assistant", "text": kam.GREETINGS["en-IN"]}]
    result = findings_mod.compute_findings(history)
    assert [f["id"] for f in result] == ["stockouts", "cancellations", "pricing"]
    assert _statuses(history) == {
        "stockouts": "checking",
        "cancellations": "checking",
        "pricing": "checking",
    }
    # Details are the brief's real numbers only — nothing invented.
    assert "18 vs 5 (3.6×)" in result[0]["detail"]
    assert "14% (up from 8%)" in result[1]["detail"]
    assert "₹500" in result[2]["detail"]


def test_findings_stockouts_confirmed_by_seller():
    history = [{"role": "user", "text": "Yes, many of my best sellers are out of stock."}]
    assert _statuses(history)["stockouts"] == "confirmed"
    assert _statuses(history)["cancellations"] == "checking"


def test_findings_cancellations_confirmed_by_any_mention():
    history = [{"role": "assistant", "text": "Your cancellation rate has gone up from 8% to 14%."}]
    assert _statuses(history)["cancellations"] == "confirmed"


def test_findings_pricing_confirmed_by_seller_saying_unchanged():
    history = [{"role": "user", "text": "I have not changed my prices."}]
    assert _statuses(history)["pricing"] == "confirmed"


def test_findings_pricing_confirmed_by_asha_ruling_it_out():
    history = [{"role": "assistant", "text": "Your price is flat, so this is not a pricing problem."}]
    assert _statuses(history)["pricing"] == "confirmed"


def test_findings_all_confirmed_after_typical_second_turn():
    history = [
        {"role": "assistant", "text": kam.GREETINGS["en-IN"]},
        {"role": "user", "text": "My sales have dropped this month, I don't know why."},
        {"role": "assistant", "text": "Which of your products have gone out of stock — are they your usual bestsellers?"},
        {"role": "user", "text": "Yes, many of my best sellers are out of stock, and I have had a lot of cancellations too. I have not changed my prices."},
    ]
    assert _statuses(history) == {
        "stockouts": "confirmed",
        "cancellations": "confirmed",
        "pricing": "confirmed",
    }


def test_findings_in_start_and_talk_responses(voice_on, llm_on, monkeypatch):
    """start returns all-checking findings; talk progresses them."""
    body = _start("en-IN")
    assert body["findings"] == [
        {"id": "stockouts", "label": "Stock-outs",
         "detail": "Out-of-stock products 18 vs 5 (3.6×) — primary cause",
         "status": "checking"},
        {"id": "cancellations", "label": "Cancellations",
         "detail": "Cancellation rate 14% (up from 8%) — secondary cause",
         "status": "checking"},
        {"id": "pricing", "label": "Pricing",
         "detail": "Pricing ruled out — average selling price flat at ₹500",
         "status": "checking"},
    ]

    # Seller confirms all three in one message; Asha's reply also rules out pricing.
    monkeypatch.setattr(
        voice, "transcribe_audio",
        lambda *a, **k: "Yes, my best sellers are out of stock and I have had a lot of cancellations too. I have not changed my prices.",
    )
    res = _talk(body["session_id"])
    assert res.status_code == 200, res.text
    statuses = {f["id"]: f["status"] for f in res.json()["findings"]}
    assert statuses == {
        "stockouts": "confirmed",
        "cancellations": "confirmed",
        "pricing": "confirmed",
    }
