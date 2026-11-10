"""Gnani voice backend for the Snapdeal Virtual KAM voice bot.

Wraps the official ``gnani-vachana`` SDK (Prisma v2.5 speech-to-text,
Timbre v2.5 text-to-speech), following the same pattern as BolInterview:

* The API key comes ONLY from the ``GNANI_API_KEY`` environment variable
  — it is never hardcoded, logged, or shipped.
* Clients are created lazily and cached; the module still imports when the
  SDK is absent (voice functions then report unavailable).
* Generation never raises — every failure returns an empty result (""/b"")
  and the caller decides what the user sees.

Voice choice (both verified present in the timbre-v2.5 catalog via
``GnaniTTSClient.supported_voices(model="timbre-v2.5")``):
* English (en-IN) -> Kaveri (default — the seller conversation is in
  English/Hinglish)
* Hindi (hi-IN) -> Nalini

No live Gnani call is ever made in tests — monkeypatch
``_get_stt_client`` / ``_get_tts_client`` with fakes.
"""

import logging
import os
import time

logger = logging.getLogger(__name__)

# Prisma v2.5 is the server-side default STT model; never request v3.0.
STT_DEFAULT_LANGUAGE = "en-IN"
TTS_MODEL = "timbre-v2.5"
# Language -> Timbre v2.5 voice (both verified in the v2.5 catalog).
TTS_VOICES = {
    "hi-IN": "Nalini",
    "en-IN": "Kaveri",
}
TTS_DEFAULT_LANGUAGE = "en-IN"

try:
    from gnani.stt import GnaniSTTClient
except Exception:  # pragma: no cover - SDK absent
    GnaniSTTClient = None

try:
    from gnani.tts import AudioConfig, GnaniTTSClient
except Exception:  # pragma: no cover - SDK absent
    AudioConfig = None
    GnaniTTSClient = None

_stt_client = None
_tts_client = None


def _get_gnani_api_key() -> str:
    """Read the Gnani API key from the environment (never hardcoded)."""
    return os.getenv("GNANI_API_KEY", "").strip()


def voice_available() -> bool:
    """True when a Gnani API key is configured (voice calls can be attempted)."""
    return bool(_get_gnani_api_key())


def _get_stt_client():
    """Return a cached Prisma STT client, or None when unavailable.

    The SDK constructor raises when no key is set, so the env var is
    checked first; any other failure also degrades to None.
    """
    global _stt_client
    if _stt_client is not None:
        return _stt_client
    if GnaniSTTClient is None or not _get_gnani_api_key():
        return None
    try:
        _stt_client = GnaniSTTClient()
    except Exception as exc:  # pragma: no cover - constructor failure
        logger.warning("Gnani STT client unavailable (%s).", exc)
        return None
    return _stt_client


def _get_tts_client():
    """Return a cached Timbre TTS client, or None when unavailable."""
    global _tts_client
    if _tts_client is not None:
        return _tts_client
    if GnaniTTSClient is None or not _get_gnani_api_key():
        return None
    try:
        _tts_client = GnaniTTSClient()
    except Exception as exc:  # pragma: no cover - constructor failure
        logger.warning("Gnani TTS client unavailable (%s).", exc)
        return None
    return _tts_client


def _call_with_retry(fn, *, label: str):
    """Call fn(), retrying transient Gnani failures with backoff.

    The Vachana API occasionally returns HTTP 504 (gateway time-out) and
    then HTTP 429 ("Only 1 active request is allowed at a time") while the
    timed-out request is still counted as active — both seen live on
    Render 8 Nov 2026. A single call failing that way is transient, so
    retry twice (after 5 s, then 15 s) before giving up. Non-transient
    errors are raised immediately. Returns fn()'s result.
    """
    delays = (5.0, 15.0)
    last_exc = None
    for attempt in range(3):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            msg = str(exc)
            transient = ("504" in msg or "429" in msg or "RATE_LIMITED" in msg
                         or "time-out" in msg.lower() or "timeout" in msg.lower())
            if not transient or attempt == 2:
                raise
            delay = delays[attempt]
            logger.warning("%s failed (%s); retrying in %.0fs.", label, exc, delay)
            time.sleep(delay)
    raise last_exc


def voice_for_language(language: str) -> str:
    """Timbre v2.5 voice for a language code (Nalini for Hindi, Kaveri for English)."""
    return TTS_VOICES.get(language, TTS_VOICES[TTS_DEFAULT_LANGUAGE])


def transcribe_audio(audio_bytes: bytes, filename: str = "clip.m4a",
                     language_code: str = STT_DEFAULT_LANGUAGE) -> str:
    """Transcribe spoken audio with Prisma v2.5 (REST, 30 s limit).

    Returns the transcript string, or "" when there is no key, no audio,
    or the call fails for any reason. Never raises.
    """
    if not audio_bytes:
        return ""
    client = _get_stt_client()
    if client is None:
        return ""
    try:
        # model=None -> the server-side default (Prisma v2.5).
        result = _call_with_retry(
            lambda: client.transcribe_bytes(
                audio_bytes, filename=filename or "clip.m4a",
                language_code=language_code,
            ),
            label="Prisma transcription",
        )
    except Exception as exc:
        logger.warning("Prisma transcription failed (%s).", exc)
        return ""
    if not isinstance(result, dict):
        return ""
    return (result.get("transcript") or "").strip()


def synthesize_speech(text: str, language: str = TTS_DEFAULT_LANGUAGE) -> bytes:
    """Speak text with Timbre v2.5 (mp3).

    Voice follows the language: Nalini for Hindi, Kaveri for English.
    Returns raw mp3 audio bytes, or b"" when there is no key, no text,
    or the call fails for any reason. Never raises.
    """
    if not text or not text.strip():
        return b""
    client = _get_tts_client()
    if client is None:
        return b""
    try:
        audio_config = None
        if AudioConfig is not None:
            audio_config = AudioConfig(
                container="mp3", sample_rate=48000, bitrate="128k"
            )
        audio = _call_with_retry(
            lambda: client.synthesize(
                text,
                voice=voice_for_language(language),
                model=TTS_MODEL,
                language=language,
                speed=1.0,
                audio_config=audio_config,
            ),
            label="Timbre synthesis",
        )
    except Exception as exc:
        logger.warning("Timbre synthesis failed (%s).", exc)
        return b""
    if not audio:
        return b""
    return bytes(audio)
