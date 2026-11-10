"""Snapdeal Virtual KAM — a browser-based voice bot for Snapdeal sellers.

FastAPI backend + a single static frontend served by the same app.

Endpoints:
    GET    /                          the frontend (static/index.html)
    GET    /api/health                liveness + which keys are configured
    POST   /api/kam/start             {language} -> session + spoken greeting
    POST   /api/kam/talk              multipart audio + session_id -> transcript,
                                      KAM spoken reply, full history
    POST   /api/kam/summary/{id}     end the call -> issues, recommendations,
                                      next steps + spoken summary
    GET    /api/kam/history/{id}     conversation history for a session

Voice: Gnani Prisma v2.5 STT + Timbre v2.5 TTS (key from GNANI_API_KEY only).
Brain: Groq openai/gpt-oss-120b (key from GROQ_API_KEY only). When Groq is
not configured the KAM answers with a clearly-flagged, data-only fallback
built from the mock seller data — never presented as an LLM reply.
"""

from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import kam, llm, voice

APP_VERSION = "1.0.0"

app = FastAPI(title="Snapdeal Virtual KAM", version=APP_VERSION)

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


class StartRequest(BaseModel):
    language: str = "en-IN"


def _map_error(exc: kam.KamError) -> HTTPException:
    return HTTPException(status_code=exc.status_code, detail=exc.detail)


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "version": APP_VERSION,
        "gnani_configured": voice.voice_available(),
        "groq_configured": llm.llm_available(),
    }


@app.post("/api/kam/start")
def start(req: StartRequest) -> dict:
    try:
        return kam.start_session(req.language)
    except kam.KamError as exc:
        raise _map_error(exc)


@app.post("/api/kam/talk")
async def talk(session_id: str, audio: UploadFile = File(...)) -> dict:
    data = await audio.read()
    try:
        return kam.talk(session_id, data, audio.filename or "clip.m4a")
    except kam.KamError as exc:
        raise _map_error(exc)


@app.post("/api/kam/summary/{session_id}")
def summary(session_id: str) -> dict:
    try:
        return kam.end_call(session_id)
    except kam.KamError as exc:
        raise _map_error(exc)


@app.get("/api/kam/history/{session_id}")
def history(session_id: str) -> dict:
    try:
        return kam.get_history(session_id)
    except kam.KamError as exc:
        raise _map_error(exc)


app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
