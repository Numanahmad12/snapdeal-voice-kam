# Snapdeal Virtual KAM — Mini AI Voice KAM

A browser-based voice bot that plays a Snapdeal Virtual Key Account Manager
("Asha"). A seller calls because sales dropped; Asha greets them, holds a
natural two-way voice conversation, asks follow-up questions, uses the mock
seller data to diagnose the problem, recommends 2–3 practical actions, and
summarises the issues, recommendations and next steps at the end.

Built for the **Snapdeal AI First Intern** practical assignment
(deadline 11 Nov 2026, 11:15 IST). Architecture follows
[BolInterview](https://github.com/Numanahmad12/bolinterview) — the same
FastAPI + static-frontend + Gnani voice + Groq LLM stack, adapted from a
fixed-question interviewer to an open-ended KAM conversation.

## How it works

```
Seller speaks (mic) ──► Prisma v2.5 STT ──► transcript
        │                                        │
        │   Groq openai/gpt-oss-120b (KAM prompt + full history)
        │                                        ▼
        └── Timbre v2.5 TTS ◄── KAM reply (spoken, Hinglish-friendly)
```

- **Voice in:** Gnani Prisma v2.5 speech-to-text (30 s clips, mic → mp4/m4a).
- **Brain:** Groq `openai/gpt-oss-120b` with `reasoning_effort: "low"`
  (without it, this reasoning model's internal reasoning can eat the whole
  token budget and return empty content). The system prompt carries the mock
  seller data and the conversation flow: greet → ask 2–3 targeted follow-ups
  → diagnose → recommend 2–3 actions → summarise.
- **Voice out:** Gnani Timbre v2.5 text-to-speech (Kaveri for English,
  Nalini for Hindi), mp3 played back in the browser.
- **Frontend:** single static page — big mic button, live waveform while
  recording, spoken replies, and a visible transcript of the whole call.

## The mock seller data (from the assignment brief — the only data used)

| Metric | Last Month | This Month |
| --- | --- | --- |
| Sales | ₹2,00,000 | ₹1,35,000 (−32.5%) |
| Orders | 400 | 270 (−32.5%) |
| Cancellation rate | 8% | 14% |
| Average selling price | ₹500 | ₹500 (flat) |
| Out-of-stock products | 5 | 18 (3.6x) |

**Diagnosis the bot reaches:** a volume + fulfilment problem, not pricing.
Stock-outs (18 products, likely including bestsellers) plus the cancellation
spike (probably dispatch delays / stale stock counts from the same stock
problem). ASP is flat, so price is not the issue.

**Recommended actions:** (1) urgently restock the out-of-stock bestsellers
first; (2) fix cancellations — faster dispatch, accurate stock counts,
proactive buyer messages; (3) push visibility on in-stock products while
the restock is underway.

## Run locally

```bash
cd voice-kam
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

export GNANI_API_KEY="..."   # Gnani key (Prisma STT + Timbre TTS)
export GROQ_API_KEY="..."    # Groq key (openai/gpt-oss-120b)

uvicorn app.main:app --host 0.0.0.0 --port 8000
# open http://localhost:8000
```

Keys come from environment variables only — they are never hardcoded or
written to any file. Without `GROQ_API_KEY` the KAM still talks, using a
clearly-flagged deterministic fallback built only from the mock data above
(the UI shows a note, and the API returns `"fallback": true`). Without
`GNANI_API_KEY` the voice loop cannot start and `/api/kam/start` returns
503 with an honest message.

## API

| Method | Endpoint | What it does |
| --- | --- | --- |
| GET | `/api/health` | Liveness + which keys are configured |
| POST | `/api/kam/start` | `{language}` → session + spoken greeting |
| POST | `/api/kam/talk?session_id=…` | multipart `audio` → transcript, KAM reply (text + audio), full history |
| POST | `/api/kam/summary/{session_id}` | Ends the call → issues, recommendations, next steps + spoken summary |
| GET | `/api/kam/history/{session_id}` | Conversation history |

## Tests

```bash
pytest -q
```

All Gnani + Groq calls are mocked — zero live calls. Covers the
start/talk/summary flow, the no-LLM fallback path, validation errors
(unknown session, empty/oversized audio, empty transcript, talk after the
call ended), and language selection.

## Deploy (Render, Docker — same as BolInterview)

```bash
docker build -t voice-kam .
docker run -e GNANI_API_KEY="..." -e GROQ_API_KEY="..." -p 8000:8000 voice-kam
```

On Render: create a Web Service from the repo, Docker runtime, and set
`GNANI_API_KEY` and `GROQ_API_KEY` in Environment Variables. The Dockerfile
runs `uvicorn app.main:app --host 0.0.0.0 --port $PORT`.

## Files

```
app/main.py    FastAPI app + endpoints
app/kam.py     Conversation engine: sessions, KAM prompt, greeting, summary
app/llm.py     Groq chat calls + deterministic data-only fallback
app/voice.py   Gnani Prisma STT + Timbre TTS (keys from env only)
static/        Frontend (index.html + style.css)
tests/         pytest suite (mocked voice/LLM)
requirements.txt
Dockerfile
```

## Submission checklist (for the assignment)

- [ ] Deploy to Render (or run locally) and get the demo link.
- [ ] Record a 1–2 minute screen recording of a real voice conversation.
- [ ] Write the one-page note: models (Prisma v2.5 STT, Timbre v2.5 TTS,
  Groq gpt-oss-120b), workflow (mic → STT → LLM → TTS → speaker), and
  the diagnosis above.
