# EchoFlow

A voice-to-voice AI assistant: you speak, it listens, thinks and speaks back over a live WebSocket, with latency kept as low as possible.

## Why this exists
A project to learn what it takes to make a real-time voice assistant feel responsive: voice activity detection, turn-taking, interruptions and streaming audio in both directions.

## What it does
- Streams microphone audio from the browser to a FastAPI backend over one WebSocket (`/ws/voice`)
- Detects speech with voice activity detection and runs a conversation state machine (idle, listening, thinking, speaking)
- Runs speech-to-text, the LLM and text-to-speech for each utterance and streams the reply audio back
- Uses Gemini for all three steps by default; each step can switch to OpenAI on its own through env settings
- Keeps session state across reconnects and logs latency for each step

## Project structure
```
EchoFlow/
├── backend/
│   ├── main.py         Server entry point
│   ├── app.py          FastAPI app and WebSocket route
│   ├── core/           Session state and store
│   ├── voice/          State machine, VAD, pipeline
│   ├── stt/            Speech-to-text adapters
│   ├── llm/            LLM adapters (Gemini, OpenAI)
│   ├── tts/            Text-to-speech adapters
│   ├── websocket/      Wire protocol and connections
│   ├── services/       Latency metrics logging
│   ├── utils/          Audio resampling and WAV helpers
│   ├── spike/          Gemini latency probe script
│   └── tests/          pytest suite, providers mocked
└── frontend/
    ├── app/            Main page
    ├── hooks/          Mic, playback, session hooks
    ├── lib/            Protocol, state machine, WS client
    └── stores/         Client state
```

## Getting started
Requirements: Python 3.12+, Node.js 18+, a Gemini or OpenAI API key, a microphone.
```bash
git clone https://github.com/abhijha8287/EchoFlow.git
cd EchoFlow

cd backend                                   # terminal 1
pip install -e .
cp .env.example .env                         # add GEMINI_API_KEY
python main.py                               # http://localhost:8000/health

cd frontend                                  # terminal 2
npm install
cp .env.local.example .env.local
npm run dev                                  # http://localhost:3000
```
Open the page, allow microphone access and talk.

## Configuration
| Variable | Purpose |
|:--|:--|
| `GEMINI_API_KEY` | Default provider for all three steps |
| `OPENAI_API_KEY` | Needed if any step uses OpenAI |
| `STT_PROVIDER`, `LLM_PROVIDER`, `TTS_PROVIDER` | `gemini` or `openai`, set per step |
| `NEXT_PUBLIC_WS_URL` (frontend) | Backend WebSocket, default `ws://localhost:8000/ws/voice` |

## Tests
```bash
cd backend && python -m pytest
cd frontend && npm run test
```

## Tech stack
Python, FastAPI, WebSockets, Gemini API, OpenAI API, Next.js, React, TypeScript, Zustand, Vitest

## How a conversation actually flows

1. Browser opens a WebSocket to the backend and sends `session.start`.
2. Backend replies `session.hello`, and the session sits in `LISTENING`.
3. You talk — mic audio streams up as binary frames, the VAD detects speech and then silence (the "endpoint").
4. Backend runs speech-to-text on what you said, then sends the transcript to the LLM, then turns the reply into speech.
5. The generated audio streams back down as binary frames and plays in your browser.
6. If you start talking again while it's still replying, that's a barge-in — the backend cancels the in-flight response and goes back to listening.

## Notes on the current state

- STT/LLM/TTS default to Gemini but each has an OpenAI adapter too, selectable per role via `STT_PROVIDER`/`LLM_PROVIDER`/`TTS_PROVIDER`. Swapping in yet another provider (ElevenLabs, etc.) just means writing a new adapter behind the same interface — nothing else has to change.
- VAD is a simple volume-threshold detector, not a real speech-detection model yet.
- The LLM doesn't remember earlier turns in the conversation yet — every request is stateless.
- The UI is intentionally plain right now (just shows the state and transcript), no animations or polish yet.
