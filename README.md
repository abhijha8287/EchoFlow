# EchoFlow

A voice-to-voice AI assistant. You talk, it listens, thinks, and talks back — over a live WebSocket connection, aiming for the lowest latency I could reasonably get it to.

Right now it runs on Gemini for all three legs (speech-to-text, the LLM, and text-to-speech). Other providers can be plugged in later, the code is written so a provider is just something that implements a small interface.

Made by Abhishek.

## What's actually in here

- **`backend/`** — Python (FastAPI + WebSockets). Owns the conversation: voice activity detection, a state machine (idle → listening → thinking → speaking, etc.), and the STT → LLM → TTS pipeline.
- **`frontend/`** — Next.js + React. Captures your mic audio in the browser, streams it to the backend, and plays back whatever audio comes back.

They talk to each other over one WebSocket: `ws://localhost:8000/ws/voice`. Binary frames carry raw audio, JSON text frames carry everything else (state changes, transcripts, errors).

## Running it locally

You need two terminals — one for the backend, one for the frontend.

### 1. Backend

```bash
cd backend
pip install -e .
```

Create a `backend/.env` file (copy `backend/.env.example`) and put your Gemini key in it:

```
GEMINI_API_KEY=your-key-here
```


Then run it:

```bash
python main.py
```

It boots on `http://localhost:8000`. Check `http://localhost:8000/health` — you should see `{"status":"ok"}`.

### 2. Frontend

```bash
cd frontend
npm install
```

Copy `frontend/.env.local.example` to `frontend/.env.local` — it just points the frontend at your local backend:

```
NEXT_PUBLIC_WS_URL=ws://localhost:8000/ws/voice
```

Then run it:

```bash
npm run dev
```

Open `http://localhost:3000`, allow microphone access, and talk.

### Running the tests

```bash
# backend
cd backend && python -m pytest

# frontend
cd frontend && npm run test
```

## Folder structure

```
voice-to-voice/
├── backend/
│   ├── main.py              # real boot entrypoint — starts the actual server
│   ├── app.py                # FastAPI app + WebSocket route wiring
│   ├── core/
│   │   └── session.py         # Session + SessionStore (conversation state, survives reconnects)
│   ├── voice/
│   │   ├── fsm.py              # the conversation state machine
│   │   ├── vad.py              # voice-activity-detection interface
│   │   ├── simple_vad.py       # simple energy-based VAD (placeholder for real Silero VAD)
│   │   ├── pipeline.py         # runs STT -> LLM -> TTS for one utterance
│   │   └── session_runtime.py  # the big orchestrator, ties everything together
│   ├── stt/                   # speech-to-text
│   │   ├── provider.py          # STT interface + shared errors
│   │   ├── algorithm.py         # chunked vs. streaming STT logic
│   │   └── gemini.py            # real Gemini STT adapter
│   ├── llm/
│   │   └── gemini.py            # real Gemini LLM adapter
│   ├── tts/
│   │   └── gemini.py            # real Gemini TTS adapter
│   ├── websocket/
│   │   ├── codec.py              # the wire protocol (binary audio frames + JSON text frames)
│   │   ├── connection.py         # the WebSocket connection manager
│   │   └── errors.py             # wire error codes
│   ├── services/
│   │   └── metrics.py           # latency/metrics logging
│   ├── utils/
│   │   └── audio.py             # audio resampling + WAV helpers
│   ├── spike/
│   │   └── gemini_spike.py      # standalone script used to probe real Gemini latency/behavior
│   └── tests/                  # pytest suite (all providers are mocked here, no real API calls)
│
└── frontend/
    ├── app/
    │   └── page.tsx             # the main (and only, for now) page
    ├── hooks/
    │   ├── useMicrophone.ts       # captures mic audio via an AudioWorklet
    │   ├── usePlayback.ts         # plays back audio from the server
    │   ├── useVoiceSession.ts     # ties the WebSocket + mic + playback + state together
    │   └── useKeyboardShortcuts.ts
    ├── lib/
    │   ├── protocol.ts            # wire protocol encode/decode (mirrors the backend's codec.py)
    │   ├── fsm.ts                 # state machine (mirrors the backend's fsm.py)
    │   ├── ws-client.ts           # WebSocket client with reconnect logic
    │   └── audio.ts               # audio format helpers
    ├── stores/
    │   ├── session.ts             # Zustand store for conversation state
    │   └── connection.ts          # Zustand store for connection/reconnect state
    └── public/worklets/
        └── pcm-capture-processor.js  # AudioWorklet that captures raw mic PCM
```

## How a conversation actually flows

1. Browser opens a WebSocket to the backend and sends `session.start`.
2. Backend replies `session.hello`, and the session sits in `LISTENING`.
3. You talk — mic audio streams up as binary frames, the VAD detects speech and then silence (the "endpoint").
4. Backend runs speech-to-text on what you said, then sends the transcript to the LLM, then turns the reply into speech.
5. The generated audio streams back down as binary frames and plays in your browser.
6. If you start talking again while it's still replying, that's a barge-in — the backend cancels the in-flight response and goes back to listening.

## Notes on the current state

- STT/LLM/TTS all use Gemini right now. Swapping in another provider (OpenAI, ElevenLabs, etc.) just means writing a new adapter behind the same interface — nothing else has to change.
- VAD is a simple volume-threshold detector, not a real speech-detection model yet.
- The LLM doesn't remember earlier turns in the conversation yet — every request is stateless.
- The UI is intentionally plain right now (just shows the state and transcript), no animations or polish yet.
