# Realtime Mock Interview Coach

A Flask-based prototype for practicing technical interviews with a voice-first workflow. Upload a resume and hold a live conversation with Gemini 2.5 Flash Lite – no typing required.

## Features

- Resume ingestion (PDF/TXT) with extraction on the server.
- Voice input powered by the browser's SpeechRecognition API.
- Instant Kokoro-powered speech synthesis for lifelike interviewer tone (no browser fallback).
- Session-aware Gemini prompts that incorporate resume context (resume + history kept server-side per session).
- Optional OpenAI Realtime WebSocket support for text generation (audio still handled by Kokoro).

## Project layout

- `app/` – Flask backend, templates, and static assets.
- `docs/` – architecture/design notes that were previously at the repo root.
- `scripts/` – helper utilities such as `run_server.sh` for local development.
- `storage/resumes/` – runtime-only folder (gitignored) where uploaded resumes are stored.

## Prerequisites

- Python 3.10+
- A modern Chromium-based browser (Chrome, Edge) for the voice APIs.
- Google AI Studio API access to Gemini 2.5 Flash Lite.
- OpenAI API access if you want to use OpenAI models (optional).
- NVIDIA GPU with CUDA drivers (Kokoro will automatically use it when available).

## Setup

1. Install dependencies:

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

2. Provide your credentials in a `.env` file (or export them in your shell):

   ```bash
   echo "FLASK_SECRET_KEY=$(python -c 'import secrets; print(secrets.token_hex(16))')" > .env
   echo "GEMINI_API_KEY=YOUR_API_KEY_HERE" >> .env
   echo "GEMINI_MODEL=gemini-2.5-flash-lite" >> .env
   echo "GEMINI_PHASE_MODEL=gemini-1.5-pro-latest" >> .env
   echo "OPENAI_API_KEY=YOUR_OPENAI_KEY_HERE" >> .env
   ```

3. Run the development server on the requested port:

   ```bash
   flask --app app.app run --host 0.0.0.0 --port 1111 --debug
   ```

   or use the helper script (it ensures dependencies are installed and requires `GEMINI_API_KEY` to be set):

   ```bash
   export GEMINI_API_KEY=your_api_key
   ./scripts/run_server.sh
   ```

4. Open `http://localhost:1111` and:
   - Upload your resume (PDF or TXT).
   - (Optional) Fill in the "Target company & role" card so OOD sessions match that context.
   - Start the interview (grant microphone access once).
   - Speak naturally; Gemini will respond out loud.

### Kokoro voice

- The `kokoro` package downloads voice weights on first use; keep the server running while it builds the cache.
- If you prefer a different timbre, change `voice="af_sarah"` in `app/tts.py`.
- Kokoro must succeed for every reply: any synthesis failure propagates to the UI so you can debug instead of silently degrading.

## Notes on Real-time Conversation

- The prototype keeps the microphone open (once permitted) and auto-loops recognition so you can speak naturally without pushing buttons.
- Gemini replies stream back with Kokoro-generated audio that plays immediately; barge-in is supported, so speaking will interrupt the model mid-sentence.
- The backend keeps the full conversation context and resume copy so each Gemini response remains grounded.

## Next Ideas

- Add WebRTC support for full-duplex streaming with Gemini Realtime.
- Explore OpenAI Realtime voice sessions for end-to-end audio when you want lower latency than the current STT/TTS loop.
- Swap the browser STT with a low-latency server-side model (Whisper V3, Deepgram, etc.) and voice-activity detection for true barge-in support.
- Persist sessions and resumes in a database for multi-user support.
- Integrate code execution challenges during the interview.
