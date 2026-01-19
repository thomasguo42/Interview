from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any, Optional
from uuid import uuid4

from flask import (
    Flask,
    Response,
    jsonify,
    render_template,
    request,
    session,
)

import requests
from requests import RequestException

from .config import config
from .gemini_client import GeminiClient, GeminiError
from .resume_processor import allowed_file, extract_text
from .whisper_service import get_whisper_transcriber
from .state import (
    append_turn,
    clear_conversation,
    get_conversation,
    get_resume,
    has_resume,
    store_resume,
    # Structured interview functions
    start_interview,
    get_interview_state,
    update_interview_phase,
    update_code,
    get_current_code,
    update_last_speech,
    set_problem_presented,
    set_coding_question,
    set_code_evaluation,
    calculate_time_in_phase,
    calculate_total_time,
    set_ood_question,
    get_ood_question,
    set_design_phase_complete,
    set_company_context,
    get_company_context,
    set_interview_ended,
    is_interview_ended,
    set_interview_report,
    get_interview_report,
    PHASE_INTRO,
    PHASE_RESUME,
    PHASE_CODING,
    PHASE_QUESTIONS,
    PHASE_OOD_DESIGN,
    PHASE_OOD_IMPLEMENTATION,
    SUPPORTED_LANGUAGES,
)
from .ood_questions import get_hard_question, get_stock_match_engine_question
from .question_bank import get_random_coding_question
from .question_tests import ensure_tests_for_question, get_cached_tests, format_signatures_for_prompt
from .test_runner import run_code_tests
from .tts import KokoroUnavailable, kokoro


def create_app() -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config.from_object(config)

    resume_storage = Path(__file__).resolve().parent.parent / "storage" / "resumes"
    resume_storage.mkdir(exist_ok=True, parents=True)
    app.config["UPLOAD_FOLDER"] = resume_storage

    def _ensure_session_id() -> str:
        session_id = session.get("session_id")
        if not session_id:
            session_id = uuid4().hex
            session["session_id"] = session_id
        return session_id

    @app.route("/")
    def index() -> str:
        return render_template("index.html")

    @app.route("/api/status", methods=["GET"])
    def status():
        session_id = session.get("session_id")
        resume_loaded = False
        conversation_started = False
        if session_id:
            resume_loaded = has_resume(session_id)
            conversation_started = resume_loaded and bool(get_conversation(session_id))

        return jsonify(
            {
                "resumeLoaded": resume_loaded,
                "conversationStarted": conversation_started,
            }
        )

    @app.route("/api/upload_resume", methods=["POST"])
    def upload_resume():
        if "file" not in request.files:
            return jsonify({"error": "No file part in the request."}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No file selected."}), 400

        if not allowed_file(file.filename):
            return jsonify(
                {"error": "Unsupported file type. Upload a PDF or TXT resume."}
            ), 400

        try:
            resume_text = extract_text(file)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        except Exception:
            return jsonify({"error": "Failed to process the resume file."}), 500

        session_id = _ensure_session_id()
        store_resume(session_id, resume_text)

        return jsonify({"message": "Resume uploaded successfully."})

    @app.route("/api/company_context", methods=["GET", "POST"])
    def company_context_endpoint():
        session_id = session.get("session_id") or _ensure_session_id()

        if request.method == "GET":
            context = get_company_context(session_id)
            return jsonify(context)

        payload = request.get_json(silent=True) or {}
        company = (payload.get("company") or "").strip()
        role = (payload.get("role") or "").strip()
        details = (payload.get("details") or "").strip()

        set_company_context(session_id, company=company, role=role, details=details)
        return jsonify({"message": "Company & role context saved."})

    @app.route("/api/chat", methods=["POST"])
    def chat():
        payload = request.get_json(silent=True) or {}
        user_message = (payload.get("message") or "").strip()
        if not user_message:
            return jsonify({"error": "Message is required."}), 400

        session_id = session.get("session_id")
        if not session_id:
            return (
                jsonify({"error": "Upload a resume before starting the interview."}),
                400,
            )

        # Check if structured interview is active
        interview_state = get_interview_state(session_id)

        resume_text = get_resume(session_id)

        # Resume is optional for coding_only mode, required for full mode
        interview_mode = interview_state.get("mode") if interview_state else None
        if not resume_text:
            resume_required = not interview_state or interview_mode in {"full"}
            if resume_required:
                return jsonify({"error": "Upload a resume before starting the interview."}), 400
        current_code = payload.get("code", "")
        code_changed = payload.get("code_changed", False)

        code_supported = bool(
            interview_state and interview_state.get("mode") in {"full", "coding_only", "ood"}
        )

        if code_supported and interview_state and not current_code:
            stored_code = get_current_code(session_id)
            if stored_code:
                current_code = stored_code
                code_changed = False
        # Preserve existing code snapshot so Gemini still sees editor content even if frontend omits it
        print("[CHAT DEBUG] No code payload received; using last stored code snapshot.")

        print(f"[CHAT DEBUG] User message: '{user_message}'")
        print(f"[CHAT DEBUG] Code received - length: {len(current_code)}, changed: {code_changed}")
        if current_code:
            print(f"[CHAT DEBUG] Code preview: {current_code[:150]}...")

        # Update state
        if interview_state:
            update_last_speech(session_id)
            if code_supported and current_code:
                update_code(session_id, current_code)
                if code_changed and interview_state.get("current_phase") == PHASE_CODING:
                    coding_question = interview_state.get("coding_question") or {}
                    question_title = coding_question.get("title", "")
                    if question_title:
                        test_spec = get_cached_tests(question_title)
                        if test_spec:
                            evaluation = run_code_tests(
                                interview_state.get("language", ""),
                                current_code,
                                test_spec,
                            )
                            set_code_evaluation(session_id, evaluation)

        conversation: List[Dict[str, Any]] = get_conversation(session_id)
        try:
            client = GeminiClient()

            # Use structured interview prompts if interview is active
            if interview_state:
                reply_text = client.generate_structured_interview_reply(
                    session_id=session_id,
                    conversation=conversation,
                    resume_text=resume_text,
                    user_message=user_message,
                    current_code=current_code,
                    code_changed=code_changed,
                )
            else:
                reply_text = client.generate_interview_reply(
                    conversation=conversation,
                    resume_text=resume_text,
                    user_message=user_message,
                )
        except (ValueError, GeminiError) as exc:
            return jsonify({"error": str(exc)}), 500

        print(f"[CHAT DEBUG] Gemini reply length: {len(reply_text)}")
        print(f"[CHAT DEBUG] Gemini reply preview: {reply_text[:200]}...")
        print(f"[CHAT DEBUG] Contains CODE_START: {'[CODE_START]' in reply_text}")
        print(f"[CHAT DEBUG] Contains CODE_END: {'[CODE_END]' in reply_text}")

        phase_decision = None
        if interview_state:
            enriched_conversation = list(conversation)
            enriched_conversation.append({"role": "user", "parts": [{"text": user_message}]})
            enriched_conversation.append({"role": "model", "parts": [{"text": reply_text}]})

            phase_decision = client.decide_phase_transition(
                session_id=session_id,
                interview_state=interview_state,
                conversation=enriched_conversation,
                user_message=user_message,
                model_reply=reply_text,
            )

            _apply_phase_decision(session_id, interview_state, phase_decision, user_message, reply_text)
            _handle_problem_presentation(session_id, interview_state, user_message, reply_text, phase_decision)

        # Strip code markers before TTS (code will be displayed in editor, not spoken)
        tts_text = _strip_code_markers(reply_text)

        try:
            print(f"[BACKEND DEBUG] About to call kokoro.synthesize_base64")
            print(f"[BACKEND DEBUG] Reply text: '{tts_text[:100]}...'")
            print(f"[BACKEND DEBUG] Reply text length: {len(tts_text)}")
            reply_audio = kokoro.synthesize_base64(tts_text)
            print(f"[BACKEND DEBUG] Audio generated successfully, length: {len(reply_audio)}")
        except KokoroUnavailable as exc:
            print(f"[BACKEND DEBUG] KokoroUnavailable error: {exc}")
            return jsonify({"error": str(exc)}), 500
        except Exception as exc:
            print(f"[BACKEND DEBUG] Unexpected error in TTS: {exc}")
            import traceback
            print(f"[BACKEND DEBUG] Traceback: {traceback.format_exc()}")
            return jsonify({"error": f"TTS error: {exc}"}), 500

        append_turn(session_id, user_message, reply_text)

        interview_ended = False
        if interview_state:
            total_time = calculate_total_time(session_id)
            mode = interview_state.get("mode", "full")
            current_phase = interview_state.get("current_phase")
            if total_time >= 40.0 and mode in {"full", "coding_only", "ood"}:
                set_interview_ended(session_id, True)
                interview_ended = True
            elif _detect_interview_close(reply_text):
                set_interview_ended(session_id, True)
                interview_ended = True

        response_phase = interview_state.get("current_phase") if interview_state else None
        return jsonify({
            "reply": reply_text,
            "replyAudio": reply_audio,
            "phase": response_phase,
            "interviewEnded": interview_ended,
        })

    @app.route("/api/live/session", methods=["POST"])
    def live_session():
        if not config.GEMINI_API_KEY:
            return jsonify({"error": "Gemini API key is not configured on the server."}), 500

        payload = request.get_json(silent=True) or {}
        model = payload.get("model") or config.GEMINI_MODEL or "models/gemini-1.5-pro-latest"
        language_code = payload.get("languageCode") or "en-US"
        voice_name = payload.get("voiceName") or "Poppy"

        session_payload: Dict[str, Any] = {
            "model": model,
            "languageCode": language_code,
            "voiceConfig": {
                "speechConfig": {
                    "voiceName": voice_name,
                }
            },
        }

        try:
            response = requests.post(
                "https://generativelanguage.googleapis.com/v1beta/sessions",
                headers={
                    "Content-Type": "application/json",
                    "x-goog-api-key": config.GEMINI_API_KEY,
                },
                json=session_payload,
                timeout=30,
            )
        except RequestException as exc:
            return jsonify({"error": f"Failed to reach Gemini Live API: {exc}"}), 502

        if response.status_code != 200:
            try:
                error_payload = response.json()
                message = error_payload.get("error", error_payload)
            except ValueError:
                message = response.text
            return jsonify({"error": f"Gemini Live API error: {message}"}), response.status_code

        return jsonify(response.json())

    @app.route("/api/live/connect", methods=["POST"])
    def live_connect():
        if not config.GEMINI_API_KEY:
            return jsonify({"error": "Gemini API key is not configured on the server."}), 500

        payload = request.get_json(silent=True) or {}
        session_name = (payload.get("session") or "").strip()
        client_sdp = (payload.get("sdp") or "").strip()

        if not session_name or not client_sdp:
            return jsonify({"error": "Both session and sdp are required."}), 400

        endpoint = f"https://generativelanguage.googleapis.com/v1beta/{session_name}:connect"

        try:
            response = requests.post(
                endpoint,
                headers={
                    "Content-Type": "application/sdp",
                    "x-goog-api-key": config.GEMINI_API_KEY,
                },
                data=client_sdp,
                timeout=30,
            )
        except RequestException as exc:
            return jsonify({"error": f"Failed to connect to Gemini Live session: {exc}"}), 502

        if response.status_code != 200:
            try:
                error_payload = response.json()
                message = error_payload.get("error", error_payload)
            except ValueError:
                message = response.text
            return jsonify({"error": f"Gemini Live connect error: {message}"}), response.status_code

        return Response(response.text, mimetype="application/sdp")

    @app.route("/api/intervene", methods=["POST"])
    def intervene():
        payload = request.get_json(silent=True) or {}
        partial_text = (payload.get("partial") or "").strip()
        if not partial_text:
            return jsonify({"error": "Partial transcript required."}), 400

        session_id = session.get("session_id")
        if not session_id:
            return jsonify({"error": "Upload a resume before starting the interview."}), 400

        resume_text = get_resume(session_id)
        if not resume_text:
            return jsonify({"error": "Upload a resume before starting the interview."}), 400

        conversation: List[Dict[str, Any]] = get_conversation(session_id)
        intervention_prompt = (
            "Coach, interrupt the candidate constructively. You just heard this partial thought:\n"
            f"{partial_text}\n"
            "Politely interject if they are rambling or off track. Be brief (<=2 sentences) and specific."
        )

        try:
            client = GeminiClient()
            reply_text = client.generate_interview_reply(
                conversation=conversation,
                resume_text=resume_text,
                user_message=intervention_prompt,
                temperature=0.6,
            )
        except (ValueError, GeminiError) as exc:
            return jsonify({"error": str(exc)}), 500

        try:
            print(f"Generating intervention audio for: {reply_text[:100]}...")
            reply_audio = kokoro.synthesize_base64(reply_text)
            print(f"Intervention audio generated successfully, length: {len(reply_audio)}")
        except KokoroUnavailable as exc:
            print(f"Kokoro intervention error: {exc}")
            return jsonify({"error": str(exc)}), 500
        except Exception as exc:
            print(f"Unexpected error in intervention TTS: {exc}")
            return jsonify({"error": f"TTS error: {exc}"}), 500

        append_turn(session_id, f"[intervention] {partial_text}", reply_text)

        return jsonify({"reply": reply_text, "replyAudio": reply_audio})

    @app.route("/api/reset", methods=["POST"])
    def reset():
        session_id = session.get("session_id")
        if session_id:
            clear_conversation(session_id)
        return jsonify({"message": "Conversation reset."})

    # ============================================================================
    # STRUCTURED INTERVIEW ENDPOINTS
    # ============================================================================

    @app.route("/api/start_interview", methods=["POST"])
    def start_structured_interview():
        """Initialize a structured interview session"""
        payload = request.get_json(silent=True) or {}
        language = payload.get("language", "python").lower()
        mode = payload.get("mode", "full")  # "full", "coding_only", or "ood"

        print(f"[START INTERVIEW] Received request - language: {language}, mode: {mode}")

        if language not in SUPPORTED_LANGUAGES:
            return jsonify({"error": f"Unsupported language. Choose from: {', '.join(SUPPORTED_LANGUAGES)}"}), 400

        if mode not in ["full", "coding_only", "ood"]:
            return jsonify({"error": "Invalid mode. Choose 'full', 'coding_only', or 'ood'"}), 400

        session_id = _ensure_session_id()
        if not session_id:
            return jsonify({"error": "Failed to create session"}), 500

        resume_text = get_resume(session_id)
        company_context = get_company_context(session_id)
        # Resume is optional for coding_only and ood modes
        if not resume_text and mode in {"full"}:
            return jsonify({"error": "Upload a resume before starting this interview."}), 400

        # Initialize structured interview state
        interview_state = start_interview(session_id, language, mode)

        # For OOD mode, force the stock matching engine scenario
        if mode == "ood":
            question = get_stock_match_engine_question()
            set_ood_question(session_id, question)
            context_summary = company_context.get("company") or company_context.get("role")
            if context_summary:
                print(
                    f"[OOD INTERVIEW] Stock matching engine question selected for context: {context_summary}"
                )
            else:
                print("[OOD INTERVIEW] Stock matching engine question selected")
        else:
            coding_question = get_random_coding_question()
            print(f"[START INTERVIEW] Selected question: {coding_question.get('title', 'UNKNOWN')}")
            if coding_question:
                question_title = coding_question.get("title", "")
                print(f"[START INTERVIEW] Generating/fetching test cases for: {question_title}")
                test_spec = ensure_tests_for_question(question_title)
                if not test_spec:
                    print(f"[START INTERVIEW ERROR] Failed to generate test cases for: {question_title}")
                    return jsonify({"error": "Failed to generate test cases for the selected question."}), 500
                print(f"[START INTERVIEW] Test cases ready - {len(test_spec.get('tests', []))} tests")
                coding_question["signatures"] = format_signatures_for_prompt(test_spec)
                set_coding_question(session_id, coding_question)

        print(f"[START INTERVIEW] Created state - phase: {interview_state['current_phase']}, mode: {interview_state['mode']}")

        mode_name = {"full": "Full", "coding_only": "Coding-only", "ood": "OOD"}
        response_data = {
            "message": f"{mode_name[mode]} interview started",
            "phase": interview_state["current_phase"],
            "language": interview_state["language"],
            "mode": interview_state["mode"],
        }

        print(f"[START INTERVIEW] Returning: {response_data}")
        return jsonify(response_data)

    @app.route("/api/interview_status", methods=["GET"])
    def interview_status():
        """Get current interview state and timing"""
        session_id = session.get("session_id")
        if not session_id:
            return jsonify({"interviewActive": False})

        interview_state = get_interview_state(session_id)
        if not interview_state:
            return jsonify({"interviewActive": False})

        return jsonify({
            "interviewActive": True,
            "phase": interview_state["current_phase"],
            "language": interview_state["language"],
            "timeInPhase": calculate_time_in_phase(session_id),
            "totalTime": calculate_total_time(session_id),
            "problemPresented": interview_state.get("problem_presented", False),
            "interviewEnded": bool(interview_state.get("ended", False)),
        })

    @app.route("/api/interview_report", methods=["POST"])
    def interview_report():
        session_id = session.get("session_id")
        if not session_id:
            return jsonify({"error": "No active session"}), 400

        interview_state = get_interview_state(session_id)
        if not interview_state:
            return jsonify({"error": "No active interview"}), 400

        cached = get_interview_report(session_id)
        if cached:
            return jsonify({"report": cached, "cached": True})

        conversation = get_conversation(session_id)
        current_code = get_current_code(session_id)
        mode = interview_state.get("mode", "full")
        language = interview_state.get("language", "python")
        code_snapshots = interview_state.get("code_snapshots", [])
        problem_presented = interview_state.get("problem_presented", False)

        try:
            client = GeminiClient()
            report = client.generate_interview_report(
                mode=mode,
                language=language,
                conversation=conversation,
                current_code=current_code,
                code_snapshots=code_snapshots,
                problem_presented=problem_presented,
            )
        except (ValueError, GeminiError) as exc:
            return jsonify({"error": str(exc)}), 500

        set_interview_report(session_id, report)
        return jsonify({"report": report, "cached": False})

    @app.route("/api/end_interview", methods=["POST"])
    def end_interview():
        session_id = session.get("session_id")
        if not session_id:
            return jsonify({"error": "No active session"}), 400

        interview_state = get_interview_state(session_id)
        if not interview_state:
            return jsonify({"error": "No active interview"}), 400

        set_interview_ended(session_id, True)
        report = get_interview_report(session_id)
        if report:
            return jsonify({"report": report, "cached": True})

        conversation = get_conversation(session_id)
        current_code = get_current_code(session_id)
        mode = interview_state.get("mode", "full")
        language = interview_state.get("language", "python")
        code_snapshots = interview_state.get("code_snapshots", [])
        problem_presented = interview_state.get("problem_presented", False)

        try:
            client = GeminiClient()
            report = client.generate_interview_report(
                mode=mode,
                language=language,
                conversation=conversation,
                current_code=current_code,
                code_snapshots=code_snapshots,
                problem_presented=problem_presented,
            )
        except (ValueError, GeminiError) as exc:
            return jsonify({"error": str(exc)}), 500

        set_interview_report(session_id, report)
        return jsonify({"report": report, "cached": False})

    @app.route("/api/update_code", methods=["POST"])
    def update_code_endpoint():
        """Update current code snapshot"""
        payload = request.get_json(silent=True) or {}
        code = payload.get("code", "")

        session_id = session.get("session_id")
        if not session_id:
            return jsonify({"error": "No active session"}), 400

        interview_state = get_interview_state(session_id)
        if not interview_state:
            return jsonify({"error": "No active interview"}), 400

        update_code(session_id, code)
        return jsonify({"message": "Code updated"})

    @app.route("/api/transcribe", methods=["POST"])
    def transcribe_audio():
        """
        Transcribe audio using Whisper for accurate speech-to-text.
        Accepts audio blob from frontend (WebM, WAV, etc.)
        """
        import time
        import logging

        start_time = time.time()

        try:
            # Get audio data from request
            if 'audio' not in request.files:
                return jsonify({"error": "No audio file provided"}), 400

            audio_file = request.files['audio']
            audio_format = request.form.get('format', 'webm')

            # Read audio bytes
            audio_bytes = audio_file.read()

            if not audio_bytes:
                return jsonify({"error": "Empty audio file"}), 400

            audio_size_kb = len(audio_bytes) / 1024
            logging.info(f"[WHISPER] Received {audio_size_kb:.1f} KB audio file ({audio_format})")

            # Get Whisper transcriber instance
            transcriber = get_whisper_transcriber(model_size="base")

            # Transcribe audio
            transcribe_start = time.time()
            result = transcriber.transcribe_bytes(
                audio_bytes,
                audio_format=audio_format,
                language="en"
            )
            transcribe_time = time.time() - transcribe_start

            total_time = time.time() - start_time

            logging.info(
                f"[WHISPER] Transcription complete in {transcribe_time:.3f}s "
                f"(total: {total_time:.3f}s) - Device: {transcriber.device} - "
                f"Text: \"{result['text'][:50]}...\""
            )

            return jsonify({
                "text": result["text"],
                "raw_text": result["raw_text"],
                "language": result["language"],
                "duration": result["duration"],
                "transcription_time": transcribe_time,
                "device": transcriber.device,
                "success": True
            })

        except Exception as e:
            logging.error(f"[WHISPER] Transcription error: {e}", exc_info=True)
            return jsonify({
                "error": str(e),
                "success": False
            }), 500

    @app.route("/api/transition_phase", methods=["POST"])
    def transition_phase():
        """Manually transition to next phase"""
        payload = request.get_json(silent=True) or {}
        new_phase = payload.get("phase")

        valid_phases = [
            PHASE_INTRO,
            PHASE_RESUME,
            PHASE_CODING,
            PHASE_QUESTIONS,
            PHASE_OOD_DESIGN,
            PHASE_OOD_IMPLEMENTATION,
        ]
        if new_phase not in valid_phases:
            return jsonify({"error": "Invalid phase"}), 400

        session_id = session.get("session_id")
        if not session_id:
            return jsonify({"error": "No active session"}), 400

        interview_state = get_interview_state(session_id)
        if not interview_state:
            return jsonify({"error": "No active interview"}), 400

        if interview_state.get("mode") == "full":
            total_time = calculate_total_time(session_id)
            if new_phase == PHASE_CODING and total_time < 5.0:
                return jsonify({"error": "Cannot enter coding before 5 minutes total time."}), 400
            if new_phase == PHASE_QUESTIONS and total_time < 35.0:
                return jsonify({"error": "Cannot enter questions before 35 minutes total time."}), 400

        update_interview_phase(session_id, new_phase)
        return jsonify({"message": f"Transitioned to {new_phase} phase"})

    def _strip_code_markers(text: str) -> str:
        """Remove code markers from text before TTS"""
        # Find and remove [CODE_START]...[CODE_END] blocks
        import re
        cleaned = re.sub(r'\[CODE_START\].*?\[CODE_END\]', '', text, flags=re.DOTALL)
        cleaned = re.sub(r'\[CODE_START\].*$', '', cleaned, flags=re.DOTALL)
        cleaned = re.sub(r'\[PROBLEM_START\].*?\[PROBLEM_END\]', '', cleaned, flags=re.DOTALL)
        cleaned = cleaned.strip()
        return _strip_markdown_symbols(cleaned)

    def _detect_interview_close(reply_text: str) -> bool:
        if not reply_text:
            return False
        reply_lower = reply_text.lower()
        closing_phrases = [
            "thanks for your time",
            "we'll be in touch",
            "best of luck",
            "that concludes",
            "this concludes",
            "we are out of time",
            "we're out of time",
            "wrap up here",
            "goodbye",
            "have a great day",
        ]
        return any(phrase in reply_lower for phrase in closing_phrases)

    def _strip_markdown_symbols(text: str) -> str:
        """Remove common markdown symbols so TTS doesn't speak them."""
        import re
        if not text:
            return ""
        cleaned = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
        cleaned = re.sub(r"`([^`]+)`", r"\1", cleaned)
        cleaned = re.sub(r"\*\*(.*?)\*\*", r"\1", cleaned)
        cleaned = re.sub(r"__(.*?)__", r"\1", cleaned)
        cleaned = re.sub(r"\*(.*?)\*", r"\1", cleaned)
        cleaned = re.sub(r"_(.*?)_", r"\1", cleaned)
        cleaned = re.sub(r"^#{1,6}\s+", "", cleaned, flags=re.MULTILINE)
        return cleaned.strip()

    def _apply_phase_decision(
        session_id: str,
        interview_state: Dict[str, Any],
        decision: Optional[Dict[str, Any]],
        user_message: str,
        model_reply: str,
    ) -> None:
        """Apply phase decision returned by the coordinator model with safe fallbacks."""
        if not interview_state:
            return

        current_phase = interview_state.get("current_phase")
        if not current_phase:
            return
        interview_mode = interview_state.get("mode", "full")
        total_time = calculate_total_time(session_id)

        if interview_mode == "full":
            if total_time >= 35.0 and current_phase != PHASE_QUESTIONS:
                print(f"[PHASE HARD STOP] Forcing questions at {total_time:.1f} min total")
                update_interview_phase(session_id, PHASE_QUESTIONS)
                interview_state["current_phase"] = PHASE_QUESTIONS
                return
            if total_time >= 10.0 and current_phase in {PHASE_INTRO, PHASE_RESUME}:
                print(f"[PHASE HARD STOP] Forcing coding at {total_time:.1f} min total")
                update_interview_phase(session_id, PHASE_CODING)
                interview_state["current_phase"] = PHASE_CODING
                return

        should_transition = False
        if decision is not None:
            raw_flag = decision.get("should_transition", False)
            if isinstance(raw_flag, str):
                should_transition = raw_flag.lower() == "true"
            else:
                should_transition = bool(raw_flag)

        if should_transition and decision:
            next_phase = decision.get("next_phase")
            reason = decision.get("reason", "")
            if next_phase and next_phase != current_phase:
                if interview_mode == "full":
                    if total_time < 5.0 and next_phase in {PHASE_CODING, PHASE_QUESTIONS}:
                        print(f"[PHASE GUARD] Blocked early transition to {next_phase} at {total_time:.1f} min")
                        return
                    if total_time < 35.0 and next_phase == PHASE_QUESTIONS:
                        print(f"[PHASE GUARD] Blocked questions before 35 min at {total_time:.1f} min")
                        return
                print(f"[PHASE DECISION] {current_phase} → {next_phase} (reason: {reason})")
                update_interview_phase(session_id, next_phase)
                interview_state["current_phase"] = next_phase
            return

        keyword_transition = _detect_content_transition(current_phase, user_message, model_reply)
        if keyword_transition:
            next_phase, reason = keyword_transition
            if interview_mode == "full":
                if total_time < 5.0 and next_phase in {PHASE_CODING, PHASE_QUESTIONS}:
                    print(f"[PHASE GUARD] Blocked early transition to {next_phase} at {total_time:.1f} min")
                    return
                if total_time < 35.0 and next_phase == PHASE_QUESTIONS:
                    print(f"[PHASE GUARD] Blocked questions before 35 min at {total_time:.1f} min")
                    return
            print(f"[PHASE CONTENT] {current_phase} → {next_phase} (reason: {reason})")
            update_interview_phase(session_id, next_phase)
            interview_state["current_phase"] = next_phase
            return

        # Fallback: guard against getting stuck in a phase for far too long
        if interview_mode == "full":
            fallback_thresholds = {
                PHASE_INTRO: 5.0,
                PHASE_RESUME: 5.0,
                PHASE_CODING: 25.0,
            }
        else:
            fallback_thresholds = {
                PHASE_INTRO: 7.0,
                PHASE_RESUME: 18.0,
                PHASE_CODING: 45.0,
                PHASE_OOD_DESIGN: 20.0,  # 20 minutes for design phase
                PHASE_OOD_IMPLEMENTATION: 20.0,  # 20 minutes for implementation phase
            }
        time_in_phase = calculate_time_in_phase(session_id)
        if current_phase in fallback_thresholds and time_in_phase > fallback_thresholds[current_phase]:
            next_phase_map = {
                PHASE_INTRO: PHASE_RESUME,
                PHASE_RESUME: PHASE_CODING,
                PHASE_CODING: PHASE_QUESTIONS,
                PHASE_OOD_DESIGN: PHASE_OOD_IMPLEMENTATION,  # Auto-transition to implementation
            }
            next_phase = next_phase_map.get(current_phase)
            if next_phase:
                if interview_mode == "full":
                    total_time = calculate_total_time(session_id)
                    if total_time < 5.0 and next_phase == PHASE_CODING:
                        return
                    if total_time < 35.0 and next_phase == PHASE_QUESTIONS:
                        return
                print(f"[PHASE FALLBACK] {current_phase} → {next_phase} (time_in_phase={time_in_phase:.1f} min)")
                update_interview_phase(session_id, next_phase)
                interview_state["current_phase"] = next_phase

                # Mark design phase complete when transitioning to implementation
                if current_phase == PHASE_OOD_DESIGN and next_phase == PHASE_OOD_IMPLEMENTATION:
                    set_design_phase_complete(session_id, True)

    def _handle_problem_presentation(
        session_id: str,
        interview_state: Dict[str, Any],
        user_message: str,
        reply_text: str,
        decision: Optional[Dict[str, Any]],
    ) -> None:
        """Track when the coding problem has been formally presented."""
        if not interview_state or interview_state.get("current_phase") != PHASE_CODING:
            return

        if interview_state.get("problem_presented"):
            return

        mark_problem = False
        if decision is not None:
            raw_mark = decision.get("mark_problem_presented", False)
            if isinstance(raw_mark, str):
                mark_problem = raw_mark.lower() == "true"
            else:
                mark_problem = bool(raw_mark)

        if mark_problem:
            print("[PROBLEM PRESENTED] Marked via phase decision model")
            set_problem_presented(session_id, True)
            interview_state["problem_presented"] = True
            return

        reply_lower = reply_text.lower()
        user_lower = (user_message or "").lower()
        problem_keywords = [
            "given an array",
            "given a",
            "design a",
            "implement a",
            "find the",
            "return",
            "write a function",
            "create a function",
            "you need to",
            "your task is",
            "the problem is",
        ]
        if any(phrase in reply_lower for phrase in problem_keywords):
            print("[PROBLEM PRESENTED] Detected problem presentation keywords (fallback)")
            set_problem_presented(session_id, True)
            interview_state["problem_presented"] = True
            return

        # Candidate explicitly acknowledging problem can also mark it
        candidate_ack = [
            "okay what's the problem",
            "ready for the problem",
            "sounds good let's code",
            "yes let's do the coding problem",
            "ready for coding",
        ]
        if any(phrase in user_lower for phrase in candidate_ack):
            print("[PROBLEM PRESENTED] Candidate acknowledged problem (fallback)")
            set_problem_presented(session_id, True)
            interview_state["problem_presented"] = True

    def _detect_content_transition(current_phase: str, user_message: str, model_reply: str) -> Optional[tuple[str, str]]:
        """Use conversational keywords to trigger coherent transitions."""
        if not current_phase:
            return None

        user_lower = (user_message or "").lower()
        reply_lower = (model_reply or "").lower()

        if current_phase == PHASE_INTRO:
            intro_to_resume = [
                "tell me about yourself",
                "walk me through your background",
                "let's start with your background",
                "let's talk about your resume",
                "start with your experience",
            ]
            if any(phrase in reply_lower for phrase in intro_to_resume):
                return (PHASE_RESUME, "Interviewer invited resume discussion")

        if current_phase == PHASE_RESUME:
            resume_to_coding = [
                "let's move to coding",
                "let's move on to coding",
                "let's move on to the coding",
                "let's work on a coding problem",
                "time for a coding challenge",
                "let's switch to the coding problem",
                "let's start the coding challenge",
            ]
            candidate_ready = [
                "ready for the coding problem",
                "can we start coding",
                "i'm ready to start coding",
                "let's start coding",
                "let's jump into coding",
            ]
            if any(phrase in reply_lower for phrase in resume_to_coding):
                return (PHASE_CODING, "Interviewer prompted coding phase")
            if any(phrase in user_lower for phrase in candidate_ready):
                return (PHASE_CODING, "Candidate requested coding phase")

        if current_phase == PHASE_CODING:
            coding_to_questions = [
                "what questions do you have for me",
                "do you have any questions for me",
                "any questions about the team",
                "your questions for me",
                "ask me anything about the role",
            ]
            if any(phrase in reply_lower for phrase in coding_to_questions):
                return (PHASE_QUESTIONS, "Interviewer invited candidate questions")

        return None

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=1111, debug=True)
