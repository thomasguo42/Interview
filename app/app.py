from __future__ import annotations

from datetime import datetime
from uuid import uuid4
from pathlib import Path
from typing import Any, Dict, List
import hashlib
import logging

from flask import (
    Flask,
    jsonify,
    render_template,
    request,
    session,
    redirect,
    url_for,
    flash,
    send_file,
    abort,
)
from sqlalchemy import select
from sqlalchemy.orm import Session
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

from .auth import login_required, get_current_user
from .config import config
from .db import init_db, SessionLocal
from .gemini_client import GeminiError
from .openai_client import OpenAIError
from .llm_client import get_llm_client
from .models import User, Resume, CompanyProfile, Interview
from .ood_questions import get_stock_match_engine_question
from .question_bank import get_random_coding_question
from .question_tests import (
    ensure_tests_for_question,
    get_cached_tests,
    format_signatures_for_prompt,
    build_starter_code,
)
from .resume_processor import allowed_file, extract_text
from .test_runner import run_code_tests
from .tts import KokoroUnavailable, kokoro
from .whisper_service import get_whisper_transcriber
from .state import (
    PHASE_INTRO_RESUME,
    PHASE_CODING,
    PHASE_QUESTIONS,
    PHASE_OOD_DESIGN,
    PHASE_OOD_IMPLEMENTATION,
    SUPPORTED_LANGUAGES,
)


PHASE_COMPLETE_TOKEN = "[PHASE_COMPLETE]"

FULL_PHASE_RULES = {
    PHASE_INTRO_RESUME: {"min": 5.0, "max": 10.0, "next": PHASE_CODING},
    PHASE_CODING: {"min": 0.0, "max": 30.0, "next": PHASE_QUESTIONS},
    PHASE_QUESTIONS: {"min": 0.0, "max": 5.0, "next": None},
}


def create_app() -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config.from_object(config)

    resume_storage = Path(__file__).resolve().parent.parent / "storage" / "resumes"
    resume_storage.mkdir(exist_ok=True, parents=True)
    app.config["UPLOAD_FOLDER"] = resume_storage

    def _setup_debug_logging() -> logging.Logger:
        log_dir = Path(__file__).resolve().parent.parent / "storage" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "debug.log"

        logger = logging.getLogger("interview")
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

        if not any(isinstance(h, logging.FileHandler) and h.baseFilename == str(log_path) for h in logger.handlers):
            file_handler = logging.FileHandler(log_path, encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(formatter)
            logger.addHandler(stream_handler)

        return logger

    logger = _setup_debug_logging()

    init_db()

    def _db() -> Session:
        return SessionLocal()

    def _ensure_logged_in(db: Session) -> User:
        user = get_current_user(db)
        if not user:
            abort(401)
        return user

    def _load_interview(db: Session, interview_id: int, user_id: int) -> Interview:
        interview = db.get(Interview, interview_id)
        if not interview or interview.user_id != user_id or interview.status == "deleted":
            abort(404)
        return interview

    def _now() -> datetime:
        return datetime.utcnow()

    def _set_updated(interview: Interview) -> None:
        interview.updated_at = _now()

    def _extract_candidate_name(resume_text: str) -> str:
        if not resume_text:
            return ""
        lines = [line.strip() for line in resume_text.splitlines() if line.strip()]
        for line in lines[:5]:
            lowered = line.lower()
            if "@" in line or "http" in lowered or "www" in lowered:
                continue
            if any(char.isdigit() for char in line):
                continue
            words = [w for w in line.split() if w]
            if 2 <= len(words) <= 4 and all(word.isalpha() for word in words):
                return line
        return ""

    def _append_turn(interview: Interview, user_message: str, model_message: str) -> None:
        conversation = list(interview.conversation or [])
        conversation.append({"role": "user", "parts": [{"text": user_message}]})
        conversation.append({"role": "model", "parts": [{"text": model_message}]})
        if len(conversation) > 40:
            conversation = conversation[-40:]
        interview.conversation = conversation

    def _update_code(interview: Interview, code: str) -> None:
        interview.current_code = code
        interview.last_code_change_at = _now()
        snapshots = list(interview.code_snapshots or [])
        snapshots.append({"code": code, "timestamp": _now().isoformat()})
        if len(snapshots) > 10:
            snapshots = snapshots[-10:]
        interview.code_snapshots = snapshots

    def _calculate_time_in_phase(interview: Interview) -> float:
        if not interview.phase_started_at:
            return 0.0
        return (_now() - interview.phase_started_at).total_seconds() / 60.0

    def _calculate_total_time(interview: Interview) -> float:
        if not interview.started_at:
            return 0.0
        return (_now() - interview.started_at).total_seconds() / 60.0

    def _calculate_silence_duration(interview: Interview) -> float:
        if not interview.last_speech_at:
            return 0.0
        return (_now() - interview.last_speech_at).total_seconds()

    def _calculate_code_idle_duration(interview: Interview) -> float:
        if not interview.last_code_change_at:
            return 0.0
        return (_now() - interview.last_code_change_at).total_seconds()

    def _extract_phase_complete(text: str) -> tuple[bool, str]:
        if not text:
            return False, ""
        if PHASE_COMPLETE_TOKEN in text:
            cleaned = text.replace(PHASE_COMPLETE_TOKEN, "").strip()
            return True, cleaned
        return False, text

    def _strip_markdown_symbols(text: str) -> str:
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

    def _strip_code_markers(text: str) -> str:
        import re
        cleaned = re.sub(r"\[CODE_START\].*?\[CODE_END\]", "", text, flags=re.DOTALL)
        cleaned = re.sub(r"\[CODE_START\].*$", "", cleaned, flags=re.DOTALL)
        cleaned = re.sub(r"\[PROBLEM_START\].*?\[PROBLEM_END\]", "", cleaned, flags=re.DOTALL)
        cleaned = cleaned.replace(PHASE_COMPLETE_TOKEN, "")
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

    def _should_run_tests_fallback(message: str) -> bool:
        msg = (message or "").lower()
        keywords = [
            "i am done",
            "i'm done",
            "finished",
            "ready to test",
            "run the tests",
            "run tests",
            "check my code",
            "check if the code is correct",
            "can you check if the code is correct",
            "can you check my code",
            "can you verify",
            "verify my solution",
            "submit",
        ]
        return any(kw in msg for kw in keywords)

    def _code_looks_complete(code: str, language: str) -> bool:
        if not code or "TODO" in code or "pass" in code:
            return False
        language = (language or "").lower()
        if language == "python":
            return "def " in code and "return" in code
        if language == "java":
            return "class " in code and "return" in code
        if language == "cpp":
            return "return" in code
        return False

    def _build_coding_summary(interview: Interview) -> Dict[str, Any]:
        summary: Dict[str, Any] = {}
        if interview.coding_question:
            summary["question_title"] = interview.coding_question.get("title", "")
        if interview.code_evaluation:
            summary["status"] = interview.code_evaluation.get("status", "")
            summary["summary"] = interview.code_evaluation.get("summary", "")
        return summary

    def _handle_problem_presentation(interview: Interview, reply_text: str) -> None:
        if not reply_text:
            return
        if interview.current_phase != PHASE_CODING:
            return
        if interview.problem_presented:
            return
        if "[PROBLEM_START]" in reply_text and "[PROBLEM_END]" in reply_text:
            interview.problem_presented = True

    def _calculate_phase_signal(interview: Interview, time_in_phase: float) -> str:
        """
        Calculate phase signal based on time constraints.
        Returns: "keep_going", "end_if_you_want", or "must_end_now"
        """
        if interview.mode != "full":
            return "end_if_you_want"

        current_phase = interview.current_phase
        if not current_phase:
            return "end_if_you_want"

        rules = FULL_PHASE_RULES.get(current_phase)
        if not rules:
            return "end_if_you_want"

        min_minutes = float(rules.get("min", 0.0))
        max_minutes = float(rules.get("max", 0.0))

        # Must end if we've hit the maximum time
        if max_minutes > 0.0 and time_in_phase >= max_minutes:
            return "must_end_now"

        # Must continue if we haven't hit the minimum time
        if min_minutes > 0.0 and time_in_phase < min_minutes:
            return "keep_going"

        # In the allowed window - model can decide
        return "end_if_you_want"

    def _apply_phase_completion(interview: Interview, phase_complete: bool) -> None:
        if interview.mode != "full":
            return
        current_phase = interview.current_phase
        if not current_phase:
            return
        rules = FULL_PHASE_RULES.get(current_phase)
        if not rules:
            return
        time_in_phase = _calculate_time_in_phase(interview)
        min_minutes = float(rules.get("min", 0.0))
        max_minutes = float(rules.get("max", 0.0))
        next_phase = rules.get("next")

        if phase_complete and time_in_phase < min_minutes:
            logger.warning(
                "[PHASE BLOCKED] Model tried to end %s at %.1fm but minimum is %.1fm (%.1fm remaining)",
                current_phase,
                time_in_phase,
                min_minutes,
                min_minutes - time_in_phase,
            )
            phase_complete = False

        force_end = max_minutes > 0.0 and time_in_phase >= max_minutes
        if not (phase_complete or force_end):
            return

        if current_phase == PHASE_CODING and phase_complete and not force_end:
            evaluation = interview.code_evaluation or {}
            if str(evaluation.get("status", "")).strip() != "pass":
                return

        if next_phase:
            if current_phase == PHASE_CODING:
                summary = _build_coding_summary(interview)
                if summary:
                    interview.coding_summary = summary
            logger.info(
                "[PHASE SHIFT] full mode %s -> %s (phase_complete=%s force_end=%s)",
                current_phase,
                next_phase,
                phase_complete,
                force_end,
            )
            interview.current_phase = next_phase
            interview.phase_started_at = _now()
            interview.phase_turn_start_index = len(interview.conversation or [])
            return

        logger.info("[PHASE SHIFT] full mode %s -> ended", current_phase)
        interview.status = "ended"
        interview.ended_at = _now()

    def _transition_ood_phase(interview: Interview) -> None:
        if interview.mode != "ood":
            return
        if interview.current_phase != PHASE_OOD_DESIGN:
            return
        if _calculate_time_in_phase(interview) >= 20.0:
            logger.info("[PHASE SHIFT] ood_design -> ood_implementation")
            interview.current_phase = PHASE_OOD_IMPLEMENTATION
            interview.phase_started_at = _now()
            interview.phase_turn_start_index = len(interview.conversation or [])

    def _ensure_interview_started(interview: Interview) -> None:
        now = _now()
        if not interview.started_at:
            interview.started_at = now
        if not interview.phase_started_at:
            interview.phase_started_at = now
        if not interview.last_speech_at:
            interview.last_speech_at = now
        if not interview.last_code_change_at:
            interview.last_code_change_at = now

    @app.route("/")
    def home():
        if session.get("user_id"):
            return redirect(url_for("dashboard"))
        return redirect(url_for("login"))

    @app.route("/register", methods=["GET", "POST"])
    def register():
        if request.method == "GET":
            return render_template("register.html")

        email = (request.form.get("email") or "").strip().lower()
        password = request.form.get("password") or ""
        confirm = request.form.get("confirm") or ""

        if not email or not password:
            flash("Email and password are required.", "error")
            return render_template("register.html")
        if password != confirm:
            flash("Passwords do not match.", "error")
            return render_template("register.html")
        if len(password) < 8:
            flash("Password must be at least 8 characters.", "error")
            return render_template("register.html")

        with _db() as db:
            existing = db.scalar(select(User).where(User.email == email))
            if existing:
                flash("An account with that email already exists.", "error")
                return render_template("register.html")
            user = User(email=email, password_hash=generate_password_hash(password))
            db.add(user)
            db.commit()
            session["user_id"] = user.id
            return redirect(url_for("dashboard"))

    @app.route("/login", methods=["GET", "POST"])
    def login():
        if request.method == "GET":
            return render_template("login.html")

        email = (request.form.get("email") or "").strip().lower()
        password = request.form.get("password") or ""

        with _db() as db:
            user = db.scalar(select(User).where(User.email == email))
            if not user or not check_password_hash(user.password_hash, password):
                flash("Invalid email or password.", "error")
                return render_template("login.html")
            session["user_id"] = user.id
            return redirect(url_for("dashboard"))

    @app.route("/logout", methods=["POST"])
    def logout():
        session.clear()
        return redirect(url_for("login"))

    @app.route("/dashboard")
    @login_required
    def dashboard():
        with _db() as db:
            user = _ensure_logged_in(db)
            resumes = db.scalars(select(Resume).where(Resume.user_id == user.id)).all()
            companies = db.scalars(select(CompanyProfile).where(CompanyProfile.user_id == user.id)).all()
            interviews = db.scalars(
                select(Interview).where(Interview.user_id == user.id, Interview.status != "deleted").order_by(Interview.updated_at.desc())
            ).all()
            return render_template(
                "dashboard.html",
                user=user,
                resumes=resumes,
                companies=companies,
                interviews=interviews,
                supported_models=list(config.SUPPORTED_MODELS.keys()),
            )

    @app.route("/resumes/upload", methods=["POST"])
    @login_required
    def upload_resume():
        if "file" not in request.files:
            flash("No file part in the request.", "error")
            return redirect(url_for("dashboard"))

        file = request.files["file"]
        if file.filename == "":
            flash("No file selected.", "error")
            return redirect(url_for("dashboard"))

        if not allowed_file(file.filename):
            flash("Unsupported file type. Upload a PDF or TXT resume.", "error")
            return redirect(url_for("dashboard"))

        safe_name = secure_filename(file.filename)
        file_id = uuid4().hex

        with _db() as db:
            user = _ensure_logged_in(db)
            user_dir = app.config["UPLOAD_FOLDER"] / str(user.id)
            user_dir.mkdir(parents=True, exist_ok=True)
            stored_name = f"{file_id}_{safe_name}"
            stored_path = user_dir / stored_name
            file.save(stored_path)

            try:
                file.stream.seek(0)
                resume_text = extract_text(file)
            except Exception as exc:
                stored_path.unlink(missing_ok=True)
                flash(f"Failed to process resume: {exc}", "error")
                return redirect(url_for("dashboard"))

            resume = Resume(
                user_id=user.id,
                filename=safe_name,
                content_type=file.mimetype or "application/octet-stream",
                size_bytes=stored_path.stat().st_size,
                storage_path=str(stored_path),
                text=resume_text.strip(),
            )
            db.add(resume)
            db.commit()

        flash("Resume uploaded successfully.", "success")
        return redirect(url_for("dashboard"))

    @app.route("/resumes/<int:resume_id>/download")
    @login_required
    def download_resume(resume_id: int):
        with _db() as db:
            user = _ensure_logged_in(db)
            resume = db.get(Resume, resume_id)
            if not resume or resume.user_id != user.id:
                abort(404)
            return send_file(resume.storage_path, as_attachment=True, download_name=resume.filename)

    @app.route("/companies/create", methods=["POST"])
    @login_required
    def create_company_profile():
        company = (request.form.get("company") or "").strip()
        role = (request.form.get("role") or "").strip()
        details = (request.form.get("details") or "").strip()

        if not company and not role:
            flash("Company or role is required.", "error")
            return redirect(url_for("dashboard"))

        with _db() as db:
            user = _ensure_logged_in(db)
            profile = CompanyProfile(
                user_id=user.id,
                company=company,
                role=role,
                details=details,
            )
            db.add(profile)
            db.commit()

        flash("Company profile saved.", "success")
        return redirect(url_for("dashboard"))

    @app.route("/interviews/create", methods=["POST"])
    @login_required
    def create_interview():
        mode = (request.form.get("mode") or "full").strip()
        language = (request.form.get("language") or "python").strip().lower()
        model_name = (request.form.get("model") or config.GEMINI_MODEL or "").strip()
        resume_id = request.form.get("resume_id")
        company_profile_id = request.form.get("company_profile_id")

        if language not in SUPPORTED_LANGUAGES:
            flash("Unsupported language.", "error")
            return redirect(url_for("dashboard"))
        if mode not in {"full", "coding_only", "ood"}:
            flash("Invalid interview mode.", "error")
            return redirect(url_for("dashboard"))
        if model_name not in config.SUPPORTED_MODELS:
            flash("Unsupported model.", "error")
            return redirect(url_for("dashboard"))

        with _db() as db:
            user = _ensure_logged_in(db)

            resume = None
            if resume_id:
                resume = db.get(Resume, int(resume_id))
                if not resume or resume.user_id != user.id:
                    flash("Selected resume not found.", "error")
                    return redirect(url_for("dashboard"))
            if not resume and mode == "full":
                flash("Resume is required for full interviews.", "error")
                return redirect(url_for("dashboard"))

            company_profile = None
            if company_profile_id:
                company_profile = db.get(CompanyProfile, int(company_profile_id))
                if not company_profile or company_profile.user_id != user.id:
                    flash("Selected company profile not found.", "error")
                    return redirect(url_for("dashboard"))

            candidate_name = _extract_candidate_name(resume.text if resume else "")

            interview = Interview(
                user_id=user.id,
                resume_id=resume.id if resume else None,
                company_profile_id=company_profile.id if company_profile else None,
                mode=mode,
                language=language,
                model=model_name,
                status="created",
                current_phase=None,
                candidate_name=candidate_name,
                company_context={
                    "company": company_profile.company if company_profile else "",
                    "role": company_profile.role if company_profile else "",
                    "details": company_profile.details if company_profile else "",
                },
            )
            db.add(interview)
            db.commit()
            return redirect(url_for("interview_page", interview_id=interview.id))

    @app.route("/interviews/delete", methods=["POST"])
    @login_required
    def delete_interview():
        interview_id = request.form.get("interview_id")
        if not interview_id:
            return redirect(url_for("dashboard"))
        with _db() as db:
            user = _ensure_logged_in(db)
            interview = _load_interview(db, int(interview_id), user.id)
            interview.status = "deleted"
            interview.ended_at = _now()
            _set_updated(interview)
            db.commit()
        return redirect(url_for("dashboard"))

    @app.route("/interviews/<int:interview_id>")
    @login_required
    def interview_page(interview_id: int):
        with _db() as db:
            user = _ensure_logged_in(db)
            interview = _load_interview(db, interview_id, user.id)
            resume = interview.resume
            company_profile = interview.company_profile
            return render_template(
                "interview.html",
                interview=interview,
                resume=resume,
                company_profile=company_profile,
            )

    @app.route("/api/interviews/<int:interview_id>/status", methods=["GET"])
    @login_required
    def interview_status(interview_id: int):
        with _db() as db:
            user = _ensure_logged_in(db)
            interview = _load_interview(db, interview_id, user.id)
            return jsonify({
                "interviewActive": interview.status == "active",
                "status": interview.status,
                "phase": interview.current_phase,
                "language": interview.language,
                "mode": interview.mode,
                "timeInPhase": _calculate_time_in_phase(interview),
                "totalTime": _calculate_total_time(interview),
                "problemPresented": bool(interview.problem_presented),
                "interviewEnded": interview.status == "ended",
            })

    @app.route("/api/interviews/<int:interview_id>/start", methods=["POST"])
    @login_required
    def start_interview(interview_id: int):
        with _db() as db:
            user = _ensure_logged_in(db)
            interview = _load_interview(db, interview_id, user.id)

            if interview.status in {"ended", "deleted"}:
                return jsonify({"error": "Interview already ended."}), 400

            was_started = interview.started_at is not None

            if interview.status != "active":
                interview.status = "active"

            if not interview.candidate_name and interview.resume and interview.resume.text:
                interview.candidate_name = _extract_candidate_name(interview.resume.text)

            if interview.mode == "ood":
                interview.current_phase = interview.current_phase or PHASE_OOD_DESIGN
            elif interview.mode == "coding_only":
                interview.current_phase = interview.current_phase or PHASE_CODING
            else:
                interview.current_phase = interview.current_phase or PHASE_INTRO_RESUME

            _ensure_interview_started(interview)
            if not was_started and interview.phase_started_at:
                interview.phase_turn_start_index = len(interview.conversation or [])

            if interview.mode == "ood" and not interview.ood_question:
                interview.ood_question = get_stock_match_engine_question()

            coding_question = None
            if interview.mode != "ood" and not interview.coding_question:
                coding_question = get_random_coding_question()
                if coding_question:
                    question_title = coding_question.get("title", "")
                    test_spec = ensure_tests_for_question(question_title)
                    if not test_spec:
                        return jsonify({"error": "Failed to generate test cases for the selected question."}), 500
                    coding_question["signatures"] = format_signatures_for_prompt(test_spec)
                    coding_question["starter_code"] = build_starter_code(test_spec, interview.language)
                    interview.coding_question = coding_question

            _set_updated(interview)
            db.commit()

            response_data = {
                "message": "Interview started",
                "phase": interview.current_phase,
                "language": interview.language,
                "mode": interview.mode,
                "model": interview.model,
            }
            if interview.mode != "ood":
                coding_question = interview.coding_question or {}
                response_data["codingQuestion"] = {
                    "title": coding_question.get("title", ""),
                    "signature": (coding_question.get("signatures") or {}).get(interview.language, ""),
                    "starterCode": coding_question.get("starter_code", ""),
                }
            return jsonify(response_data)

    @app.route("/api/interviews/<int:interview_id>/pause", methods=["POST"])
    @login_required
    def pause_interview(interview_id: int):
        with _db() as db:
            user = _ensure_logged_in(db)
            interview = _load_interview(db, interview_id, user.id)
            if interview.status == "active":
                interview.status = "paused"
                _set_updated(interview)
                db.commit()
            return jsonify({"message": "Interview paused."})

    @app.route("/api/interviews/<int:interview_id>/resume", methods=["POST"])
    @login_required
    def resume_interview(interview_id: int):
        with _db() as db:
            user = _ensure_logged_in(db)
            interview = _load_interview(db, interview_id, user.id)
            if interview.status in {"created", "paused"}:
                interview.status = "active"
                _ensure_interview_started(interview)
                _set_updated(interview)
                db.commit()
            return jsonify({"message": "Interview resumed."})

    @app.route("/api/interviews/<int:interview_id>/delete", methods=["POST"])
    @login_required
    def delete_interview_api(interview_id: int):
        with _db() as db:
            user = _ensure_logged_in(db)
            interview = _load_interview(db, interview_id, user.id)
            interview.status = "deleted"
            interview.ended_at = _now()
            _set_updated(interview)
            db.commit()
            return jsonify({"message": "Interview deleted."})

    @app.route("/api/interviews/<int:interview_id>/reset", methods=["POST"])
    @login_required
    def reset_interview(interview_id: int):
        with _db() as db:
            user = _ensure_logged_in(db)
            interview = _load_interview(db, interview_id, user.id)
            interview.conversation = []
            interview.current_code = ""
            interview.code_snapshots = []
            interview.code_evaluation = {}
            interview.coding_summary = {}
            interview.report = {}
            interview.problem_presented = False
            interview.phase_turn_start_index = 0
            interview.phase_started_at = _now() if interview.current_phase else None
            _set_updated(interview)
            db.commit()
            return jsonify({"message": "Interview reset."})

    @app.route("/api/chat", methods=["POST"])
    @login_required
    def chat():
        import time

        payload = request.get_json(silent=True) or {}
        user_message = (payload.get("message") or "").strip()
        interview_id = payload.get("interview_id")
        if not user_message:
            return jsonify({"error": "Message is required."}), 400
        if not interview_id:
            return jsonify({"error": "Interview ID is required."}), 400

        with _db() as db:
            user = _ensure_logged_in(db)
            interview = _load_interview(db, int(interview_id), user.id)

            if interview.status != "active":
                return jsonify({"error": "Interview is not active."}), 400

            full_resume_text = interview.resume.text if interview.resume else ""
            interview_state = {
                "current_phase": interview.current_phase,
                "language": interview.language,
                "mode": interview.mode,
                "model": interview.model,
                "problem_presented": interview.problem_presented,
                "code_snapshots": interview.code_snapshots,
                "coding_question": interview.coding_question,
                "code_evaluation": interview.code_evaluation,
                "coding_summary": interview.coding_summary,
                "company_context": interview.company_context,
                "ood_question": interview.ood_question,
                "candidate_name": interview.candidate_name,
                "phase_turn_start_index": interview.phase_turn_start_index,
            }
            resume_text = full_resume_text if interview.current_phase in {PHASE_INTRO_RESUME, PHASE_QUESTIONS} else None

            current_code = payload.get("code", "")
            code_changed = payload.get("code_changed", False)

            if not current_code:
                current_code = interview.current_code or ""
                code_changed = False

            if current_code:
                _update_code(interview, current_code)

            interview.last_speech_at = _now()

            conversation: List[Dict[str, Any]] = list(interview.conversation or [])
            perf_enabled = bool(interview.mode == "coding_only")
            perf_t0 = time.perf_counter() if perf_enabled else 0.0
            test_gate_time = 0.0
            test_run_time = 0.0
            llm_reply_time = 0.0
            tts_time = 0.0
            try:
                client = get_llm_client(interview.model)

                if interview.current_phase == PHASE_CODING:
                    coding_question = interview.coding_question or {}
                    question_title = coding_question.get("title", "")
                    if question_title and current_code.strip():
                        code_hash = hashlib.sha256(current_code.encode("utf-8")).hexdigest()
                        existing_eval = interview.code_evaluation or {}
                        if existing_eval.get("code_hash") != code_hash:
                            interview.code_evaluation = {}
                        fallback_hit = _should_run_tests_fallback(user_message)
                        complete_hit = _code_looks_complete(current_code, interview.language)
                        test_gate_start = time.perf_counter() if perf_enabled else 0.0
                        model_hit = client.should_run_tests(
                            interview_state=interview_state,
                            conversation=conversation,
                            user_message=user_message,
                            current_code=current_code,
                        )
                        if perf_enabled:
                            test_gate_time = time.perf_counter() - test_gate_start
                        logger.info(
                            "[TEST GATE] phase=%s fallback=%s complete=%s model=%s",
                            interview.current_phase,
                            fallback_hit,
                            complete_hit,
                            model_hit,
                        )
                        should_run = fallback_hit or complete_hit or model_hit
                        should_rerun = fallback_hit or model_hit
                        can_run = should_run and (
                            existing_eval.get("code_hash") != code_hash or not existing_eval or should_rerun
                        )
                        if can_run:
                            test_spec = get_cached_tests(question_title)
                            if test_spec:
                                logger.info("[TEST RUN] phase=%s question=%s", interview.current_phase, question_title)
                                test_run_start = time.perf_counter() if perf_enabled else 0.0
                                evaluation = run_code_tests(
                                    interview.language,
                                    current_code,
                                    test_spec,
                                )
                                if perf_enabled:
                                    test_run_time = time.perf_counter() - test_run_start
                                if str(evaluation.get("status", "")).strip() == "pass":
                                    pass_turn_count = existing_eval.get("pass_turn_count")
                                    if pass_turn_count is None:
                                        pass_turn_count = len(conversation) // 2
                                    evaluation["pass_turn_count"] = pass_turn_count
                                evaluation["code_hash"] = code_hash
                                interview.code_evaluation = evaluation
                                interview_state["code_evaluation"] = evaluation
                                logger.info(
                                    "[TEST RUN] phase=%s status=%s summary=%s",
                                    interview.current_phase,
                                    evaluation.get("status"),
                                    evaluation.get("summary"),
                                )

                time_in_phase = _calculate_time_in_phase(interview)
                total_time = _calculate_total_time(interview)
                silence_duration = _calculate_silence_duration(interview)
                code_idle_duration = _calculate_code_idle_duration(interview)

                # Calculate phase signal based on time constraints
                phase_signal = _calculate_phase_signal(interview, time_in_phase)

                logger.info(
                    "[MODEL REQUEST] phase=%s mode=%s time_in_phase=%.1fm phase_signal=%s user_message=%s",
                    interview.current_phase,
                    interview.mode,
                    time_in_phase,
                    phase_signal,
                    user_message[:100] + "..." if len(user_message) > 100 else user_message,
                )
                llm_start = time.perf_counter() if perf_enabled else 0.0
                reply_text = client.generate_structured_interview_reply(
                    interview_state=interview_state,
                    conversation=conversation,
                    resume_text=resume_text,
                    user_message=user_message,
                    current_code=current_code,
                    code_changed=code_changed,
                    time_in_phase=time_in_phase,
                    total_time=total_time,
                    silence_duration=silence_duration,
                    code_idle_duration=code_idle_duration,
                    phase_signal=phase_signal,
                )
                if perf_enabled:
                    llm_reply_time = time.perf_counter() - llm_start
                logger.info(
                    "[MODEL RESPONSE] phase=%s reply_len=%s llm_time=%.3fs",
                    interview.current_phase,
                    len(reply_text or ""),
                    llm_reply_time,
                )
            except (ValueError, GeminiError, OpenAIError) as exc:
                logger.error("[MODEL ERROR] %s", exc)
                return jsonify({"error": str(exc)}), 500

            phase_complete = False
            phase_before_transition = interview.current_phase
            phase_complete, reply_text = _extract_phase_complete(reply_text)
            if interview.mode == "full" and interview.current_phase == PHASE_CODING:
                evaluation = interview.code_evaluation or {}
                if str(evaluation.get("status", "")).strip() == "pass":
                    pass_turn_count = evaluation.get("pass_turn_count")
                    if isinstance(pass_turn_count, int):
                        projected_turns = (len(conversation) // 2) + 1
                        turns_since_pass = projected_turns - pass_turn_count
                        if turns_since_pass >= 10:
                            phase_complete = True
            logger.info(
                "[PHASE CHECK] phase=%s phase_complete=%s time_in_phase=%.1fm",
                phase_before_transition,
                phase_complete,
                time_in_phase,
            )
            _handle_problem_presentation(interview, reply_text)
            _apply_phase_completion(interview, phase_complete)
            _transition_ood_phase(interview)
            if interview.current_phase != phase_before_transition:
                logger.info(
                    "[PHASE TRANSITION] %s -> %s",
                    phase_before_transition,
                    interview.current_phase,
                )

            tts_text = _strip_code_markers(reply_text)

            try:
                tts_start = time.perf_counter() if perf_enabled else 0.0
                reply_audio = kokoro.synthesize_base64(tts_text)
                if perf_enabled:
                    tts_time = time.perf_counter() - tts_start
            except KokoroUnavailable as exc:
                logger.error("[TTS ERROR] %s", exc)
                return jsonify({"error": str(exc)}), 500
            except Exception as exc:
                logger.error("[TTS ERROR] %s", exc)
                return jsonify({"error": f"TTS error: {exc}"}), 500

            _append_turn(interview, user_message, reply_text)

            interview_ended = False
            if interview.mode in {"coding_only", "ood"}:
                total_time = _calculate_total_time(interview)
                if interview.mode == "coding_only":
                    evaluation = interview.code_evaluation or {}
                    if str(evaluation.get("status", "")).strip() == "pass":
                        pass_turn_count = evaluation.get("pass_turn_count")
                        if isinstance(pass_turn_count, int):
                            projected_turns = (len(interview.conversation or []) // 2)
                            turns_since_pass = projected_turns - pass_turn_count
                            if turns_since_pass >= 10:
                                interview.status = "ended"
                                interview.ended_at = _now()
                if interview.mode == "coding_only" and phase_complete:
                    evaluation = interview.code_evaluation or {}
                    if str(evaluation.get("status", "")).strip() == "pass":
                        interview.status = "ended"
                        interview.ended_at = _now()
                if total_time >= 40.0:
                    interview.status = "ended"
                    interview.ended_at = _now()
                elif _detect_interview_close(reply_text):
                    interview.status = "ended"
                    interview.ended_at = _now()
            interview_ended = interview.status == "ended"

            if perf_enabled:
                total_time = time.perf_counter() - perf_t0
                logger.info(
                    "[PERF][coding_only] total=%.3fs gate=%.3fs test=%.3fs llm=%.3fs tts=%.3fs",
                    total_time,
                    test_gate_time,
                    test_run_time,
                    llm_reply_time,
                    tts_time,
                )

            _set_updated(interview)
            db.commit()

            return jsonify({
                "reply": reply_text,
                "replyAudio": reply_audio,
                "phase": interview.current_phase,
                "interviewEnded": interview_ended,
            })

    @app.route("/api/update_code", methods=["POST"])
    @login_required
    def update_code_endpoint():
        payload = request.get_json(silent=True) or {}
        code = payload.get("code", "")
        interview_id = payload.get("interview_id")
        if not interview_id:
            return jsonify({"error": "Interview ID is required."}), 400

        with _db() as db:
            user = _ensure_logged_in(db)
            interview = _load_interview(db, int(interview_id), user.id)
            _update_code(interview, code)
            _set_updated(interview)
            db.commit()
            return jsonify({"message": "Code updated"})

    @app.route("/api/interviews/<int:interview_id>/end", methods=["POST"])
    @login_required
    def end_interview(interview_id: int):
        with _db() as db:
            user = _ensure_logged_in(db)
            interview = _load_interview(db, interview_id, user.id)

            interview.status = "ended"
            interview.ended_at = _now()

            if interview.report:
                _set_updated(interview)
                db.commit()
                return jsonify({"report": interview.report, "cached": True})

            try:
                client = get_llm_client(interview.model)
                report = client.generate_interview_report(
                    mode=interview.mode,
                    language=interview.language,
                    conversation=interview.conversation or [],
                    current_code=interview.current_code or "",
                    code_snapshots=interview.code_snapshots or [],
                    problem_presented=bool(interview.problem_presented),
                )
            except (ValueError, GeminiError, OpenAIError) as exc:
                return jsonify({"error": str(exc)}), 500

            interview.report = report
            _set_updated(interview)
            db.commit()
            return jsonify({"report": report, "cached": False})

    @app.route("/api/interviews/<int:interview_id>/report", methods=["POST"])
    @login_required
    def interview_report(interview_id: int):
        with _db() as db:
            user = _ensure_logged_in(db)
            interview = _load_interview(db, interview_id, user.id)

            if interview.report:
                return jsonify({"report": interview.report, "cached": True})

            try:
                client = get_llm_client(interview.model)
                report = client.generate_interview_report(
                    mode=interview.mode,
                    language=interview.language,
                    conversation=interview.conversation or [],
                    current_code=interview.current_code or "",
                    code_snapshots=interview.code_snapshots or [],
                    problem_presented=bool(interview.problem_presented),
                )
            except (ValueError, GeminiError, OpenAIError) as exc:
                return jsonify({"error": str(exc)}), 500

            interview.report = report
            _set_updated(interview)
            db.commit()
            return jsonify({"report": report, "cached": False})

    @app.route("/api/transition_phase", methods=["POST"])
    @login_required
    def transition_phase():
        payload = request.get_json(silent=True) or {}
        new_phase = payload.get("phase")
        interview_id = payload.get("interview_id")
        if not interview_id:
            return jsonify({"error": "Interview ID is required."}), 400

        valid_phases = [
            PHASE_INTRO_RESUME,
            PHASE_CODING,
            PHASE_QUESTIONS,
            PHASE_OOD_DESIGN,
            PHASE_OOD_IMPLEMENTATION,
        ]
        if new_phase not in valid_phases:
            return jsonify({"error": "Invalid phase"}), 400

        with _db() as db:
            user = _ensure_logged_in(db)
            interview = _load_interview(db, int(interview_id), user.id)

            if interview.mode == "full":
                time_in_phase = _calculate_time_in_phase(interview)
                if new_phase == PHASE_CODING and time_in_phase < FULL_PHASE_RULES[PHASE_INTRO_RESUME]["min"]:
                    return jsonify({"error": "Cannot enter coding before 5 minutes in intro/resume."}), 400
                if new_phase == PHASE_QUESTIONS and time_in_phase < FULL_PHASE_RULES[PHASE_CODING]["min"]:
                    return jsonify({"error": "Cannot enter questions before coding begins."}), 400

            interview.current_phase = new_phase
            interview.phase_started_at = _now()
            interview.phase_turn_start_index = len(interview.conversation or [])
            _set_updated(interview)
            db.commit()
            return jsonify({"message": f"Transitioned to {new_phase} phase"})

    @app.route("/api/transcribe", methods=["POST"])
    @login_required
    def transcribe_audio():
        import time
        start_time = time.time()

        try:
            if "audio" not in request.files:
                return jsonify({"error": "No audio file provided"}), 400

            audio_file = request.files["audio"]
            audio_format = request.form.get("format", "webm")
            audio_bytes = audio_file.read()
            if not audio_bytes:
                return jsonify({"error": "Empty audio file"}), 400

            transcriber = get_whisper_transcriber(model_size="base")
            transcribe_start = time.time()
            result = transcriber.transcribe_bytes(
                audio_bytes,
                audio_format=audio_format,
                language="en",
            )
            transcribe_time = time.time() - transcribe_start

            total_time = time.time() - start_time
            logging.info(
                "[WHISPER] Transcription complete in %.3fs (total: %.3fs)",
                transcribe_time,
                total_time,
            )

            return jsonify({
                "text": result["text"],
                "raw_text": result["raw_text"],
                "language": result["language"],
                "duration": result["duration"],
                "transcription_time": transcribe_time,
                "device": transcriber.device,
                "success": True,
            })

        except Exception as e:
            logging.error("[WHISPER] Transcription error: %s", e, exc_info=True)
            return jsonify({"error": str(e), "success": False}), 500

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=1111, debug=False)
