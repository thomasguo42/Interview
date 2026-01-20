from __future__ import annotations

from threading import Lock
from typing import Dict, Any, List, Optional
from datetime import datetime


MAX_RESUME_CHARS = 20000
MAX_CONVERSATION_MESSAGES = 40  # 20 turns
MAX_COMPANY_FIELD_CHARS = 200
MAX_COMPANY_DETAILS_CHARS = 1000

_state: Dict[str, Dict[str, Any]] = {}
_lock = Lock()

# Interview phases
PHASE_INTRO = "intro"
PHASE_RESUME = "resume"
PHASE_CODING = "coding"
PHASE_QUESTIONS = "questions"

# OOD Interview phases
PHASE_OOD_DESIGN = "ood_design"
PHASE_OOD_IMPLEMENTATION = "ood_implementation"

# Supported programming languages
SUPPORTED_LANGUAGES = ["python", "java", "cpp"]


def store_resume(session_id: str, resume_text: str) -> None:
    truncated_resume = resume_text.strip()
    if len(truncated_resume) > MAX_RESUME_CHARS:
        truncated_resume = truncated_resume[:MAX_RESUME_CHARS]

    with _lock:
        context = _state.setdefault(session_id, {"resume": None, "conversation": []})
        context["resume"] = truncated_resume
        context["conversation"] = []


def get_resume(session_id: str) -> Optional[str]:
    with _lock:
        context = _state.get(session_id)
        if not context:
            return None
        return context.get("resume")


def has_resume(session_id: str) -> bool:
    with _lock:
        context = _state.get(session_id)
        return bool(context and context.get("resume"))


def set_company_context(session_id: str, company: str = "", role: str = "", details: str = "") -> None:
    """Store the target company/role context for tailoring prompts."""

    def _sanitize(value: str, limit: int) -> str:
        text = (value or "").strip()
        if len(text) > limit:
            text = text[:limit]
        return text

    company_clean = _sanitize(company, MAX_COMPANY_FIELD_CHARS)
    role_clean = _sanitize(role, MAX_COMPANY_FIELD_CHARS)
    details_clean = _sanitize(details, MAX_COMPANY_DETAILS_CHARS)

    context_payload = {
        "company": company_clean,
        "role": role_clean,
        "details": details_clean,
    }

    with _lock:
        context = _state.setdefault(session_id, {"resume": None, "conversation": []})
        context["company_context"] = context_payload
        interview = context.get("interview")
        if interview is not None:
            interview["company_context"] = context_payload


def get_company_context(session_id: str) -> Dict[str, str]:
    """Return the stored company/role context, if any."""
    with _lock:
        context = _state.get(session_id)
        if not context:
            return {"company": "", "role": "", "details": ""}
        stored = context.get("company_context")
        if not stored:
            return {"company": "", "role": "", "details": ""}
        return dict(stored)


def get_conversation(session_id: str) -> List[Dict[str, Any]]:
    with _lock:
        context = _state.get(session_id)
        if not context:
            return []
        return list(context.get("conversation", []))


def append_turn(session_id: str, user_message: str, model_message: str) -> None:
    user_entry = {"role": "user", "parts": [{"text": user_message}]}
    model_entry = {"role": "model", "parts": [{"text": model_message}]}

    with _lock:
        context = _state.setdefault(session_id, {"resume": None, "conversation": []})
        conversation: List[Dict[str, Any]] = context.setdefault("conversation", [])
        conversation.append(user_entry)
        conversation.append(model_entry)
        if len(conversation) > MAX_CONVERSATION_MESSAGES:
            context["conversation"] = conversation[-MAX_CONVERSATION_MESSAGES:]


def clear_conversation(session_id: str) -> None:
    with _lock:
        context = _state.get(session_id)
        if context:
            context["conversation"] = []


# ============================================================================
# STRUCTURED INTERVIEW SESSION MANAGEMENT
# ============================================================================

def start_interview(
    session_id: str,
    language: str = "python",
    mode: str = "full",
    model: str | None = None,
) -> Dict[str, Any]:
    """Initialize a structured interview session"""
    if language not in SUPPORTED_LANGUAGES:
        language = "python"

    # Determine starting phase based on mode
    if mode == "ood":
        starting_phase = PHASE_OOD_DESIGN
    elif mode == "coding_only":
        starting_phase = PHASE_CODING
    else:
        starting_phase = PHASE_INTRO

    with _lock:
        context = _state.setdefault(session_id, {"resume": None, "conversation": []})
        interview_state = {
            "started_at": datetime.now(),
            "current_phase": starting_phase,
            "phase_started_at": datetime.now(),
            "language": language,
            "mode": mode,  # Store the mode
            "model": model,
            "current_code": "",
            "last_speech_at": datetime.now(),
            "last_code_change_at": datetime.now(),
            "intervention_count": 0,
            "problem_presented": False,
            "code_snapshots": [],
            "coding_question": None,
            "code_evaluation": None,
        }

        company_context = context.get("company_context")
        if company_context:
            interview_state["company_context"] = dict(company_context)

        # Add OOD-specific state
        if mode == "ood":
            interview_state.update({
                "ood_question": None,
                "design_phase_complete": False,
            })

        context["interview"] = interview_state
        return context["interview"]


def get_interview_state(session_id: str) -> Optional[Dict[str, Any]]:
    """Get current interview state"""
    with _lock:
        context = _state.get(session_id)
        if not context:
            return None
        return context.get("interview")


def update_interview_phase(session_id: str, new_phase: str) -> None:
    """Transition to a new interview phase"""
    with _lock:
        context = _state.get(session_id)
        if context and "interview" in context:
            context["interview"]["current_phase"] = new_phase
            context["interview"]["phase_started_at"] = datetime.now()


def update_code(session_id: str, code: str) -> None:
    """Update current code and track change time"""
    with _lock:
        context = _state.get(session_id)
        if context and "interview" in context:
            context["interview"]["current_code"] = code
            context["interview"]["last_code_change_at"] = datetime.now()
            # Keep last 10 snapshots for analysis
            snapshots = context["interview"].setdefault("code_snapshots", [])
            snapshots.append({
                "code": code,
                "timestamp": datetime.now()
            })
            if len(snapshots) > 10:
                context["interview"]["code_snapshots"] = snapshots[-10:]


def get_current_code(session_id: str) -> str:
    """Get current code"""
    with _lock:
        context = _state.get(session_id)
        if context and "interview" in context:
            return context["interview"].get("current_code", "")
        return ""


def update_last_speech(session_id: str) -> None:
    """Update last speech timestamp"""
    with _lock:
        context = _state.get(session_id)
        if context and "interview" in context:
            context["interview"]["last_speech_at"] = datetime.now()


def increment_intervention_count(session_id: str) -> None:
    """Track number of interventions during coding"""
    with _lock:
        context = _state.get(session_id)
        if context and "interview" in context:
            context["interview"]["intervention_count"] = context["interview"].get("intervention_count", 0) + 1


def set_problem_presented(session_id: str, presented: bool = True) -> None:
    """Mark that coding problem has been presented"""
    with _lock:
        context = _state.get(session_id)
        if context and "interview" in context:
            context["interview"]["problem_presented"] = presented


def set_coding_question(session_id: str, question: Dict[str, str]) -> None:
    """Set the coding question metadata for the interview."""
    with _lock:
        context = _state.get(session_id)
        if context and "interview" in context:
            context["interview"]["coding_question"] = dict(question)


def get_coding_question(session_id: str) -> Dict[str, str]:
    """Get the stored coding question metadata."""
    with _lock:
        context = _state.get(session_id)
        if context and "interview" in context:
            stored = context["interview"].get("coding_question")
            if stored:
                return dict(stored)
        return {}


def set_code_evaluation(session_id: str, evaluation: Dict[str, Any]) -> None:
    """Store the most recent code evaluation results."""
    with _lock:
        context = _state.get(session_id)
        if context and "interview" in context:
            context["interview"]["code_evaluation"] = dict(evaluation)


def get_code_evaluation(session_id: str) -> Dict[str, Any]:
    """Get the last stored code evaluation results."""
    with _lock:
        context = _state.get(session_id)
        if context and "interview" in context:
            stored = context["interview"].get("code_evaluation")
            if stored:
                return dict(stored)
        return {}


def calculate_time_in_phase(session_id: str) -> float:
    """Calculate minutes elapsed in current phase"""
    with _lock:
        context = _state.get(session_id)
        if context and "interview" in context:
            phase_start = context["interview"].get("phase_started_at")
            if phase_start:
                delta = datetime.now() - phase_start
                return delta.total_seconds() / 60.0
        return 0.0


def calculate_total_time(session_id: str) -> float:
    """Calculate total interview minutes elapsed"""
    with _lock:
        context = _state.get(session_id)
        if context and "interview" in context:
            start = context["interview"].get("started_at")
            if start:
                delta = datetime.now() - start
                return delta.total_seconds() / 60.0
        return 0.0


def calculate_silence_duration(session_id: str) -> float:
    """Calculate seconds since last speech"""
    with _lock:
        context = _state.get(session_id)
        if context and "interview" in context:
            last_speech = context["interview"].get("last_speech_at")
            if last_speech:
                delta = datetime.now() - last_speech
                return delta.total_seconds()
        return 0.0


def calculate_code_idle_duration(session_id: str) -> float:
    """Calculate seconds since last code change"""
    with _lock:
        context = _state.get(session_id)
        if context and "interview" in context:
            last_code = context["interview"].get("last_code_change_at")
            if last_code:
                delta = datetime.now() - last_code
                return delta.total_seconds()
        return 0.0


# ============================================================================
# OOD INTERVIEW MANAGEMENT
# ============================================================================


def set_ood_question(session_id: str, question: Dict[str, str]) -> None:
    """Set the OOD question for the interview"""
    with _lock:
        context = _state.get(session_id)
        if context and "interview" in context:
            context["interview"]["ood_question"] = dict(question)


def get_ood_question(session_id: str) -> Dict[str, str]:
    """Get the stored OOD question metadata"""
    with _lock:
        context = _state.get(session_id)
        if context and "interview" in context:
            stored = context["interview"].get("ood_question")
            if stored:
                return dict(stored)
        return {}


def set_design_phase_complete(session_id: str, complete: bool = True) -> None:
    """Mark design phase as complete"""
    with _lock:
        context = _state.get(session_id)
        if context and "interview" in context:
            context["interview"]["design_phase_complete"] = complete


def set_interview_ended(session_id: str, ended: bool = True) -> None:
    """Mark interview as ended to trigger report generation."""
    with _lock:
        context = _state.get(session_id)
        if context and "interview" in context:
            context["interview"]["ended"] = ended


def is_interview_ended(session_id: str) -> bool:
    """Check if interview has ended."""
    with _lock:
        context = _state.get(session_id)
        if context and "interview" in context:
            return bool(context["interview"].get("ended", False))
        return False


def set_interview_report(session_id: str, report: Dict[str, Any]) -> None:
    """Cache generated interview report."""
    with _lock:
        context = _state.get(session_id)
        if context and "interview" in context:
            context["interview"]["report"] = dict(report)


def get_interview_report(session_id: str) -> Optional[Dict[str, Any]]:
    """Get cached interview report, if any."""
    with _lock:
        context = _state.get(session_id)
        if context and "interview" in context:
            stored = context["interview"].get("report")
            if stored:
                return dict(stored)
        return None
