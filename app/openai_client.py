from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

import requests
from requests import RequestException

from .config import config
from .gemini_client import GeminiClient
from . import state


class OpenAIError(RuntimeError):
    """Raised when the OpenAI API returns an unexpected response."""


class OpenAIClient:
    API_URL = "https://api.openai.com/v1/chat/completions"

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        self.api_key = api_key or config.OPENAI_API_KEY
        self.model = model or config.OPENAI_MODEL
        if not self.api_key:
            raise ValueError(
                "OpenAI API key is not configured. Set the OPENAI_API_KEY environment variable."
            )
        # Reuse Gemini prompt construction without hitting Gemini APIs.
        self._builder = GeminiClient(api_key="unused", model="unused")

    def generate_interview_reply(
        self,
        conversation: List[Dict[str, Any]],
        resume_text: str | None,
        user_message: str,
        temperature: float = 0.7,
    ) -> str:
        payload = self._builder._build_payload(conversation, resume_text, user_message, temperature)
        return self._post_for_text(payload)

    def generate_structured_interview_reply(
        self,
        session_id: str,
        conversation: List[Dict[str, Any]],
        resume_text: str | None,
        user_message: str,
        current_code: str = "",
        code_changed: bool = False,
        temperature: float = 0.7,
    ) -> str:
        interview_state = state.get_interview_state(session_id)
        if not interview_state:
            return self.generate_interview_reply(conversation, resume_text, user_message, temperature)

        payload = self._builder._build_structured_payload(
            session_id,
            interview_state,
            conversation,
            resume_text,
            user_message,
            current_code,
            code_changed,
            temperature,
        )

        system_prompt = payload.get("systemInstruction", {}).get("parts", [{}])[0].get("text", "")
        logging.info("[OPENAI DEBUG] System prompt length: %s", len(system_prompt))
        return self._post_for_text(payload)

    def decide_phase_transition(
        self,
        session_id: str,
        interview_state: Dict[str, Any],
        conversation: List[Dict[str, Any]],
        user_message: str,
        model_reply: str,
    ) -> Optional[Dict[str, Any]]:
        time_in_phase = state.calculate_time_in_phase(session_id)
        total_time = state.calculate_total_time(session_id)
        silence_duration = state.calculate_silence_duration(session_id)
        code_idle_duration = state.calculate_code_idle_duration(session_id)

        payload = self._builder._build_phase_decision_payload(
            interview_state=interview_state,
            conversation=conversation,
            user_message=user_message,
            model_reply=model_reply,
            time_in_phase=time_in_phase,
            total_time=total_time,
            silence_duration=silence_duration,
            code_idle_duration=code_idle_duration,
        )

        decision_text = self._post_for_text(payload, max_tokens=200, temperature=0.1)
        decision = self._builder._parse_phase_decision(decision_text)
        if not decision:
            logging.warning("Unable to parse OpenAI phase decision JSON: %s", decision_text)
            return None

        next_phase = decision.get("next_phase")
        if next_phase and next_phase not in {
            state.PHASE_INTRO_RESUME,
            state.PHASE_CODING,
            state.PHASE_QUESTIONS,
        }:
            logging.warning("Phase decision returned invalid next_phase: %s", decision)
            return None

        return decision

    def should_run_tests(
        self,
        session_id: str,
        interview_state: Dict[str, Any],
        conversation: List[Dict[str, Any]],
        user_message: str,
        current_code: str,
    ) -> bool:
        current_phase = interview_state.get("current_phase", state.PHASE_INTRO_RESUME)
        if current_phase != state.PHASE_CODING:
            return False

        snippet = self._builder._format_conversation_snippet(conversation)
        payload = self._builder._build_test_gate_payload(
            user_message=user_message,
            current_code=current_code,
            conversation_snippet=snippet,
        )
        decision_text = self._post_for_text(payload, max_tokens=120, temperature=0.1)
        decision = self._builder._parse_phase_decision(decision_text)
        if not decision:
            logging.warning("Unable to parse OpenAI test gate JSON: %s", decision_text)
            return False

        flag = decision.get("should_run_tests", False)
        logging.info("[TEST GATE MODEL] raw=%s parsed=%s", decision_text, flag)
        if isinstance(flag, str):
            return flag.lower() == "true"
        return bool(flag)

    def generate_interview_report(
        self,
        mode: str,
        language: str,
        conversation: List[Dict[str, Any]],
        current_code: str,
        code_snapshots: List[Dict[str, Any]],
        problem_presented: bool,
    ) -> Dict[str, Any]:
        system_prompt = (
            "You are a senior engineering interviewer writing a concise evaluation report. "
            "Assess the candidate like a real SDE interview. "
            "Return STRICT JSON only, no extra commentary."
        )

        rubric_details = {
            "problem_solving": "How well they structured the approach, validated assumptions, and decomposed the problem.",
            "technical_depth": "Quality of algorithm/data structure reasoning and trade-off analysis.",
            "coding_correctness": "Correctness of implementation vs. requirements.",
            "edge_cases": "Coverage of tricky inputs and boundary conditions.",
            "efficiency": "Time/space complexity choices and optimization awareness.",
            "communication": "Clarity, structure, and collaboration throughout.",
            "code_quality": "Readability, naming, organization, and maintainability.",
        }

        max_conversation_turns = min(40, len(conversation) if conversation else 0)
        context = {
            "mode": mode,
            "language": language,
            "problem_presented": problem_presented,
            "conversation_summary": self._builder._format_conversation_snippet(
                conversation,
                max_turns=max_conversation_turns,
            ),
            "total_conversation_turns": len(conversation) if conversation else 0,
            "current_code": current_code or "[empty]",
            "code_snapshot_count": len(code_snapshots),
        }

        instructions = (
            "Return JSON with EXACTLY this structure (no additional fields):\n"
            "{\n"
            '  "overall_score": number 1-5,\n'
            '  "recommendation": "strong_hire" | "hire" | "lean_hire" | "lean_no_hire" | "no_hire",\n'
            '  "summary": "Brief 2-sentence evaluation",\n'
            '  "category_scores": {\n'
            '     "problem_solving": 1-5,\n'
            '     "technical_depth": 1-5,\n'
            '     "coding_correctness": 1-5,\n'
            '     "edge_cases": 1-5,\n'
            '     "efficiency": 1-5,\n'
            '     "communication": 1-5,\n'
            '     "code_quality": 1-5\n'
            "  },\n"
            '  "strengths": ["concise point 1", "concise point 2", "concise point 3"],\n'
            '  "improvements": ["concise point 1", "concise point 2", "concise point 3"],\n'
            '  "notable_moments": ["brief moment 1", "brief moment 2", "brief moment 3"]\n'
            "}\n\n"
            "IMPORTANT:\n"
            "- Output ONLY the JSON structure above with NO additional fields\n"
            "- Do NOT add explanations for category scores (scores are self-explanatory)\n"
            "- Keep each strength/improvement/moment to ONE sentence max\n"
            "- Be specific but concise\n"
            "- Output raw JSON only (no markdown, no code fences, no commentary)"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": instructions + "\n\nCONTEXT:\n" + json.dumps(context, ensure_ascii=False),
            },
        ]

        response_text = self._post_messages(messages, max_tokens=2048, temperature=0.2, top_p=0.9)
        report = self._builder._parse_json_from_text(response_text)
        report["rubric_details"] = rubric_details
        return report

    def _post_for_text(
        self,
        gemini_payload: Dict[str, Any],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        system_prompt = gemini_payload.get("systemInstruction", {}).get("parts", [{}])[0].get("text", "")
        contents = gemini_payload.get("contents", [])
        messages = [{"role": "system", "content": system_prompt}]
        for item in contents:
            role = item.get("role")
            parts = item.get("parts", [])
            text = " ".join(part.get("text", "") for part in parts if isinstance(part, dict)).strip()
            if not text:
                continue
            messages.append(
                {
                    "role": "assistant" if role == "model" else "user",
                    "content": text,
                }
            )

        gen_config = gemini_payload.get("generationConfig", {})
        return self._post_messages(
            messages,
            max_tokens=max_tokens or gen_config.get("maxOutputTokens", 800),
            temperature=temperature if temperature is not None else gen_config.get("temperature", 0.7),
            top_p=gen_config.get("topP", 0.95),
        )

    def _post_messages(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }

        try:
            response = requests.post(
                self.API_URL,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=payload,
                timeout=30,
            )
        except RequestException as exc:  # type: ignore[name-defined]
            raise OpenAIError(f"OpenAI API request failed: {exc}") from exc

        if response.status_code != 200:
            logging.error(
                "OpenAI API error: status=%s body=%s",
                response.status_code,
                response.text,
            )
            raise OpenAIError(f"OpenAI API error {response.status_code}: {response.text}")

        data = response.json()
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            logging.error("Unexpected OpenAI API payload: %s", data)
            raise OpenAIError("OpenAI API returned an unexpected payload") from exc
        return (content or "").strip()
