from __future__ import annotations

import logging
import json
import re
from typing import List, Dict, Any, Optional

import requests
from requests import RequestException

from .config import config
from . import state


class GeminiError(RuntimeError):
    """Raised when the Gemini API returns an unexpected response."""


class GeminiClient:
    API_URL_TEMPLATE = (
        "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    )

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        self.api_key = api_key or config.GEMINI_API_KEY
        self.model = model or config.GEMINI_MODEL
        self.phase_model = self.model
        if not self.api_key:
            raise ValueError(
                "Gemini API key is not configured. Set the GEMINI_API_KEY environment variable."
            )

    def generate_interview_reply(
        self,
        conversation: List[Dict[str, Any]],
        resume_text: str | None,
        user_message: str,
        temperature: float = 0.7,
    ) -> str:
        """
        Sends the conversation history and the latest user message to Gemini and returns the reply text.

        Args:
            conversation: Existing conversation history formatted for the Gemini API.
            resume_text: Optional resume text to ground the interview context.
            user_message: Latest user utterance to append to the conversation.
            temperature: Sampling temperature for response creativity.
        """
        payload = self._build_payload(conversation, resume_text, user_message, temperature)
        response = requests.post(
            self.API_URL_TEMPLATE.format(model=self.model),
            params={"key": self.api_key},
            json=payload,
            timeout=30,
        )
        if response.status_code != 200:
            logging.error(
                "Gemini API error: status=%s body=%s",
                response.status_code,
                response.text,
            )
            raise GeminiError(f"Gemini API error {response.status_code}: {response.text}")

        data = response.json()
        try:
            parts = data["candidates"][0]["content"]["parts"]
            combined_text = "\n".join(part.get("text", "") for part in parts if "text" in part).strip()
            if not combined_text:
                raise KeyError("Empty response text")
            return combined_text
        except (KeyError, IndexError) as exc:
            logging.error("Unexpected Gemini API payload: %s", data)
            raise GeminiError("Gemini API returned an unexpected payload") from exc

    def _build_payload(
        self,
        conversation: List[Dict[str, Any]],
        resume_text: str | None,
        user_message: str,
        temperature: float,
    ) -> Dict[str, Any]:
        system_prompt = (
            "You are an experienced senior engineer conducting a LIVE VOICE mock technical interview. "
            "This is a natural, back-and-forth CONVERSATION, not a written exchange.\n\n"
            "CRITICAL CONVERSATION RULES:\n"
            "- Keep responses SHORT and NATURAL - like how people actually talk (1-3 sentences usually)\n"
            "- Vary your response length based on context: sometimes just 'Got it', 'Makes sense', 'Tell me more', "
            "sometimes a follow-up question, sometimes detailed feedback\n"
            "- ANALYZE the conversation history and user's last message to determine the appropriate response type:\n"
            "  * Short acknowledgment (e.g., 'Okay', 'I see', 'Right')\n"
            "  * Brief follow-up question (e.g., 'What about error handling?')\n"
            "  * Deeper probe (e.g., 'Walk me through your approach to...')\n"
            "  * Constructive feedback (when they finish explaining something)\n"
            "  * Topic switch (when current topic is exhausted)\n"
            "- Speak like a human in conversation: use contractions (I'm, you're, that's), casual phrasing, "
            "natural pauses indicated by punctuation\n"
            "- NEVER write essay-style responses - you're SPEAKING, not writing\n"
            "- Listen actively: reference what they just said, build on it naturally\n"
            "- Don't explain everything at once - let the conversation flow organically\n"
            "- Ask ONE question at a time, not multiple\n"
            "- Balance between being supportive and challenging them appropriately\n\n"
            "CRITICAL - TEXT-TO-SPEECH OUTPUT:\n"
            "- Your output will be converted to SPEECH by a TTS system\n"
            "- NEVER use markdown symbols: NO asterisks (*), NO hashtags (#), NO underscores (_)\n"
            "- NEVER use formatting: NO **bold**, NO *italics*, NO `code blocks`\n"
            "- NEVER use filler words like 'um', 'uh', 'hmm', 'well', 'so'\n"
            "- Speak clearly and directly - write EXACTLY what should be spoken out loud\n"
            "- For code or technical terms, just say them plainly"
            "\nIMPORTANT:\n"
            "- The interviewer always initiates the conversation. Do not wait for the candidate to greet.\n"
        )
        if resume_text:
            system_prompt += (
                "\nThe candidate resume you have is below. Use it to tailor your questions "
                "and feedback. Focus on relevant technologies, accomplishments, and gaps.\n"
                f"Resume:\n{resume_text}\n"
            )

        # Clone conversation to avoid mutating caller data
        conversation_payload = list(conversation)
        conversation_payload.append(
            {
                "role": "user",
                "parts": [
                    {
                        "text": user_message,
                    }
                ],
            }
        )

        payload: Dict[str, Any] = {
            "systemInstruction": {
                "role": "system",
                "parts": [
                    {
                        "text": system_prompt,
                    }
                ],
            },
            "contents": conversation_payload,
            "generationConfig": {
                "temperature": temperature,
                "topP": 0.95,
                "topK": 40,
                "maxOutputTokens": 800,  # Increased to allow code responses
            },
        }
        return payload

    def generate_structured_interview_reply(
        self,
        interview_state: Dict[str, Any],
        conversation: List[Dict[str, Any]],
        resume_text: str | None,
        user_message: str,
        current_code: str = "",
        code_changed: bool = False,
        time_in_phase: float = 0.0,
        total_time: float = 0.0,
        silence_duration: float = 0.0,
        code_idle_duration: float = 0.0,
        phase_signal: str = "end_if_you_want",
        temperature: float = 0.7,
    ) -> str:
        """
        Generate reply for structured interview with phase awareness.

        Args:
            interview_state: Current interview state
            conversation: Existing conversation history
            resume_text: Resume text for context
            user_message: Latest user utterance
            current_code: Current code in editor
            code_changed: Whether code changed recently
            time_in_phase: Minutes elapsed in current phase
            total_time: Total interview minutes elapsed
            silence_duration: Seconds since candidate spoke
            code_idle_duration: Seconds since code changed
            temperature: Sampling temperature
        """
        if not interview_state:
            return self.generate_interview_reply(conversation, resume_text, user_message, temperature)

        payload = self._build_structured_payload(
            interview_state,
            conversation,
            resume_text,
            user_message,
            current_code,
            code_changed,
            time_in_phase,
            total_time,
            silence_duration,
            code_idle_duration,
            phase_signal,
            temperature,
        )

        # Debug logging
        current_phase = interview_state.get("current_phase", "unknown")
        mode = interview_state.get("mode", "unknown")
        system_prompt = payload.get("systemInstruction", {}).get("parts", [{}])[0].get("text", "")
        code_in_prompt = bool('CURRENT CODE IN EDITOR' in system_prompt)
        logging.info(f"[GEMINI DEBUG] phase={current_phase} mode={mode} time_in_phase={time_in_phase:.1f}m")
        logging.info(f"[GEMINI DEBUG] phase={current_phase} current_code_length={len(current_code)}")
        logging.info(f"[GEMINI DEBUG] phase={current_phase} code_included_in_prompt={code_in_prompt}")
        if current_phase == "coding":
            evaluation = interview_state.get("code_evaluation") or {}
            test_status = str(evaluation.get("status", "")).strip() or "not_run"
            logging.info(f"[GEMINI DEBUG] phase={current_phase} test_status={test_status} code_access_granted={code_in_prompt}")
        logging.info(f"[GEMINI DEBUG] phase={current_phase} system_prompt_length={len(system_prompt)}")

        response = requests.post(
            self.API_URL_TEMPLATE.format(model=self.model),
            params={"key": self.api_key},
            json=payload,
            timeout=30,
        )

        if response.status_code != 200:
            logging.error(
                "Gemini API error: status=%s body=%s",
                response.status_code,
                response.text,
            )
            raise GeminiError(f"Gemini API error {response.status_code}: {response.text}")

        data = response.json()
        try:
            parts = data["candidates"][0]["content"]["parts"]
            combined_text = "\n".join(part.get("text", "") for part in parts if "text" in part).strip()
            if not combined_text:
                raise KeyError("Empty response text")
            return combined_text
        except (KeyError, IndexError) as exc:
            logging.error("Unexpected Gemini API payload: %s", data)
            raise GeminiError("Gemini API returned an unexpected payload") from exc

    def generate_interview_report(
        self,
        mode: str,
        language: str,
        conversation: List[Dict[str, Any]],
        current_code: str,
        code_snapshots: List[Dict[str, Any]],
        problem_presented: bool,
    ) -> Dict[str, Any]:
        """Generate a structured feedback report for the completed interview."""
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

        # For report generation, include significant conversation history but not everything
        # to avoid overwhelming the context window. 40 turns should capture key moments
        # while leaving room for detailed output.
        max_conversation_turns = min(40, len(conversation) if conversation else 0)

        context = {
            "mode": mode,
            "language": language,
            "problem_presented": problem_presented,
            "conversation_summary": self._format_conversation_snippet(
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

        payload = {
            "systemInstruction": {
                "role": "system",
                "parts": [{"text": system_prompt}],
            },
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": instructions
                            + "\n\nCONTEXT:\n"
                            + json.dumps(context, ensure_ascii=False),
                        }
                    ],
                }
            ],
            "generationConfig": {
                "temperature": 0.2,
                "topP": 0.9,
                "topK": 40,
                "maxOutputTokens": 2048,
                "responseMimeType": "application/json",
            },
        }

        response = requests.post(
            self.API_URL_TEMPLATE.format(model=self.model),
            params={"key": self.api_key},
            json=payload,
            timeout=30,
        )
        if response.status_code != 200:
            logging.error(
                "Gemini API error: status=%s body=%s",
                response.status_code,
                response.text,
            )
            raise GeminiError(f"Gemini API error {response.status_code}: {response.text}")

        data = response.json()

        # Check if response was truncated due to token limit
        finish_reason = data.get("candidates", [{}])[0].get("finishReason", "")
        if finish_reason == "MAX_TOKENS":
            logging.warning(
                "Report generation hit MAX_TOKENS limit. Retrying with higher limit. "
                "Original response: %s",
                data
            )
            # Retry with higher token limit
            payload["generationConfig"]["maxOutputTokens"] = 3072
            response = requests.post(
                self.API_URL_TEMPLATE.format(model=self.model),
                params={"key": self.api_key},
                json=payload,
                timeout=30,
            )
            if response.status_code != 200:
                logging.error(
                    "Gemini API error on retry: status=%s body=%s",
                    response.status_code,
                    response.text,
                )
                raise GeminiError(f"Gemini API error {response.status_code}: {response.text}")
            data = response.json()
            finish_reason = data.get("candidates", [{}])[0].get("finishReason", "")
            if finish_reason == "MAX_TOKENS":
                logging.error(
                    "Report generation still hitting MAX_TOKENS even with 3072 limit. "
                    "Response: %s",
                    data
                )

        try:
            parts = data["candidates"][0]["content"]["parts"]
            combined_text = " ".join(part.get("text", "") for part in parts if "text" in part).strip()
            if not combined_text:
                raise KeyError("Empty response text")
            report = self._parse_json_from_text(combined_text)
        except (KeyError, IndexError, json.JSONDecodeError) as exc:
            logging.error("Unexpected Gemini report payload: %s", data)
            if finish_reason == "MAX_TOKENS":
                raise GeminiError(
                    "Report generation exceeded token limit. The interview may be too long. "
                    "Please try ending the interview earlier or contact support."
                ) from exc
            raise GeminiError("Gemini API returned an invalid report payload") from exc

        report["rubric_details"] = rubric_details
        return report

    def generate_coding_wrapup_reply(
        self,
        user_message: str,
        language: str,
        mode: str,
    ) -> str:
        system_prompt = (
            "You are a senior engineer conducting a LIVE VOICE coding interview. "
            "This is mid-interview and ALL TESTS ARE PASSING. "
            "Your goal is to wrap up: ask 1 brief question about edge cases or optimizations, "
            "then be ready to conclude. Keep it to 1-2 short sentences. "
            "Do NOT mention tests or counts. Do NOT use markdown."
        )
        prompt = (
            f"Language: {language}\n"
            f"Mode: {mode}\n"
            f"Candidate just said: {user_message}\n"
            "Respond now."
        )
        payload = {
            "systemInstruction": {
                "role": "system",
                "parts": [{"text": system_prompt}],
            },
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}],
                }
            ],
            "generationConfig": {
                "temperature": 0.4,
                "topP": 0.9,
                "topK": 40,
                "maxOutputTokens": 200,
            },
        }
        response = requests.post(
            self.API_URL_TEMPLATE.format(model=self.model),
            params={"key": self.api_key},
            json=payload,
            timeout=30,
        )
        if response.status_code != 200:
            logging.error(
                "Gemini API error: status=%s body=%s",
                response.status_code,
                response.text,
            )
            raise GeminiError(f"Gemini API error {response.status_code}: {response.text}")
        data = response.json()
        try:
            parts = data["candidates"][0]["content"]["parts"]
            combined_text = "\n".join(part.get("text", "") for part in parts if "text" in part).strip()
            if not combined_text:
                raise KeyError("Empty response text")
            return combined_text
        except (KeyError, IndexError) as exc:
            logging.error("Unexpected Gemini wrapup payload: %s", data)
            raise GeminiError("Gemini API returned an invalid wrapup payload") from exc

    def generate_problem_description(
        self,
        question: Dict[str, Any],
        language: str,
    ) -> str:
        system_prompt = (
            "You are generating a coding problem description for an interview. "
            "Return ONLY the problem description content (no markers, no markdown, no extra commentary)."
        )
        title = str(question.get("title", "") or "").strip()
        difficulty = str(question.get("difficulty", "") or "").strip()
        signature = ""
        signatures = question.get("signatures") if isinstance(question, dict) else None
        if isinstance(signatures, dict):
            signature = str(signatures.get(language, "") or "").strip()
        instructions = (
            "Write a clear problem statement and include EXACTLY ONE example with input and output. "
            "The example must be a general case (not an edge case) and include a brief walkthrough "
            "explaining why the output is correct.\n"
            "Include the required function signature verbatim.\n"
            "Do NOT include multiple examples. Do NOT include solution steps.\n"
            "Output as plain text with blank lines between sections.\n\n"
            f"Title: {title}\n"
            f"Difficulty: {difficulty}\n"
            f"Required signature: {signature}\n"
        )
        payload = {
            "systemInstruction": {
                "role": "system",
                "parts": [{"text": system_prompt}],
            },
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": instructions}],
                }
            ],
            "generationConfig": {
                "temperature": 0.3,
                "topP": 0.9,
                "topK": 40,
                "maxOutputTokens": 800,
            },
        }
        response = requests.post(
            self.API_URL_TEMPLATE.format(model=self.model),
            params={"key": self.api_key},
            json=payload,
            timeout=30,
        )
        if response.status_code != 200:
            logging.error(
                "Gemini API error: status=%s body=%s",
                response.status_code,
                response.text,
            )
            raise GeminiError(f"Gemini API error {response.status_code}: {response.text}")
        data = response.json()
        try:
            parts = data["candidates"][0]["content"]["parts"]
            combined_text = "\n".join(part.get("text", "") for part in parts if "text" in part).strip()
            if not combined_text:
                raise KeyError("Empty response text")
            return combined_text
        except (KeyError, IndexError) as exc:
            logging.error("Unexpected Gemini problem payload: %s", data)
            raise GeminiError("Gemini API returned an invalid problem payload") from exc

    def _parse_json_from_text(self, text: str) -> Dict[str, Any]:
        """Parse JSON from text, handling markdown code fences and extra whitespace."""
        cleaned = text.strip()
        # Remove markdown code fences if present
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```\w*\n?|```$", "", cleaned, flags=re.DOTALL).strip()
        # Extract JSON object between first { and last }
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            cleaned = cleaned[start:end + 1]
        return json.loads(cleaned)

    def _build_structured_payload(
        self,
        interview_state: Dict[str, Any],
        conversation: List[Dict[str, Any]],
        resume_text: str | None,
        user_message: str,
        current_code: str,
        code_changed: bool,
        time_in_phase: float,
        total_time: float,
        silence_duration: float,
        code_idle_duration: float,
        phase_signal: str,
        temperature: float,
    ) -> Dict[str, Any]:
        """Build payload with phase-aware prompting for structured interview"""

        current_phase = interview_state.get("current_phase", state.PHASE_INTRO_RESUME)
        language = interview_state.get("language", "python")
        problem_presented = interview_state.get("problem_presented", False)
        mode = interview_state.get("mode", "full")
        company_context = interview_state.get("company_context")
        ood_question = interview_state.get("ood_question")
        coding_question = interview_state.get("coding_question")
        code_evaluation = interview_state.get("code_evaluation") or {}
        coding_summary = interview_state.get("coding_summary") or {}
        optimization = interview_state.get("optimization") or {}
        candidate_name = str(interview_state.get("candidate_name") or "")

        normalized_code = self._normalize_editor_code(current_code)

        # Build comprehensive system prompt
        system_prompt = self._build_phase_prompt(
            current_phase,
            language,
            resume_text,
            candidate_name,
            normalized_code,
            code_changed,
            problem_presented,
            time_in_phase,
            total_time,
            silence_duration,
            code_idle_duration,
            phase_signal,
            mode,
            company_context,
            ood_question,
            coding_question,
            code_evaluation,
            coding_summary,
            optimization,
        )

        # Build conversation payload (limit to current phase turns)
        phase_start_index = int(interview_state.get("phase_turn_start_index", 0) or 0)
        conversation_payload = list(conversation[phase_start_index:])
        enriched_user_message = self._build_user_payload_with_editor_context(
            user_message=user_message,
            normalized_code=normalized_code,
        )
        conversation_payload.append(
            {
                "role": "user",
                "parts": [{"text": enriched_user_message}],
            }
        )

        payload: Dict[str, Any] = {
            "systemInstruction": {
                "role": "system",
                "parts": [{"text": system_prompt}],
            },
            "contents": conversation_payload,
            "generationConfig": {
                "temperature": temperature,
                "topP": 0.95,
                "topK": 40,
                "maxOutputTokens": 800,  # Increased to allow code responses
            },
        }
        return payload

    def decide_phase_transition(
        self,
        interview_state: Dict[str, Any],
        conversation: List[Dict[str, Any]],
        user_message: str,
        model_reply: str,
        time_in_phase: float,
        total_time: float,
        silence_duration: float,
        code_idle_duration: float,
    ) -> Optional[Dict[str, Any]]:
        """Use a lightweight model to evaluate whether the interview should transition phases."""

        payload = self._build_phase_decision_payload(
            interview_state=interview_state,
            conversation=conversation,
            user_message=user_message,
            model_reply=model_reply,
            time_in_phase=time_in_phase,
            total_time=total_time,
            silence_duration=silence_duration,
            code_idle_duration=code_idle_duration,
        )

        try:
            response = requests.post(
                self.API_URL_TEMPLATE.format(model=self.phase_model),
                params={"key": self.api_key},
                json=payload,
                timeout=20,
            )
        except RequestException as exc:  # type: ignore[name-defined]
            logging.error("Phase decision request failed: %s", exc)
            return None

        if response.status_code != 200:
            logging.warning(
                "Phase decision model error: status=%s body=%s",
                response.status_code,
                response.text,
            )
            return None

        data = response.json()
        try:
            parts = data["candidates"][0]["content"]["parts"]
            decision_text = " ".join(part.get("text", "") for part in parts if "text" in part).strip()
        except (KeyError, IndexError) as exc:
            logging.warning("Phase decision payload missing text: %s", data)
            return None

        if decision_text:
            logging.info("[PHASE DECISION RAW] %s", decision_text)

        decision = self._parse_phase_decision(decision_text)
        if not decision:
            logging.warning("Unable to parse phase decision JSON: %s", decision_text)
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
        interview_state: Dict[str, Any],
        conversation: List[Dict[str, Any]],
        user_message: str,
        current_code: str,
    ) -> bool:
        """Decide if the candidate appears finished and tests should run."""
        current_phase = interview_state.get("current_phase", state.PHASE_INTRO_RESUME)
        if current_phase != state.PHASE_CODING:
            return False

        snippet = self._format_conversation_snippet(conversation)
        payload = self._build_test_gate_payload(
            user_message=user_message,
            current_code=current_code,
            conversation_snippet=snippet,
        )
        try:
            response = requests.post(
                self.API_URL_TEMPLATE.format(model=self.model),
                params={"key": self.api_key},
                json=payload,
                timeout=20,
            )
        except RequestException as exc:  # type: ignore[name-defined]
            logging.error("Test gate request failed: %s", exc)
            return False

        if response.status_code != 200:
            logging.warning(
                "Test gate model error: status=%s body=%s",
                response.status_code,
                response.text,
            )
            return False

        data = response.json()
        try:
            parts = data["candidates"][0]["content"]["parts"]
            decision_text = " ".join(part.get("text", "") for part in parts if "text" in part).strip()
        except (KeyError, IndexError):
            logging.warning("Test gate payload missing text: %s", data)
            return False

        decision = self._parse_phase_decision(decision_text)
        if not decision:
            logging.warning("Unable to parse test gate JSON: %s", decision_text)
            return False

        flag = decision.get("should_run_tests", False)
        logging.info("[TEST GATE MODEL] raw=%s parsed=%s", decision_text, flag)
        if isinstance(flag, str):
            return flag.lower() == "true"
        return bool(flag)

    def _build_test_gate_payload(
        self,
        user_message: str,
        current_code: str,
        conversation_snippet: str,
    ) -> Dict[str, Any]:
        system_prompt = (
            "You are the coding interview test gate.\n"
            "Decide if the candidate appears finished and the solution should be tested now.\n"
            "Be lenient: return true if they imply they are done, "
            "ask to run/verify tests, or the solution looks complete.\n"
            "Return strict JSON only, no extra text."
        )
        context = {
            "recent_candidate_message": user_message,
            "conversation_excerpt": conversation_snippet,
            "current_code": current_code,
        }
        instructions = (
            "Return JSON with the following keys:\n"
            '{ "should_run_tests": true | false, "reason": "concise explanation" }\n'
            "If they are still implementing or discussing approach, return false."
        )
        payload: Dict[str, Any] = {
            "systemInstruction": {
                "role": "system",
                "parts": [{"text": system_prompt}],
            },
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": instructions + "\n\nCONTEXT:\n" + json.dumps(context, ensure_ascii=False)}],
                }
            ],
            "generationConfig": {
                "temperature": 0.1,
                "topP": 0.8,
                "topK": 32,
                "maxOutputTokens": 120,
            },
        }
        return payload

    def _build_phase_decision_payload(
        self,
        interview_state: Dict[str, Any],
        conversation: List[Dict[str, Any]],
        user_message: str,
        model_reply: str,
        time_in_phase: float,
        total_time: float,
        silence_duration: float,
        code_idle_duration: float,
    ) -> Dict[str, Any]:
        current_phase = interview_state.get("current_phase", state.PHASE_INTRO_RESUME)
        mode = interview_state.get("mode", "full")
        time_in_phase = float(time_in_phase or 0.0)
        total_time = float(total_time or 0.0)
        problem_presented = bool(interview_state.get("problem_presented", False))

        snippet = self._format_conversation_snippet(conversation)

        context = {
            "current_phase": current_phase,
            "mode": mode,
            "time_in_phase_minutes": round(time_in_phase, 2),
            "total_time_minutes": round(total_time, 2),
            "minutes_until_coding_window": round(max(0.0, 5.0 - total_time), 2),
            "minutes_until_questions_window": round(max(0.0, 35.0 - total_time), 2),
            "minutes_remaining_total": round(max(0.0, 40.0 - total_time), 2),
            "seconds_since_candidate_spoke": round(float(silence_duration or 0.0), 1),
            "seconds_since_code_change": round(float(code_idle_duration or 0.0), 1),
            "problem_already_presented": problem_presented,
            "recent_candidate_message": user_message,
            "model_reply_being_sent": model_reply,
            "conversation_excerpt": snippet,
            "phase_options": [
                state.PHASE_INTRO_RESUME,
                state.PHASE_CODING,
                state.PHASE_QUESTIONS,
            ],
        }

        system_prompt = (
            "You are the phase coordinator for a live technical interview.\n"
            "Decide whether the interviewer should switch phases based on the conversation flow, "
            "the time spent in the current phase, and whether the candidate appears ready.\n"
            "Only recommend a transition when it feels natural and consistent with the discussion.\n"
            "If the current dialogue still fits the phase objectives, do not transition yet.\n"
            "For full mode, enforce hard timing windows: do not transition to coding before 5 minutes total, "
            "and transition to questions around 35 minutes total.\n"
            "Respond ONLY with strict JSON. Do not include any extra commentary."
        )

        instructions = (
            "Return JSON with the following keys:\n"
            '{\n'
            '  "should_transition": true | false,\n'
            '  "next_phase": "intro" | "resume" | "coding" | "questions" | null,\n'
            '  "reason": "concise explanation",\n'
            '  "confidence": number between 0 and 1,\n'
            '  "mark_problem_presented": true | false\n'
            '}\n'
            "If should_transition is false, set next_phase to null. "
            "Only set mark_problem_presented true if the coding problem was clearly introduced."
        )

        payload: Dict[str, Any] = {
            "systemInstruction": {
                "role": "system",
                "parts": [{"text": system_prompt}],
            },
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": instructions + "\n\nCONTEXT:\n" + json.dumps(context, ensure_ascii=False),
                        }
                    ],
                }
            ],
            "generationConfig": {
                "temperature": 0.1,
                "topP": 0.8,
                "topK": 32,
                "maxOutputTokens": 200,
            },
        }
        return payload

    def _format_conversation_snippet(
        self,
        conversation: List[Dict[str, Any]],
        max_turns: int = 8,
    ) -> str:
        if not conversation:
            return ""

        if max_turns and max_turns > 0:
            recent_turns = conversation[-max_turns:]
        else:
            recent_turns = conversation
        lines: List[str] = []
        for turn in recent_turns:
            role = turn.get("role")
            parts = turn.get("parts", [])
            text_segments = []
            for part in parts:
                if isinstance(part, dict):
                    text_segments.append(part.get("text", ""))
            text = " ".join(text_segments).strip()
            if not text:
                continue
            speaker = "Interviewer" if role == "model" else "Candidate"
            lines.append(f"{speaker}: {text}")
        return "\n".join(lines)

    def _parse_phase_decision(self, text: str) -> Optional[Dict[str, Any]]:
        if not text:
            return None
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if not match:
                return None
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return None

    def _build_phase_prompt(
        self,
        current_phase: str,
        language: str,
        resume_text: Optional[str],
        candidate_name: str,
        current_code: str,
        code_changed: bool,
        problem_presented: bool,
        time_in_phase: float,
        total_time: float,
        silence_duration: float,
        code_idle_duration: float,
        phase_signal: str,
        mode: str = "full",
        company_context: Optional[Dict[str, str]] = None,
        ood_question: Optional[Dict[str, Any]] = None,
        coding_question: Optional[Dict[str, Any]] = None,
        code_evaluation: Optional[Dict[str, Any]] = None,
        coding_summary: Optional[Dict[str, Any]] = None,
        optimization: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Build phase-specific system prompt"""

        # Base prompt - applies to all phases
        if mode == "coding_only":
            base_prompt = """You are a senior software engineer conducting a LIVE VOICE coding interview.
This is a CODING-ONLY session (35-40 minutes). Jump straight into the problem.

🎯 SPEECH OUTPUT - THIS IS CRITICAL:
Your responses will be SPOKEN ALOUD by a text-to-speech system to the candidate.
- Think like you're having a phone conversation, not writing an email
- Write ONLY what should be spoken - no formatting, no markdown, no visual cues
- NEVER use: *, #, _, **, `, or any formatting symbols
- NEVER use filler words: "um", "uh", "hmm", "well", "so"
- Say technical terms plainly: "use a for loop" NOT "use a *for loop*"

💬 BE EXTREMELY CONCISE:
- DEFAULT LENGTH: 1-2 sentences (10-20 words total)
- Most responses should be under 15 words
- Only go longer (3-4 sentences) when:
  * Explaining a complex concept they specifically asked about
  * Presenting a coding problem (use [PROBLEM_START]...[PROBLEM_END] markers)
  * Giving feedback on completed solution
- If you find yourself writing more than 4 sentences, STOP and cut it down
- Every word costs time - be ruthless about brevity

🗣️ NATURAL CONVERSATION:
- Use contractions: "I'm", "you're", "that's", "let's"
- ONE thought per response - don't chain multiple ideas
- Let them speak - don't fill every silence
- Sound like a real person, not a textbook
- Vary your phrasing - avoid repetitive patterns
- Never do long step-by-step simulations or traces out loud
- If you must reference a case, summarize in 1-2 sentences
- Do NOT repeat the same question over and over; rephrase once, then move on
- Do NOT get stuck on one question or one experience for too long

"""
        else:
            base_prompt = """You are a senior software engineer conducting a LIVE VOICE technical interview.
This is a single-context session. You ONLY handle the current context and do NOT assume anything beyond it.

🎯 SPEECH OUTPUT - THIS IS CRITICAL:
Your responses will be SPOKEN ALOUD by a text-to-speech system to the candidate.
- Think like you're having a phone conversation, not writing an email
- Write ONLY what should be spoken - no formatting, no markdown, no visual cues
- NEVER use: *, #, _, **, `, or any formatting symbols
- NEVER use filler words: "um", "uh", "hmm", "well", "so"
- Say technical terms plainly: "use a for loop" NOT "use a *for loop*"

💬 BE EXTREMELY CONCISE:
- DEFAULT LENGTH: 1-2 sentences (10-25 words total)
- Most responses should be under 20 words
- Only go longer (3-5 sentences) when:
  * Explaining a complex concept they specifically asked about
  * Presenting a coding problem (use [PROBLEM_START]...[PROBLEM_END] markers)
  * Giving detailed feedback on their work
- If you find yourself writing more than 5 sentences, STOP and cut it down
- Every word costs time - be ruthless about brevity

🗣️ NATURAL CONVERSATION:
- Use contractions: "I'm", "you're", "that's", "let's"
- ONE thought or question per response
- Let them speak - don't fill every silence
- Sound like a real person, not a textbook
- Vary your phrasing - avoid repetitive patterns
- Never do long step-by-step simulations or traces out loud
- If you must reference a case, summarize in 1-2 sentences
- Do NOT repeat the same question over and over; rephrase once, then move on
- Do NOT get stuck on one question or one experience for too long

"""

        # Interview structure context
        if mode == "coding_only":
            structure = f"""SESSION TYPE: CODING-ONLY MODE (35-40 minutes)
Present a coding problem immediately and keep the conversation focused on solving it.

CURRENT STATE:
- Time elapsed: {total_time:.1f} minutes
- Seconds since candidate spoke: {silence_duration:.0f}s
- Seconds since code changed: {code_idle_duration:.0f}s

"""
        elif mode == "full":
            phase_min = {
                state.PHASE_INTRO_RESUME: 5.0,
                state.PHASE_CODING: 0.0,
                state.PHASE_QUESTIONS: 0.0,
            }
            phase_max = {
                state.PHASE_INTRO_RESUME: 10.0,
                state.PHASE_CODING: 30.0,
                state.PHASE_QUESTIONS: 5.0,
            }
            min_minutes = phase_min.get(current_phase, 0.0)
            max_minutes = phase_max.get(current_phase, 0.0)
            remaining = max(0.0, max_minutes - time_in_phase) if max_minutes else 0.0
            structure = f"""CURRENT CONTEXT:
- Minutes elapsed in this segment: {time_in_phase:.1f}
- Minutes remaining in this segment: {remaining:.1f}
- Minimum minutes before ending this segment: {min_minutes:.1f}
- Seconds since candidate spoke: {silence_duration:.0f}s
- Seconds since code changed: {code_idle_duration:.0f}s

"""
        else:
            structure = f"""SESSION TYPE: OOD MODE (40 minutes total)

CURRENT STATE:
- Time in this segment: {time_in_phase:.1f} minutes
- Total interview time: {total_time:.1f} minutes
- Seconds since candidate spoke: {silence_duration:.0f}s
- Seconds since code changed: {code_idle_duration:.0f}s

"""

        company_lines: List[str] = []
        if company_context:
            company_value = (company_context.get("company") or "").strip()
            role_value = (company_context.get("role") or "").strip()
            details_value = (company_context.get("details") or "").strip()
            if company_value:
                company_lines.append(f"Company: {company_value}")
            if role_value:
                company_lines.append(f"Role: {role_value}")
            if details_value:
                company_lines.append(f"Notes: {details_value}")

        if company_lines:
            structure += "COMPANY & ROLE CONTEXT:\n" + "\n".join(f"- {line}" for line in company_lines) + "\n\n"

        # Phase-specific instructions
        if current_phase == state.PHASE_INTRO_RESUME:
            resume_context = f"\n\nCANDIDATE'S RESUME:\n{resume_text}\n" if resume_text else "\n[No resume provided]"
            interviewer_name = "Thomas"
            interviewer_company = "Google"
            if company_context:
                company_value = (company_context.get("company") or "").strip()
                if company_value:
                    interviewer_company = company_value
            name_line = f"Candidate name: {candidate_name}" if candidate_name else "Candidate name: [unknown]"
            interviewer_line = f"Interviewer name: {interviewer_name}\nInterviewer company: {interviewer_company}"

            # Build phase ending instructions based on signal
            if phase_signal == "keep_going":
                ending_instructions = """
═══════════════════════════════════════════════════════════════════
CONTINUE THE DISCUSSION - DO NOT END YET
═══════════════════════════════════════════════════════════════════
The current signal is: "keep_going"

This means you MUST continue the resume discussion. DO NOT end this discussion yet.

WHAT YOU MUST DO:
✓ Continue asking questions about their resume and experience
✓ Ask one follow-up question based on what they just said
✓ Keep the conversation flowing naturally
✓ Stay engaged and curious about their background

WHAT YOU MUST NOT DO:
✗ Do NOT mention any upcoming activities
✗ Do NOT say "let's move on"
✗ Do NOT append [SECTION_COMPLETE]
✗ Do NOT wrap up or conclude the conversation
✗ Do NOT present any programming problems

CONTINUE THE CONVERSATION by asking another relevant question about their experience.
═══════════════════════════════════════════════════════════════════
"""
            elif phase_signal == "wrap_up_soon":
                ending_instructions = """
═══════════════════════════════════════════════════════════════════
WRAP UP SOON - PLAN TO END
═══════════════════════════════════════════════════════════════════
The current signal is: "wrap_up_soon"

This means you should START WRAPPING UP within the next 2-3 turns.

WHAT YOU MUST DO:
✓ Ask at most one more focused question
✓ Avoid starting new long topics
✓ Prepare to end the discussion naturally

WHAT YOU MUST NOT DO:
✗ Do NOT repeat prior questions
✗ Do NOT introduce new deep topics
✗ Do NOT append [SECTION_COMPLETE] in this response unless it already feels complete

NEXT STEP:
- If their answer is sufficient, end the discussion on your next turn with [SECTION_COMPLETE]
═══════════════════════════════════════════════════════════════════
"""
            elif phase_signal == "must_end_now":
                ending_instructions = """
═══════════════════════════════════════════════════════════════════
ENDING NOW - YOU MUST END
═══════════════════════════════════════════════════════════════════
The current signal is: "must_end_now"

This means you MUST end this discussion immediately in your next response.

WHAT YOU MUST DO RIGHT NOW:
1. Provide a brief transition sentence (OPTIONAL, one sentence max):
   Example: "Thanks for sharing your background."
   Example: "Great, I have a good sense of your experience."

2. IMMEDIATELY append this exact token on a NEW line:
   [SECTION_COMPLETE]

EXAMPLE RESPONSE FORMAT:
"Thanks for walking me through your experience.
[SECTION_COMPLETE]"

OR simply:
"[SECTION_COMPLETE]"

CRITICAL RULES:
✓ The [SECTION_COMPLETE] token MUST be on its own line
✓ You MUST include [SECTION_COMPLETE] in this response
✓ Do NOT ask any more questions
✓ Do NOT continue the resume discussion
✓ Keep any transition text to ONE sentence maximum

DO THIS NOW - END THE DISCUSSION.
═══════════════════════════════════════════════════════════════════
"""
            else:  # end_if_you_want
                ending_instructions = """
═══════════════════════════════════════════════════════════════════
ENDING - OPTIONAL (YOUR CHOICE)
═══════════════════════════════════════════════════════════════════
The current signal is: "end_if_you_want"

This means you have the CHOICE to either continue OR end this discussion based on the conversation flow.

OPTION 1: CONTINUE THE DISCUSSION
If you feel there are still important topics to cover about their resume:
✓ Ask another question about their experience
✓ Follow up on something they mentioned
✓ Explore their technical background more deeply
✓ Do NOT append [SECTION_COMPLETE]

OPTION 2: END THE DISCUSSION
If you feel you've adequately covered their background and experience:
1. Optionally say a brief transition (one sentence max):
   Example: "Thanks, I have a good sense of your background."
2. APPEND this exact token on a NEW line:
   [SECTION_COMPLETE]

HOW TO DECIDE:
- Have you asked about their recent projects? If NO, continue
- Have you asked about their technical skills? If NO, continue
- Have you asked about a technical challenge they solved? If NO, continue
- Have you covered the main points on their resume? If YES, you can end
- Does the conversation feel complete? If YES, you can end

EXAMPLE - CONTINUING:
"What technologies are you most comfortable working with?"
(Do NOT include [SECTION_COMPLETE])

EXAMPLE - ENDING:
"Thanks for sharing your background.
[SECTION_COMPLETE]"

Your choice - evaluate the conversation and decide.
═══════════════════════════════════════════════════════════════════
"""

            phase_instructions = f"""
═══════════════════════════════════════════════════════════════════
INTRO + RESUME DISCUSSION
═══════════════════════════════════════════════════════════════════

WHAT IS THIS CONTEXT:
This is the current session context. Your job is to:
1. Introduce yourself to the candidate
2. Get to know their background by reviewing their resume
3. Ask questions about their experience, projects, and skills
4. Build rapport and set context for the technical discussion

This discussion typically lasts up to 6 minutes. You will receive signals about when to end.

CONTEXT:
{interviewer_line}
{name_line}{resume_context}

═══════════════════════════════════════════════════════════════════
FIRST RESPONSE ONLY - OPENING SCRIPT
═══════════════════════════════════════════════════════════════════

If this is the very first turn of the interview (conversation is empty or just starting):

STEP 1: Greet and introduce yourself (ONE sentence)
Say: "Hi, I'm {interviewer_name}, senior engineer at {interviewer_company}."

STEP 2: Start with an open question (ONE sentence)
Say: "Tell me about yourself."

TOTAL: Exactly 2 sentences, under 20 words total.

Example complete opening:
"Hi, I'm {interviewer_name}, senior engineer at {interviewer_company}. Tell me about yourself."

DO NOT:
✗ Do NOT say multiple greetings
✗ Do NOT explain the interview structure
✗ Do NOT mention internal rules or time limits
✗ Do NOT present programming problems
✗ Keep it short and natural

═══════════════════════════════════════════════════════════════════
DURING THE CONVERSATION - RESPONSE STYLE
═══════════════════════════════════════════════════════════════════

YOUR ROLE: You are a friendly, curious interviewer learning about the candidate.

LENGTH: Keep responses concise but flexible
- Target: 1-2 sentences per turn
- Allow 3-4 sentences only when you need to clarify or follow up on multiple details
- Keep each sentence short and focused
- Listen MORE than you talk

CONVERSATION FLOW:
1. Listen to what the candidate says
2. Ask ONE follow-up question based on their response
3. Let them elaborate
4. Ask another question
5. Repeat
6. If a topic stalls, rephrase once, then switch to another resume item

DEPTH GUIDANCE:
- Use details from their last response to dig deeper (scope, impact, tradeoffs, decisions, ownership)
- Ask for specifics: "What part did you own?" "What was the impact?" "Why did you choose that approach?"
- If they mention a project or tech, follow the thread for 1-2 more turns before switching topics

TONE:
✓ Conversational and natural
✓ Curious and engaged
✓ Professional but friendly
✓ Use contractions: "I'm", "you're", "that's"

WHAT TO ASK ABOUT:
✓ Recent projects and their impact
✓ Technical challenges they've solved
✓ Technologies and tools they're comfortable with
✓ What they're excited to work on next
✓ Specific items mentioned on their resume
✓ Their role and contributions in past projects

EXAMPLE GOOD QUESTIONS:
- "Tell me about your most recent project."
- "What was the biggest technical challenge you faced there?"
- "What technologies did you use?"
- "What was your specific role on the team?"
- "What are you most comfortable working with?"
- "What kind of work are you excited about next?"

WHAT NOT TO DO:
✗ Do NOT ask multiple questions at once
✗ Do NOT present programming problems or technical exercises
✗ Do NOT mention any upcoming activities
✗ Do NOT give long explanations or lectures
✗ Do NOT use filler words like "um", "well", "so"
✗ Do NOT use markdown formatting (*, #, _, etc.)

RESPONSE EXAMPLES:

Candidate: "I worked on a payment processing system at my last company."
Good Response: "What was your role in building that system?"
Bad Response: "That's interesting! Payment systems are complex. What technologies did you use, and what were some of the challenges you faced?"

Candidate: "I mostly used Python and JavaScript."
Good Response: "What did you build with Python?"
Bad Response: "Great! Python and JavaScript are both very useful languages. How long have you been using them?"

═══════════════════════════════════════════════════════════════════
QUESTIONS TO COVER
═══════════════════════════════════════════════════════════════════

Try to touch on these topics during the conversation:
1. Recent projects and their business impact
2. Technical challenges they've solved and how
3. Technologies and programming languages they're comfortable with
4. What kind of work excites them or what they want to do next
5. Specific experiences mentioned on their resume

You don't need to ask all of these in order. Follow the natural flow of conversation.

{ending_instructions}

═══════════════════════════════════════════════════════════════════
CRITICAL REMINDERS
═══════════════════════════════════════════════════════════════════

✓ This is ONLY the resume discussion
✓ Do NOT present programming problems in this discussion
✓ Do NOT mention interview structure or internal steps
✓ Keep responses SHORT - one question at a time
✓ Pay attention to the current signal above - it tells you when to end
✓ To end the discussion, append [SECTION_COMPLETE] on a NEW line
✓ Do NOT append [SECTION_COMPLETE] unless instructed to end this discussion

═══════════════════════════════════════════════════════════════════
"""

        elif current_phase == state.PHASE_CODING:
            language_display = {
                "python": "Python",
                "java": "Java",
                "c": "C",
                "cpp": "C++"
            }.get(language, language)
            if mode == "full":
                name_line = "Candidate name: [unknown]"
            else:
                name_line = f"Candidate name: {candidate_name}" if candidate_name else "Candidate name: [unknown]"

            # Always show code; write access only in final 2 minutes if tests are failing
            max_phase_time = 30.0
            time_remaining = max(0.0, max_phase_time - time_in_phase)
            tests_passing = (code_evaluation and str(code_evaluation.get("status", "")).strip() == "pass")
            can_write_code = (time_remaining <= 2.0 and not tests_passing)

            # Show code without markdown formatting (model should not output markdown)
            if current_code.strip():
                code_display = f"\n=== CURRENT CODE IN EDITOR ===\n{current_code}\n=== END OF CODE ===\n"
            else:
                code_display = "\n=== CURRENT CODE IN EDITOR ===\n[No code written yet]\n=== END OF CODE ===\n"

            visibility_instructions = f"""CODE VISIBILITY & EDITOR ACCESS:
- Time remaining: {time_remaining:.1f} minutes
- Tests status: {"PASSING" if tests_passing else "FAILING"}
- You can SEE the code at all times, but you CANNOT write or edit it
- The snapshot between the 'CURRENT CODE IN EDITOR' markers above is EXACTLY what the candidate has typed
- Only if time remaining <= 2 minutes AND tests are failing, you MAY write code to help
- To write code (only when allowed), use: [CODE_START]...code...[CODE_END]

"""

            evaluation_block = ""
            if code_evaluation:
                status = str(code_evaluation.get("status", "")).strip()
                summary = str(code_evaluation.get("summary", "")).strip()
                if status:
                    evaluation_block = (
                        "INTERNAL CODE EVALUATION (DO NOT MENTION TESTS OR COUNTS):\n"
                        f"- Status: {status}\n"
                    )
                    if summary:
                        evaluation_block += f"- Summary: {summary}\n"
                    if status == "pass":
                        evaluation_block += (
                            "Guidance: Treat the solution as correct. Do not question correctness. "
                            "Ask only about optimizations, complexity, or clarifications.\n"
                        )
                    elif status in {"fail", "error"}:
                        evaluation_block += (
                            "Guidance: The solution has issues. Focus on debugging and fixing the failures. "
                            "Do NOT ask about complexity, edge cases, or optimizations until tests pass.\n"
                        )
                    evaluation_block += "Use this only to guide your feedback.\n\n"

            optimization_block = ""
            if optimization:
                opt_status = str(optimization.get("status", "")).strip()
                opt_feedback = str(optimization.get("feedback", "")).strip()
                if opt_status:
                    optimization_block = (
                        "OPTIMIZATION REVIEW (DO NOT MENTION THIS LABEL):\n"
                        f"- Status: {opt_status}\n"
                    )
                    if opt_feedback:
                        optimization_block += f"- Feedback: {opt_feedback}\n"
                    if opt_status == "needs_improvement":
                        optimization_block += (
                            "Guidance: Ask for improvements and tradeoffs. Push them to optimize.\n"
                        )
                    elif opt_status == "optimized":
                        optimization_block += (
                            "Guidance: The solution is optimized. Wrap up in a few turns.\n"
                        )
                    optimization_block += "Use this only to guide your feedback.\n\n"

            # Build phase ending instructions based on signal
            if phase_signal == "must_end_now":
                timing_block = """ENDING NOW - TIME IS UP:
- You MUST end now
- Provide brief closing remarks if needed
- Append this token on a NEW line: [SECTION_COMPLETE]

"""
            else:  # end_if_you_want (coding has no minimum time)
                timing_block = """ENDING:
- You can end when the coding problem is complete and tests pass
- To end, append this token on a NEW line: [SECTION_COMPLETE]
- Do NOT end if tests are still failing (unless time runs out)

"""
            completion_window = ""
            if mode == "coding_only":
                completion_window = """AFTER TESTS PASS:
- End the interview within 10 turns
- Ask at most 1-2 brief wrap-up questions
- Then append [SECTION_COMPLETE] on a NEW line

"""
            elif mode == "full":
                completion_window = """AFTER TESTS PASS:
- Wrap up within 10 turns
- Ask at most 1-2 brief wrap-up questions
- Then append [SECTION_COMPLETE] on a NEW line

"""

            selected_question_block = ""
            if coding_question and coding_question.get("title"):
                title = str(coding_question.get("title", "")).strip()
                difficulty = str(coding_question.get("difficulty", "")).strip()
                acceptance = str(coding_question.get("acceptance", "")).strip()
                signature = ""
                signatures = coding_question.get("signatures") if isinstance(coding_question, dict) else None
                if isinstance(signatures, dict):
                    signature = str(signatures.get(language, "")).strip()
                details = [f"- Title: {title}"]
                if difficulty:
                    details.append(f"- Difficulty: {difficulty}")
                if acceptance:
                    details.append(f"- Acceptance: {acceptance}")
                if signature:
                    details.append(f"- Required function signature ({language_display}): {signature}")
                selected_question_block = "USE THIS QUESTION FROM THE BANK:\n" + "\n".join(details) + "\nDo NOT choose another question.\n\n"

            if not problem_presented:
                # Different instructions for coding_only mode
                if mode == "coding_only":
                    phase_instructions = f"""CODING-ONLY MODE - PRESENT PROBLEM IMMEDIATELY:

LANGUAGE: {language_display}
{completion_window}{code_display}{visibility_instructions}{evaluation_block}{optimization_block}
{selected_question_block}
PROBLEM DESCRIPTION (READ VERBATIM INSIDE MARKERS):
{coding_question.get("problem_description", "") if isinstance(coding_question, dict) else ""}

FIRST RESPONSE ONLY:
1. Brief greeting: "Hi! Ready for a coding problem?"
2. Present problem in [PROBLEM_START]...[PROBLEM_END] markers
   - Use the provided PROBLEM DESCRIPTION verbatim
   - Do NOT alter or add to it
3. Outside markers, keep spoken text under 15 words: "Take a minute to think about your approach."
4. Ask: "Any questions on the problem?"

AFTER THEY UNDERSTAND:
- Guide approach discussion BEFORE coding
- Short prompts: "What's your approach?" "What about edge cases?" "Time complexity?"
- Do NOT repeat the same question; rephrase once, then switch to a different probe

═══════════════════════════════════════════════════════════════════
STRICT NO-SOLUTION POLICY - READ CAREFULLY:
═══════════════════════════════════════════════════════════════════
You are an INTERVIEWER, not a teacher or tutor. Your job is to EVALUATE, not to HELP.

ABSOLUTELY FORBIDDEN - DO NOT:
❌ Give explicit algorithms: "Use two pointers" "Use a hash map" "Try dynamic programming"
❌ Give implementation hints: "Store it in a dictionary" "Loop from the end" "Use a sliding window"
❌ Suggest data structures: "What about using a heap?" "Maybe try a set?"
❌ Give complexity hints: "This needs to be O(n)" "You need O(1) lookup"
❌ Explain solutions: "The trick is to..." "The key insight is..." "You should..."
❌ Write ANY code (you do not have write access)
❌ Say "good idea" or validate their approach (unless it's actually correct)

WHAT YOU CAN DO:
✅ Ask questions: "What's your approach?" "How would you handle duplicates?" "What's the time complexity?"
✅ Point out logical gaps: "What if the array is empty?" "Does that handle negative numbers?"
✅ Challenge their thinking: "Walk me through an example" "What happens when N is 1?"
✅ Stay silent and let them think
✅ Acknowledge: "Okay" "I see" "Keep going"
✅ If they ask for help, you may give a gentle, high-level hint or direction (not a full solution)
✅ Hints must be abstract (e.g., "Consider ordering", "Think about constraints") and never name a specific data structure or algorithm

IF THEY ASK FOR HELP:
- "What part are you stuck on?"
- "Walk me through your thinking"
- "What have you tried so far?"
 - If still stuck and they explicitly ask for a hint, give ONE high-level hint only

IF THEY BEG FOR THE ANSWER:
- "I can't give you the solution - what ideas do you have?"
- "Let's think through it together - where would you start?"

REMEMBER: Struggling IS the interview. Let them struggle.
═══════════════════════════════════════════════════════════════════

DEFAULT: 1-2 sentence responses. Let them work."""
                else:
                    phase_instructions = f"""CODING SESSION - PROBLEM PRESENTATION:

{name_line}
LANGUAGE: {language_display}
{timing_block}{completion_window}{code_display}{visibility_instructions}{evaluation_block}{optimization_block}
{selected_question_block}
PROBLEM DESCRIPTION (READ VERBATIM INSIDE MARKERS):
{coding_question.get("problem_description", "") if isinstance(coding_question, dict) else ""}

YOUR TASK NOW:
1. Brief greeting: "Hi! Let's do a coding problem."
2. Present the problem inside [PROBLEM_START]...[PROBLEM_END] markers:
   - Copy the provided PROBLEM DESCRIPTION verbatim
   - Do NOT add or remove examples
   - Ask if they have clarifying questions
   - Do not read the examples out loud; keep spoken text brief outside the markers

3. GUIDE THEM TO DISCUSS APPROACH FIRST:
   - Don't let them jump to coding immediately
   - Make them explain their approach
   - Ask about edge cases
   - Discuss time/space complexity
   - Do NOT repeat the same question; rephrase once, then switch to a different probe

═══════════════════════════════════════════════════════════════════
STRICT NO-SOLUTION POLICY - READ CAREFULLY:
═══════════════════════════════════════════════════════════════════
You are an INTERVIEWER, not a teacher or tutor. Your job is to EVALUATE, not to HELP.

ABSOLUTELY FORBIDDEN - DO NOT:
❌ Give explicit algorithms: "Use two pointers" "Use a hash map" "Try dynamic programming"
❌ Give implementation hints: "Store it in a dictionary" "Loop from the end" "Use a sliding window"
❌ Suggest data structures: "What about using a heap?" "Maybe try a set?"
❌ Give complexity hints: "This needs to be O(n)" "You need O(1) lookup"
❌ Explain solutions: "The trick is to..." "The key insight is..." "You should..."
❌ Write ANY code unless you have write access (see editor access policy above)
❌ Say "good idea" or validate their approach (unless it's actually correct)

WHAT YOU CAN DO:
✅ Ask probing questions: "What's your approach?" "How would you handle duplicates?" "What's the time complexity?"
✅ Point out logical gaps: "What if the array is empty?" "Does that handle negative numbers?"
✅ Challenge their thinking: "Walk me through an example" "What happens when N is 1?"
✅ Stay silent and let them think
✅ Acknowledge without helping: "Okay" "I see" "Keep going"
✅ If they ask for help, you may give a gentle, high-level hint or direction (not a full solution)
✅ Hints must be abstract (e.g., "Consider ordering", "Think about constraints") and never name a specific data structure or algorithm

IF THEY ASK FOR HELP:
- "What part are you stuck on?"
- "Walk me through your thinking"
- "What have you tried so far?"
 - If still stuck and they explicitly ask for a hint, give ONE high-level hint only

IF THEY BEG FOR THE ANSWER:
- "I can't give you the solution - what ideas do you have?"
- "This is your interview - I need to see how you approach it"
- "Let's think through it together - where would you start?"

IF THEY ASK YOU TO WRITE CODE:
- "I can see your code, but I can't write for you"
- Only in the last 2 minutes with failing tests can you write code

REMEMBER: Struggling IS the interview. Let them struggle. Silence is okay.
═══════════════════════════════════════════════════════════════════

REMEMBER: After presenting, I will mark problem as presented."""

            else:
                phase_instructions = f"""CODING SESSION - ACTIVE SOLVING:

{name_line}
LANGUAGE: {language_display}
{timing_block}{completion_window}{code_display}{visibility_instructions}{evaluation_block}{optimization_block}

═══════════════════════════════════════════════════════════════════
STRICT NO-SOLUTION POLICY - READ CAREFULLY:
═══════════════════════════════════════════════════════════════════
You are an INTERVIEWER, not a teacher or tutor. Your job is to EVALUATE, not to HELP.

ABSOLUTELY FORBIDDEN - DO NOT:
❌ Give explicit algorithms: "Use two pointers" "Use a hash map" "Try dynamic programming"
❌ Give implementation hints: "Store it in a dictionary" "Loop from the end" "Use a sliding window"
❌ Suggest data structures: "What about using a heap?" "Maybe try a set?"
❌ Give complexity hints: "This needs to be O(n)" "You need O(1) lookup"
❌ Explain solutions: "The trick is to..." "The key insight is..." "You should..."
❌ Point out specific bugs in code you can't see
❌ Say "good idea" or validate approaches (unless actually correct)

WHAT YOU CAN DO:
✅ Ask questions: "What's your approach?" "How does that handle edge cases?" "What's the complexity?"
✅ Challenge: "Walk me through an example" "What if the input is empty?" "Does that work for negatives?"
✅ Listen: Stay quiet and let them code
✅ Acknowledge: "Okay" "I see" "Keep going"
✅ Check in: "How's it going?" (every 2-3 minutes)
✅ Vary your questions; rephrase once if needed, then move on
✅ If tests are failing or there is an error, focus on fixing it; do NOT ask about complexity or edge cases until tests pass
✅ If they ask for help, you may give a gentle, high-level hint or direction (not a full solution)
✅ Hints must be abstract (e.g., "Consider ordering", "Think about constraints") and never name a specific data structure or algorithm

CODE EDITOR ACCESS:
{visibility_instructions.strip()}

IF THEY ASK FOR HELP:
- "What part are you stuck on?"
- "Talk through your approach"
- "What have you tried?"
 - If still stuck and they explicitly ask for a hint, give ONE high-level hint only

IF THEY ASK FOR THE ANSWER:
- "I can't give you the solution"
- "What ideas do you have?"
- "This is your interview - keep working"

IF THEY'RE SILENT/STUCK:
- Wait 30+ seconds before speaking
- Then: "What are you thinking?" or "Where are you stuck?"
- Don't rescue them immediately

IF TIME REMAINING <= 2 MINUTES AND TESTS FAILING:
- You now have write access (see above)
- You MAY write code using [CODE_START]...[CODE_END]
- Format: "Let me show you. [CODE_START]...code...[CODE_END] This uses..."

PROBLEM SWITCH REQUESTS:
- "We need to finish this problem" or "Let's stick with this one"
- Do NOT end when they ask to switch

ENDING NOW:
- ONLY end if: (1) Tests pass, OR (2) Time is up (30 minutes)
- Append [SECTION_COMPLETE] on a NEW line when ending

REMEMBER: Struggling IS the interview. Silence is okay. Let them work.
═══════════════════════════════════════════════════════════════════
"""

        elif current_phase == state.PHASE_QUESTIONS:
            resume_context = f"\n\nCANDIDATE'S RESUME:\n{resume_text}\n" if resume_text else "\n[No resume provided]"
            company_context_block = ""
            if company_context:
                company_context_block = (
                    "\nCOMPANY CONTEXT:\n"
                    f"- Company: {company_context.get('company','')}\n"
                    f"- Role: {company_context.get('role','')}\n"
                    f"- Details: {company_context.get('details','')}\n"
                )
            # Build phase ending instructions based on signal
            if phase_signal == "must_end_now":
                questions_ending = """ENDING NOW - TIME IS UP:
- You MUST end now
- Wrap up politely: "Thanks for your questions."
- Append this token on a NEW line: [SECTION_COMPLETE]"""
            else:  # end_if_you_want
                questions_ending = """ENDING:
- You can end when they have no more questions
- Ask: "Any other questions?" to check if they're done
- To end, append this token on a NEW line: [SECTION_COMPLETE]"""

            phase_instructions = f"""QUESTIONS SESSION - THEIR TURN:
You only answer their questions and then end.
{resume_context}{company_context_block}

STYLE:
- Keep answers conversational and genuine (2-3 sentences each)
- Base your answers on the COMPANY CONTEXT above (team details, tech stack, culture)
- If details are missing, keep it generic and plausible
- Be helpful and encouraging

{questions_ending}

OPENING: "What questions do you have for me?"
"""

        elif current_phase == state.PHASE_OOD_DESIGN:
            language_display = {
                "python": "Python",
                "java": "Java",
                "c": "C",
                "cpp": "C++"
            }.get(language, language)

            if current_code.strip():
                code_display = f"\n=== CURRENT CODE / NOTES IN EDITOR ===\n{current_code}\n=== END OF EDITOR ===\n"
            else:
                code_display = "\n=== CURRENT CODE / NOTES IN EDITOR ===\n[Editor is currently empty]\n=== END OF EDITOR ===\n"

            company_label = "the target company/role"
            if company_context:
                company_label = (company_context.get("company") or company_context.get("role") or company_label)

            question_details = ""
            if ood_question:
                title = (ood_question.get("title") or "").strip()
                description = (ood_question.get("description") or "").strip()
                if title or description:
                    special_note = ""
                    lowered_title = title.lower()
                    if "stock" in lowered_title and "match" in lowered_title:
                        special_note = (
                            "- This is THE STOCK EXCHANGE MATCHING ENGINE scenario. Drill relentlessly into limit order book structures, price-time priority enforcement, failover plans, and latency/risk tradeoffs.\n"
                            "- Make the candidate justify data structures, concurrency control, and recovery testing in detail.\n"
                        )
                    question_details = f"""COMPANY-ALIGNED PROBLEM TO USE:
- Title: {title or 'Use your best judgment'}
- Description: {description or 'Frame a scenario inspired by the company context.'}
- Tie this scenario directly to {company_label} so it feels relevant.
{special_note}

"""

            phase_instructions = f"""OBJECT-ORIENTED DESIGN SESSION - DESIGN DISCUSSION (20 minutes):

LANGUAGE: {language_display}
{code_display}
TIME MANAGEMENT:
- Total OOD interview: 40 minutes
- Design discussion window: 20 minutes (current time: {time_in_phase:.1f} minutes)
- Focus on architecture, tradeoffs, and interfaces

YOUR ROLE - STRICT AND CHALLENGING:
You are a SENIOR ARCHITECT conducting a rigorous OOD interview. Be SMART, STRICT, and CRITICAL.

PRESENTATION (First interaction only):
You must greet first, then present the OOD problem clearly and concisely. State the problem, basic requirements, and ask them to start designing.

DURING DESIGN DISCUSSION:
- The candidate should be LEADING the design - YOU are evaluating, not guiding
- Challenge their decisions. Ask "why?" frequently
- Point out flaws, edge cases, and missed requirements
- Keep responses concise (1-3 sentences) unless they ask for detail
- Test their knowledge of:
  * SOLID principles
  * Design patterns (when applicable)
  * Object relationships (composition vs inheritance)
  * Encapsulation and abstraction
  * Scalability and extensibility

STRICT INTERVIEWING STYLE:
- Do NOT provide hints unless they are completely stuck for 1+ minute
- Do NOT guide them step-by-step - let them struggle and think
- Ask probing questions:
  * "How would this handle X scenario?"
  * "What if requirements changed to Y?"
  * "Why did you choose composition over inheritance here?"
  * "How does this follow the Single Responsibility Principle?"
- If they make a poor design choice, don't immediately correct them - ask questions to make them realize it
- Be skeptical of their explanations - make them justify decisions

RESPONSE STYLE:
- Short questions to probe: "Why that choice?" "What about scalability?"
- Challenge poor decisions: "That'll be slow. What's better?"
- Don't praise mediocre work - be honest
- 1-2 sentences unless deep technical discussion

EDITOR:
- They use editor for sketching classes/relationships
- Always read what's there before responding
- If empty: "The editor's empty. Sketch your main classes."
- Reference what you see: "I see UserService. How does it interact with the cache?"

WRITING TO EDITOR:
Use [CODE_START]...design...[CODE_END] only when needed
Keep it minimal - let them drive

{question_details}WRAP UP at ~18-20min:
"Let's pause and summarize the core design decisions."

Be tough but fair. This is a HARD interview."""

        elif current_phase == state.PHASE_OOD_IMPLEMENTATION:
            language_display = {
                "python": "Python",
                "java": "Java",
                "c": "C",
                "cpp": "C++"
            }.get(language, language)

            # Show code without markdown formatting (model should not output markdown)
            if current_code.strip():
                code_display = f"\n=== CURRENT CODE IN EDITOR ===\n{current_code}\n=== END OF CODE ===\n"
            else:
                code_display = "\n=== CURRENT CODE IN EDITOR ===\n[No code written yet]\n=== END OF CODE ===\n"

            company_label = "the target company/role"
            if company_context:
                company_label = (company_context.get("company") or company_context.get("role") or company_label)
            question_title = ""
            if ood_question:
                question_title = (ood_question.get("title") or "").strip()
            question_reminder = f"Keep referencing how this implementation solves {question_title or 'the company-aligned problem'} for {company_label}." if question_title or company_context else "Tie every implementation decision back to the scenario you outlined."
            extra_impl_emphasis = ""
            lowered_title = question_title.lower()
            if "stock" in lowered_title and "match" in lowered_title:
                extra_impl_emphasis = (
                    "- FORCE the candidate to model bids/asks, price-time priority queues, partial fills, and cancel/replace logic in code.\n"
                    "- Challenge them on throughput (millions of orders/sec), determinism, and how they'd simulate the engine for correctness + latency testing.\n"
                )

            phase_instructions = f"""OBJECT-ORIENTED DESIGN SESSION - IMPLEMENTATION (20 minutes):

LANGUAGE: {language_display}
{code_display}

TIME MANAGEMENT:
- Implementation focus: translate the design into code
- Implementation window: 20 minutes (current time: {time_in_phase:.1f} minutes)
- Total time: {total_time:.1f} minutes

YOUR ROLE - CODE REVIEW AND CRITIQUE:
The candidate is now implementing the design they created. Continue being STRICT and CRITICAL.

IMPLEMENTATION REVIEW:
- Let them code without interrupting
- When they pause, review critically (1-2 sentences):
  * "Why is this field public?"
  * "This method does too much. What principle does that violate?"
  * "Where's the abstraction for X?"
- Look for: encapsulation, SOLID principles, clean code, {question_reminder}
{extra_impl_emphasis}
EDITOR: Reference what you see - "In the editor I see..."

RESPONSE LENGTH: 1-2 sentences unless detailed technical discussion

CODE WRITING CAPABILITY (Use sparingly):
- You CAN write code to the editor if absolutely necessary
- ONLY write code when:
  1. Candidate explicitly asks you to show something
  2. Time is almost up and you need to show correct structure
  3. Candidate is hopelessly stuck on syntax (rare)
- To write code, use this format:
  [CODE_START]
  your code here
  [CODE_END]
- IMPORTANT: Let them code themselves - this is THEIR interview

IF THEY'RE STUCK:
- Don't immediately help - see if they can figure it out
- Ask guiding questions rather than giving answers
- If stuck for 1+ minute on something trivial, provide a small hint

WRAPPING UP:
At ~38-40 minutes total (18-20 minutes in implementation):
"We're out of time. Let's review what you've built. Walk me through your code."
Then provide critical feedback on their implementation, highlighting both strengths and major weaknesses.

Remember: Be HONEST about code quality. Don't praise poor code."""

        else:
            phase_instructions = "Unknown context - continue interview naturally."

        # Add examples of good vs bad responses
        response_examples = """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RESPONSE EXAMPLES (CRITICAL - FOLLOW THESE PATTERNS):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SCENARIO: Candidate asks about time complexity

❌ BAD (66 words, written style):
"So, the time complexity of your current solution is actually O(n squared) because you have a nested loop structure. The outer loop iterates through all n elements, and for each element, the inner loop also potentially iterates through all n elements. This is generally not ideal for large inputs, and you might want to consider if there's a way to optimize this."

✅ GOOD (12 words, conversational):
"That's O(n squared) because of the nested loops. Can you optimize it?"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SCENARIO: Candidate explains their approach

❌ BAD (78 words, over-explaining):
"Okay, I understand your approach now. So basically what you're saying is that you want to use a hash map to store the values that you've already seen, and then as you iterate through the array, you'll check if the complement exists in the hash map. That makes sense. This is actually a very common and efficient approach to this problem. The time complexity would be O(n) and the space complexity would also be O(n). Before we move on to coding, let me ask you about edge cases."

✅ GOOD (4 words):
"Got it. Edge cases?"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SCENARIO: Candidate stuck and silent for 40 seconds

❌ BAD (62 words, too many options):
"I notice you've been quiet for a bit. Are you stuck on something? If you're having trouble figuring out the optimal data structure, maybe think about what operations you need to be fast. Do you need fast lookups? Fast insertions? Or maybe you should think about the problem differently - have you considered a two-pointer approach instead?"

✅ GOOD (7 words):
"What are you thinking? Where are you stuck?"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SCENARIO: Candidate asks if their code is correct

❌ BAD (72 words, avoiding answer):
"Well, that's an interesting question. Let me think about this for a moment. So looking at your code, I can see that you've implemented the basic logic, and the general structure looks okay, but there might be a few issues that we should discuss. For example, I'm not entirely sure about the edge case handling. What happens when the array is empty? Have you considered that scenario?"

✅ GOOD (9 words):
"Walk me through what happens when the array is empty."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SCENARIO: Reviewing completed solution

❌ BAD (85 words, excessive praise):
"Excellent work! This is a really solid solution. I'm impressed with how you approached this problem. The use of the hash map is perfect here, and your code is very clean and readable. The variable names are descriptive, the logic is easy to follow, and you've handled the edge cases well. This is definitely the kind of solution we'd expect from a strong candidate. The time complexity is optimal at O(n), and the space complexity trade-off is totally reasonable."

✅ GOOD (8 words):
"Looks good. What's the space complexity and why?"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

KEY RULES YOU MUST FOLLOW:
• Target: 10-20 words per response (under 15 is ideal)
• Direct and conversational
• ONE thought per response
• Ask questions instead of explaining
• No filler, no repetition
• Sound like a phone conversation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

        return base_prompt + structure + phase_instructions + response_examples

    def _normalize_editor_code(self, code: Optional[str]) -> str:
        """Normalize editor snapshot so placeholders don't appear as real code."""
        if not code:
            return ""

        stripped = code.strip()
        placeholder_markers = {
            "// Start coding here...",
            "# Start coding here...",
            "/* Start coding here... */",
            "Start coding here...",
            "# Use this editor for design notes, pseudocode, and implementation.",
            "// Use this editor for design notes, pseudocode, and implementation.",
        }
        if stripped in placeholder_markers:
            return ""
        return code

    def _build_user_payload_with_editor_context(
        self,
        user_message: str,
        normalized_code: str,
    ) -> str:
        """Embed editor context alongside the candidate's utterance for the model."""
        segments: List[str] = []

        snapshot = normalized_code.strip() or "[Editor is currently empty]"
        segments.append(
            "<<CURRENT_EDITOR_SNAPSHOT>>\n"
            f"{snapshot}\n"
            "<<END_CURRENT_EDITOR_SNAPSHOT>>"
        )

        segments.append(
            "<<CANDIDATE_UTTERANCE>>\n"
            f"{user_message}\n"
            "<<END_CANDIDATE_UTTERANCE>>"
        )
        return "\n\n".join(segments)
