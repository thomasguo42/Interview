from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

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
        if not interview_state:
            return self.generate_interview_reply(conversation, resume_text, user_message, temperature)

        payload = self._builder._build_structured_payload(
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

        current_phase = interview_state.get("current_phase", "unknown")
        mode = interview_state.get("mode", "unknown")
        system_prompt = payload.get("systemInstruction", {}).get("parts", [{}])[0].get("text", "")
        code_in_prompt = bool('CURRENT CODE IN EDITOR' in system_prompt)
        logging.info(f"[OPENAI DEBUG] phase={current_phase} mode={mode} time_in_phase={time_in_phase:.1f}m")
        logging.info(f"[OPENAI DEBUG] phase={current_phase} current_code_length={len(current_code)}")
        logging.info(f"[OPENAI DEBUG] phase={current_phase} code_included_in_prompt={code_in_prompt}")
        if current_phase == "coding":
            evaluation = interview_state.get("code_evaluation") or {}
            test_status = str(evaluation.get("status", "")).strip() or "not_run"
            logging.info(f"[OPENAI DEBUG] phase={current_phase} test_status={test_status} code_access_granted={code_in_prompt}")
        logging.info(f"[OPENAI DEBUG] phase={current_phase} system_prompt_length={len(system_prompt)}")
        return self._post_for_text(payload)

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
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        return self._post_messages(messages, max_tokens=200, temperature=0.4, top_p=0.9)

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
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": instructions},
        ]
        return self._post_messages(messages, max_tokens=800, temperature=0.3, top_p=0.9)

    def evaluate_code_optimization(
        self,
        question: Dict[str, Any],
        language: str,
        code: str,
    ) -> Dict[str, Any]:
        system_prompt = (
            "You are a senior software engineer reviewing a candidate solution. "
            "Determine whether the solution is asymptotically optimal for the problem. "
            "Return STRICT JSON only."
        )

        question_title = str(question.get("title", "") or "").strip()
        difficulty = str(question.get("difficulty", "") or "").strip()
        signature = ""
        signatures = question.get("signatures") if isinstance(question, dict) else None
        if isinstance(signatures, dict):
            signature = str(signatures.get(language, "") or "").strip()

        prompt = (
            "Evaluate the solution's optimality.\n\n"
            f"Problem title: {question_title or 'Unknown'}\n"
            f"Difficulty: {difficulty or 'Unknown'}\n"
            f"Required signature: {signature or 'Unknown'}\n"
            f"Language: {language}\n\n"
            "Candidate code:\n"
            f"```{language}\n{code}\n```\n\n"
            "Return JSON with EXACTLY this structure:\n"
            '{\n  "optimized": true | false,\n  "feedback": "1-2 sentences, concrete and actionable"\n}\n'
            "Rules:\n"
            "- optimized=true only if time/space complexity is asymptotically optimal\n"
            "- If constraints are unclear, use standard interview expectations\n"
            "- Do NOT include any extra fields or commentary\n"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        response_text = self._post_messages(messages, max_tokens=300, temperature=0.1, top_p=1.0)
        parsed = self._builder._parse_json_from_text(response_text)
        if not isinstance(parsed, dict) or "optimized" not in parsed:
            raise OpenAIError(f"Invalid optimization payload: {response_text}")
        return parsed

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
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
        }

        model_name = str(self.model or "")
        if model_name.startswith("gpt-5"):
            payload["max_completion_tokens"] = max_tokens
        else:
            payload["max_tokens"] = max_tokens

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


class OpenAIRealtimeClient(OpenAIClient):
    WS_URL_TEMPLATE = "wss://api.openai.com/v1/realtime?model={model}"

    def _post_messages(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> str:
        instructions, input_items = self._messages_to_realtime_input(messages)
        if not input_items:
            raise OpenAIError("Realtime request requires at least one user message.")

        response_event = {
            "type": "response.create",
            "response": {
                "conversation": "none",
                "output_modalities": ["text"],
                "instructions": instructions,
                "input": input_items,
                "max_output_tokens": max_tokens,
            },
        }

        try:
            import websocket  # type: ignore
        except ImportError as exc:
            raise OpenAIError(
                "websocket-client is required for OpenAI Realtime API support. "
                "Install it with `pip install websocket-client`."
            ) from exc

        url = self.WS_URL_TEMPLATE.format(model=self.model)
        headers = [f"Authorization: Bearer {self.api_key}"]

        try:
            ws = websocket.create_connection(url, header=headers, timeout=30)
        except Exception as exc:
            raise OpenAIError(f"Failed to connect to OpenAI Realtime API: {exc}") from exc

        collected: List[str] = []
        last_event_time = time.time()
        try:
            ws.send(json.dumps({"type": "session.update", "session": {"type": "realtime"}}))
            ws.send(json.dumps(response_event))

            while True:
                raw = ws.recv()
                if not raw:
                    if time.time() - last_event_time > 10:
                        break
                    continue
                last_event_time = time.time()
                try:
                    event = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                event_type = event.get("type")
                if event_type == "response.output_text.delta":
                    delta = event.get("delta") or ""
                    if delta:
                        collected.append(delta)
                elif event_type == "response.output_text.done":
                    break
                elif event_type == "response.done":
                    if collected:
                        break
                elif event_type == "error":
                    error_payload = event.get("error") or {}
                    message = error_payload.get("message") or str(error_payload) or "Unknown error"
                    raise OpenAIError(f"OpenAI Realtime error: {message}")
        except Exception as exc:
            raise OpenAIError(f"OpenAI Realtime request failed: {exc}") from exc
        finally:
            try:
                ws.close()
            except Exception:
                pass

        return "".join(collected).strip()

    @staticmethod
    def _messages_to_realtime_input(
        messages: List[Dict[str, str]],
    ) -> Tuple[str, List[Dict[str, Any]]]:
        instructions = ""
        input_items: List[Dict[str, Any]] = []
        for message in messages:
            role = message.get("role")
            content = (message.get("content") or "").strip()
            if not content:
                continue
            if role == "system":
                instructions = content
                continue
            if role not in {"user", "assistant"}:
                continue
            content_type = "input_text" if role == "user" else "output_text"
            input_items.append(
                {
                    "type": "message",
                    "role": role,
                    "content": [{"type": content_type, "text": content}],
                }
            )
        return instructions, input_items
