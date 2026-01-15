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
    """Raised when the model API returns an unexpected response."""


class GeminiClient:
    API_URL = "https://api.deepseek.com/v1/chat/completions"

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        self.api_key = api_key or config.DEEPSEEK_API_KEY
        self.model = model or config.DEEPSEEK_MODEL
        self.phase_model = config.DEEPSEEK_PHASE_MODEL or self.model
        if not self.api_key:
            raise ValueError(
                "DeepSeek API key is not configured. Set the DEEPSEEK_API_KEY environment variable."
            )

    def generate_interview_reply(
        self,
        conversation: List[Dict[str, Any]],
        resume_text: str | None,
        user_message: str,
        temperature: float = 0.7,
    ) -> str:
        """
        Sends the conversation history and the latest user message to DeepSeek and returns the reply text.

        Args:
            conversation: Existing conversation history formatted for the app state.
            resume_text: Optional resume text to ground the interview context.
            user_message: Latest user utterance to append to the conversation.
            temperature: Sampling temperature for response creativity.
        """
        system_prompt, messages = self._build_payload(conversation, resume_text, user_message)
        return self._post_chat(
            messages=messages,
            model=self.model,
            temperature=temperature,
            top_p=0.95,
            max_tokens=800,
            system_prompt=system_prompt,
        )

    def _build_payload(
        self,
        conversation: List[Dict[str, Any]],
        resume_text: str | None,
        user_message: str,
    ) -> tuple[str, List[Dict[str, str]]]:
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
        )
        if resume_text:
            system_prompt += (
                "\nThe candidate resume you have is below. Use it to tailor your questions "
                "and feedback. Focus on relevant technologies, accomplishments, and gaps.\n"
                f"Resume:\n{resume_text}\n"
            )

        messages = self._build_messages(conversation, system_prompt, user_message)
        return system_prompt, messages

    def _build_messages(
        self,
        conversation: List[Dict[str, Any]],
        system_prompt: str,
        user_message: str,
    ) -> List[Dict[str, str]]:
        messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
        messages.extend(self._convert_conversation(conversation))
        messages.append({"role": "user", "content": user_message})
        return messages

    def _convert_conversation(self, conversation: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        messages: List[Dict[str, str]] = []
        for turn in conversation:
            role = turn.get("role")
            parts = turn.get("parts", [])
            text_segments = []
            for part in parts:
                if isinstance(part, dict):
                    text_segments.append(part.get("text", ""))
            content = " ".join(text_segments).strip()
            if not content:
                continue
            mapped_role = "assistant" if role == "model" else "user"
            messages.append({"role": mapped_role, "content": content})
        return messages

    def _post_chat(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        system_prompt: str,
    ) -> str:
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }
        try:
            response = requests.post(
                self.API_URL,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=30,
            )
        except RequestException as exc:
            logging.error("DeepSeek API request failed: %s", exc)
            raise GeminiError("DeepSeek API request failed") from exc

        if response.status_code != 200:
            logging.error(
                "DeepSeek API error: status=%s body=%s",
                response.status_code,
                response.text,
            )
            raise GeminiError(f"DeepSeek API error {response.status_code}: {response.text}")

        data = response.json()
        try:
            content = data["choices"][0]["message"]["content"]
            if not content:
                raise KeyError("Empty response text")
            return content.strip()
        except (KeyError, IndexError, TypeError) as exc:
            logging.error("Unexpected DeepSeek API payload: %s", data)
            logging.error("System prompt size for debugging: %s", len(system_prompt))
            raise GeminiError("DeepSeek API returned an unexpected payload") from exc

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
        """
        Generate reply for structured interview with phase awareness.

        Args:
            session_id: Session identifier
            conversation: Existing conversation history
            resume_text: Resume text for context
            user_message: Latest user utterance
            current_code: Current code in editor
            code_changed: Whether code changed recently
            temperature: Sampling temperature
        """
        interview_state = state.get_interview_state(session_id)
        if not interview_state:
            # Fallback to regular interview if no structured state
            return self.generate_interview_reply(conversation, resume_text, user_message, temperature)

        payload = self._build_structured_payload(
            session_id,
            interview_state,
            conversation,
            resume_text,
            user_message,
            current_code,
            code_changed,
            temperature,
        )

        # Debug logging
        system_prompt, messages = payload
        logging.info(f"[MODEL DEBUG] Current code length: {len(current_code)}")
        if current_code:
            logging.info(f"[MODEL DEBUG] Code is included in prompt: {bool('CURRENT CODE IN EDITOR' in system_prompt)}")
        logging.info(f"[MODEL DEBUG] System prompt length: {len(system_prompt)}")
        return self._post_chat(
            messages=messages,
            model=self.model,
            temperature=temperature,
            top_p=0.95,
            max_tokens=800,
            system_prompt=system_prompt,
        )

    def _build_structured_payload(
        self,
        session_id: str,
        interview_state: Dict[str, Any],
        conversation: List[Dict[str, Any]],
        resume_text: str | None,
        user_message: str,
        current_code: str,
        code_changed: bool,
        temperature: float,
    ) -> tuple[str, List[Dict[str, str]]]:
        """Build payload with phase-aware prompting for structured interview"""

        current_phase = interview_state.get("current_phase", state.PHASE_INTRO)
        language = interview_state.get("language", "python")
        problem_presented = interview_state.get("problem_presented", False)
        mode = interview_state.get("mode", "full")
        company_context = interview_state.get("company_context")
        ood_question = interview_state.get("ood_question")

        normalized_code = self._normalize_editor_code(current_code)

        # Calculate timing information
        time_in_phase = state.calculate_time_in_phase(session_id)
        total_time = state.calculate_total_time(session_id)
        silence_duration = state.calculate_silence_duration(session_id)
        code_idle_duration = state.calculate_code_idle_duration(session_id)

        # Build comprehensive system prompt
        system_prompt = self._build_phase_prompt(
            current_phase,
            language,
            resume_text,
            normalized_code,
            code_changed,
            problem_presented,
            time_in_phase,
            total_time,
            silence_duration,
            code_idle_duration,
            mode,
            company_context,
            ood_question,
        )

        # Build conversation payload
        conversation_payload = list(conversation)
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

        messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
        messages.extend(self._convert_conversation(conversation_payload))
        return system_prompt, messages

    def decide_phase_transition(
        self,
        session_id: str,
        interview_state: Dict[str, Any],
        conversation: List[Dict[str, Any]],
        user_message: str,
        model_reply: str,
    ) -> Optional[Dict[str, Any]]:
        """Use a lightweight model to evaluate whether the interview should transition phases."""

        time_in_phase = state.calculate_time_in_phase(session_id)
        total_time = state.calculate_total_time(session_id)
        silence_duration = state.calculate_silence_duration(session_id)
        code_idle_duration = state.calculate_code_idle_duration(session_id)

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

        system_prompt, messages = payload
        try:
            decision_text = self._post_chat(
                messages=messages,
                model=self.phase_model,
                temperature=0.1,
                top_p=0.8,
                max_tokens=200,
                system_prompt=system_prompt,
            )
        except GeminiError as exc:
            logging.warning("Phase decision model error: %s", exc)
            return None

        if decision_text:
            logging.info("[PHASE DECISION RAW] %s", decision_text)

        decision = self._parse_phase_decision(decision_text)
        if not decision:
            logging.warning("Unable to parse phase decision JSON: %s", decision_text)
            return None

        next_phase = decision.get("next_phase")
        if next_phase and next_phase not in {
            state.PHASE_INTRO,
            state.PHASE_RESUME,
            state.PHASE_CODING,
            state.PHASE_QUESTIONS,
        }:
            logging.warning("Phase decision returned invalid next_phase: %s", decision)
            return None

        return decision

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
    ) -> tuple[str, List[Dict[str, str]]]:
        current_phase = interview_state.get("current_phase", state.PHASE_INTRO)
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
            "seconds_since_candidate_spoke": round(float(silence_duration or 0.0), 1),
            "seconds_since_code_change": round(float(code_idle_duration or 0.0), 1),
            "problem_already_presented": problem_presented,
            "recent_candidate_message": user_message,
            "model_reply_being_sent": model_reply,
            "conversation_excerpt": snippet,
            "phase_options": [
                state.PHASE_INTRO,
                state.PHASE_RESUME,
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

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": instructions + "\n\nCONTEXT:\n" + json.dumps(context, ensure_ascii=False),
            },
        ]
        return system_prompt, messages

    def _format_conversation_snippet(
        self,
        conversation: List[Dict[str, Any]],
        max_turns: int = 8,
    ) -> str:
        if not conversation:
            return ""

        recent_turns = conversation[-max_turns:]
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
        current_code: str,
        code_changed: bool,
        problem_presented: bool,
        time_in_phase: float,
        total_time: float,
        silence_duration: float,
        code_idle_duration: float,
        mode: str = "full",
        company_context: Optional[Dict[str, str]] = None,
        ood_question: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Build phase-specific system prompt"""

        # Base prompt - applies to all phases
        if mode == "coding_only":
            base_prompt = """You are a senior software engineer conducting a LIVE VOICE coding interview.
This is a CODING-ONLY session (35-40 minutes) - no intro, no resume discussion, just pure technical problem-solving.

CRITICAL CONVERSATION RULES:
- Speak naturally like in a real conversation, NOT written text
- Keep responses SHORT (1-3 sentences usually) - you're SPEAKING, not writing essays
- Vary response length based on context and what the candidate just said
- Use contractions (I'm, you're, that's), casual professional phrasing
- Reference the conversation history - what was discussed earlier matters
- ONE question or point at a time
- Let silences happen when they're thinking or coding - don't fill every gap
- Be supportive but maintain interview professionalism

CRITICAL - TEXT-TO-SPEECH OUTPUT:
- Your output will be converted to SPEECH by a TTS system
- NEVER use markdown symbols: NO asterisks (*), NO hashtags (#), NO underscores (_)
- NEVER use formatting: NO **bold**, NO *italics*, NO `code blocks`
- NEVER use filler words like "um", "uh", "hmm", "well", "so"
- Speak clearly and directly - write EXACTLY what should be spoken out loud
- For code or technical terms, just say them plainly (e.g., "use a for loop" not "use a *for loop*")

"""
        else:
            base_prompt = """You are a senior software engineer conducting a LIVE VOICE technical interview.
This is a 60-minute structured interview with natural, back-and-forth CONVERSATION.

CRITICAL CONVERSATION RULES:
- Speak naturally like in a real conversation, NOT written text
- Keep responses SHORT (1-3 sentences usually) - you're SPEAKING, not writing essays
- Vary response length based on context and what the candidate just said
- Use contractions (I'm, you're, that's), casual professional phrasing
- Reference the conversation history - what was discussed earlier matters
- ONE question or point at a time
- Let silences happen when they're thinking or coding - don't fill every gap
- Be supportive but maintain interview professionalism

CRITICAL - TEXT-TO-SPEECH OUTPUT:
- Your output will be converted to SPEECH by a TTS system
- NEVER use markdown symbols: NO asterisks (*), NO hashtags (#), NO underscores (_)
- NEVER use formatting: NO **bold**, NO *italics*, NO `code blocks`
- NEVER use filler words like "um", "uh", "hmm", "well", "so"
- Speak clearly and directly - write EXACTLY what should be spoken out loud
- For code or technical terms, just say them plainly (e.g., "use a for loop" not "use a *for loop*")

"""

        # Interview structure context
        if mode == "coding_only":
            structure = f"""SESSION TYPE: CODING-ONLY MODE (35-40 minutes)
You skip all intro and resume discussion. Jump STRAIGHT to presenting a coding problem.

CURRENT STATE:
- Time elapsed: {total_time:.1f} minutes
- Seconds since candidate spoke: {silence_duration:.0f}s
- Seconds since code changed: {code_idle_duration:.0f}s

"""
        else:
            structure = f"""INTERVIEW STRUCTURE (60 minutes total):
Phase 1: Introduction (0-5min) - Greet, explain format
Phase 2: Resume Discussion (5-15min) - Background deep-dive
Phase 3: Coding Problem (15-50min) - Technical assessment
Phase 4: Candidate Questions (50-60min) - Reverse interview

CURRENT STATE:
- Phase: {current_phase.upper()}
- Time in this phase: {time_in_phase:.1f} minutes
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
        if current_phase == state.PHASE_INTRO:
            phase_instructions = """INTRODUCTION PHASE - CURRENT OBJECTIVE:
- If just starting: Introduce yourself briefly as a senior engineer (make up a name/company)
- Explain the interview format in 2-3 sentences
- Be warm and put them at ease
- Ask if they have questions before starting
- WHEN READY (after ~3-5 min): Naturally transition to resume discussion
  Example: "Great! Let's start with your background. Tell me about yourself."

KEEP IT BRIEF - This is just the intro, not the main interview yet."""

        elif current_phase == state.PHASE_RESUME:
            resume_context = f"\n\nCANDIDATE'S RESUME:\n{resume_text}\n" if resume_text else "\n[No resume provided]"
            phase_instructions = f"""RESUME DISCUSSION PHASE - CURRENT OBJECTIVE:{resume_context}

WHAT TO DO:
- Ask about their experience, recent projects, technologies they've used
- Dig deeper into interesting areas with follow-up questions
- Assess technical depth through probing questions
- Keep it conversational, like getting to know a colleague
- Reference what they mentioned earlier in THIS interview

TIMING:
- Target: ~10 minutes for this phase
- Current time in phase: {time_in_phase:.1f} minutes

CRITICAL - WHEN TO TRANSITION TO CODING:
After about 10 minutes (or after 3-4 resume questions), you MUST transition to the coding phase.
To trigger the transition, you MUST include one of these EXACT phrases in your response:
  - "Let's move to coding"
  - "Let's work on a coding problem"
  - "Time for a coding challenge"

Example transition: "Thanks for sharing that. Let's move to coding now. I'll present a problem and I want you to talk through your approach first before coding. Ready?"

NOTE: The system detects these keywords to show the code editor. Without them, the editor won't appear!"""

        elif current_phase == state.PHASE_CODING:
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

            visibility_instructions = """CODE VISIBILITY & EDITOR ACCESS:
- The snapshot between the 'CURRENT CODE IN EDITOR' markers above is EXACTLY what the candidate has typed.
- YOU CAN READ IT DIRECTLY. Never say you cannot see their code or editor.
- If it literally shows "[No code written yet]", tell the candidate their editor is empty instead of claiming you lack access.
- When code exists, reference concrete details from it (functions, variables, logic) before giving feedback.
- Some user messages also include <<CURRENT_EDITOR_SNAPSHOT>> ... <<END_CURRENT_EDITOR_SNAPSHOT>> markers. Treat those as a live view of the editor right before their utterance.
- When you need to modify their code, output a [CODE_START] ... [CODE_END] block to overwrite the editor.

"""

            if not problem_presented:
                # Different instructions for coding_only mode
                if mode == "coding_only":
                    phase_instructions = f"""CODING-ONLY MODE - START IMMEDIATELY WITH PROBLEM:

LANGUAGE: {language_display}
{code_display}{visibility_instructions}

YOU ARE IN A CODING-ONLY SESSION. NO INTRO, NO RESUME DISCUSSION, NO SMALL TALK.

WHEN THE CANDIDATE GREETS YOU (says "Hi", "Hello", "Ready", etc.):
YOUR FIRST RESPONSE SHOULD:
1. Brief greeting (1 sentence): "Hi! Ready to solve a coding problem?"
2. IMMEDIATELY present a LeetCode-style problem from your knowledge:
   - Difficulty: MEDIUM-HARD preferred, or EASY with MEDIUM-HARD follow-ups
   - Examples: Two Sum, Merge Intervals, LRU Cache, Design Twitter, Valid Parentheses, etc.
   - State the problem naturally and clearly
   - Give example inputs/outputs
   - Keep it concise - you're speaking, not writing

3. Ask if they have clarifying questions

AFTER they understand the problem:
- GUIDE THEM TO DISCUSS APPROACH FIRST before coding
- Make them explain their approach
- Ask about edge cases
- Discuss time/space complexity

CODE WRITING CAPABILITY:
- You CAN write code directly to the candidate's editor when necessary
- ONLY write code in these situations:
  1. The candidate explicitly asks you to write code or put something in the editor
  2. Time is almost up and you need to show the correct solution
  3. The candidate is very stuck and asks for the complete answer
- CRITICAL: Before writing code, ALWAYS look at what they already have
- You can modify their existing code or provide a complete replacement
- To write code to the editor, you MUST use this EXACT format:
  1. First, say what you're doing (e.g., "Let me write that for you")
  2. Then add the code block with these EXACT markers:

  [CODE_START]
  your complete code here
  [CODE_END]

  3. Then optionally explain what you wrote
- Example response when asked to write code:
  "Okay, let me write a solution for you. [CODE_START]
  def two_sum(nums, target):
      seen = {{}}
      for i, num in enumerate(nums):
          if target - num in seen:
              return [seen[target - num], i]
          seen[num] = i
  [CODE_END] This solution uses a hash map for O(n) time complexity."
- The code between markers will replace ALL content in the editor
- IMPORTANT: Only use this when truly necessary - let them code themselves!

REMEMBER: Be concise. This is a voice conversation, not a written document.
NOTE: The candidate speaks first (they start the conversation), you respond with the problem."""
                else:
                    phase_instructions = f"""CODING PHASE - PROBLEM PRESENTATION STAGE:

LANGUAGE: {language_display}
{code_display}{visibility_instructions}

YOUR TASK NOW:
1. Select a LeetCode-style coding problem from your knowledge
   - Difficulty: MEDIUM-HARD preferred, or EASY with MEDIUM-HARD follow-ups
   - Pick something relevant to their resume/experience if possible
   - Examples: Two Sum, Merge Intervals, LRU Cache, Design Twitter, etc.

2. Present the problem clearly and conversationally:
   - State the problem naturally (don't just copy-paste)
   - Give example inputs/outputs
   - Ask if they have clarifying questions

3. GUIDE THEM TO DISCUSS APPROACH FIRST:
   - Don't let them jump to coding immediately
   - Make them explain their approach
   - Ask about edge cases
   - Discuss time/space complexity

CODE WRITING CAPABILITY:
- You CAN write code directly to the candidate's editor when necessary
- ONLY write code in these situations:
  1. The candidate explicitly asks you to write code or put something in the editor
  2. Time is almost up and you need to show the correct solution
  3. The candidate is very stuck and asks for the complete answer
- CRITICAL: Before writing code, ALWAYS look at what they already have
- You can modify their existing code or provide a complete replacement
- To write code to the editor, you MUST use this EXACT format:
  1. First, say what you're doing (e.g., "Let me write that for you")
  2. Then add the code block with these EXACT markers:

  [CODE_START]
  your complete code here
  [CODE_END]

  3. Then optionally explain what you wrote
- Example response when asked to write code:
  "Okay, let me write a solution for you. [CODE_START]
  def two_sum(nums, target):
      seen = {{}}
      for i, num in enumerate(nums):
          if target - num in seen:
              return [seen[target - num], i]
          seen[num] = i
  [CODE_END] This solution uses a hash map for O(n) time complexity."
- The code between markers will replace ALL content in the editor
- IMPORTANT: Only use this when truly necessary - let them code themselves!

REMEMBER: After presenting, I will mark problem as presented."""

            else:
                phase_instructions = f"""CODING PHASE - ACTIVE PROBLEM SOLVING:

LANGUAGE: {language_display}
{code_display}{visibility_instructions}

CODING STAGE GUIDANCE:
The candidate is now working on the problem. Your role depends on what's happening:

**IF CANDIDATE JUST SPOKE:**
- Respond to what they said
- Answer questions they ask
- Provide hints if they explicitly ask for help
- Give feedback on their approach if they're explaining

**IF THEY'RE ACTIVELY CODING (code changing recently):**
- Let them work! Brief check-ins only every 2-3 minutes
- Example: "How's it going?" or "Making progress?"

**IF THEY'RE STUCK (silence >30s AND no code changes >30s):**
Current stuck indicators: {silence_duration:.0f}s silence, {code_idle_duration:.0f}s code idle
- Offer help: "Walk me through what you're thinking" or "What's blocking you?"
- Never give direct solutions - only hints or guiding questions

**IF THEY'RE ON WRONG PATH:**
- Gently redirect with questions: "What about the case when...?"
- Don't tell them the answer - guide them to realize it

**CODE ANALYSIS:**
- ALWAYS look at their current code in the "CURRENT CODE IN EDITOR" section above
- Assess correctness based on logic, approach, edge cases
- Identify bugs or issues you see by looking at their actual code
- When they say they're done: Review their actual code together, discuss edge cases, optimizations
- Reference specific lines or patterns you see in their code

CODE READING & WRITING CAPABILITY:
- You can SEE the current code in the editor (shown above in "CURRENT CODE IN EDITOR")
- ALWAYS read and understand their existing code before responding to ANY question or comment
- You can reference specific parts of their code in your responses
- Example: "I see you've started with a hash map approach. Good choice!"
- You MUST look at their code when they ask questions about it or say they're stuck

- You CAN write code directly to the candidate's editor when necessary
- ONLY write code in these situations:
  1. The candidate explicitly asks you to write code or put something in the editor
  2. Time is almost up and you need to show the correct solution
  3. The candidate is very stuck and asks for the complete answer
- CRITICAL: Before writing code, ALWAYS look at what they already have
- You can modify their existing code or provide a complete replacement
- To write code to the editor, you MUST use this EXACT format:
  1. First, say what you're doing (e.g., "Let me write that for you")
  2. Then add the code block with these EXACT markers:

  [CODE_START]
  your complete code here
  [CODE_END]

  3. Then optionally explain what you wrote
- Example response when asked to write code:
  "Okay, let me write a solution for you. [CODE_START]
  def two_sum(nums, target):
      seen = {{}}
      for i, num in enumerate(nums):
          if target - num in seen:
              return [seen[target - num], i]
          seen[num] = i
  [CODE_END] This solution uses a hash map for O(n) time complexity."
- The code between markers will replace ALL content in the editor
- IMPORTANT: Only use this when truly necessary - let them code themselves!

INTERVENTION RULES:
- Code changed recently: {code_changed}
- Brief responses when they're coding
- Longer feedback when reviewing or they ask for help

TIME MANAGEMENT:
- Current time in coding phase: {time_in_phase:.1f} minutes
- Target: ~35 minutes for this phase
- If approaching ~35 min: Guide toward wrapping up and testing edge cases
- WHEN READY TO MOVE ON (~40-45 min total time): Transition to questions phase
  Example: "Nice work. We have a few minutes left - what questions do you have for me?"
"""

        elif current_phase == state.PHASE_QUESTIONS:
            phase_instructions = """CANDIDATE QUESTIONS PHASE - REVERSE INTERVIEW:

YOUR ROLE NOW:
- Ask what questions they have for you
- Answer as a helpful, genuine senior engineer
- Be informative about team, culture, technologies (make up reasonable details)
- This is their chance to learn about the "company"

TIMING:
- This is the final phase (~10 minutes)
- Current time in phase: {time_in_phase:.1f} minutes
- When approaching 60 min total: Thank them and close warmly
  Example: "Thanks for your time today. We'll be in touch soon. Best of luck!"
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

            phase_instructions = f"""OBJECT-ORIENTED DESIGN PHASE - DESIGN DISCUSSION (20 minutes):

LANGUAGE: {language_display}
{code_display}
TIME MANAGEMENT:
- Total OOD interview: 40 minutes
- Design phase: 20 minutes (current time: {time_in_phase:.1f} minutes)
- Implementation phase: 20 minutes (auto-transition at 20 min mark)

YOUR ROLE - STRICT AND CHALLENGING:
You are a SENIOR ARCHITECT conducting a rigorous OOD interview. Be SMART, STRICT, and CRITICAL.

PRESENTATION (First interaction only):
When the candidate greets you, present the OOD problem clearly and concisely. State the problem, basic requirements, and ask them to start designing.

DURING DESIGN DISCUSSION:
- The candidate should be LEADING the design - YOU are evaluating, not guiding
- Challenge their decisions. Ask "why?" frequently
- Point out flaws, edge cases, and missed requirements
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

WORKSPACE (NO WHITEBOARD):
- There is ONE shared code editor for outlining, pseudocode, and implementation
- You see every update in the "CURRENT CODE / NOTES" section above
- Reference what you see: "I see you've drafted a UserService class. Explain how it interacts with the cache"
- If the editor is empty or vague, call it out: "The editor is empty. Start by sketching your main classes"
- Encourage structured thinking: class lists, relationships, invariants, design patterns

CODE EDITOR RULES:
- Always read existing notes before responding
- It's okay if they write bullet lists or comments there; keep them accountable for clarity
- You can modify the editor when needed. Use this exact sequence:
  1. Say what you're about to write ("Let me capture the core entities")
  2. Provide the replacement using markers:
     [CODE_START]
     ...their design or pseudocode...
     [CODE_END]
  3. Briefly explain what changed
- The code between markers replaces the ENTIRE editor, so use it intentionally

WHAT YOU SHOULD NOT DO:
- Don't give away the solution
- Don't provide step-by-step instructions
- Don't praise mediocre work - be honest about weaknesses
- Don't move on until they have a solid core design

{question_details}TRANSITION TO IMPLEMENTATION:
At ~18-20 minutes, transition to implementation phase if they have a reasonable design:
"We're running short on time. Let's move to implementing your design. You'll have 20 minutes to code the core classes."

Remember: This is a CHALLENGING interview. Be tough but fair."""

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

            phase_instructions = f"""OBJECT-ORIENTED DESIGN PHASE - IMPLEMENTATION (20 minutes):

LANGUAGE: {language_display}
{code_display}

TIME MANAGEMENT:
- Design phase complete (spent ~20 minutes)
- Implementation phase: 20 minutes (current time: {time_in_phase:.1f} minutes)
- Total time: {total_time:.1f} minutes

YOUR ROLE - CODE REVIEW AND CRITIQUE:
The candidate is now implementing the design they created. Continue being STRICT and CRITICAL.

IMPLEMENTATION EXPECTATIONS:
- Clean, readable code following OOP best practices
- Proper encapsulation (private fields, public methods)
- Meaningful names for classes, methods, variables
- Key design patterns implemented correctly
- Handle core functionality (not every edge case)
- {question_reminder}
{extra_impl_emphasis}

WORKSPACE REMINDER:
- There is still ONE editor. It contains their latest notes plus working code
- Reference it constantly: "In the editor I see..."
- If they forget to update the editor, tell them to write their changes so you can review them

CODE REVIEW APPROACH:
- Let them code without constant interruption
- When they pause or ask questions, review their code critically:
  * "Why is this field public? Should it be private?"
  * "This method is doing too much - what principle does that violate?"
  * "Where's the abstraction for X?"
- Point out violations of SOLID principles
- Question implementation choices
- Ask about missing critical pieces

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
            phase_instructions = "Unknown phase - continue interview naturally."

        return base_prompt + structure + phase_instructions

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
