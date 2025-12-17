# Structured Interview System - Comprehensive Design Plan

## Table of Contents
1. [Requirements Breakdown](#1-requirements-breakdown)
2. [Key Technical Challenges](#2-key-technical-challenges)
3. [Proposed Architecture](#3-proposed-architecture)
4. [Intelligent Behavior System](#4-intelligent-behavior-system)
5. [Data Flow Examples](#5-data-flow-examples)
6. [Implementation Phases](#6-implementation-phases)
7. [Open Questions & Decisions Needed](#7-open-questions--decisions-needed)
8. [Recommendations](#8-recommendations)

---

## 1. Requirements Breakdown

### Interview Structure (60 minutes total)
- **Phase 1: Introduction (5 min)** - Greeting, format explanation
- **Phase 2: Resume Discussion (10 min)** - Background, experience deep-dive
- **Phase 3: Coding Problem (35 min)** - Main technical assessment
  - **3a. Problem Presentation** - Clear problem statement, clarifying questions
  - **3b. Approach Discussion** - High-level thinking, edge cases, complexity
  - **3c. Implementation** - Actual coding with minimal interruption
  - **3d. Review & Testing** - Edge cases, optimization discussion
- **Phase 4: Candidate Questions (10 min)** - Reverse interview

### Core Capabilities Needed
1. **Time awareness** - Track phase durations, trigger transitions
2. **Code visibility** - LLM sees what's being written in real-time
3. **Activity detection** - Know when user is typing, speaking, or stuck
4. **Intelligent interruption** - Speak at right moments (not random)
5. **Context awareness** - Phase objectives, problem state, code quality
6. **Natural guidance** - Hints without spoiling, feedback without lecturing

---

## 2. Key Technical Challenges

### Challenge A: When to Interrupt vs. Let Them Think
**The Problem:** Too many interruptions = annoying; too few = user feels abandoned

**Solution Strategy:**
- **Track multiple signals:**
  - Time since last speech
  - Time since last code change
  - Current phase (planning allows longer silence than debugging)
  - User's emotional state (frustrated phrases like "hmm", "I don't know")

- **Intervention rules:**
  - **During approach discussion**: Wait 45-60s before offering help
  - **During active coding**: Wait 30s if no code changes OR user says they're stuck
  - **After user asks question**: Always respond immediately
  - **User on wrong path**: Gentle redirect within 10-15s
  - **Time pressure**: Proactive warnings at phase boundaries

### Challenge B: Real-time Code Awareness
**The Problem:** Gemini needs to see code to provide relevant feedback

**Options:**

| Approach | Pros | Cons |
|----------|------|------|
| **Send code with every voice message** | Simple, works with existing flow | Only sees code when user speaks |
| **Periodic snapshots (every 10s)** | Balanced, detects stuck patterns | 10s delay in awareness |
| **WebSocket streaming** | True real-time | Complex, overkill for this use case |

**Recommended:** Hybrid approach:
1. Send code snapshot with every voice interaction (always up-to-date when responding)
2. Also send periodic snapshots every 15-20s during active typing
3. Backend can detect "user stuck" pattern and proactively check in

### Challenge C: Phase Transitions
**The Problem:** Natural flow vs. strict timing

**Solution:**
- **Soft transitions** (±3 min flexibility)
- Gemini gets time warnings: "You have 5 minutes left in this phase"
- Prompted to transition naturally: "Let's move to the coding problem now"
- Can override if user in middle of important thought

### Challenge D: Code Analysis & Grading
**The Problem:** How to assess correctness without running code

**Solution - Multi-layer approach:**

1. **Basic validation** (Backend Python):
   ```python
   import ast
   try:
       ast.parse(code)  # Check Python syntax
       is_valid = True
   except SyntaxError:
       is_valid = False
   ```

2. **Structural analysis** (Pattern matching):
   - Detect common patterns: loops, hash maps, recursion
   - Check function signatures match problem
   - Count edge case handling

3. **LLM analysis** (Gemini sees the code):
   - Assess approach correctness
   - Identify logical errors
   - Suggest improvements
   - Compare to optimal solution (internally, not shown to user)

---

## 3. Proposed Architecture

### A. Frontend Changes

**New UI Components:**
```
┌─────────────────────────────────────────┐
│  Interview Timer: Phase 3/4 - 25:34    │ <- Timer + Phase indicator
├─────────────────────────────────────────┤
│  Conversation Transcript                │
│  [User]: "I'll use a hash map..."      │ <- Scrollable history
│  [Gemini]: "Good. What's the lookup?" │
├─────────────────────────────────────────┤
│  Live Caption: "Let me implement..."   │ <- Real-time caption
├─────────────────────────────────────────┤
│  Problem Statement (Coding Phase)      │ <- Show during phase 3
│  [Two Sum: Given an array, find...]    │
├─────────────────────────────────────────┤
│  Code Editor                           │ <- Monaco Editor or CodeMirror
│  def two_sum(nums, target):            │
│      hash_map = {}                     │
│      ...                               │
└─────────────────────────────────────────┘
```

**New Frontend State:**
```javascript
const interviewState = {
  phase: 'intro', // 'intro' | 'resume' | 'coding' | 'questions'
  phaseStartTime: timestamp,
  totalStartTime: timestamp,
  currentProblem: {
    title: "Two Sum",
    description: "...",
    difficulty: "medium"
  },
  code: "",
  lastCodeChange: timestamp,
  lastSpeech: timestamp,
};
```

**Code Editor Choice:**
- **Recommendation: Monaco Editor**
  - Same editor as VS Code
  - Syntax highlighting, auto-indent
  - Lightweight CDN version available
  - Easy integration: `<script src="monaco-cdn.js">`

### B. Backend Changes

**Enhanced State Management** (`state.py`):
```python
interview_sessions = {
    'session_id': {
        'started_at': datetime,
        'current_phase': 'intro',
        'phase_started_at': datetime,
        'problem': {
            'id': 'two_sum',
            'title': '...',
            'description': '...',
            'hints': [...],
            'optimal_approach': 'hash_map',
        },
        'current_code': "",
        'code_snapshots': [],  # History for analysis
        'last_speech_at': datetime,
        'last_code_change_at': datetime,
        'intervention_count': 0,
    }
}
```

**New API Endpoints:**
```python
POST /api/start_interview
  → Initializes structured interview session
  → Returns: { phase: 'intro', message: "Hi! Ready to start?" }

POST /api/update_code
  Body: { code: "...", timestamp: ... }
  → Updates code snapshot
  → Triggers stuck detection if needed

GET /api/interview_status
  → Returns: { phase, time_elapsed, time_in_phase, problem }

POST /api/chat (enhanced)
  Body: {
    message: "user speech",
    code: "current code",  # NEW
    code_changed: true/false  # NEW
  }
  → Includes interview context in Gemini prompt
```

**Problem Bank** (`problems.py`):
```python
PROBLEMS = {
    'easy': [
        {
            'id': 'two_sum',
            'title': 'Two Sum',
            'description': 'Given an array of integers nums and an integer target...',
            'hints': [
                'Think about what you need to find for each number',
                'Consider using a hash map to store complements',
            ],
            'edge_cases': ['empty array', 'no solution', 'duplicates'],
            'optimal_tc': 'O(n)',
            'optimal_sc': 'O(n)',
        },
        # More easy problems...
    ],
    'medium': [...],
    'hard': [...],
}
```

### C. Gemini Prompt Engineering

**Multi-Section System Prompt:**

```python
def build_interview_prompt(session, user_message, code):
    # Section 1: Core Role
    base_prompt = """
You are a senior software engineer conducting a LIVE VOICE technical interview.
This is a 60-minute structured interview with distinct phases.
Speak naturally, guide intelligently, assess fairly.
"""

    # Section 2: Interview Structure + Current State
    structure = f"""
INTERVIEW STRUCTURE:
- Phase 1: Introduction (0-5min) - Greet, explain format
- Phase 2: Resume Discussion (5-15min) - Background deep-dive
- Phase 3: Coding Problem (15-50min) - Technical assessment
- Phase 4: Candidate Questions (50-60min) - Reverse interview

CURRENT STATE:
Phase: {session['current_phase']}
Time in phase: {calculate_phase_time(session)} minutes
Total time elapsed: {calculate_total_time(session)} minutes
"""

    # Section 3: Phase-Specific Instructions
    if session['current_phase'] == 'intro':
        phase_instructions = """
INTRODUCTION PHASE:
- Introduce yourself briefly (fictional senior engineer)
- Ask if they're ready
- Explain the interview format clearly
- Be warm and put them at ease
- TRANSITION when ready: "Let's start with your background"
"""

    elif session['current_phase'] == 'resume':
        phase_instructions = f"""
RESUME DISCUSSION PHASE:
Resume: {session.get('resume_text', 'Not provided')}

- Ask about their experience, projects, technologies
- Dig deeper into interesting areas
- Assess technical depth through follow-ups
- Keep it conversational, not interrogation
- TRANSITION at ~10min: "Let's move to a coding problem"
"""

    elif session['current_phase'] == 'coding':
        problem = session['problem']
        phase_instructions = f"""
CODING PROBLEM PHASE:

PROBLEM:
{problem['description']}

STAGES (guide them through these):
1. Problem Clarification - Ensure they understand, answer questions
2. Approach Discussion - Make them explain approach BEFORE coding
   - Ask about edge cases
   - Discuss time/space complexity
   - Don't let them start coding until approach is clear
3. Implementation - Let them code with minimal interruption
   - Brief check-ins every 2-3 minutes
   - Only interrupt if stuck (30s+ silence, no code progress)
   - Never give direct solutions, only hints
4. Review - Discuss edge cases, optimizations, testing

CURRENT CODE:
{code if code else "[No code written yet]"}

HINTS (use sparingly, only if stuck):
{problem['hints']}

INTERVENTION RULES:
- User just spoke: Respond to what they said
- User actively coding (code changing): Let them work, brief check-ins only
- User stuck (silence + no code changes >30s): Offer help
- User wrong approach: Gently redirect with questions
- User asks for hint: Provide one progressive hint
- Time running low (<5min): Guide to wrap up

CODE CHANGED RECENTLY: {code_changed}
TIME SINCE LAST SPEECH: {seconds_since_speech}s
"""

    elif session['current_phase'] == 'questions':
        phase_instructions = """
CANDIDATE QUESTIONS PHASE:
- Ask what questions they have for you
- Answer as a helpful senior engineer
- Be genuine and informative
- Close warmly at end
"""

    # Section 4: Conversation Style
    style_guide = """
CONVERSATION STYLE:
- SHORT responses when they're actively working
- LONGER explanations when presenting problems or answering questions
- Use natural speech patterns, contractions
- One question/point at a time
- Let silence happen when they're thinking/coding
- Be supportive but maintain professionalism
"""

    return base_prompt + structure + phase_instructions + style_guide
```

---

## 4. Intelligent Behavior System

**Decision Engine** (when to speak proactively):

```python
def should_proactive_intervention(session):
    """Determines if Gemini should speak without user prompting"""

    current_time = datetime.now()
    seconds_since_speech = (current_time - session['last_speech_at']).seconds
    seconds_since_code = (current_time - session['last_code_change_at']).seconds
    phase = session['current_phase']
    phase_time = calculate_phase_time(session)

    # Never interrupt if user spoke recently
    if seconds_since_speech < 5:
        return False, None

    # Phase transition needed
    if phase == 'intro' and phase_time > 5:
        return True, "transition_resume"
    if phase == 'resume' and phase_time > 12:
        return True, "transition_coding"
    if phase == 'coding' and phase_time > 35:
        return True, "transition_questions"

    # Stuck detection (coding phase)
    if phase == 'coding':
        no_activity = min(seconds_since_speech, seconds_since_code)

        # Completely stuck
        if no_activity > 40:
            return True, "offer_help"

        # Moderate pause, check in
        if no_activity > 25 and session['intervention_count'] < 3:
            return True, "check_in"

    return False, None
```

---

## 5. Data Flow Examples

### Example 1: Normal Conversation During Coding
```
1. User speaks: "I think I'll use a hash map to store complements"

2. Frontend captures speech → transcript

3. Frontend sends POST /api/chat:
   {
     message: "I think I'll use a hash map to store complements",
     code: "def two_sum(nums, target):\n    # TODO",
     code_changed: false,
     timestamp: ...
   }

4. Backend builds enhanced prompt:
   - Phase: coding
   - Problem: Two Sum description
   - Current code: (shows the TODO comment)
   - User's message
   - Time context: 22 minutes elapsed

5. Gemini receives context, responds:
   "Good choice. What exactly will you store as the key and value in your hash map?"

6. TTS converts to audio, plays back

7. User responds, cycle continues
```

### Example 2: Stuck Detection & Intervention
```
1. User has been silent for 35 seconds
   AND no code changes for 35 seconds
   (User is stuck)

2. Frontend detects via timer check every 5 seconds

3. Frontend sends POST /api/check_in:
   {
     type: "stuck_detected",
     code: "def two_sum(nums, target):\n    hash_map = {}\n    # ???",
     silence_duration: 35
   }

4. Backend builds prompt with context:
   "User has been silent and not coding for 35 seconds. Code shows they started hash map but stopped. They may be stuck on the logic."

5. Gemini responds empathetically:
   "Are you thinking through the logic? Walk me through what you're trying to accomplish in the loop."

6. Audio plays, user un-stuck
```

### Example 3: Phase Transition
```
1. Resume phase time: 11 minutes

2. User finishes explaining a project

3. Backend sees:
   - Phase time > 10 minutes
   - User finished speaking (natural pause)
   - Should transition to coding

4. Gemini prompted with:
   "Time to transition to coding phase. Do so naturally."

5. Gemini says:
   "Great, I have a good sense of your background. Let's work on a coding problem together. I'm going to present a problem, and I want you to first talk through your approach before writing any code. Sound good?"

6. Phase updated to 'coding'
   Problem selected and stored in session
```

---

## 6. Implementation Phases

### Phase 1: State & Structure (Backend)
- [ ] Create `problems.py` with problem bank
- [ ] Enhance `state.py` with interview session tracking
- [ ] Add `POST /api/start_interview` endpoint
- [ ] Add `GET /api/interview_status` endpoint
- [ ] Add `POST /api/update_code` endpoint

### Phase 2: Prompt Engineering
- [ ] Create phase-specific prompt templates
- [ ] Add interview context builder
- [ ] Enhance `gemini_client.py` with structured prompts
- [ ] Test prompt variations for natural transitions

### Phase 3: Frontend - Basic UI
- [ ] Add phase indicator display
- [ ] Add interview timer display
- [ ] Create problem statement panel (hidden until coding phase)
- [ ] Update layout for code editor space

### Phase 4: Frontend - Code Editor
- [ ] Integrate Monaco Editor or CodeMirror
- [ ] Wire up code change tracking
- [ ] Send code snapshots to backend
- [ ] Implement auto-save every 15s

### Phase 5: Intelligent Behavior
- [ ] Create intervention decision logic
- [ ] Implement stuck detection
- [ ] Add proactive check-in system
- [ ] Wire up phase transition triggers

### Phase 6: Testing & Refinement
- [ ] Test full interview flow end-to-end
- [ ] Refine intervention thresholds
- [ ] Adjust prompt based on behavior
- [ ] Polish transitions and timing

---

## 7. Open Questions & Decisions Needed

### Q1: Code Editor Preference
**Option A:** Monaco Editor (VS Code's editor)
- ✅ Pros: Professional, full-featured, syntax highlighting, autocomplete
- ❌ Cons: ~5MB CDN load

**Option B:** CodeMirror
- ✅ Pros: Lighter weight (~500KB)
- ❌ Cons: Less polished

**Option C:** Simple `<textarea>` with syntax highlighting library
- ✅ Pros: Minimal overhead
- ❌ Cons: Basic features only

**👉 YOUR DECISION:** _______________

---

### Q2: Programming Language Support
Should we support multiple languages or start with one?

**Option A:** Python only (simplest for MVP)
- Start simple, validate the concept
- Easy syntax validation with `ast.parse()`
- Add more languages later

**Option B:** Python + JavaScript
- Cover two major languages
- More complex validation logic

**Option C:** Let user choose from 3-4 languages (Python, JS, Java, C++)
- Most flexible
- Significantly more complex

**👉 YOUR DECISION:** _______________

---

### Q3: Problem Selection
How should we select the coding problem?

**Option A:** Random from difficulty tier (easy/medium based on resume)
- Backend analyzes resume, picks appropriate difficulty
- Random selection within that tier

**Option B:** Gemini chooses based on resume analysis
- More intelligent matching
- LLM call overhead

**Option C:** User selects difficulty at start
- User control
- Might not match their actual level

**👉 YOUR DECISION:** _______________

---

### Q4: Code Submission
How does the user "submit" their code for review?

**Option A:** Gemini just sees it in real-time, no explicit submission
- Most natural flow
- User says "I'm done" or similar

**Option B:** User says "I'm done" verbally (voice command)
- Clear signal
- Requires speech detection

**Option C:** Click a "Submit for Review" button
- Explicit control
- Less natural for voice interview

**👉 YOUR DECISION:** _______________

---

### Q5: Stuck Detection Thresholds
How long should we wait before interrupting?

**During approach discussion (thinking time):**
- Suggested: 45 seconds
- **👉 YOUR PREFERENCE:** _____ seconds

**During active coding (implementation):**
- Suggested: 30 seconds (if no code changes)
- **👉 YOUR PREFERENCE:** _____ seconds

**Max interventions per coding phase:**
- Suggested: 4-5 times
- **👉 YOUR PREFERENCE:** _____ times

**Phase transition flexibility:**
- Suggested: ±3 minutes
- **👉 YOUR PREFERENCE:** ±_____ minutes

---

### Q6: Initial Problem Set Size
How many problems should we include in the initial version?

**Option A:** 3 problems (1 easy, 1 medium, 1 hard)
- Minimal viable set
- Quick to implement

**Option B:** 10 problems (4 easy, 4 medium, 2 hard)
- Good variety
- Moderate effort

**Option C:** 20+ problems (comprehensive bank)
- Full experience
- Significant upfront work

**👉 YOUR DECISION:** _______________

---

### Q7: Resume Upload Requirement
Should resume upload be required to start the interview?

**Option A:** Required - no resume, no interview
- More realistic experience
- Gemini has better context

**Option B:** Optional - can proceed without resume
- Faster to test
- Generic questions if no resume

**👉 YOUR DECISION:** _______________

---

### Q8: Code Execution
Should we actually run the code, or just have Gemini review it?

**Option A:** No execution - Gemini reviews only
- Simpler implementation
- Focus on approach and logic

**Option B:** Execute with test cases in sandbox
- More realistic
- Requires secure sandboxing (docker, etc.)

**Option C:** Hybrid - Execute simple test cases, Gemini reviews approach
- Best of both worlds
- Moderate complexity

**👉 YOUR DECISION:** _______________

---

## 8. Recommendations

Based on the analysis, here are my recommendations:

| Question | Recommendation | Rationale |
|----------|----------------|-----------|
| **Q1: Code Editor** | Monaco Editor | Professional feel worth the load time |
| **Q2: Languages** | Python only (MVP) | Simpler validation, can expand later |
| **Q3: Problem Selection** | Random from appropriate tier | Balances simplicity with intelligence |
| **Q4: Code Submission** | No explicit submission | Most natural for voice interview |
| **Q5: Thresholds** | 45s planning / 30s coding / max 4 interventions | Balances help vs. autonomy |
| **Q6: Problem Set** | 10 problems initially | Good variety without excessive work |
| **Q7: Resume** | Required | Better context for realistic interview |
| **Q8: Code Execution** | Gemini review only (MVP) | Focus on conversation quality first |

---

## 9. Next Steps - Your Action Items

Please review this document and provide:

1. **Answers to Q1-Q8** (fill in your decisions above)

2. **Priority ordering** - Which implementation phases should we tackle first?
   - Phase 1: State & Structure (Backend)
   - Phase 2: Prompt Engineering
   - Phase 3: Frontend - Basic UI
   - Phase 4: Frontend - Code Editor
   - Phase 5: Intelligent Behavior
   - Phase 6: Testing & Refinement

3. **Any modifications** to the proposed architecture or flows

4. **Green light to start** - Once you've reviewed, I'll begin implementation based on your decisions

---

---

## 10. FINAL DECISIONS (User Approved)

### Core Architecture Decisions
- **Challenge B (Code Awareness)**: ✅ Hybrid approach - send code with voice + periodic snapshots every 15-20s
- **Challenge D (Code Analysis)**: ✅ Rely primarily on LLM to determine correctness, with basic syntax validation

### Feature Decisions
- **Q1 - Code Editor**: ✅ Monaco Editor (minimal features, NO autocomplete or fancy features)
- **Q2 - Languages**: ✅ Python, Java, C, C++ (no compilation/execution, LLM analysis only)
- **Q3 - Problem Selection**: ✅ Gemini chooses from its knowledge (medium-hard difficulty, or easy with medium-hard follow-ups)
- **Q4 - Code Submission**: ✅ Option A - No explicit submission, Gemini sees code in real-time
- **Q5 - Stuck Detection**: ✅ 45s planning / 30s coding / max 4 interventions / ±3min phase flexibility
- **Q6 - Problem Set**: ✅ NO problem bank - Gemini pulls LeetCode questions from its own knowledge
- **Q7 - Resume**: ✅ Required for better context
- **Q8 - Code Execution**: ✅ No execution - LLM review only

### Implementation Approach
- **ALL PHASES IMPLEMENTED TOGETHER** - Complete end-to-end system since components are interdependent

### Critical Requirements
1. **Gemini must be aware of conversation history** throughout interview
2. **Conversational output only** - speak naturally, not written form
3. **Context-aware responses** - reference what was discussed earlier
4. **Medium-hard difficulty bias** for coding problems

---

**Document Version:** 2.0 (APPROVED)
**Date:** 2025-10-26
**Status:** ✅ Ready for Implementation
