# How the LLM Sees and Edits Code in the Editor

## Overview
This document explains the complete flow of how the Gemini LLM (Large Language Model) interacts with the Monaco code editor in this mock interview application.

## Architecture Flow

### 1. Code Editor (Frontend)
- **Monaco Editor**: A browser-based code editor loaded from CDN
- **Location**: `app/templates/index.html` (lines 87-90, 106-107)
- **Initialization**: `app/static/js/app.js` (lines 377-423)
- **Language Support**: Python, Java, C, C++
- **Visibility**: Editor is hidden until coding phase, or immediately shown in coding-only mode

### 2. Code Reading: How LLM Sees User's Code

#### Step 1: User Types Code
- User types code in the Monaco editor
- Changes are tracked via `monacoEditor.onDidChangeModelContent()` (line 415)
- `lastCodeChangeTime` is updated on every change (line 416)

#### Step 2: Code Snapshot to Backend
Two mechanisms send code to the backend:

**A. Periodic Snapshots (Every 15 seconds)**
```javascript
// app/static/js/app.js:524-539
async function sendCodeSnapshot() {
  const code = monacoEditor.getValue();
  await fetch("/api/update_code", {
    method: "POST",
    body: JSON.stringify({ code }),
  });
}
```

**B. On User Speech (Real-time)**
```javascript
// app/static/js/app.js:595-605
if (interviewActive && monacoEditor) {
  currentCode = monacoEditor.getValue();
  codeChanged = (Date.now() - lastCodeChangeTime) < 5000; // Changed in last 5 seconds
}
```

#### Step 3: Backend Stores Code
```python
# app/app.py:139-159
current_code = payload.get("code", "")
code_changed = payload.get("code_changed", False)

# If no code in payload, use stored snapshot
if interview_state and not current_code:
    stored_code = get_current_code(session_id)
    if stored_code:
        current_code = stored_code

# Update state
if current_code:
    update_code(session_id, current_code)
```

```python
# app/state.py:125-139
def update_code(session_id: str, code: str) -> None:
    """Update current code and track change time"""
    context["interview"]["current_code"] = code
    context["interview"]["last_code_change_at"] = datetime.now()
    # Keep last 10 snapshots for analysis
    snapshots.append({
        "code": code,
        "timestamp": datetime.now()
    })
```

#### Step 4: Code Included in LLM Prompt
```python
# app/gemini_client.py:477-835
def _build_phase_prompt(...):
    # Code is embedded in the system prompt
    if current_code.strip():
        code_display = f"\n=== CURRENT CODE IN EDITOR ===\n{current_code}\n=== END OF CODE ===\n"
    else:
        code_display = "\n=== CURRENT CODE IN EDITOR ===\n[No code written yet]\n=== END OF CODE ===\n"
```

The LLM receives the code in this format within the system prompt:
```
=== CURRENT CODE IN EDITOR ===
def two_sum(nums, target):
    seen = {}
    for i, num in enumerate(nums):
        if target - num in seen:
            return [seen[target - num], i]
        seen[num] = i
=== END OF CODE ===
```

#### Step 5: LLM Instructions for Reading Code
```python
# app/gemini_client.py:762-773
"""
CODE ANALYSIS:
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
"""
```

### 3. Code Writing: How LLM Edits User's Code

#### Step 1: LLM Generates Code with Markers
The LLM is instructed to write code using special markers:
```python
# app/gemini_client.py:656-673
"""
To write code to the editor, you MUST use this EXACT format:
1. First, say what you're doing (e.g., "Let me write that for you")
2. Then add the code block with these EXACT markers:

[CODE_START]
your complete code here
[CODE_END]

3. Then optionally explain what you wrote

Example response when asked to write code:
"Okay, let me write a solution for you. [CODE_START]
def two_sum(nums, target):
    seen = {}
    for i, num in enumerate(nums):
        if target - num in seen:
            return [seen[target - num], i]
        seen[num] = i
[CODE_END] This solution uses a hash map for O(n) time complexity."
"""
```

#### Step 2: Backend Strips Code from TTS
```python
# app/app.py:206-207, 463-469
# Strip code markers before TTS (code will be displayed in editor, not spoken)
tts_text = _strip_code_markers(reply_text)

def _strip_code_markers(text: str) -> str:
    """Remove code markers from text before TTS"""
    pattern = r'\[CODE_START\].*?\[CODE_END\]'
    cleaned = re.sub(pattern, '', text, flags=re.DOTALL)
    return cleaned.strip()
```

#### Step 3: Frontend Extracts Code
```javascript
// app/static/js/app.js:731-756
function extractGeminiCode(replyText) {
  const codeStartMarker = '[CODE_START]';
  const codeEndMarker = '[CODE_END]';
  
  const startIndex = replyText.indexOf(codeStartMarker);
  const endIndex = replyText.indexOf(codeEndMarker);
  
  if (startIndex !== -1 && endIndex !== -1 && endIndex > startIndex) {
    // Extract the code between markers
    const code = replyText.substring(
      startIndex + codeStartMarker.length,
      endIndex
    ).trim();
    
    // Extract the text (everything except the code block)
    const textBefore = replyText.substring(0, startIndex).trim();
    const textAfter = replyText.substring(endIndex + codeEndMarker.length).trim();
    const text = (textBefore + ' ' + textAfter).trim();
    
    return { text, code };
  }
  
  return { text: replyText, code: null };
}
```

#### Step 4: Code Applied to Editor
```javascript
// app/static/js/app.js:631-648
// Extract and apply code if Gemini wrote code
const { text, code } = extractGeminiCode(data.reply);

// Display the text part (without code markers)
appendMessage("Gemini", text, "model");

// If Gemini wrote code, update the editor
if (code && monacoEditor) {
  console.log("Gemini wrote code to editor:", code.substring(0, 100) + "...");
  monacoEditor.setValue(code);  // REPLACES ALL EDITOR CONTENT
  // Immediately sync the updated code back to the backend
  await sendCodeSnapshot();
  appendSystemMessage("📝 Gemini updated the code editor");
}
```

**Important**: `monacoEditor.setValue(code)` **replaces ALL content** in the editor. The LLM is instructed to provide complete code, not partial edits.

### 4. Complete Data Flow Diagram

```
┌─────────────────┐
│  Monaco Editor  │  User types code
│  (Frontend)     │
└────────┬────────┘
         │
         │ 1. User types/changes code
         ▼
┌─────────────────┐
│  JavaScript     │  Track changes, send snapshots
│  app.js         │  - Every 15 seconds
└────────┬────────┘  - On user speech
         │
         │ 2. POST /api/chat or /api/update_code
         │    { code: "...", code_changed: true/false }
         ▼
┌─────────────────┐
│  Flask Backend  │  Store in session state
│  app.py         │  - update_code(session_id, code)
└────────┬────────┘  - get_current_code(session_id)
         │
         │ 3. Include in LLM prompt
         │    "=== CURRENT CODE IN EDITOR ==="
         ▼
┌─────────────────┐
│  Gemini LLM     │  Reads code, generates response
│  gemini_client  │  - Analyzes code
└────────┬────────┘  - May write code: [CODE_START]...code...[CODE_END]
         │
         │ 4. Response with code markers
         ▼
┌─────────────────┐
│  Flask Backend  │  Strip code from TTS text
│  app.py         │  - _strip_code_markers()
└────────┬────────┘
         │
         │ 5. JSON response
         │    { reply: "...", replyAudio: "..." }
         ▼
┌─────────────────┐
│  JavaScript     │  Extract code, update editor
│  app.js         │  - extractGeminiCode()
└────────┬────────┘  - monacoEditor.setValue(code)
         │
         │ 6. Update Monaco Editor
         ▼
┌─────────────────┐
│  Monaco Editor  │  Code displayed to user
│  (Frontend)     │
└─────────────────┘
```

## Key Features

### Code Visibility for LLM
- **Always Current**: Code is sent on every user speech + periodic snapshots
- **Fallback**: If frontend omits code, backend uses last stored snapshot
- **Explicit Format**: Code is clearly marked in prompt: `=== CURRENT CODE IN EDITOR ===`
- **Instructions**: LLM is explicitly told to read and analyze code before responding

### Code Editing by LLM
- **Marker-Based**: Uses `[CODE_START]...[CODE_END]` markers
- **Complete Replacement**: `setValue()` replaces all editor content
- **Immediate Sync**: Updated code is immediately sent back to backend
- **TTS Separation**: Code is stripped from audio (not spoken)

### State Management
- **Session Storage**: Code stored per session in `state.py`
- **Change Tracking**: Tracks when code last changed (`last_code_change_at`)
- **Snapshots**: Keeps last 10 code snapshots for analysis
- **Persistence**: Code persists even if frontend doesn't send it in a request

## Limitations & Considerations

### Current Limitations
1. **Complete Replacement Only**: LLM cannot make partial edits - it must provide complete code
2. **No Diff/Patch**: No mechanism for incremental updates
3. **No Syntax Highlighting in Prompt**: LLM sees plain text, not formatted code
4. **Token Limits**: Code is included in system prompt (counts toward token limits)

### Design Decisions
1. **Snapshot Strategy**: 15-second snapshots ensure LLM always has recent code
2. **Marker-Based Editing**: Simple, reliable mechanism for code extraction
3. **Immediate Sync**: Code written by LLM is immediately synced back to backend
4. **Fallback Mechanism**: Backend preserves last code snapshot if frontend omits it

## Files Involved

### Frontend
- `app/templates/index.html`: Monaco editor HTML container
- `app/static/js/app.js`: 
  - Editor initialization (lines 377-423)
  - Code snapshot sending (lines 524-539)
  - Code extraction from LLM response (lines 731-756)
  - Editor update (lines 642-648)

### Backend
- `app/app.py`: 
  - Code reception (lines 139-159)
  - Code marker stripping (lines 463-469)
  - Code update endpoint (lines 426-441)
- `app/gemini_client.py`: 
  - Prompt building with code (lines 477-835)
  - Code reading instructions (lines 762-773)
  - Code writing instructions (lines 656-673, 710-728, 782-800)
- `app/state.py`: 
  - Code storage (lines 125-139)
  - Code retrieval (lines 142-148)

## Testing the Flow

1. **Test Code Reading**:
   - Type code in editor
   - Ask: "What do you think of my code?"
   - LLM should reference your specific code

2. **Test Code Writing**:
   - Ask: "Can you write a solution for me?"
   - LLM should use `[CODE_START]...[CODE_END]` markers
   - Code should appear in editor
   - System message should show "📝 Gemini updated the code editor"

3. **Test Code Persistence**:
   - Type code
   - Wait 15 seconds (snapshot sent)
   - Speak without mentioning code
   - LLM should still see your code (from stored snapshot)

## Future Improvements

1. **Partial Edits**: Support for diff/patch-based edits
2. **Syntax Highlighting**: Include formatted code in prompts
3. **Multi-file Support**: Support for multiple files in editor
4. **Cursor Position**: Track and restore cursor position
5. **Undo/Redo**: Better integration with editor undo/redo
6. **Real-time Sync**: WebSocket-based real-time code sync

