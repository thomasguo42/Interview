# Phase Transition & Monaco Editor Fixes

## Issue Reported
1. ❌ Monaco Editor not showing up
2. ❌ AI keeps asking resume questions without transitioning to coding

## Root Cause
- **Phase auto-detection was too strict** - looking for exact keyword matches
- **Gemini wasn't being explicit enough** about using transition keywords
- **No fallback mechanism** if auto-detection failed

---

## ✅ Fixes Implemented

### 1. **Improved Phase Detection** (`app/app.py`)

**Before:** Only 3-4 keywords per phase
```python
if any(phrase in reply_lower for phrase in ["coding problem", "let's work on"]):
```

**After:** 10+ keywords + time-based fallback
```python
coding_keywords = ["coding problem", "let's work on", "technical problem",
                  "algorithm", "write some code", "solve a problem",
                  "coding challenge", "let's code", "move to coding",
                  "move to the coding", "technical question"]

if any(phrase in reply_lower for phrase in coding_keywords) or time_in_phase > 12:
    update_interview_phase(session_id, PHASE_CODING)
```

**Time-based fallbacks:**
- **Intro → Resume**: After 5 minutes
- **Resume → Coding**: After 12 minutes
- **Coding → Questions**: After 40 minutes

**Debug logging added:**
```python
print(f"[PHASE TRANSITION] Resume → Coding (keyword match or time: {time_in_phase:.1f} min)")
```

---

### 2. **Explicit Gemini Prompts** (`app/gemini_client.py`)

**Added to Resume Phase Prompt:**
```
CRITICAL - WHEN TO TRANSITION TO CODING:
After about 10 minutes (or after 3-4 resume questions), you MUST transition to the coding phase.
To trigger the transition, you MUST include one of these EXACT phrases in your response:
  - "Let's move to coding"
  - "Let's work on a coding problem"
  - "Time for a coding challenge"

Example transition: "Thanks for sharing that. Let's move to coding now. I'll present a problem and I want you to talk through your approach first before coding. Ready?"

NOTE: The system detects these keywords to show the code editor. Without them, the editor won't appear!
```

This tells Gemini **exactly what to say** to trigger the transition.

---

### 3. **Manual Force Transition Button** (New UI Feature)

**What:** A "Force Coding Phase →" button appears in the timer header

**When:** Automatically shows after **8 minutes** in Resume phase

**Where:** In the sticky timer header at the top

**How it works:**
1. User clicks button
2. Sends `POST /api/transition_phase` with `{ phase: "coding" }`
3. Backend updates phase immediately
4. Frontend shows code editor + problem panel
5. System message: "Manually transitioned to coding phase. Code editor is now available."

**Button styling:**
- Semi-transparent white background
- White border
- Appears seamlessly in timer header
- Hover effect

---

### 4. **Improved Monaco Editor Initialization**

**Enhanced `updatePhaseUI()` function:**

```javascript
// Show/hide code editor and problem panel based on phase
if (currentPhase === 'coding') {
  console.log("CODING PHASE ACTIVE - showing code editor");
  elements.codeEditorSection.style.display = "block";
  elements.forceCodingBtn.style.display = "none";

  // Initialize Monaco if not already done
  if (!monacoEditor) {
    console.log("Monaco Editor not initialized, initializing now...");
    const language = document.getElementById("language-select")?.value || "python";
    await initializeMonacoEditor(language);
  }

  if (status.problemPresented) {
    elements.problemPanel.style.display = "block";
  }
}
```

**Key improvements:**
- ✅ Checks if coding phase is active
- ✅ Shows code editor section
- ✅ Initializes Monaco if not already initialized
- ✅ Shows problem panel when problem is presented
- ✅ Adds debug logging

---

## 🔍 How to Debug Phase Transitions

### Check Server Logs
The backend now prints phase transitions:
```
[PHASE TRANSITION] Intro → Resume (keyword match or time: 3.2 min)
[PHASE TRANSITION] Resume → Coding (keyword match or time: 10.5 min)
[PROBLEM PRESENTED] Detected problem presentation keywords
```

### Check Browser Console
The frontend logs phase changes:
```
CODING PHASE ACTIVE - showing code editor
Monaco Editor not initialized, initializing now...
Monaco Editor initialized successfully
Showing force coding button - stuck in resume phase for 9.2 minutes
```

### Manual Override
If auto-detection fails:
1. Wait 8 minutes in Resume phase
2. "Force Coding Phase →" button appears in timer
3. Click it to manually transition
4. Code editor appears immediately

---

## 🎯 Expected Behavior Now

### Normal Flow (Auto-Detection)
1. **Upload resume** → Start interview
2. **0-5 min**: Introduction phase
3. **5-15 min**: Resume discussion
   - Gemini asks 3-4 questions about your background
   - Around 10 minutes, Gemini says: "Let's move to coding now..."
   - **Phase automatically transitions** to Coding
4. **Code Editor appears!**
   - Monaco Editor loads
   - Problem statement panel visible
   - You can start coding

### Fallback Flow (Manual Override)
1. If Gemini doesn't transition after **8 minutes** in Resume phase
2. **"Force Coding Phase →" button appears** in timer header
3. **Click the button**
4. **Instant transition** to Coding phase
5. Code editor + problem panel appear

### Guaranteed Transition (Time-based)
Even if keywords aren't detected and you don't click the button:
- **After 12 minutes in Resume phase** → Auto-transitions to Coding
- **After 40 minutes in Coding phase** → Auto-transitions to Questions

---

## 🧪 Testing Instructions

### Test 1: Verify Auto-Detection
1. Start a new interview
2. Answer resume questions
3. Watch for Gemini saying "Let's move to coding" or similar
4. **Check:** Code editor should appear automatically
5. **Check console:** Should see "CODING PHASE ACTIVE - showing code editor"

### Test 2: Verify Manual Override
1. Start a new interview
2. Spend 8+ minutes in Resume phase (answer questions slowly)
3. **Check:** "Force Coding Phase →" button should appear in timer
4. Click the button
5. **Check:** Code editor appears immediately
6. **Check:** System message appears in conversation log

### Test 3: Verify Time-based Fallback
1. Start a new interview
2. Stay silent or just say "uh huh" in Resume phase
3. Wait 12+ minutes
4. **Check:** Should auto-transition to Coding even without keywords

### Test 4: Verify Monaco Editor Works
1. Get to Coding phase (any method above)
2. **Check:** Code editor container is visible
3. **Check:** Can type code in the editor
4. **Check:** Syntax highlighting works
5. **Check:** Code sends to Gemini when you speak

---

## 📊 Changes Summary

### Files Modified
1. **`/workspace/app/app.py`** (Lines 412-453)
   - Added 10+ keywords per phase
   - Added time-based fallbacks
   - Added debug logging

2. **`/workspace/app/gemini_client.py`** (Lines 335-344)
   - Made prompt MUCH more explicit
   - Added exact phrases Gemini must use
   - Added note about why keywords are needed

3. **`/workspace/app/templates/index.html`** (Lines 21-23)
   - Added force transition button to timer header

4. **`/workspace/app/static/css/style.css`** (Lines 234-250)
   - Added styling for force transition button

5. **`/workspace/app/static/js/app.js`** (Multiple locations)
   - Added force transition button element
   - Added click handler (lines 297-320)
   - Improved `updatePhaseUI()` (lines 406-456)
   - Added Monaco initialization check in coding phase

### Lines of Code Changed
- **Backend**: ~50 lines modified/added
- **Frontend**: ~70 lines modified/added
- **Total**: ~120 lines

---

## ✅ Verification Checklist

After restarting the server and clearing cache:

- [ ] Timer appears when interview starts
- [ ] Phase indicator shows "Phase 2: Resume Discussion"
- [ ] Resume questions are asked (3-4 questions)
- [ ] After ~10 min, Gemini says "Let's move to coding" or similar
- [ ] Phase indicator changes to "Phase 3: Coding Problem"
- [ ] **Code editor section appears** with Monaco Editor
- [ ] Can type code in the editor
- [ ] Problem statement panel shows the coding problem

**If auto-transition doesn't work:**
- [ ] After 8 minutes in Resume, "Force Coding Phase →" button appears
- [ ] Clicking button transitions to Coding phase immediately
- [ ] Code editor appears
- [ ] System message confirms manual transition

**Fallback guarantee:**
- [ ] After 12 minutes in Resume, automatically transitions even without keywords

---

## 🚀 Next Steps

1. **Restart Flask server:**
   ```bash
   ./run_mock_interview.sh
   ```

2. **Clear browser cache:**
   - Hard refresh: `Ctrl + Shift + R` (Windows/Linux) or `Cmd + Shift + R` (Mac)

3. **Test the interview:**
   - Upload a real resume
   - Start interview
   - Answer resume questions
   - **Watch for phase transition** around 10 minutes
   - **Or click "Force Coding Phase →" button** if it appears
   - Verify code editor shows up

4. **Check logs:**
   - Server terminal: Look for `[PHASE TRANSITION]` messages
   - Browser console (F12): Look for "CODING PHASE ACTIVE" messages

---

## 💡 Pro Tips

### To Force Quick Transition (for testing):
- Answer resume questions quickly
- After 3-4 questions, wait 8 minutes
- Click "Force Coding Phase →" button
- Or just wait 12 minutes for automatic transition

### To Debug:
- Open browser DevTools (F12) → Console tab
- Watch for phase change messages
- Check server terminal for backend logs

### If Monaco Editor Still Doesn't Show:
1. Check console for errors
2. Verify phase is actually "coding" (check phase indicator)
3. Try manually calling `updatePhaseUI()` in console
4. Check if `elements.codeEditorSection` exists in console

---

**Document Version:** 1.0
**Date:** 2025-10-26
**Status:** ✅ Ready for Testing
