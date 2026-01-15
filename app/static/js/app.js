console.log("app.js starting to load...");

// Wait for DOM to be fully loaded
document.addEventListener('DOMContentLoaded', function() {
  console.log("DOM loaded, initializing structured interview app...");

  const elements = {
    resumeForm: document.getElementById("resume-form"),
    resumeFile: document.getElementById("resume-file"),
    resumeStatus: document.getElementById("resume-status"),
    companyRoleForm: document.getElementById("company-role-form"),
    companyInput: document.getElementById("company-input"),
    roleInput: document.getElementById("role-input"),
    companyRoleDetails: document.getElementById("company-role-details"),
    companyRoleStatus: document.getElementById("company-role-status"),
    languageSelect: document.getElementById("language-select"),
    startButton: document.getElementById("start-btn"),
    stopButton: document.getElementById("stop-btn"),
    resetButton: document.getElementById("reset-btn"),
    conversationLog: document.getElementById("conversation-log"),
    messageTemplate: document.getElementById("message-template"),
    liveCaption: document.getElementById("live-caption"),
    interviewTimer: document.getElementById("interview-timer"),
    phaseIndicator: document.getElementById("phase-indicator"),
    timeDisplay: document.getElementById("time-display"),
    problemPanel: document.getElementById("problem-panel"),
    problemStatement: document.getElementById("problem-statement"),
    codeEditorSection: document.getElementById("code-editor-section"),
    codeEditorDiv: document.getElementById("code-editor"),
    forceCodingBtn: document.getElementById("force-coding-btn"),
    // OOD mode elements
    oodWorkspace: document.getElementById("ood-workspace"),
    oodCodeEditorDiv: document.getElementById("ood-code-editor"),
  };

  console.log("Elements found:", elements);

  loadCompanyRoleContext();

  const SpeechRecognition =
    window.SpeechRecognition || window.webkitSpeechRecognition || null;

  let recognition;

  // ============================================================================
  // AUDIO-LEVEL ECHO DETECTION - Define early so it's accessible everywhere
  // ============================================================================

  /**
   * Initialize microphone stream with echo cancellation enabled
   * and set up audio level monitoring
   */
  async function setupAudioMonitoring() {
    try {
      // Request microphone access with echo cancellation
      microphoneStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,  // Browser-level echo cancellation!
          noiseSuppression: true,
          autoGainControl: true,
          sampleRate: 48000
        }
      });

      console.log("[AUDIO] Microphone stream obtained with echo cancellation enabled");

      // Create audio analyzer for input monitoring
      if (!audioContext) {
        audioContext = new AudioContext();
      }

      const micSource = audioContext.createMediaStreamSource(microphoneStream);
      audioAnalyzer = audioContext.createAnalyser();
      audioAnalyzer.fftSize = 256;
      micSource.connect(audioAnalyzer);

      console.log("[AUDIO] Audio analyzer connected");

      // Start monitoring input levels
      startLevelMonitoring();

    } catch (error) {
      console.error("[AUDIO] Failed to set up audio monitoring:", error);
    }
  }

  /**
   * Monitor microphone input levels in real-time
   */
  function startLevelMonitoring() {
    if (levelMonitorInterval) return;

    const bufferLength = audioAnalyzer?.frequencyBinCount || 0;
    const dataArray = new Uint8Array(bufferLength);

    levelMonitorInterval = setInterval(() => {
      if (!audioAnalyzer) return;

      audioAnalyzer.getByteFrequencyData(dataArray);

      // Calculate RMS (root mean square) level
      let sum = 0;
      for (let i = 0; i < bufferLength; i++) {
        sum += dataArray[i] * dataArray[i];
      }
      const rms = Math.sqrt(sum / bufferLength);
      micInputLevel = rms / 255; // Normalize to 0-1

      // Log when significant audio detected
      if (micInputLevel > 0.1) {
        console.log(`[AUDIO LEVEL] Mic input: ${(micInputLevel * 100).toFixed(1)}%, AI output: ${(aiOutputLevel * 100).toFixed(1)}%`);
      }
    }, 100); // Check every 100ms
  }

  function stopLevelMonitoring() {
    if (levelMonitorInterval) {
      clearInterval(levelMonitorInterval);
      levelMonitorInterval = null;
    }
  }

  /**
   * Check if current audio input is likely echo based on level comparison
   */
  function isAudioLevelEcho() {
    if (!isAiSpeaking) return false;
    if (aiOutputLevel < 0.1) return false; // AI not loud enough to cause echo
    if (micInputLevel < 0.1) return false; // No significant input

    // If mic input level is suspiciously close to AI output level, it's likely echo
    const levelRatio = micInputLevel / Math.max(aiOutputLevel, 0.01);

    if (levelRatio > AUDIO_LEVEL_ECHO_THRESHOLD && levelRatio < 1.5) {
      console.log(`[AUDIO ECHO] Level ratio: ${levelRatio.toFixed(2)} (mic: ${(micInputLevel * 100).toFixed(1)}%, AI: ${(aiOutputLevel * 100).toFixed(1)}%)`);
      return true;
    }

    return false;
  }

  /**
   * Pause speech recognition when AI starts speaking.
   * Will only resume when AI is completely done speaking.
   * NO BARGE-IN - Simple and reliable.
   */
  function pauseRecognitionForAI() {
    if (!recognition || recognitionPausedForEcho) return;

    console.log(`[NO BARGE-IN] Pausing speech recognition - AI is speaking`);

    recognitionPausedForEcho = true;

    // Stop recognition completely
    if (isListening) {
      try {
        recognition.stop();
      } catch (err) {
        console.warn("[NO BARGE-IN] Error stopping recognition:", err);
      }
    }

    // DO NOT RESUME - will only resume when AI finishes (in playReply finally block)
  }

  /**
   * Resume speech recognition after AI is done speaking
   */
  function resumeRecognitionAfterAI() {
    recognitionPausedForEcho = false;

    if (micEnabled && !isAiSpeaking) {
      console.log("[NO BARGE-IN] AI finished - resuming speech recognition");
      tryStartListening();
    }
  }

  // Now declare all state variables
  let micEnabled = false;
  let isListening = false;
  let isAiSpeaking = false;
  let awaitingResponse = false;
  let lastInterim = "";
  let audioContext;
  let currentAudioSource;
  let currentChatController = null;

  // Audio monitoring for echo detection
  let microphoneStream = null;
  let audioAnalyzer = null;
  let micInputLevel = 0;
  let aiOutputLevel = 0;
  let levelMonitorInterval = null;

  // Echo cancellation state
  let currentAiText = "";
  let aiSpeechStartTime = 0;
  let consecutiveInterimCount = 0;
  let lastEchoCheckText = "";
  let recognitionPausedForEcho = false;
  let pendingFinalTranscript = "";
  let pendingFinalTimer = null;

  const FINAL_TRANSCRIPT_DELAY_MS = 500;

  const CODE_SUPPORTED_MODES = new Set(["full", "coding_only", "ood"]);

  function modeSupportsEditor(mode = currentMode) {
    return CODE_SUPPORTED_MODES.has(mode);
  }

  // Tunable parameters
  const ECHO_PROTECTION_PAUSE_MS = 2000; // Pause recognition for 2 seconds when AI starts
  const ECHO_GRACE_PERIOD_MS = 1000;
  const MIN_BARGE_IN_CONFIDENCE = 0.7;
  const MAX_TEXT_SIMILARITY = 0.35;
  const MIN_CONSECUTIVE_INTERIM = 3;
  const ECHO_EXTENDED_PERIOD_MS = 3000;
  const AUDIO_LEVEL_ECHO_THRESHOLD = 0.7; // If input level is 70%+ of output level, likely echo

  // Structured interview state
  let monacoEditor = null;
  let interviewActive = false;
  let interviewPaused = false;
  let currentPhase = "intro";
  let currentMode = "full"; // Track interview mode
  let interviewStartTime = null;
  let phaseStartTime = null;
  let timerInterval = null;
  let codeSnapshotInterval = null;
  let stuckDetectionInterval = null;
  let lastCodeChangeTime = Date.now();
  let codeSyncTimeout = null;
  let lastSyncedCode = "";
  let lastSpeechTime = Date.now();
  let problemStatementSet = false; // Track if initial problem is set
  let problemUpdatesCount = 0; // Track number of updates

  // OOD interview state
  let oodMonacoEditor = null; // Separate editor for OOD sessions

  if (!SpeechRecognition) {
    setStatus(
      "Your browser does not support continuous speech recognition. Use Chrome/Edge on desktop.",
      "error"
    );
    console.error("SpeechRecognition not available:", {
      SpeechRecognition: window.SpeechRecognition,
      webkitSpeechRecognition: window.webkitSpeechRecognition,
      userAgent: navigator.userAgent
    });
  } else {
    console.log("SpeechRecognition available, initializing...");
    recognition = new SpeechRecognition();
    recognition.lang = "en-US";
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.maxAlternatives = 1;

    recognition.onstart = () => {
      console.log("Speech recognition started");
      isListening = true;
      elements.liveCaption.textContent = "Listening…";
      updateControlStates();
    };

    recognition.onerror = (event) => {
      console.error("Speech recognition error:", event.error);

      // Handle different error types differently
      if (event.error === 'no-speech') {
        // No-speech is not a critical error - just means silence was detected
        // Don't disable the mic, let onend handler restart it
        console.log("No speech detected, recognition will auto-restart");
        return;
      }

      if (event.error === 'aborted') {
        // Aborted is expected when we manually stop
        console.log("Speech recognition aborted (expected)");
        return;
      }

      // For actual errors (not-allowed, network, etc.), disable and show error
      setStatus(
        "Speech recognition failed. Refresh the page and re-enable the microphone.",
        "error"
      );
      micEnabled = false;
      isListening = false;
      updateControlStates();
    };

    recognition.onend = () => {
      isListening = false;
      if (micEnabled && !recognitionPausedForEcho) {
        setTimeout(() => tryStartListening(), 150);
      } else {
        elements.liveCaption.textContent = recognitionPausedForEcho ? "Recognition paused (echo protection)..." : "Microphone idle.";
        updateControlStates();
      }
    };

  // ============================================================================
  // ECHO CANCELLATION FUNCTIONS
  // ============================================================================
  /**
   * MULTI-LAYER ECHO CANCELLATION SYSTEM
   *
   * Problem: The AI's voice from speakers gets picked up by the microphone,
   * causing the system to think the user is speaking and interrupt itself.
   *
   * Solution: Audio-based + text-based multi-layer defense:
   *
   * Hardware/OS Layer:
   *   - Browser echo cancellation (echoCancellation: true in getUserMedia)
   *   - Processes audio at driver level before reaching our code
   *
   * Layer 0 (Recognition Pause): Pause speech recognition when AI starts
   *   - First 2 seconds: recognition completely disabled
   *   - Prevents any echo from being detected at all
   *   - After 2 seconds: resume for barge-in capability
   *
   * Layer 1 (Audio Level Detection): Monitor and compare audio levels
   *   - Track microphone input level (RMS)
   *   - Compare to AI output level
   *   - If levels suspiciously similar (70%+ ratio) → block (echo!)
   *
   * Layer 2 (Grace Period): Ignore all speech for first second after resume
   *   - Additional protection after pause ends
   *
   * Layer 3 (Substring/Sequential Match): Check for partial echo
   *   - Is recognized text a substring of AI's text?
   *   - Do words appear in sequence within AI's text?
   *   - If yes → block (echo!)
   *
   * Layer 4 (Prefix Match): Check beginning of AI speech
   *   - Do first words match AI's response?
   *   - If yes → block (echo!)
   *
   * Layer 5 (Confidence Threshold): Stricter during first 3 seconds
   *   - Normal: 0.7, First 3s: 0.85
   *
   * Layer 6 (General Similarity): Final backstop
   *   - Jaccard similarity check
   *
   * Layer 7 (Length Check): Short utterances suspicious
   *
   * Layer 8 (Consecutive Detection): Require 3+ consecutive interim results
   *
   * All layers work together for robust echo rejection while preserving barge-in.
   */

  /**
   * Calculate word overlap similarity between two texts
   * Returns value between 0 (no match) and 1 (identical)
   */
  function calculateTextSimilarity(text1, text2) {
    if (!text1 || !text2) return 0;

    // Normalize and extract words (filter out short words)
    const words1 = text1.toLowerCase().split(/\s+/).filter(w => w.length > 2);
    const words2 = text2.toLowerCase().split(/\s+/).filter(w => w.length > 2);

    if (words1.length === 0 || words2.length === 0) return 0;

    const set1 = new Set(words1);
    const set2 = new Set(words2);

    // Count matching words
    let matchCount = 0;
    set2.forEach(word => {
      if (set1.has(word)) matchCount++;
    });

    // Calculate Jaccard similarity
    return matchCount / Math.max(set1.size, set2.size);
  }

  /**
   * Check if recognized text appears as a substring or sequential words in AI's text
   * This catches partial echoes (e.g., AI saying "hello how are you" and mic picking up "hello how")
   */
  function isSequentialMatch(recognizedText, aiText) {
    if (!recognizedText || !aiText) return false;

    const recognized = recognizedText.toLowerCase().trim();
    const ai = aiText.toLowerCase().trim();

    // Direct substring check (very strong indicator of echo)
    if (ai.includes(recognized)) {
      console.log(`[ECHO DETECT] Direct substring match found`);
      return true;
    }

    // Extract words from recognized text
    const recognizedWords = recognized.split(/\s+/).filter(w => w.length > 2);
    if (recognizedWords.length < 2) return false; // Need at least 2 words for sequential check

    // Check if words appear in sequence within AI's text
    const aiWords = ai.split(/\s+/);

    // Try to find all recognized words in order within AI's text
    let aiIndex = 0;
    let matchedWords = 0;

    for (let recWord of recognizedWords) {
      let found = false;
      // Look for this word starting from current position in AI text
      for (let i = aiIndex; i < aiWords.length; i++) {
        if (aiWords[i].includes(recWord) || recWord.includes(aiWords[i])) {
          found = true;
          aiIndex = i + 1; // Move past this match
          matchedWords++;
          break;
        }
      }
      if (!found) break; // Words not in sequence
    }

    // If 80%+ of recognized words appear in sequence in AI text, it's echo
    const sequenceRatio = matchedWords / recognizedWords.length;
    if (sequenceRatio >= 0.8) {
      console.log(`[ECHO DETECT] Sequential match: ${(sequenceRatio * 100).toFixed(0)}% of words in order`);
      return true;
    }

    return false;
  }

  /**
   * Check for word-level prefix match (first N words match)
   */
  function isPrefixMatch(recognizedText, aiText) {
    if (!recognizedText || !aiText) return false;

    const recWords = recognizedText.toLowerCase().trim().split(/\s+/).filter(w => w.length > 2);
    const aiWords = aiText.toLowerCase().trim().split(/\s+/).filter(w => w.length > 2);

    if (recWords.length < 2) return false;

    // Check if first N words of AI text match recognized words
    let matches = 0;
    const checkLength = Math.min(recWords.length, 5); // Check first 5 words max

    for (let i = 0; i < checkLength && i < aiWords.length; i++) {
      if (recWords[i] && aiWords[i] &&
          (recWords[i] === aiWords[i] ||
           recWords[i].includes(aiWords[i]) ||
           aiWords[i].includes(recWords[i]))) {
        matches++;
      }
    }

    // If 70%+ of first words match, it's likely echo
    const matchRatio = matches / checkLength;
    if (matchRatio >= 0.7) {
      console.log(`[ECHO DETECT] Prefix match: ${(matchRatio * 100).toFixed(0)}% of initial words match`);
      return true;
    }

    return false;
  }

  /**
   * Determine if detected speech is legitimate barge-in or echo
   * Returns true only if all anti-echo checks pass
   */
  function shouldAllowBargeIn(transcript, confidence = 1.0) {
    if (!isAiSpeaking) return false;
    if (recognitionPausedForEcho) {
      console.log(`[ECHO FILTER] Recognition currently paused for echo protection`);
      return false;
    }

    const timeSinceAiStarted = Date.now() - aiSpeechStartTime;
    const inExtendedPeriod = timeSinceAiStarted < ECHO_EXTENDED_PERIOD_MS;

    // Layer 0: Audio level check - if input level matches output level, it's echo
    if (isAudioLevelEcho()) {
      console.log(`[ECHO FILTER] Audio level echo detected`);
      return false;
    }

    // Layer 1: Absolute grace period - NEVER allow barge-in in first second
    if (timeSinceAiStarted < ECHO_GRACE_PERIOD_MS) {
      console.log(`[ECHO FILTER] Absolute grace period (${timeSinceAiStarted}ms < ${ECHO_GRACE_PERIOD_MS}ms)`);
      return false;
    }

    // Layer 2: Prevent duplicate detections of same text
    if (transcript === lastEchoCheckText) {
      console.log(`[ECHO FILTER] Duplicate text detection (same as last check)`);
      return false;
    }
    lastEchoCheckText = transcript;

    // Layer 3: Substring/Sequential match check - MOST IMPORTANT for partial echo
    // This catches when AI says "hello how are you" and mic picks up "hello how"
    if (isSequentialMatch(transcript, currentAiText)) {
      console.log(`[ECHO FILTER] Sequential/substring match detected`);
      console.log(`[ECHO FILTER] AI text: "${currentAiText.substring(0, 80)}..."`);
      console.log(`[ECHO FILTER] Recognized: "${transcript}"`);
      return false;
    }

    // Layer 4: Prefix match check - catches echo of beginning of AI speech
    if (isPrefixMatch(transcript, currentAiText)) {
      console.log(`[ECHO FILTER] Prefix match detected`);
      console.log(`[ECHO FILTER] AI text: "${currentAiText.substring(0, 80)}..."`);
      console.log(`[ECHO FILTER] Recognized: "${transcript}"`);
      return false;
    }

    // Layer 5: Confidence check - stricter during extended period
    const minConfidence = inExtendedPeriod ? 0.85 : MIN_BARGE_IN_CONFIDENCE;
    if (confidence < minConfidence) {
      console.log(`[ECHO FILTER] Low confidence (${confidence.toFixed(2)} < ${minConfidence.toFixed(2)})`);
      return false;
    }

    // Layer 6: General similarity check as final backstop
    const similarity = calculateTextSimilarity(transcript, currentAiText);
    const maxSimilarity = inExtendedPeriod ? 0.2 : MAX_TEXT_SIMILARITY; // Stricter in first 3 seconds
    if (similarity > maxSimilarity) {
      console.log(`[ECHO FILTER] High similarity (${(similarity * 100).toFixed(1)}% > ${(maxSimilarity * 100).toFixed(1)}%)`);
      console.log(`[ECHO FILTER] AI text: "${currentAiText.substring(0, 60)}..."`);
      console.log(`[ECHO FILTER] Recognized: "${transcript}"`);
      return false;
    }

    // Layer 7: Length check - very short utterances during AI speech are suspicious
    if (transcript.trim().split(/\s+/).length < 3 && inExtendedPeriod) {
      console.log(`[ECHO FILTER] Too short during extended period (${transcript.trim().split(/\s+/).length} words)`);
      return false;
    }

    // ALL checks passed - this appears to be legitimate user speech
    console.log(`[BARGE-IN ALLOWED] ✓ Legitimate user interruption detected`);
    console.log(`[BARGE-IN] Time: ${timeSinceAiStarted}ms, Confidence: ${confidence.toFixed(2)}, Similarity: ${(similarity * 100).toFixed(1)}%`);
    console.log(`[BARGE-IN] User said: "${transcript}"`);
    return true;
  }

  recognition.onresult = (event) => {
    console.log("Speech recognition result:", event);
    let interimTranscript = "";

    for (let i = event.resultIndex; i < event.results.length; i += 1) {
      const result = event.results[i];
      const transcript = result[0].transcript.trim();
      const confidence = result[0].confidence || 1.0; // Some browsers don't provide confidence for interim

      console.log("Transcript result:", {
        transcript,
        isFinal: result.isFinal,
        confidence: confidence.toFixed(2)
      });

      if (!transcript) {
        continue;
      }

      if (result.isFinal) {
        console.log("Final transcript:", transcript);
        lastInterim = "";
        consecutiveInterimCount = 0; // Reset on final result
        lastSpeechTime = Date.now();
        queueFinalTranscript(transcript);
      } else {
        console.log("Interim transcript:", transcript);

        // NO BARGE-IN: Just accumulate interim transcript, never interrupt AI
        interimTranscript += `${transcript} `;
      }
    }

    lastInterim = interimTranscript.trim();
    if (lastInterim) {
      elements.liveCaption.textContent = lastInterim;
    } else if (micEnabled) {
      elements.liveCaption.textContent = "Listening…";
    }
  };

}

// ============================================================================
// INITIALIZATION & STATUS
// ============================================================================

async function fetchStatus() {
  try {
    const response = await fetch("/api/status");
    if (!response.ok) throw new Error("Failed to fetch status");
    const status = await response.json();
    if (status.resumeLoaded) {
      enableInterviewControls();
      if (elements.resetButton) {
        elements.resetButton.disabled = false;
      }
      setStatus(
        "Resume loaded. Choose mode and click start.",
        "success"
      );
    }
  } catch (error) {
    console.warn("Status check failed:", error);
  }
}

// Enable start button for coding-only mode even without resume
function updateStartButtonForMode() {
  const modeRadio = document.querySelector('input[name="interview-mode"]:checked');
  const mode = modeRadio ? modeRadio.value : "full";

  // For coding_only mode, enable start button even without resume
  if (mode === "coding_only" && !interviewActive) {
    if (elements.startButton) {
      elements.startButton.disabled = false;
    }
  }
}

// Listen for mode changes
document.querySelectorAll('input[name="interview-mode"]').forEach(radio => {
  radio.addEventListener('change', updateStartButtonForMode);
});

// Check mode on page load
setTimeout(updateStartButtonForMode, 100);

// ============================================================================
// COMPANY / ROLE CONTEXT
// ============================================================================

async function loadCompanyRoleContext() {
  if (!elements.companyInput || !elements.roleInput || !elements.companyRoleDetails) {
    return;
  }
  try {
    const response = await fetch("/api/company_context", {
      method: "GET",
      credentials: "same-origin",
    });
    if (!response.ok) return;
    const data = await response.json();
    elements.companyInput.value = data.company || "";
    elements.roleInput.value = data.role || "";
    elements.companyRoleDetails.value = data.details || "";
  } catch (error) {
    console.warn("Failed to load company context:", error);
  }
}

async function saveCompanyRoleContext() {
  if (!elements.companyRoleForm) return;
  const company = elements.companyInput?.value.trim() || "";
  const role = elements.roleInput?.value.trim() || "";
  const details = elements.companyRoleDetails?.value.trim() || "";

  if (elements.companyRoleStatus) {
    elements.companyRoleStatus.textContent = "Saving role context...";
    elements.companyRoleStatus.className = "status-message info";
  }

  try {
    const response = await fetch("/api/company_context", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      credentials: "same-origin",
      body: JSON.stringify({ company, role, details }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Failed to save role context");
    if (elements.companyRoleStatus) {
      elements.companyRoleStatus.textContent = data.message || "Saved";
      elements.companyRoleStatus.className = "status-message success";
    }
  } catch (error) {
    console.error("Company context error:", error);
    if (elements.companyRoleStatus) {
      elements.companyRoleStatus.textContent = error.message || "Failed to save";
      elements.companyRoleStatus.className = "status-message error";
    }
  }
}

// ============================================================================
// RESUME UPLOAD
// ============================================================================

elements.resumeForm?.addEventListener("submit", async (event) => {
  event.preventDefault();
  const formData = new FormData(elements.resumeForm);
  setStatus("Uploading resume...", "info");
  disableInterviewControls();

  try {
    const response = await fetch("/api/upload_resume", {
      method: "POST",
      body: formData,
      credentials: "same-origin",
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Upload failed");
    setStatus(data.message || "Resume uploaded.", "success");
    enableInterviewControls();
    if (elements.resetButton) {
      elements.resetButton.disabled = false;
    }
  } catch (error) {
    console.error(error);
    setStatus(error.message, "error");
  }
});

elements.companyRoleForm?.addEventListener("submit", async (event) => {
  event.preventDefault();
  await saveCompanyRoleContext();
});

// ============================================================================
// INTERVIEW CONTROLS
// ============================================================================

elements.startButton?.addEventListener("click", async () => {
  console.log("Start button clicked");
  if (interviewActive && interviewPaused) {
    console.log("Resuming paused interview");
    interviewPaused = false;
    micEnabled = true;
    updateControlStates();
    startTimers();
    await setupAudioMonitoring();
    tryStartListening();
    setStatus("Interview resumed. Pick up where you left off.", "success");
    return;
  }
  if (!SpeechRecognition) {
    console.error("SpeechRecognition not available when start button clicked");
    return;
  }

  // Get selected mode and language
  const modeRadio = document.querySelector('input[name="interview-mode"]:checked');
  const mode = modeRadio ? modeRadio.value : "full";
  const language = elements.languageSelect.value;
  console.log("Starting interview with mode:", mode, "language:", language);

  try {
    const response = await fetch("/api/start_interview", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ language, mode }),
      credentials: "same-origin",
    });

    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Failed to start interview");

    console.log("=== INTERVIEW START DEBUG ===");
    console.log("Mode sent:", mode);
    console.log("Response data:", data);
    console.log("Starting phase:", data.phase);
    console.log("Starting mode:", data.mode);

    interviewActive = true;
    interviewPaused = false;
    currentPhase = data.phase;
    currentMode = data.mode; // Store the mode
    interviewStartTime = Date.now();
    phaseStartTime = Date.now();

    // Show interview timer
    elements.interviewTimer.style.display = "block";

    // Initialize based on mode
    if (data.mode === "ood") {
      console.log("OOD MODE - initializing unified OOD code editor");
      await initializeOodCodeEditor(language);
      // Show OOD workspace
      elements.oodWorkspace.style.display = "block";
      // Hide regular code editor and problem panel
      elements.codeEditorSection.style.display = "none";
      elements.problemPanel.style.display = "none";
    } else {
      // Initialize Monaco Editor FIRST (before updatePhaseUI)
      await initializeMonacoEditor(language);
      console.log("Monaco Editor initialized");

      // Hide OOD workspace
      if (elements.oodWorkspace) {
        elements.oodWorkspace.style.display = "none";
      }
    }

    // Update phase UI first to set the currentPhase correctly
    await updatePhaseUI();

    // For coding-only mode or coding phase, ensure editor is shown
    if (data.mode === "coding_only" || data.phase === "coding" || currentPhase === "coding") {
      console.log("CODING MODE DETECTED - ensuring code editor and problem panel are visible");
      elements.codeEditorSection.style.display = "block";
      elements.problemPanel.style.display = "block";
    }

    // Start timers
    startTimers();

    // Enable microphone and set up audio monitoring
    micEnabled = true;
    updateControlStates();

    // Initialize audio monitoring with echo cancellation
    await setupAudioMonitoring();

    tryStartListening();
    setStatus("Interview started! Speak when ready.", "success");

    // Add system message
    if (data.mode === "coding_only") {
      appendSystemMessage("Coding-only session started. Say hello to get your coding problem.");
    } else if (data.mode === "ood") {
      appendSystemMessage("OOD interview started. Say hello to receive your design question. Use the unified code editor for outlining your design, pseudocode, and final implementation.");
    } else {
      appendSystemMessage("Structured interview started. The interviewer will introduce themselves.");
    }

  } catch (error) {
    console.error("Failed to start interview:", error);
    setStatus(error.message, "error");
  }
});

elements.stopButton?.addEventListener("click", () => {
  micEnabled = false;
  if (recognition && isListening) {
    recognition.stop();
  }

  if (pendingFinalTimer) {
    clearTimeout(pendingFinalTimer);
    pendingFinalTimer = null;
  }
  pendingFinalTranscript = "";

  stopTimers();
  stopLevelMonitoring();
  if (interviewActive) {
    interviewPaused = true;
  }
  updateControlStates();
  setStatus("Interview paused. Press start to resume.", "info");
});

elements.resetButton?.addEventListener("click", async () => {
  try {
    const response = await fetch("/api/reset", {
      method: "POST",
      credentials: "same-origin",
    });
    if (!response.ok) throw new Error("Failed to reset conversation");
    elements.conversationLog.innerHTML = "";

    // Reset interview state
    interviewActive = false;
    interviewPaused = false;
    currentPhase = "intro";
    currentMode = "full"; // Reset mode
    problemStatementSet = false; // Reset problem tracking
    problemUpdatesCount = 0;
    elements.interviewTimer.style.display = "none";
    elements.problemPanel.style.display = "none";
    elements.codeEditorSection.style.display = "none";
    elements.forceCodingBtn.style.display = "none";
    elements.problemStatement.innerHTML = "Problem will appear here during the coding phase...";

    // Reset OOD workspace
    if (elements.oodWorkspace) {
      elements.oodWorkspace.style.display = "none";
    }

    stopTimers();
    if (pendingFinalTimer) {
      clearTimeout(pendingFinalTimer);
      pendingFinalTimer = null;
    }
    pendingFinalTranscript = "";
    if (codeSyncTimeout) {
      clearTimeout(codeSyncTimeout);
      codeSyncTimeout = null;
    }
    lastSyncedCode = "";

    setStatus("Conversation reset. Resume remains loaded.", "info");
  } catch (error) {
    setStatus(error.message, "error");
  }
});

// Force transition to coding phase button
elements.forceCodingBtn?.addEventListener("click", async () => {
  try {
    console.log("Force transition to coding phase");
    const response = await fetch("/api/transition_phase", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ phase: "coding" }),
      credentials: "same-origin",
    });

    if (!response.ok) throw new Error("Failed to transition phase");

    // Immediately update UI
    currentPhase = "coding";
    updatePhaseUI();
    elements.forceCodingBtn.style.display = "none";

    appendSystemMessage("Manually transitioned to coding phase. Code editor is now available.");
    console.log("Successfully transitioned to coding phase");
  } catch (error) {
    console.error("Failed to force transition:", error);
    setStatus(error.message, "error");
  }
});

// ============================================================================
// MONACO EDITOR INITIALIZATION
// ============================================================================

async function initializeMonacoEditor(language) {
  console.log("Initializing Monaco Editor for language:", language);

  return new Promise((resolve, reject) => {
    require.config({
      paths: { vs: 'https://cdnjs.cloudflare.com/ajax/libs/monaco-editor/0.45.0/min/vs' }
    });

    require(['vs/editor/editor.main'], function() {
      if (monacoEditor) {
        try {
          monacoEditor.dispose();
        } catch (err) {
          console.warn("Failed to dispose previous Monaco instance:", err);
        }
      }

      const languageMap = {
        python: 'python',
        java: 'java',
        c: 'c',
        cpp: 'cpp'
      };

      const monacoLanguage = languageMap[language] || 'python';

      monacoEditor = monaco.editor.create(elements.codeEditorDiv, {
        value: '',
        language: monacoLanguage,
        theme: 'vs',
        automaticLayout: true,
        minimap: { enabled: false },
        fontSize: 14,
        lineNumbers: 'on',
        scrollBeyondLastLine: false,
        wordWrap: 'on',
        // Disable autocomplete and suggestions for minimal features
        quickSuggestions: false,
        suggestOnTriggerCharacters: false,
        acceptSuggestionOnCommitCharacter: false,
        tabCompletion: 'off',
        parameterHints: { enabled: false },
        suggest: { enabled: false }
      });

      // Track code changes
      monacoEditor.onDidChangeModelContent(() => {
        lastCodeChangeTime = Date.now();
        scheduleCodeSync();
      });

      lastSyncedCode = monacoEditor.getValue();
      console.log("Monaco Editor initialized successfully");
      resolve();
    });
  });
}

// ============================================================================
// INTERVIEW TIMING & PHASE MANAGEMENT
// ============================================================================

function startTimers() {
  // Update timer display every second
  timerInterval = setInterval(updateTimerDisplay, 1000);

  // Send code snapshot every 15 seconds
  codeSnapshotInterval = setInterval(sendCodeSnapshot, 15000);

  // Check for stuck detection every 5 seconds
  stuckDetectionInterval = setInterval(checkStuckDetection, 5000);
}

function stopTimers() {
  if (timerInterval) clearInterval(timerInterval);
  if (codeSnapshotInterval) clearInterval(codeSnapshotInterval);
  if (stuckDetectionInterval) clearInterval(stuckDetectionInterval);
  if (codeSyncTimeout) {
    clearTimeout(codeSyncTimeout);
    codeSyncTimeout = null;
  }
}

function updateTimerDisplay() {
  if (!interviewActive || !interviewStartTime) return;

  const totalElapsed = Math.floor((Date.now() - interviewStartTime) / 1000);
  const totalMinutes = Math.floor(totalElapsed / 60);
  const totalSeconds = totalElapsed % 60;

  elements.timeDisplay.textContent =
    `${totalMinutes}:${totalSeconds.toString().padStart(2, '0')} / 60:00`;
}

async function updatePhaseUI() {
  // Fetch current phase from server
  try {
    const response = await fetch("/api/interview_status");
    if (!response.ok) return;

    const status = await response.json();
    if (!status.interviewActive) return;

    currentPhase = status.phase;
    const phaseNames = {
      intro: "Phase 1: Introduction",
      resume: "Phase 2: Resume Discussion",
      coding: "Phase 3: Coding Problem",
      questions: "Phase 4: Your Questions",
      ood_design: "OOD: Design Phase",
      ood_implementation: "OOD: Implementation Phase"
    };

    elements.phaseIndicator.textContent = phaseNames[currentPhase] || currentPhase;

    // Show force coding button if stuck in resume phase for >8 minutes
    if (currentPhase === 'resume' && status.timeInPhase > 8) {
      elements.forceCodingBtn.style.display = "block";
      console.log("Showing force coding button - stuck in resume phase for", status.timeInPhase, "minutes");
    } else {
      elements.forceCodingBtn.style.display = "none";
    }

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

      // Always show problem panel in coding phase (even if problem not formally "presented" yet)
      elements.problemPanel.style.display = "block";
    } else {
      // For coding_only mode, NEVER hide the code editor (it starts in coding phase)
      if (currentMode !== "coding_only") {
        elements.codeEditorSection.style.display = "none";
        elements.problemPanel.style.display = "none";
      } else {
        console.log("Coding-only mode - keeping code editor visible");
      }
    }

  } catch (error) {
    console.error("Failed to update phase UI:", error);
  }
}

// Update phase UI every 2 seconds to catch transitions
setInterval(() => {
  if (interviewActive) {
    updatePhaseUI();
  }
}, 2000);

// ============================================================================
// CODE SNAPSHOT & STUCK DETECTION
// ============================================================================

async function sendCodeSnapshot(options = {}) {
  if (!interviewActive || !modeSupportsEditor()) return;

  // Use the appropriate editor based on mode
  const activeEditor = (currentMode === "ood") ? oodMonacoEditor : monacoEditor;
  if (!activeEditor) return;

  const { skipIfUnchanged = false } = options;
  const code = activeEditor.getValue();

  if (skipIfUnchanged && code === lastSyncedCode) {
    return;
  }

  try {
    await fetch("/api/update_code", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ code }),
      credentials: "same-origin",
    });
    lastSyncedCode = code;
  } catch (error) {
    console.error("Failed to send code snapshot:", error);
  }
}

function scheduleCodeSync() {
  if (!interviewActive || !modeSupportsEditor()) return;
  const activeEditor = (currentMode === "ood") ? oodMonacoEditor : monacoEditor;
  if (!activeEditor) return;
  if (codeSyncTimeout) {
    clearTimeout(codeSyncTimeout);
  }
  codeSyncTimeout = setTimeout(() => {
    sendCodeSnapshot({ skipIfUnchanged: true });
  }, 2000);
}

async function checkStuckDetection() {
  if (!interviewActive || currentPhase !== 'coding') return;
  if (awaitingResponse || isAiSpeaking) return;

  const silenceDuration = (Date.now() - lastSpeechTime) / 1000;
  const codeIdleDuration = (Date.now() - lastCodeChangeTime) / 1000;

  // Stuck if both silence AND no code changes for 30+ seconds
  if (silenceDuration > 30 && codeIdleDuration > 30) {
    console.log("User appears stuck - offering help");
    // The backend will detect this and DeepSeek will offer help based on context
    // We just ensure code is sent with the next interaction
  }
}

// ============================================================================
// CONVERSATION HANDLING
// ============================================================================

function queueFinalTranscript(text) {
  const trimmed = (text || "").trim();
  if (!trimmed) return;

  pendingFinalTranscript = pendingFinalTranscript
    ? `${pendingFinalTranscript} ${trimmed}`.trim()
    : trimmed;

  if (pendingFinalTimer) {
    clearTimeout(pendingFinalTimer);
  }

  pendingFinalTimer = setTimeout(() => {
    const toSend = pendingFinalTranscript.trim();
    pendingFinalTranscript = "";
    pendingFinalTimer = null;
    if (toSend) {
      processFinalTranscript(toSend);
    }
  }, FINAL_TRANSCRIPT_DELAY_MS);
}

function processFinalTranscript(transcript) {
  if (!transcript) return;
  if (!micEnabled) return;

  // NO BARGE-IN: Ignore input if AI is speaking
  if (isAiSpeaking) {
    console.log("[NO BARGE-IN] Ignoring user input while AI is speaking");
    return;
  }

  if (awaitingResponse) {
    interruptInFlight("Interrupting previous reply…");
  }
  handleUserUtterance(transcript);
}

function interruptInFlight(message) {
  let interrupted = false;
  if (currentChatController) {
    currentChatController.abort();
    currentChatController = null;
    interrupted = true;
  }
  awaitingResponse = false;
  cancelPlayback();
  if (message && interrupted) {
    setStatus(message, "info");
  }
}

async function handleUserUtterance(transcript) {
  console.log("handleUserUtterance called with:", transcript);
  lastSpeechTime = Date.now();
  appendMessage("You", transcript, "user");
  cancelPlayback();
  setStatus("Thinking...", "info");

  const controller = new AbortController();
  currentChatController = controller;
  awaitingResponse = true;

  // Get current code from the editor if available
  let currentCode = "";
  let codeChanged = false;
  if (interviewActive && modeSupportsEditor()) {
    // Use the appropriate editor based on mode
    const activeEditor = (currentMode === "ood") ? oodMonacoEditor : monacoEditor;
    if (activeEditor) {
      currentCode = activeEditor.getValue();
      codeChanged = (Date.now() - lastCodeChangeTime) < 5000; // Changed in last 5 seconds
      console.log("=== CODE BEING SENT TO DEEPSEEK ===");
      console.log("Mode:", currentMode);
      console.log("Code length:", currentCode.length);
      console.log("Code preview:", currentCode.substring(0, 200));
      console.log("Code changed recently:", codeChanged);
    }
  }

  try {
    console.log("Sending chat request to /api/chat");
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: transcript,
        code: currentCode,
        code_changed: codeChanged
      }),
      credentials: "same-origin",
      signal: controller.signal,
    });
    console.log("Chat response status:", response.status);
    const data = await response.json();
    console.log("Chat response data:", data);

    if (!response.ok) throw new Error(data.error || "Failed to get reply");
    if (!data.replyAudio) {
      throw new Error("DeepSeek did not return any audio.");
    }

    console.log("Audio received, length:", data.replyAudio.length);

    // FIRST: Extract and apply code if DeepSeek wrote code (do this before problem extraction)
    const { text, code } = extractModelCode(data.reply);
    console.log("Code extraction result:", { hasCode: !!code, textLength: text.length });

    // SECOND: Extract problem statement from the cleaned text (without code markers)
    extractProblemStatement(text);

    // THIRD: Display the text part (without code markers)
    appendMessage("DeepSeek", text, "model");

    // FOURTH: If DeepSeek wrote code, update the editor
    const activeEditor = (currentMode === "ood") ? oodMonacoEditor : monacoEditor;
    if (code && activeEditor) {
      console.log("DeepSeek wrote code to editor:", code.substring(0, 100) + "...");
      activeEditor.setValue(code);
      // Immediately sync the updated code back to the backend so DeepSeek sees the latest version
      await sendCodeSnapshot();
      appendSystemMessage("DeepSeek updated the code editor");
    }

    // FIFTH: Play AI's audio response with echo cancellation
    // Pass the text to enable echo detection
    await playReply(data.replyAudio, text);
    setStatus("Your turn. Continue the conversation.", "success");
  } catch (error) {
    if (error.name === "AbortError") {
      return;
    }
    console.error("Chat error:", error);
    setStatus(error.message || "Interview error.", "error");
  } finally {
    awaitingResponse = false;
    currentChatController = null;
    if (micEnabled && !isListening) {
      console.log("Restarting listening after response");
      tryStartListening();
    }
  }
}

function extractProblemStatement(replyText) {
  if (currentPhase !== 'coding') return;

  const replyLower = replyText.toLowerCase();

  // Detect if this is the initial problem presentation
  const initialProblemKeywords = ['given', 'array', 'return', 'implement', 'design', 'find', 'write a function',
                                   'create a function', 'your task is', 'you need to', 'the problem is'];
  const hasInitialKeywords = initialProblemKeywords.some(kw => replyLower.includes(kw));

  // If this looks like the initial problem (long text with keywords and not yet set)
  if (!problemStatementSet && hasInitialKeywords && replyText.length > 100) {
    console.log("Setting initial problem statement");
    elements.problemStatement.innerHTML = `<div class="problem-section">
      <h3>Problem</h3>
      <p>${escapeHtml(replyText)}</p>
    </div>`;
    elements.problemPanel.style.display = "block";
    problemStatementSet = true;
    return;
  }

  // If initial problem is already set, check if this is additional information to append
  if (problemStatementSet && replyText.length > 30) {
    // Detect what type of update this is
    let updateType = "Update";
    let shouldAppend = false;

    if (replyLower.includes('edge case') || replyLower.includes('corner case')) {
      updateType = "Edge Cases";
      shouldAppend = true;
    } else if (replyLower.includes('constraint') || replyLower.includes('requirement')) {
      updateType = "Constraints";
      shouldAppend = true;
    } else if (replyLower.includes('clarif') || replyLower.includes('example')) {
      updateType = "Clarification";
      shouldAppend = true;
    } else if (replyLower.includes('hint') || replyLower.includes('consider') || replyLower.includes('think about')) {
      updateType = "Hint";
      shouldAppend = true;
    } else if (replyLower.includes('complexity') || replyLower.includes('time') || replyLower.includes('space')) {
      updateType = "Complexity";
      shouldAppend = true;
    }

    if (shouldAppend) {
      console.log(`Appending ${updateType} to problem statement`);
      problemUpdatesCount++;
      const updateHtml = `<div class="problem-section problem-update">
        <h4>${updateType} #${problemUpdatesCount}</h4>
        <p>${escapeHtml(replyText)}</p>
      </div>`;
      elements.problemStatement.innerHTML += updateHtml;
    }
  }
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

function extractModelCode(replyText) {
  // Check if DeepSeek included code using [CODE_START] and [CODE_END] markers
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

  // Fallback: handle Markdown-style triple backtick blocks
  const fenceRegex = /```[\w.+-]*\s*([\s\S]*?)```/;
  const fenceMatch = fenceRegex.exec(replyText);
  if (fenceMatch) {
    const code = fenceMatch[1].trim();
    const before = replyText.slice(0, fenceMatch.index).trim();
    const after = replyText.slice(fenceMatch.index + fenceMatch[0].length).trim();
    const text = `${before} ${after}`.trim();
    if (code) {
      console.warn("Detected code block without Kokoro markers. Applying fallback to update editor.");
      return { text: text || "", code };
    }
  }

  // No code found, return original text
  return { text: replyText, code: null };
}

// ============================================================================
// AUDIO PLAYBACK
// ============================================================================

function appendMessage(speaker, text, role) {
  const fragment = elements.messageTemplate.content.cloneNode(true);
  const messageEl = fragment.querySelector(".message");
  messageEl.classList.add(role);
  fragment.querySelector(".speaker").textContent = speaker;
  fragment.querySelector(".text").textContent = text;
  elements.conversationLog.appendChild(fragment);
  elements.conversationLog.scrollTop = elements.conversationLog.scrollHeight;
}

function appendSystemMessage(text) {
  appendMessage("System", text, "system");
}

async function playReply(replyAudio, replyText = "") {
  console.log("playReply called with audio length:", replyAudio ? replyAudio.length : "null");
  console.log("playReply text preview:", replyText.substring(0, 100));

  if (!replyAudio) {
    throw new Error("Missing Kokoro audio payload.");
  }

  cancelPlayback();

  // Initialize echo cancellation tracking
  currentAiText = replyText;
  aiSpeechStartTime = Date.now();
  consecutiveInterimCount = 0;
  lastEchoCheckText = "";
  aiOutputLevel = 0.5; // Estimate output level (we can't measure it directly)
  isAiSpeaking = true;

  console.log("[NO BARGE-IN] ═══════════════════════════════════════");
  console.log("[NO BARGE-IN] AI speech started");
  console.log(`[NO BARGE-IN] Text (${replyText.length} chars): "${replyText}"`);
  console.log("[NO BARGE-IN] ═══════════════════════════════════════");

  // CRITICAL: Pause speech recognition completely - NO barge-in
  pauseRecognitionForAI();

  try {
    console.log("Starting audio playback...");
    await playKokoroAudio(replyAudio);
    console.log("Audio playback completed");
  } catch (error) {
    console.error("Audio playback failed:", error);
    throw error;
  } finally {
    isAiSpeaking = false;
    currentAiText = "";
    consecutiveInterimCount = 0;
    lastEchoCheckText = "";
    aiOutputLevel = 0;

    console.log("[NO BARGE-IN] AI speech ended");

    // Resume speech recognition now that AI is done
    resumeRecognitionAfterAI();
  }
}

async function playKokoroAudio(base64Audio) {
  console.log("playKokoroAudio called with base64 length:", base64Audio.length);

  try {
    const binary = Uint8Array.from(atob(base64Audio), (char) =>
      char.charCodeAt(0)
    );
    console.log("Binary data created, length:", binary.length);

    if (!audioContext) {
      console.log("Creating new AudioContext");
      audioContext = new AudioContext();
    }
    console.log("AudioContext state:", audioContext.state);

    if (audioContext.state === "suspended") {
      console.log("Resuming suspended AudioContext");
      await audioContext.resume();
    }

    const copy = binary.buffer.slice(
      binary.byteOffset,
      binary.byteOffset + binary.byteLength
    );
    console.log("Buffer slice created, length:", copy.byteLength);

    const audioBuffer = await new Promise((resolve, reject) => {
      console.log("Decoding audio data...");
      audioContext.decodeAudioData(copy, resolve, reject);
    });
    console.log("Audio buffer decoded successfully, duration:", audioBuffer.duration);

    const source = audioContext.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(audioContext.destination);
    currentAudioSource = source;

    console.log("Starting audio playback...");
    await new Promise((resolve) => {
      source.onended = () => {
        console.log("Audio playback ended");
        resolve();
      };
      source.start(0);
    });
    currentAudioSource = null;
    console.log("Audio playback completed successfully");
  } catch (error) {
    console.error("Error in playKokoroAudio:", error);
    throw error;
  }
}

function cancelPlayback() {
  if (currentAudioSource) {
    try {
      currentAudioSource.stop();
    } catch (error) {
      console.warn("Failed to stop audio source", error);
    }
    currentAudioSource = null;
  }
  isAiSpeaking = false;

  // Reset state
  currentAiText = "";
  consecutiveInterimCount = 0;
  lastEchoCheckText = "";
  aiOutputLevel = 0;

  console.log("[NO BARGE-IN] Playback canceled");

  // Resume recognition since AI is no longer speaking
  resumeRecognitionAfterAI();
}

// ============================================================================
// SPEECH RECOGNITION
// ============================================================================

function tryStartListening() {
  console.log("tryStartListening called", {
    SpeechRecognition: !!SpeechRecognition,
    micEnabled,
    isListening,
    recognition: !!recognition
  });

  if (!SpeechRecognition) {
    console.error("SpeechRecognition not available in tryStartListening");
    return;
  }
  if (!micEnabled) {
    console.log("Microphone not enabled");
    return;
  }
  if (isListening) {
    console.log("Already listening");
    return;
  }

  try {
    console.log("Starting speech recognition...");
    recognition.start();
  } catch (error) {
    console.error("Failed to start recognition", error);
    if (error.name !== "InvalidStateError") {
      setStatus("Unable to access microphone. Check browser permissions.", "error");
      micEnabled = false;
      updateControlStates();
    }
  }
}


// ============================================================================
// UI HELPERS
// ============================================================================

function setStatus(message, state) {
  if (!elements.resumeStatus) return;
  elements.resumeStatus.textContent = message;
  elements.resumeStatus.className = `status-message ${state || ""}`;
}

function updateControlStates() {
  if (!elements.startButton || !elements.stopButton) return;
  const shouldDisableStart = micEnabled || (interviewActive && !interviewPaused);
  elements.startButton.disabled = !!shouldDisableStart;
  elements.stopButton.disabled = !micEnabled;
}

function enableInterviewControls() {
  if (!elements.startButton || !elements.stopButton) return;
  if (!micEnabled && (!interviewActive || interviewPaused)) {
    elements.startButton.disabled = false;
  }
  elements.stopButton.disabled = !micEnabled;
}

function disableInterviewControls() {
  if (!elements.startButton || !elements.stopButton) return;
  elements.startButton.disabled = true;
  elements.stopButton.disabled = true;
  elements.resetButton.disabled = true;
}

// ============================================================================
// OOD INTERVIEW - UNIFIED CODE EDITOR
// ============================================================================

async function initializeOodCodeEditor(language) {
  console.log("Initializing OOD Code Editor for language:", language);

  return new Promise((resolve, reject) => {
    require.config({
      paths: { vs: 'https://cdnjs.cloudflare.com/ajax/libs/monaco-editor/0.45.0/min/vs' }
    });

    require(['vs/editor/editor.main'], function() {
      if (oodMonacoEditor) {
        try {
          oodMonacoEditor.dispose();
        } catch (err) {
          console.warn("Failed to dispose previous OOD Monaco instance:", err);
        }
      }

      const languageMap = {
        python: 'python',
        java: 'java',
        c: 'c',
        cpp: 'cpp'
      };

      const monacoLanguage = languageMap[language] || 'python';
      const planningPlaceholder = {
        python: '# Use this editor for design notes, pseudocode, and implementation.\n\n',
        java: '// Use this editor for design notes, pseudocode, and implementation.\n\n',
        c: '// Use this editor for design notes, pseudocode, and implementation.\n\n',
        cpp: '// Use this editor for design notes, pseudocode, and implementation.\n\n'
      };
      const initialValue = planningPlaceholder[monacoLanguage] || '';

      oodMonacoEditor = monaco.editor.create(elements.oodCodeEditorDiv, {
        value: initialValue,
        language: monacoLanguage,
        theme: 'vs',
        automaticLayout: true,
        minimap: { enabled: false },
        fontSize: 14,
        lineNumbers: 'on',
        scrollBeyondLastLine: false,
        wordWrap: 'on',
        quickSuggestions: false,
        suggestOnTriggerCharacters: false,
        acceptSuggestionOnCommitCharacter: false,
        tabCompletion: 'off',
        parameterHints: { enabled: false },
        suggest: { enabled: false }
      });

      // Track code changes
      oodMonacoEditor.onDidChangeModelContent(() => {
        lastCodeChangeTime = Date.now();
        scheduleCodeSync();
      });

      console.log("OOD Code Editor initialized successfully");
      resolve();
    });
  });
}

// Initialize app
fetchStatus();

}); // End of DOMContentLoaded event listener
