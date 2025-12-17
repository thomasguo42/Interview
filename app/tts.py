from __future__ import annotations

import base64
import io
import os
import threading
import time
from typing import Iterable, Optional

import numpy as np
import soundfile as sf
import logging

print("[KOKORO DEBUG] Starting import checks...")

try:
    print("[KOKORO DEBUG] Importing kokoro...")
    from kokoro import KPipeline
    print("[KOKORO DEBUG] ✓ kokoro imported successfully")
    
    print("[KOKORO DEBUG] Importing torch...")
    import torch
    print("[KOKORO DEBUG] ✓ torch imported successfully")
    
    print("[KOKORO DEBUG] Importing pydub...")
    from pydub import AudioSegment
    print("[KOKORO DEBUG] ✓ pydub imported successfully")
    
    KOKORO_AVAILABLE = True
    print("[KOKORO DEBUG] ✓ All imports successful, KOKORO_AVAILABLE = True")
    
except ImportError as e:  # pragma: no cover - Kokoro optional at runtime
    print(f"[KOKORO DEBUG] ✗ Import failed: {e}")
    KPipeline = None  # type: ignore[assignment]
    torch = None  # type: ignore[assignment]
    AudioSegment = None  # type: ignore[assignment]
    KOKORO_AVAILABLE = False
    print("[KOKORO DEBUG] ✗ KOKORO_AVAILABLE = False")

DEFAULT_SAMPLE_RATE = 24_000
DEFAULT_VOICE = os.getenv("KOKORO_VOICE", "af_heart")

# Voice mapping for natural conversation flow
VOICE_ALIASES = {
    "default": "af_heart",      # Natural female voice for interviewer
    "female": "af_heart",       # Female voice
    "male": "am_adam",          # Male voice
    "coach": "am_adam",         # Male coach voice
    "mentor": "af_sarah",       # Alternative female mentor voice
    "professor": "am_adam",     # Professor voice
    "interviewer": "af_heart",  # Interviewer voice
}


class KokoroUnavailable(RuntimeError):
    """Raised when the Kokoro engine is missing or fails to synthesize audio."""


class KokoroSynthesizer:
    """
    Kokoro TTS synthesizer for natural, human-like speech generation.
    
    This implementation follows the proper Kokoro usage pattern from the listening module
    to ensure maximum naturalness and quality. It generates MP3 audio with proper
    natural pauses and voice characteristics suitable for interview conversations.
    """

    def __init__(
        self,
        voice: str = DEFAULT_VOICE,
        language: str = "a",  # 'a' enables Kokoro's auto language routing
    ) -> None:
        self.voice = voice or DEFAULT_VOICE
        self.language = language
        self._pipeline: Optional[KPipeline] = None
        self._lock = threading.Lock()
        self._sample_rate = DEFAULT_SAMPLE_RATE

    def synthesize_base64(self, text: str, voice: str | None = None) -> str:
        """
        Generate natural-sounding speech using Kokoro TTS and return as base64-encoded WAV.
        
        This method follows the proper Kokoro usage pattern for maximum naturalness:
        1. Uses proper voice selection and mapping
        2. Generates audio with natural pauses based on punctuation
        3. Exports as WAV for browser compatibility
        4. Returns base64-encoded data for web playback
        """
        print(f"[KOKORO DEBUG] synthesize_base64 called with text: '{text[:50]}...'")
        print(f"[KOKORO DEBUG] Voice requested: {voice}, will resolve to: {self._resolve_voice(voice or self.voice)}")
        
        if not text.strip():
            print("[KOKORO DEBUG] ERROR: Empty text provided")
            raise KokoroUnavailable("Cannot synthesize empty text.")

        if not KOKORO_AVAILABLE:
            print("[KOKORO DEBUG] ERROR: Kokoro not available")
            raise KokoroUnavailable("Kokoro package is not installed. Install kokoro, torch, and pydub.")

        print("[KOKORO DEBUG] Ensuring pipeline is loaded...")
        pipeline = self._ensure_pipeline()
        resolved_voice = self._resolve_voice(voice or self.voice)
        print(f"[KOKORO DEBUG] Using voice: {resolved_voice}")

        try:
            print(f"[KOKORO DEBUG] Calling pipeline with text length: {len(text)}")
            # Generate audio with Kokoro - this follows the proper pattern from kokoro.py
            gen = pipeline(text, voice=resolved_voice)
            print("[KOKORO DEBUG] Pipeline generator created, processing chunks...")
            
            # Collect audio chunks properly
            audio_chunks = []
            chunk_count = 0
            for chunk_data in gen:
                chunk_count += 1
                print(f"[KOKORO DEBUG] Processing chunk {chunk_count}")
                
                # Handle different chunk formats
                if len(chunk_data) >= 3:
                    _, _, audio_data = chunk_data
                else:
                    audio_data = chunk_data
                
                print(f"[KOKORO DEBUG] Chunk {chunk_count} audio_data type: {type(audio_data)}")
                
                if isinstance(audio_data, torch.Tensor):
                    print(f"[KOKORO DEBUG] Chunk {chunk_count} is torch.Tensor, shape: {audio_data.shape}")
                    audio_chunks.append(audio_data.detach().cpu().numpy())
                else:
                    print(f"[KOKORO DEBUG] Chunk {chunk_count} is other type, converting to numpy")
                    audio_chunks.append(np.asarray(audio_data))
                
                print(f"[KOKORO DEBUG] Chunk {chunk_count} processed, total chunks: {len(audio_chunks)}")

            print(f"[KOKORO DEBUG] Total chunks collected: {len(audio_chunks)}")
            if not audio_chunks:
                print("[KOKORO DEBUG] ERROR: No audio chunks produced")
                raise KokoroUnavailable("Kokoro produced no audio output.")

            print("[KOKORO DEBUG] Concatenating audio chunks...")
            # Combine all audio chunks
            combined_audio = np.concatenate(audio_chunks)
            print(f"[KOKORO DEBUG] Combined audio shape: {combined_audio.shape}, dtype: {combined_audio.dtype}")
            print(f"[KOKORO DEBUG] Audio range: min={combined_audio.min():.4f}, max={combined_audio.max():.4f}")

            # Convert to WAV format for browser compatibility
            print("[KOKORO DEBUG] Converting to WAV format...")
            buf = io.BytesIO()
            sf.write(buf, combined_audio, self._sample_rate, format='WAV')
            wav_bytes = buf.getvalue()
            print(f"[KOKORO DEBUG] WAV bytes generated, length: {len(wav_bytes)}")
            
            # Return base64-encoded WAV (browser compatible)
            base64_audio = base64.b64encode(wav_bytes).decode("ascii")
            print(f"[KOKORO DEBUG] Base64 encoding complete, length: {len(base64_audio)}")
            print("[KOKORO DEBUG] synthesize_base64 completed successfully")
            return base64_audio

        except Exception as exc:
            print(f"[KOKORO DEBUG] ERROR in synthesize_base64: {exc}")
            print(f"[KOKORO DEBUG] Exception type: {type(exc)}")
            import traceback
            print(f"[KOKORO DEBUG] Traceback: {traceback.format_exc()}")
            raise KokoroUnavailable(f"Kokoro failed to synthesize audio: {exc}") from exc

    def _ensure_pipeline(self) -> KPipeline:
        """Ensure Kokoro pipeline is loaded with proper configuration."""
        print("[KOKORO DEBUG] _ensure_pipeline called")
        
        if not KOKORO_AVAILABLE:
            print("[KOKORO DEBUG] ERROR: KOKORO_AVAILABLE is False")
            print(f"[KOKORO DEBUG] KPipeline: {KPipeline}")
            print(f"[KOKORO DEBUG] torch: {torch}")
            print(f"[KOKORO DEBUG] AudioSegment: {AudioSegment}")
            raise KokoroUnavailable(
                "kokoro package is not installed. Install kokoro, torch, and pydub."
            )

        print(f"[KOKORO DEBUG] KOKORO_AVAILABLE: {KOKORO_AVAILABLE}")
        print(f"[KOKORO DEBUG] Current pipeline: {self._pipeline}")
        
        if self._pipeline is None:
            print("[KOKORO DEBUG] Pipeline is None, initializing...")
            with self._lock:
                if self._pipeline is None:
                    try:
                        print(f"[KOKORO DEBUG] Creating KPipeline with lang_code='{self.language}'")
                        logging.getLogger(__name__).info("Loading Kokoro pipeline...")
                        # Use the proper initialization pattern from kokoro.py
                        self._pipeline = KPipeline(lang_code=self.language)
                        print(f"[KOKORO DEBUG] KPipeline created: {self._pipeline}")
                        
                        self._sample_rate = getattr(self._pipeline, "sample_rate", DEFAULT_SAMPLE_RATE)
                        print(f"[KOKORO DEBUG] Sample rate: {self._sample_rate}")
                        
                        logging.getLogger(__name__).info(
                            "Kokoro pipeline loaded successfully (sample_rate=%s)",
                            self._sample_rate,
                        )
                        print("[KOKORO DEBUG] Pipeline initialization completed successfully")
                    except Exception as e:
                        print(f"[KOKORO DEBUG] ERROR during pipeline initialization: {e}")
                        print(f"[KOKORO DEBUG] Exception type: {type(e)}")
                        import traceback
                        print(f"[KOKORO DEBUG] Traceback: {traceback.format_exc()}")
                        logging.getLogger(__name__).error(f"Failed to load Kokoro pipeline: {e}")
                        raise KokoroUnavailable(f"Failed to load Kokoro pipeline: {e}") from e
        else:
            print("[KOKORO DEBUG] Pipeline already exists, returning existing pipeline")
            
        print(f"[KOKORO DEBUG] Returning pipeline: {self._pipeline}")
        return self._pipeline

    def _resolve_voice(self, requested: str) -> str:
        print(f"[KOKORO DEBUG] _resolve_voice called with: '{requested}'")
        key = (requested or "").strip().lower()
        print(f"[KOKORO DEBUG] Resolved key: '{key}'")
        print(f"[KOKORO DEBUG] Available voices: {list(VOICE_ALIASES.keys())}")
        
        if key in VOICE_ALIASES:
            resolved = VOICE_ALIASES[key]
            print(f"[KOKORO DEBUG] Voice resolved to: '{resolved}'")
            return resolved
        
        print(f"[KOKORO DEBUG] Voice not found in aliases, using original: '{requested}'")
        return requested



print("[KOKORO DEBUG] Creating KokoroSynthesizer instance...")
kokoro = KokoroSynthesizer()
print(f"[KOKORO DEBUG] KokoroSynthesizer created: {kokoro}")
print(f"[KOKORO DEBUG] Default voice: {kokoro.voice}")
print(f"[KOKORO DEBUG] Language: {kokoro.language}")
print("[KOKORO DEBUG] Module initialization complete")
