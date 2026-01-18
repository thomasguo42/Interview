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

KOKORO_DEBUG = os.getenv("KOKORO_DEBUG", "0") == "1"


def _debug(message: str) -> None:
    if KOKORO_DEBUG:
        print(message)


_debug("[KOKORO DEBUG] Starting import checks...")

try:
    _debug("[KOKORO DEBUG] Importing kokoro...")
    from kokoro import KPipeline
    _debug("[KOKORO DEBUG] ✓ kokoro imported successfully")
    
    _debug("[KOKORO DEBUG] Importing torch...")
    import torch
    _debug("[KOKORO DEBUG] ✓ torch imported successfully")
    
    _debug("[KOKORO DEBUG] Importing pydub...")
    from pydub import AudioSegment
    _debug("[KOKORO DEBUG] ✓ pydub imported successfully")
    
    KOKORO_AVAILABLE = True
    _debug("[KOKORO DEBUG] ✓ All imports successful, KOKORO_AVAILABLE = True")
    
except ImportError as e:  # pragma: no cover - Kokoro optional at runtime
    _debug(f"[KOKORO DEBUG] ✗ Import failed: {e}")
    KPipeline = None  # type: ignore[assignment]
    torch = None  # type: ignore[assignment]
    AudioSegment = None  # type: ignore[assignment]
    KOKORO_AVAILABLE = False
    _debug("[KOKORO DEBUG] ✗ KOKORO_AVAILABLE = False")

DEFAULT_SAMPLE_RATE = 24_000
DEFAULT_VOICE = os.getenv("KOKORO_VOICE", "af_heart")
DEFAULT_SPEED = float(os.getenv("KOKORO_SPEED", "1.1"))
KOKORO_DEVICE = os.getenv("KOKORO_DEVICE")
KOKORO_PREWARM = os.getenv("KOKORO_PREWARM", "0") == "1"

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
        self._speed = DEFAULT_SPEED

    def synthesize_base64(self, text: str, voice: str | None = None) -> str:
        """
        Generate natural-sounding speech using Kokoro TTS and return as base64-encoded WAV.
        
        This method follows the proper Kokoro usage pattern for maximum naturalness:
        1. Uses proper voice selection and mapping
        2. Generates audio with natural pauses based on punctuation
        3. Exports as WAV for browser compatibility
        4. Returns base64-encoded data for web playback
        """
        _debug(f"[KOKORO DEBUG] synthesize_base64 called with text: '{text[:50]}...'")
        _debug(f"[KOKORO DEBUG] Voice requested: {voice}, will resolve to: {self._resolve_voice(voice or self.voice)}")
        
        if not text.strip():
            raise KokoroUnavailable("Cannot synthesize empty text.")

        if not KOKORO_AVAILABLE:
            raise KokoroUnavailable("Kokoro package is not installed. Install kokoro, torch, and pydub.")

        _debug("[KOKORO DEBUG] Ensuring pipeline is loaded...")
        pipeline = self._ensure_pipeline()
        resolved_voice = self._resolve_voice(voice or self.voice)
        _debug(f"[KOKORO DEBUG] Using voice: {resolved_voice}")

        try:
            _debug(f"[KOKORO DEBUG] Calling pipeline with text length: {len(text)}")
            # Generate audio with Kokoro - this follows the proper pattern from kokoro.py
            gen = pipeline(text, voice=resolved_voice, speed=self._speed)
            _debug("[KOKORO DEBUG] Pipeline generator created, processing chunks...")
            
            # Collect audio chunks properly
            audio_chunks = []
            chunk_count = 0
            for chunk_data in gen:
                chunk_count += 1
                _debug(f"[KOKORO DEBUG] Processing chunk {chunk_count}")
                
                # Handle different chunk formats
                if len(chunk_data) >= 3:
                    _, _, audio_data = chunk_data
                else:
                    audio_data = chunk_data
                
                _debug(f"[KOKORO DEBUG] Chunk {chunk_count} audio_data type: {type(audio_data)}")
                
                if isinstance(audio_data, torch.Tensor):
                    _debug(f"[KOKORO DEBUG] Chunk {chunk_count} is torch.Tensor, shape: {audio_data.shape}")
                    audio_chunks.append(audio_data.detach().cpu().numpy())
                else:
                    _debug(f"[KOKORO DEBUG] Chunk {chunk_count} is other type, converting to numpy")
                    audio_chunks.append(np.asarray(audio_data))
                
                _debug(f"[KOKORO DEBUG] Chunk {chunk_count} processed, total chunks: {len(audio_chunks)}")

            _debug(f"[KOKORO DEBUG] Total chunks collected: {len(audio_chunks)}")
            if not audio_chunks:
                raise KokoroUnavailable("Kokoro produced no audio output.")

            _debug("[KOKORO DEBUG] Concatenating audio chunks...")
            # Combine all audio chunks
            combined_audio = np.concatenate(audio_chunks)
            _debug(f"[KOKORO DEBUG] Combined audio shape: {combined_audio.shape}, dtype: {combined_audio.dtype}")
            _debug(f"[KOKORO DEBUG] Audio range: min={combined_audio.min():.4f}, max={combined_audio.max():.4f}")

            # Convert to WAV format for browser compatibility
            _debug("[KOKORO DEBUG] Converting to WAV format...")
            buf = io.BytesIO()
            sf.write(buf, combined_audio, self._sample_rate, format='WAV')
            wav_bytes = buf.getvalue()
            _debug(f"[KOKORO DEBUG] WAV bytes generated, length: {len(wav_bytes)}")
            
            # Return base64-encoded WAV (browser compatible)
            base64_audio = base64.b64encode(wav_bytes).decode("ascii")
            _debug(f"[KOKORO DEBUG] Base64 encoding complete, length: {len(base64_audio)}")
            _debug("[KOKORO DEBUG] synthesize_base64 completed successfully")
            return base64_audio

        except Exception as exc:
            _debug(f"[KOKORO DEBUG] ERROR in synthesize_base64: {exc}")
            _debug(f"[KOKORO DEBUG] Exception type: {type(exc)}")
            import traceback
            _debug(f"[KOKORO DEBUG] Traceback: {traceback.format_exc()}")
            raise KokoroUnavailable(f"Kokoro failed to synthesize audio: {exc}") from exc

    def _ensure_pipeline(self) -> KPipeline:
        """Ensure Kokoro pipeline is loaded with proper configuration."""
        _debug("[KOKORO DEBUG] _ensure_pipeline called")
        
        if not KOKORO_AVAILABLE:
            _debug("[KOKORO DEBUG] ERROR: KOKORO_AVAILABLE is False")
            _debug(f"[KOKORO DEBUG] KPipeline: {KPipeline}")
            _debug(f"[KOKORO DEBUG] torch: {torch}")
            _debug(f"[KOKORO DEBUG] AudioSegment: {AudioSegment}")
            raise KokoroUnavailable(
                "kokoro package is not installed. Install kokoro, torch, and pydub."
            )

        _debug(f"[KOKORO DEBUG] KOKORO_AVAILABLE: {KOKORO_AVAILABLE}")
        _debug(f"[KOKORO DEBUG] Current pipeline: {self._pipeline}")
        
        if self._pipeline is None:
            _debug("[KOKORO DEBUG] Pipeline is None, initializing...")
            with self._lock:
                if self._pipeline is None:
                    try:
                        _debug(f"[KOKORO DEBUG] Creating KPipeline with lang_code='{self.language}'")
                        logging.getLogger(__name__).info("Loading Kokoro pipeline...")
                        device = KOKORO_DEVICE
                        if not device and torch is not None and torch.cuda.is_available():
                            device = "cuda:0"
                        # Use the proper initialization pattern from kokoro.py
                        self._pipeline = KPipeline(lang_code=self.language, device=device)
                        _debug(f"[KOKORO DEBUG] KPipeline created: {self._pipeline}")
                        
                        self._sample_rate = getattr(self._pipeline, "sample_rate", DEFAULT_SAMPLE_RATE)
                        _debug(f"[KOKORO DEBUG] Sample rate: {self._sample_rate}")
                        detected_device = self._detect_pipeline_device(self._pipeline)
                        logging.getLogger(__name__).info(
                            "Kokoro pipeline device: %s",
                            detected_device or "unknown",
                        )
                        _debug(f"[KOKORO DEBUG] Pipeline device: {detected_device or 'unknown'}")
                        
                        logging.getLogger(__name__).info(
                            "Kokoro pipeline loaded successfully (sample_rate=%s)",
                            self._sample_rate,
                        )
                        _debug("[KOKORO DEBUG] Pipeline initialization completed successfully")
                    except Exception as e:
                        _debug(f"[KOKORO DEBUG] ERROR during pipeline initialization: {e}")
                        _debug(f"[KOKORO DEBUG] Exception type: {type(e)}")
                        import traceback
                        _debug(f"[KOKORO DEBUG] Traceback: {traceback.format_exc()}")
                        logging.getLogger(__name__).error(f"Failed to load Kokoro pipeline: {e}")
                        raise KokoroUnavailable(f"Failed to load Kokoro pipeline: {e}") from e
        else:
            _debug("[KOKORO DEBUG] Pipeline already exists, returning existing pipeline")
            
        _debug(f"[KOKORO DEBUG] Returning pipeline: {self._pipeline}")
        return self._pipeline

    def _detect_pipeline_device(self, pipeline: KPipeline) -> Optional[str]:
        """Best-effort device detection for Kokoro pipeline modules."""
        if torch is None:
            return None

        def _iter_values(value: object) -> Iterable[object]:
            if isinstance(value, dict):
                for item in value.values():
                    yield item
            elif isinstance(value, (list, tuple)):
                for item in value:
                    yield item
            else:
                yield value

        seen = set()
        pending = [pipeline]
        while pending:
            current = pending.pop()
            if id(current) in seen:
                continue
            seen.add(id(current))

            if isinstance(current, torch.nn.Module):
                for param in current.parameters():
                    return str(param.device)

            if hasattr(current, "__dict__"):
                for child in _iter_values(vars(current)):
                    pending.append(child)

        return None

    def _resolve_voice(self, requested: str) -> str:
        _debug(f"[KOKORO DEBUG] _resolve_voice called with: '{requested}'")
        key = (requested or "").strip().lower()
        _debug(f"[KOKORO DEBUG] Resolved key: '{key}'")
        _debug(f"[KOKORO DEBUG] Available voices: {list(VOICE_ALIASES.keys())}")
        
        if key in VOICE_ALIASES:
            resolved = VOICE_ALIASES[key]
            _debug(f"[KOKORO DEBUG] Voice resolved to: '{resolved}'")
            return resolved
        
        _debug(f"[KOKORO DEBUG] Voice not found in aliases, using original: '{requested}'")
        return requested



_debug("[KOKORO DEBUG] Creating KokoroSynthesizer instance...")
kokoro = KokoroSynthesizer()
_debug(f"[KOKORO DEBUG] KokoroSynthesizer created: {kokoro}")
_debug(f"[KOKORO DEBUG] Default voice: {kokoro.voice}")
_debug(f"[KOKORO DEBUG] Language: {kokoro.language}")
_debug("[KOKORO DEBUG] Module initialization complete")

if KOKORO_PREWARM:
    try:
        _debug("[KOKORO DEBUG] Prewarming Kokoro pipeline...")
        pipeline = kokoro._ensure_pipeline()
        if pipeline and hasattr(pipeline, "load_voice"):
            pipeline.load_voice(kokoro.voice)
        _debug("[KOKORO DEBUG] Prewarm complete")
    except Exception as exc:  # pragma: no cover - best effort
        _debug(f"[KOKORO DEBUG] Prewarm failed: {exc}")
