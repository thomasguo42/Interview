"""
Whisper-based speech-to-text service with technical term correction.
Uses faster-whisper for efficient, accurate transcription.
"""

from __future__ import annotations

import logging
import os
import re
import tempfile
from typing import Optional

from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)


class WhisperTranscriber:
    """Handles audio transcription using Whisper with technical term post-processing."""

    def __init__(self, model_size: str = "base", device: str = "auto"):
        """
        Initialize the Whisper transcriber.

        Args:
            model_size: Whisper model size (tiny, base, small, medium, large)
                       base is recommended for balanced speed/accuracy
            device: Device to run on ("cpu", "cuda", or "auto")
        """
        self.model_size = model_size

        # Auto-detect GPU availability
        if device == "auto":
            try:
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"
        else:
            self.device = device

        # Compute type based on device
        self.compute_type = "float16" if self.device == "cuda" else "int8"

        logger.info(
            f"Initializing Whisper model: size={model_size}, device={self.device}, "
            f"compute_type={self.compute_type}"
        )

        # Initialize the model (will download on first use)
        # Using num_workers=1 for stability
        self.model = WhisperModel(
            model_size,
            device=self.device,
            compute_type=self.compute_type,
            num_workers=1
        )

        logger.info("Whisper model loaded successfully")

        # Technical term dictionary for post-processing
        self._init_technical_dict()

    def _init_technical_dict(self):
        """Initialize technical term correction dictionary."""

        # Common technical term corrections (case-insensitive patterns)
        # Format: (pattern, replacement, is_whole_word)
        self.tech_corrections = [
            # Data Structures
            (r"\bhash\s*ma[pt]p?\b", "HashMap", True),
            (r"\bhash\s*ma[pt]\b", "HashMap", True),
            (r"\blinked\s*list\b", "LinkedList", True),
            (r"\barrayli?st\b", "ArrayList", True),
            (r"\bbinary\s*tree\b", "binary tree", True),
            (r"\btree\s*ma[pt]\b", "TreeMap", True),
            (r"\bhash\s*set\b", "HashSet", True),
            (r"\btree\s*set\b", "TreeSet", True),
            (r"\bpriority\s*[qk]ue+\b", "priority queue", True),
            (r"\b[qk]ue+\b", "queue", True),
            (r"\bstac?k\b", "stack", True),
            (r"\bheap\b", "heap", True),
            (r"\bgraf\b", "graph", True),
            (r"\bgraphs?\b", "graph", True),

            # Algorithms & Complexity
            (r"\bbig\s*[oh0]\b", "O()", False),
            (r"\bbig\s*oh+\s*of\b", "O()", False),
            (r"\b[oh0]\s*of\s*[ne]\b", "O(n)", False),
            (r"\b[oh0]\s*of\s*log\s*[ne]\b", "O(log n)", False),
            (r"\b[oh0]\s*of\s*[ne]\s*squared?\b", "O(n²)", False),
            (r"\b[oh0]\s*of\s*[ne]\s*log\s*[ne]\b", "O(n log n)", False),
            (r"\bdepth\s*first\s*search\b", "DFS", True),
            (r"\bbreadth\s*first\s*search\b", "BFS", True),
            (r"\bdynamic\s*programming\b", "dynamic programming", True),
            (r"\bgreedy\s*algorithm\b", "greedy algorithm", True),
            (r"\bbinary\s*search\b", "binary search", True),
            (r"\bmerge\s*sort\b", "merge sort", True),
            (r"\bquick\s*sort\b", "quick sort", True),
            (r"\bbubble\s*sort\b", "bubble sort", True),

            # Programming Concepts
            (r"\bnull\s*pointer\b", "null pointer", True),
            (r"\bno\s*pointer\b", "null pointer", True),
            (r"\bnull\b", "null", True),
            (r"\bnone\b", "None", True),
            (r"\bboolean\b", "boolean", True),
            (r"\binteger\b", "integer", True),
            (r"\bstring\b", "string", True),
            (r"\barray\b", "array", True),
            (r"\bmatrix\b", "matrix", True),
            (r"\brecursion\b", "recursion", True),
            (r"\biteration\b", "iteration", True),
            (r"\bfor\s*loop\b", "for loop", True),
            (r"\bwhile\s*loop\b", "while loop", True),
            (r"\bif\s*statement\b", "if statement", True),
            (r"\belse\s*statement\b", "else statement", True),

            # Java/Python Specific
            (r"\bpublic\s*static\s*void\s*main\b", "public static void main", False),
            (r"\b__init__\b", "__init__", True),
            (r"\bdef\s+\w+\b", lambda m: m.group(0), False),  # Preserve function definitions
            (r"\bclass\s+\w+\b", lambda m: m.group(0), False),  # Preserve class definitions

            # Common Misheard Terms
            (r"\bpre\s*fix\b", "prefix", True),
            (r"\bpost\s*fix\b", "postfix", True),
            (r"\bin\s*fix\b", "infix", True),
            (r"\bedge\s*case\b", "edge case", True),
            (r"\bbase\s*case\b", "base case", True),
            (r"\brecursive\s*case\b", "recursive case", True),
            (r"\bhelper\s*function\b", "helper function", True),
            (r"\butility\s*function\b", "utility function", True),
            (r"\btime\s*complexity\b", "time complexity", True),
            (r"\bspace\s*complexity\b", "space complexity", True),

            # Code patterns
            (r"\bfor\s*i\s*in\s*range\b", "for i in range", False),
            (r"\blen\s*of\b", "len()", False),
            (r"\bappend\b", "append", True),
            (r"\bpop\b", "pop", True),
            (r"\bpush\b", "push", True),
            (r"\bpeek\b", "peek", True),
        ]

    def transcribe(
        self,
        audio_path: str,
        language: str = "en",
        task: str = "transcribe",
        initial_prompt: Optional[str] = None,
    ) -> dict:
        """
        Transcribe audio file using Whisper.

        Args:
            audio_path: Path to audio file (wav, mp3, etc.)
            language: Language code (default: "en")
            task: "transcribe" or "translate"
            initial_prompt: Optional prompt to guide transcription
                          (useful for technical context)

        Returns:
            dict with:
                - text: Transcribed text (post-processed)
                - raw_text: Original Whisper output
                - language: Detected language
                - segments: List of segments with timestamps
        """
        try:
            # Use prompt to hint at technical content
            if initial_prompt is None:
                initial_prompt = (
                    "This is a technical interview about software engineering, "
                    "algorithms, data structures, and coding. "
                    "Technical terms: HashMap, ArrayList, LinkedList, queue, stack, "
                    "binary tree, graph, DFS, BFS, O(n), recursion, iteration."
                )

            logger.info(f"Transcribing audio file: {audio_path}")

            # Transcribe with Whisper
            segments, info = self.model.transcribe(
                audio_path,
                language=language,
                task=task,
                initial_prompt=initial_prompt,
                beam_size=5,  # Better accuracy
                best_of=5,    # Better accuracy
                temperature=0.0,  # Deterministic
                vad_filter=True,  # Voice activity detection (remove silence)
                vad_parameters=dict(
                    threshold=0.5,
                    min_speech_duration_ms=250,
                    min_silence_duration_ms=500
                )
            )

            # Collect all segments
            segment_list = []
            full_text = ""

            for segment in segments:
                segment_list.append({
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text.strip(),
                    "confidence": segment.avg_logprob
                })
                full_text += segment.text + " "

            raw_text = full_text.strip()

            # Post-process with technical term corrections
            corrected_text = self._correct_technical_terms(raw_text)

            logger.info(
                f"Transcription complete. Language: {info.language}, "
                f"Duration: {info.duration:.2f}s, Segments: {len(segment_list)}"
            )

            if raw_text != corrected_text:
                logger.debug(f"Corrected transcription: {raw_text} → {corrected_text}")

            return {
                "text": corrected_text,
                "raw_text": raw_text,
                "language": info.language,
                "duration": info.duration,
                "segments": segment_list,
            }

        except Exception as e:
            logger.error(f"Whisper transcription failed: {e}", exc_info=True)
            raise

    def _correct_technical_terms(self, text: str) -> str:
        """
        Apply technical term corrections to transcribed text.

        Args:
            text: Raw transcription from Whisper

        Returns:
            Corrected text with proper technical terminology
        """
        corrected = text

        for pattern, replacement, is_whole_word in self.tech_corrections:
            if callable(replacement):
                # Dynamic replacement (e.g., preserve function names)
                corrected = re.sub(pattern, replacement, corrected, flags=re.IGNORECASE)
            else:
                if is_whole_word:
                    # Only match whole words
                    corrected = re.sub(
                        pattern,
                        replacement,
                        corrected,
                        flags=re.IGNORECASE
                    )
                else:
                    # Match anywhere in text
                    corrected = re.sub(
                        pattern,
                        replacement,
                        corrected,
                        flags=re.IGNORECASE
                    )

        return corrected

    def transcribe_bytes(
        self,
        audio_bytes: bytes,
        audio_format: str = "webm",
        language: str = "en",
    ) -> dict:
        """
        Transcribe audio from bytes (e.g., from web upload).

        Args:
            audio_bytes: Audio data as bytes
            audio_format: Audio format (webm, wav, mp3, etc.)
            language: Language code

        Returns:
            Same as transcribe()
        """
        # Save to temporary file
        with tempfile.NamedTemporaryFile(
            suffix=f".{audio_format}",
            delete=False
        ) as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_path = tmp_file.name

        try:
            result = self.transcribe(tmp_path, language=language)
            return result
        finally:
            # Clean up temp file
            try:
                os.unlink(tmp_path)
            except Exception as e:
                logger.warning(f"Failed to delete temp file {tmp_path}: {e}")


# Global instance (lazy loaded)
_whisper_instance = None


def get_whisper_transcriber(model_size: str = "base") -> WhisperTranscriber:
    """
    Get or create the global Whisper transcriber instance.

    Args:
        model_size: Whisper model size (tiny, base, small, medium, large)

    Returns:
        WhisperTranscriber instance
    """
    global _whisper_instance

    if _whisper_instance is None:
        _whisper_instance = WhisperTranscriber(model_size=model_size)

    return _whisper_instance
