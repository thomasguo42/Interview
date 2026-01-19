from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Dict, List, Optional


_DEFAULT_BANK_PATH = Path(__file__).resolve().parent.parent / "question_bank.txt"
_TITLE_PATTERN = re.compile(r"^\d+\.\s+(.+)$")
_DIFFICULTY_PATTERN = re.compile(r"\b(easy|med|medium|hard)\b", re.IGNORECASE)


def _normalize_lines(raw_lines: List[str]) -> List[str]:
    return [line.strip() for line in raw_lines if line.strip()]


def load_question_bank(path: Optional[Path] = None) -> List[Dict[str, str]]:
    bank_path = path or _DEFAULT_BANK_PATH
    if not bank_path.exists():
        return []

    try:
        raw_lines = bank_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []

    lines = _normalize_lines(raw_lines)
    questions: List[Dict[str, str]] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        match = _TITLE_PATTERN.match(line)
        if not match:
            index += 1
            continue

        title = match.group(1).strip()
        acceptance = ""
        difficulty = ""

        if index + 1 < len(lines) and "%" in lines[index + 1]:
            acceptance = lines[index + 1].strip()
            index += 1

        if index + 1 < len(lines) and _DIFFICULTY_PATTERN.search(lines[index + 1]):
            difficulty = lines[index + 1].strip()
            index += 1

        questions.append(
            {
                "title": title,
                "acceptance": acceptance,
                "difficulty": difficulty,
            }
        )
        index += 1

    return questions


def get_random_coding_question(path: Optional[Path] = None) -> Dict[str, str]:
    questions = load_question_bank(path)
    if not questions:
        return {}
    return random.choice(questions)
