from __future__ import annotations

import json
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional

from .gemini_client import GeminiClient, GeminiError


_CACHE_PATH = Path(__file__).resolve().parent.parent / "storage" / "question_tests.json"
_LOCK = Lock()


def _normalize_key(title: str) -> str:
    return " ".join((title or "").strip().lower().split())


def _load_cache() -> Dict[str, Any]:
    if not _CACHE_PATH.exists():
        return {}
    try:
        return json.loads(_CACHE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _save_cache(cache: Dict[str, Any]) -> None:
    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _CACHE_PATH.write_text(json.dumps(cache, indent=2, sort_keys=True), encoding="utf-8")


def get_cached_tests(question_title: str) -> Optional[Dict[str, Any]]:
    key = _normalize_key(question_title)
    with _LOCK:
        cache = _load_cache()
        stored = cache.get(key)
        if stored:
            return dict(stored)
        return None


def ensure_tests_for_question(question_title: str) -> Dict[str, Any]:
    key = _normalize_key(question_title)
    print(f"[TEST GEN] Looking for tests for: '{question_title}' (key: '{key}')")
    with _LOCK:
        cache = _load_cache()
        stored = cache.get(key)
        if stored:
            print(f"[TEST GEN] Found cached tests for: {question_title}")
            return dict(stored)

        print(f"[TEST GEN] No cache found, generating tests for: {question_title}")
        try:
            client = GeminiClient()
            print(f"[TEST GEN] GeminiClient initialized")
        except Exception as e:
            print(f"[TEST GEN ERROR] Failed to initialize GeminiClient: {type(e).__name__}: {e}")
            return {}

        try:
            generated = client.generate_question_tests(question_title)
            print(f"[TEST GEN] Successfully generated {len(generated.get('tests', []))} tests")
        except ValueError as e:
            print(f"[TEST GEN ERROR] ValueError: {e}")
            return {}
        except GeminiError as e:
            print(f"[TEST GEN ERROR] GeminiError: {e}")
            return {}
        except Exception as e:
            print(f"[TEST GEN ERROR] Unexpected error: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return {}

        cache[key] = generated
        _save_cache(cache)
        print(f"[TEST GEN] Saved to cache: {question_title}")
        return dict(generated)


def format_signatures_for_prompt(test_spec: Dict[str, Any]) -> Dict[str, str]:
    signatures = test_spec.get("function_signatures", {}) if isinstance(test_spec, dict) else {}
    formatted: Dict[str, str] = {}
    for language, signature in signatures.items():
        if not isinstance(signature, dict):
            continue
        if language == "python":
            func = signature.get("function_name")
            params = signature.get("parameters", [])
            args = ", ".join(param.get("name", "arg") for param in params if isinstance(param, dict))
            if func:
                formatted[language] = f"def {func}({args}):"
        elif language == "java":
            method = signature.get("method_name")
            params = signature.get("parameters", [])
            ret = signature.get("return_type", "void")
            args = ", ".join(
                f"{param.get('type', 'Object')} {param.get('name', 'arg')}"
                for param in params
                if isinstance(param, dict)
            )
            if method:
                formatted[language] = f"public static {ret} {method}({args}) inside class Solution"
        elif language == "cpp":
            func = signature.get("function_name")
            params = signature.get("parameters", [])
            ret = signature.get("return_type", "void")
            args = ", ".join(
                f"{param.get('type', 'auto')} {param.get('name', 'arg')}"
                for param in params
                if isinstance(param, dict)
            )
            if func:
                formatted[language] = f"{ret} {func}({args}) with no main"
        elif language == "c":
            func = signature.get("function_name")
            params = signature.get("parameters", [])
            ret = signature.get("return_type", "void")
            args = ", ".join(
                f"{param.get('type', 'void*')} {param.get('name', 'arg')}"
                for param in params
                if isinstance(param, dict)
            )
            if func:
                formatted[language] = f"{ret} {func}({args}) with no main"
    return formatted
