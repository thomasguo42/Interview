from __future__ import annotations

import json
import re
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional


_TESTS_DIR = Path(__file__).resolve().parent.parent / "storage" / "question_tests"
_LOCK = Lock()
_CACHE: Dict[str, Dict[str, Any]] = {}


def _normalize_key(title: str) -> str:
    return " ".join((title or "").strip().lower().split())


def _slugify(title: str) -> str:
    slug = (title or "").strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "_", slug)
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug


def _load_test_spec(title: str) -> Optional[Dict[str, Any]]:
    if not title:
        return None
    key = _normalize_key(title)
    cached = _CACHE.get(key)
    if cached:
        return dict(cached)
    filename = _slugify(title) + ".json"
    path = _TESTS_DIR / filename
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    _CACHE[key] = dict(payload)
    return dict(payload)


def get_cached_tests(question_title: str) -> Optional[Dict[str, Any]]:
    with _LOCK:
        return _load_test_spec(question_title)


def ensure_tests_for_question(question_title: str) -> Dict[str, Any]:
    print(f"[TEST GEN] Loading local tests for: '{question_title}'")
    with _LOCK:
        payload = _load_test_spec(question_title)
        if payload:
            print(f"[TEST GEN] Loaded local tests for: {question_title}")
            return dict(payload)
        print(f"[TEST GEN ERROR] Missing local tests for: {question_title}")
        return {}


def _format_class_signature(signature: Dict[str, Any], language: str) -> Optional[str]:
    class_name = signature.get("class_name")
    constructor = signature.get("constructor", {})
    methods = signature.get("methods", [])
    if not class_name:
        return None
    ctor_params = constructor.get("parameters", []) if isinstance(constructor, dict) else []
    ctor_args = ", ".join(param.get("name", "arg") for param in ctor_params if isinstance(param, dict))
    method_parts = []
    for method in methods:
        if not isinstance(method, dict):
            continue
        name = method.get("name")
        params = method.get("parameters", [])
        args = ", ".join(param.get("name", "arg") for param in params if isinstance(param, dict))
        if name:
            method_parts.append(f"{name}({args})")
    methods_text = ", ".join(method_parts) if method_parts else "methods"
    if language in {"python", "java", "cpp"}:
        return f"class {class_name}({ctor_args}) with {methods_text}"
    return None


def format_signatures_for_prompt(test_spec: Dict[str, Any]) -> Dict[str, str]:
    formatted: Dict[str, str] = {}
    if not isinstance(test_spec, dict):
        return formatted
    if test_spec.get("kind") == "class":
        signatures = test_spec.get("class_signatures", {})
        for language, signature in signatures.items():
            if not isinstance(signature, dict):
                continue
            rendered = _format_class_signature(signature, language)
            if rendered:
                formatted[language] = rendered
        return formatted

    signatures = test_spec.get("function_signatures", {})
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
    return formatted


def _starter_return_line(language: str, return_type: str) -> str:
    if language == "python":
        return "    return None"
    if language == "java":
        if return_type == "void":
            return ""
        if return_type in {"int", "long", "double"}:
            return "        return 0;"
        if return_type == "bool":
            return "        return false;"
        if return_type == "string":
            return "        return \"\";"
        return "        return null;"
    if language == "cpp":
        if return_type == "void":
            return ""
        if return_type in {"int", "long long", "double"}:
            return "    return 0;"
        if return_type == "bool":
            return "    return false;"
        if return_type == "string":
            return "    return \"\";"
        return "    return {};"
    return ""


def build_starter_code(test_spec: Dict[str, Any], language: str) -> str:
    if not isinstance(test_spec, dict) or not language:
        return ""
    language = language.lower()

    if test_spec.get("kind") == "class":
        signatures = test_spec.get("class_signatures", {})
        signature = signatures.get(language, {}) if isinstance(signatures, dict) else {}
        if not isinstance(signature, dict):
            return ""
        class_name = signature.get("class_name", "Solution")
        constructor = signature.get("constructor", {})
        methods = signature.get("methods", [])

        if language == "python":
            lines = [f"class {class_name}:"]
            ctor_params = constructor.get("parameters", []) if isinstance(constructor, dict) else []
            ctor_args = ", ".join(param.get("name", "arg") for param in ctor_params if isinstance(param, dict))
            lines.append(f"    def __init__(self{', ' + ctor_args if ctor_args else ''}):")
            lines.append("        pass")
            for method in methods:
                if not isinstance(method, dict):
                    continue
                name = method.get("name")
                params = method.get("parameters", [])
                args = ", ".join(param.get("name", "arg") for param in params if isinstance(param, dict))
                lines.append("")
                lines.append(f"    def {name}(self{', ' + args if args else ''}):")
                lines.append("        pass")
            return "\n".join(lines).strip() + "\n"

        if language == "java":
            lines = [f"public class {class_name} {{"]
            ctor_params = constructor.get("parameters", []) if isinstance(constructor, dict) else []
            ctor_args = ", ".join(
                f"{param.get('type', 'int')} {param.get('name', 'arg')}"
                for param in ctor_params
                if isinstance(param, dict)
            )
            lines.append(f"    public {class_name}({ctor_args}) {{")
            lines.append("        // TODO: initialize")
            lines.append("    }")
            for method in methods:
                if not isinstance(method, dict):
                    continue
                name = method.get("name")
                params = method.get("parameters", [])
                ret = method.get("return_type", "void")
                args = ", ".join(
                    f"{param.get('type', 'int')} {param.get('name', 'arg')}"
                    for param in params
                    if isinstance(param, dict)
                )
                lines.append("")
                lines.append(f"    public {ret} {name}({args}) {{")
                lines.append("        // TODO")
                return_line = _starter_return_line(language, ret)
                if return_line:
                    lines.append(return_line)
                lines.append("    }")
            lines.append("}")
            return "\n".join(lines).strip() + "\n"

        if language == "cpp":
            lines = [f"class {class_name} {{", "public:"]
            ctor_params = constructor.get("parameters", []) if isinstance(constructor, dict) else []
            ctor_args = ", ".join(
                f"{param.get('type', 'int')} {param.get('name', 'arg')}"
                for param in ctor_params
                if isinstance(param, dict)
            )
            lines.append(f"    {class_name}({ctor_args}) {{")
            lines.append("        // TODO")
            lines.append("    }")
            for method in methods:
                if not isinstance(method, dict):
                    continue
                name = method.get("name")
                params = method.get("parameters", [])
                ret = method.get("return_type", "void")
                args = ", ".join(
                    f"{param.get('type', 'int')} {param.get('name', 'arg')}"
                    for param in params
                    if isinstance(param, dict)
                )
                lines.append("")
                lines.append(f"    {ret} {name}({args}) {{")
                lines.append("        // TODO")
                return_line = _starter_return_line(language, ret)
                if return_line:
                    lines.append(return_line)
                lines.append("    }")
            lines.append("};")
            return "\n".join(lines).strip() + "\n"

        return ""

    signatures = test_spec.get("function_signatures", {})
    signature = signatures.get(language, {}) if isinstance(signatures, dict) else {}
    if not isinstance(signature, dict):
        return ""

    func_name = signature.get("function_name", "solve")
    params = signature.get("parameters", [])

    if language == "python":
        args = ", ".join(param.get("name", "arg") for param in params if isinstance(param, dict))
        return f"def {func_name}({args}):\n    pass\n"

    if language == "java":
        method = signature.get("method_name", func_name)
        ret = signature.get("return_type", "void")
        args = ", ".join(
            f"{param.get('type', 'int')} {param.get('name', 'arg')}"
            for param in params
            if isinstance(param, dict)
        )
        lines = ["public class Solution {", f"    public static {ret} {method}({args}) {{"]
        lines.append("        // TODO")
        return_line = _starter_return_line(language, ret)
        if return_line:
            lines.append(return_line)
        lines.append("    }")
        lines.append("}")
        return "\n".join(lines).strip() + "\n"

    if language == "cpp":
        ret = signature.get("return_type", "void")
        args = ", ".join(
            f"{param.get('type', 'int')} {param.get('name', 'arg')}"
            for param in params
            if isinstance(param, dict)
        )
        lines = [f"{ret} {func_name}({args}) {{", "    // TODO"]
        return_line = _starter_return_line(language, ret)
        if return_line:
            lines.append(return_line)
        lines.append("}")
        return "\n".join(lines).strip() + "\n"

    return ""
