from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple


def run_code_tests(language: str, code: str, test_spec: Dict[str, Any]) -> Dict[str, str]:
    language = (language or "").lower()
    if not code.strip():
        return {"status": "error", "summary": "No code provided."}

    if language == "python":
        return _run_python(code, test_spec)
    if language == "java":
        return _run_java(code, test_spec)
    if language == "cpp":
        return _run_cpp(code, test_spec)
    if language == "c":
        return _run_c(code, test_spec)

    return {"status": "error", "summary": f"Unsupported language: {language}"}


def _get_signature(test_spec: Dict[str, Any], language: str) -> Dict[str, Any]:
    signatures = test_spec.get("function_signatures", {})
    if not isinstance(signatures, dict):
        return {}
    signature = signatures.get(language)
    if isinstance(signature, dict):
        return signature
    return {}


def _ordered_args(signature: Dict[str, Any], test_input: Dict[str, Any]) -> List[Any]:
    params = signature.get("parameters", [])
    ordered = []
    if isinstance(params, list):
        for param in params:
            if isinstance(param, dict):
                ordered.append(test_input.get(param.get("name")))
    return ordered


def _compare_outputs(expected: Any, actual: Any, comparison: str) -> bool:
    if comparison != "unordered":
        return expected == actual

    return _normalize_unordered(expected) == _normalize_unordered(actual)


def _normalize_unordered(value: Any) -> Any:
    if isinstance(value, list):
        normalized = [_normalize_unordered(item) for item in value]
        try:
            return sorted(normalized, key=lambda item: json.dumps(item, sort_keys=True))
        except TypeError:
            return sorted(normalized, key=lambda item: str(item))
    return value


def _run_python(code: str, test_spec: Dict[str, Any]) -> Dict[str, str]:
    signature = _get_signature(test_spec, "python")
    func_name = signature.get("function_name")
    if not func_name:
        return {"status": "error", "summary": "Missing Python function signature."}

    tests = test_spec.get("tests", [])
    comparison = test_spec.get("comparison", "exact")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        solution_path = tmp_path / "solution.py"
        runner_path = tmp_path / "runner.py"
        tests_path = tmp_path / "tests.json"

        solution_path.write_text(code, encoding="utf-8")
        tests_payload = {"tests": tests, "function_name": func_name, "signature": signature}
        tests_path.write_text(json.dumps(tests_payload), encoding="utf-8")

        runner_path.write_text(
            _python_runner_script(),
            encoding="utf-8",
        )

        try:
            result = subprocess.run(
                ["python", str(runner_path)],
                cwd=tmpdir,
                capture_output=True,
                text=True,
                timeout=8,
            )
        except subprocess.TimeoutExpired:
            return {"status": "error", "summary": "Execution timed out."}

        if result.returncode != 0:
            return {"status": "error", "summary": _clean_error(result.stderr) or "Runtime error."}

        try:
            stdout_lines = [line for line in (result.stdout or "").splitlines() if line.strip()]
            raw = stdout_lines[-1] if stdout_lines else ""
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return {"status": "error", "summary": "Failed to parse execution output."}

        outputs = payload.get("outputs", [])
        if not isinstance(outputs, list):
            return {"status": "error", "summary": "Invalid execution output."}

        for idx, test in enumerate(tests):
            expected = test.get("output")
            actual = outputs[idx] if idx < len(outputs) else None
            if not _compare_outputs(expected, actual, comparison):
                return {"status": "fail", "summary": "Some test cases failed."}

        return {"status": "pass", "summary": "All tests passed."}


def _python_runner_script() -> str:
    return """import importlib.util
import json
import sys

def main():
    with open("tests.json", "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    func_name = payload.get("function_name")
    signature = payload.get("signature", {})
    params = signature.get("parameters", [])
    tests = payload.get("tests", [])

    spec = importlib.util.spec_from_file_location("solution", "solution.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    func = getattr(module, func_name)

    outputs = []
    for test in tests:
        test_input = test.get("input", {})
        args = []
        for param in params:
            args.append(test_input.get(param.get("name")))
        outputs.append(func(*args))

    print(json.dumps({"outputs": outputs}))

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(json.dumps({"error": str(exc)}))
        sys.exit(1)
"""


def _java_literal(value: Any, type_name: str) -> str:
    if type_name == "int":
        return str(int(value))
    if type_name == "bool":
        return "true" if value else "false"
    if type_name == "string":
        return json.dumps(str(value))
    if type_name == "int[]":
        return "new int[]{" + ", ".join(str(int(v)) for v in value) + "}"
    if type_name == "string[]":
        return "new String[]{" + ", ".join(json.dumps(str(v)) for v in value) + "}"
    if type_name == "int[][]":
        rows = ", ".join("{" + ", ".join(str(int(v)) for v in row) + "}" for row in value)
        return "new int[][]{" + rows + "}"
    if type_name == "string[][]":
        rows = ", ".join("{" + ", ".join(json.dumps(str(v)) for v in row) + "}" for row in value)
        return "new String[][]{" + rows + "}"
    raise ValueError(f"Unsupported Java type: {type_name}")


def _java_compare_code(return_type: str, comparison: str) -> Tuple[str, str]:
    helpers = []
    compare_call = "result.equals(expected)"
    if return_type == "int":
        compare_call = "result == expected"
    elif return_type == "bool":
        compare_call = "result == expected"
    elif return_type == "string":
        compare_call = "result.equals(expected)"
    elif return_type == "int[]":
        helpers.append(
            "static boolean compareIntArray(int[] a, int[] b, boolean unordered) {"
            "if (a == null || b == null) return a == b;"
            "if (unordered) {"
            "int[] ac = java.util.Arrays.copyOf(a, a.length);"
            "int[] bc = java.util.Arrays.copyOf(b, b.length);"
            "java.util.Arrays.sort(ac);"
            "java.util.Arrays.sort(bc);"
            "return java.util.Arrays.equals(ac, bc);"
            "}"
            "return java.util.Arrays.equals(a, b);"
            "}"
        )
        compare_call = f"compareIntArray(result, expected, {str(comparison == 'unordered').lower()})"
    elif return_type == "string[]":
        helpers.append(
            "static boolean compareStringArray(String[] a, String[] b, boolean unordered) {"
            "if (a == null || b == null) return a == b;"
            "if (unordered) {"
            "String[] ac = java.util.Arrays.copyOf(a, a.length);"
            "String[] bc = java.util.Arrays.copyOf(b, b.length);"
            "java.util.Arrays.sort(ac);"
            "java.util.Arrays.sort(bc);"
            "return java.util.Arrays.equals(ac, bc);"
            "}"
            "return java.util.Arrays.equals(a, b);"
            "}"
        )
        compare_call = f"compareStringArray(result, expected, {str(comparison == 'unordered').lower()})"
    elif return_type in {"int[][]", "string[][]"}:
        helpers.append(
            "static boolean compare2D(Object[] a, Object[] b, boolean unordered) {"
            "if (a == null || b == null) return a == b;"
            "if (!unordered) return java.util.Arrays.deepEquals(a, b);"
            "Object[] ac = java.util.Arrays.copyOf(a, a.length);"
            "Object[] bc = java.util.Arrays.copyOf(b, b.length);"
            "java.util.Arrays.sort(ac, java.util.Comparator.comparing(java.util.Arrays::deepToString));"
            "java.util.Arrays.sort(bc, java.util.Comparator.comparing(java.util.Arrays::deepToString));"
            "return java.util.Arrays.deepEquals(ac, bc);"
            "}"
        )
        compare_call = f"compare2D(result, expected, {str(comparison == 'unordered').lower()})"
    return "\n".join(helpers), compare_call


def _run_java(code: str, test_spec: Dict[str, Any]) -> Dict[str, str]:
    signature = _get_signature(test_spec, "java")
    method_name = signature.get("method_name")
    return_type = signature.get("return_type")
    params = signature.get("parameters", [])
    if not method_name or not return_type or not isinstance(params, list):
        return {"status": "error", "summary": "Missing Java function signature."}

    tests = test_spec.get("tests", [])
    comparison = test_spec.get("comparison", "exact")

    param_decls = []
    param_names = []
    for param in params:
        if not isinstance(param, dict):
            continue
        param_decls.append(f"{param.get('type')} {param.get('name')}")
        param_names.append(param.get("name"))

    helper_code, compare_call = _java_compare_code(return_type, comparison)

    test_blocks = []
    for idx, test in enumerate(tests):
        test_input = test.get("input", {})
        expected = test.get("output")
        assignments = []
        args = []
        for param in params:
            name = param.get("name")
            type_name = param.get("type")
            literal = _java_literal(test_input.get(name), type_name)
            assignments.append(f"{type_name} {name} = {literal};")
            args.append(name)
        expected_literal = _java_literal(expected, return_type)
        test_blocks.append(
            "\n".join(assignments)
            + f"\n{return_type} expected = {expected_literal};"
            + f"\n{return_type} result = Solution.{method_name}({', '.join(args)});"
            + f"\nif (!({compare_call})) {{"
            + f"\n  System.out.println(\"FAIL:{idx}\");"
            + "\n  return;"
            + "\n}"
        )

    runner_source = (
        "import java.util.*;\n"
        "public class Runner {\n"
        + (helper_code + "\n" if helper_code else "")
        + "public static void main(String[] args) {\n"
        + "try {\n"
        + "\n".join(test_blocks)
        + "\nSystem.out.println(\"PASS\");\n"
        + "} catch (Throwable t) {\n"
        + "System.out.println(\"ERROR:\" + t.getClass().getName());\n"
        + "}\n"
        + "}\n"
        + "}\n"
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        solution_path = tmp_path / "Solution.java"
        runner_path = tmp_path / "Runner.java"
        solution_path.write_text(code, encoding="utf-8")
        runner_path.write_text(runner_source, encoding="utf-8")

        compile_result = subprocess.run(
            ["javac", "Solution.java", "Runner.java"],
            cwd=tmpdir,
            capture_output=True,
            text=True,
            timeout=15,
        )
        if compile_result.returncode != 0:
            return {"status": "error", "summary": "Compilation error."}

        try:
            run_result = subprocess.run(
                ["java", "Runner"],
                cwd=tmpdir,
                capture_output=True,
                text=True,
                timeout=8,
            )
        except subprocess.TimeoutExpired:
            return {"status": "error", "summary": "Execution timed out."}

        output = (run_result.stdout or "").splitlines()
        last_line = output[-1].strip() if output else ""
        if last_line.startswith("PASS"):
            return {"status": "pass", "summary": "All tests passed."}
        if last_line.startswith("FAIL"):
            return {"status": "fail", "summary": "Some test cases failed."}
        return {"status": "error", "summary": "Runtime error."}


def _cpp_literal(value: Any, type_name: str) -> str:
    if type_name == "int":
        return str(int(value))
    if type_name == "bool":
        return "true" if value else "false"
    if type_name == "string":
        return json.dumps(str(value))
    if type_name == "vector<int>":
        return "std::vector<int>{" + ", ".join(str(int(v)) for v in value) + "}"
    if type_name == "vector<string>":
        return "std::vector<std::string>{" + ", ".join(json.dumps(str(v)) for v in value) + "}"
    if type_name == "vector<vector<int>>":
        rows = ", ".join("std::vector<int>{" + ", ".join(str(int(v)) for v in row) + "}" for row in value)
        return "std::vector<std::vector<int>>{" + rows + "}"
    if type_name == "vector<vector<string>>":
        rows = ", ".join("std::vector<std::string>{" + ", ".join(json.dumps(str(v)) for v in row) + "}" for row in value)
        return "std::vector<std::vector<std::string>>{" + rows + "}"
    raise ValueError(f"Unsupported C++ type: {type_name}")


def _cpp_compare_code(return_type: str, comparison: str) -> Tuple[str, str]:
    helpers = []
    compare_call = "result == expected"
    unordered = "true" if comparison == "unordered" else "false"
    if return_type in {"vector<int>", "vector<string>"}:
        helpers.append(
            "template <typename T>\n"
            "bool compareVector(std::vector<T> a, std::vector<T> b, bool unordered) {\n"
            "if (unordered) { std::sort(a.begin(), a.end()); std::sort(b.begin(), b.end()); }\n"
            "return a == b;\n"
            "}\n"
        )
        compare_call = f"compareVector(result, expected, {unordered})"
    elif return_type in {"vector<vector<int>>", "vector<vector<string>>"}:
        helpers.append(
            "template <typename T>\n"
            "bool compareNested(std::vector<std::vector<T>> a, std::vector<std::vector<T>> b, bool unordered) {\n"
            "if (!unordered) return a == b;\n"
            "for (auto& row : a) std::sort(row.begin(), row.end());\n"
            "for (auto& row : b) std::sort(row.begin(), row.end());\n"
            "auto toStr = [](const std::vector<T>& row){\n"
            "std::ostringstream oss; for (const auto& v : row) oss << v << ','; return oss.str(); };\n"
            "std::sort(a.begin(), a.end(), [&](const auto& x, const auto& y){ return toStr(x) < toStr(y); });\n"
            "std::sort(b.begin(), b.end(), [&](const auto& x, const auto& y){ return toStr(x) < toStr(y); });\n"
            "return a == b;\n"
            "}\n"
        )
        compare_call = f"compareNested(result, expected, {unordered})"
    return "\n".join(helpers), compare_call


def _run_cpp(code: str, test_spec: Dict[str, Any]) -> Dict[str, str]:
    signature = _get_signature(test_spec, "cpp")
    func_name = signature.get("function_name")
    return_type = signature.get("return_type")
    params = signature.get("parameters", [])
    if not func_name or not return_type or not isinstance(params, list):
        return {"status": "error", "summary": "Missing C++ function signature."}

    tests = test_spec.get("tests", [])
    comparison = test_spec.get("comparison", "exact")

    helper_code, compare_call = _cpp_compare_code(return_type, comparison)
    test_blocks = []

    for idx, test in enumerate(tests):
        test_input = test.get("input", {})
        expected = test.get("output")
        assignments = []
        args = []
        for param in params:
            name = param.get("name")
            type_name = param.get("type")
            literal = _cpp_literal(test_input.get(name), type_name)
            assignments.append(f"{type_name} {name} = {literal};")
            args.append(name)
        expected_literal = _cpp_literal(expected, return_type)
        test_blocks.append(
            "\n".join(assignments)
            + f"\n{return_type} expected = {expected_literal};"
            + f"\n{return_type} result = {func_name}({', '.join(args)});"
            + f"\nif (!({compare_call})) {{ std::cout << \"FAIL:{idx}\"; return 0; }}\n"
        )

    runner_source = (
        "#include <bits/stdc++.h>\n"
        "using namespace std;\n"
        + helper_code
        + "\n"
        + code
        + "\n"
        + "int main() {\n"
        + "\n".join(test_blocks)
        + "\ncout << \"PASS\";\n"
        + "return 0;\n"
        + "}\n"
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        runner_path = tmp_path / "runner.cpp"
        runner_path.write_text(runner_source, encoding="utf-8")

        compile_result = subprocess.run(
            ["g++", "runner.cpp", "-std=c++17", "-O2"],
            cwd=tmpdir,
            capture_output=True,
            text=True,
            timeout=15,
        )
        if compile_result.returncode != 0:
            return {"status": "error", "summary": "Compilation error."}

        try:
            run_result = subprocess.run(
                ["./a.out"],
                cwd=tmpdir,
                capture_output=True,
                text=True,
                timeout=8,
            )
        except subprocess.TimeoutExpired:
            return {"status": "error", "summary": "Execution timed out."}

        output = (run_result.stdout or "").splitlines()
        last_line = output[-1].strip() if output else ""
        if last_line.startswith("PASS"):
            return {"status": "pass", "summary": "All tests passed."}
        if last_line.startswith("FAIL"):
            return {"status": "fail", "summary": "Some test cases failed."}
        return {"status": "error", "summary": "Runtime error."}


def _c_literal(value: Any, type_name: str) -> str:
    if type_name == "int":
        return str(int(value))
    if type_name == "bool":
        return "true" if value else "false"
    if type_name == "string":
        return json.dumps(str(value))
    if type_name == "int[]":
        return "{" + ", ".join(str(int(v)) for v in value) + "}"
    if type_name == "string[]":
        return "{" + ", ".join(json.dumps(str(v)) for v in value) + "}"
    raise ValueError(f"Unsupported C type: {type_name}")


def _run_c(code: str, test_spec: Dict[str, Any]) -> Dict[str, str]:
    signature = _get_signature(test_spec, "c")
    func_name = signature.get("function_name")
    return_type = signature.get("return_type")
    params = signature.get("parameters", [])
    return_size_param = signature.get("return_size_param")
    if not func_name or not return_type or not isinstance(params, list):
        return {"status": "error", "summary": "Missing C function signature."}

    if return_type not in {"int", "bool", "string", "int[]", "string[]"}:
        return {"status": "error", "summary": "Unsupported C return type."}
    if return_type in {"int[]", "string[]"} and not return_size_param:
        return {"status": "error", "summary": "Missing C return size parameter."}

    tests = test_spec.get("tests", [])
    comparison = test_spec.get("comparison", "exact")
    unordered = comparison == "unordered"

    helper_code = (
        "#include <stdio.h>\n"
        "#include <stdbool.h>\n"
        "#include <string.h>\n"
        "#include <stdlib.h>\n"
        "static int cmp_int(const void* a, const void* b) { return (*(int*)a) - (*(int*)b); }\n"
        "static int cmp_str(const void* a, const void* b) { return strcmp(*(const char**)a, *(const char**)b); }\n"
        "static bool compare_int_array(int* a, int a_len, int* b, int b_len, bool unordered) {\n"
        "if (a_len != b_len) return false;\n"
        "if (unordered) { qsort(a, a_len, sizeof(int), cmp_int); qsort(b, b_len, sizeof(int), cmp_int); }\n"
        "for (int i = 0; i < a_len; i++) { if (a[i] != b[i]) return false; }\n"
        "return true; }\n"
        "static bool compare_str_array(char** a, int a_len, char** b, int b_len, bool unordered) {\n"
        "if (a_len != b_len) return false;\n"
        "if (unordered) { qsort(a, a_len, sizeof(char*), cmp_str); qsort(b, b_len, sizeof(char*), cmp_str); }\n"
        "for (int i = 0; i < a_len; i++) { if (strcmp(a[i], b[i]) != 0) return false; }\n"
        "return true; }\n"
    )

    test_blocks = []
    for idx, test in enumerate(tests):
        test_input = test.get("input", {})
        expected = test.get("output")
        assignments = []
        args = []
        for param in params:
            if not isinstance(param, dict):
                continue
            name = param.get("name")
            type_name = param.get("type")
            literal = _c_literal(test_input.get(name), type_name)
            if type_name == "int[]":
                length = len(test_input.get(name) or [])
                assignments.append(f"int {name}[] = {literal};")
                size_param = param.get("size_param")
                if size_param:
                    assignments.append(f"int {size_param} = {length};")
            elif type_name == "string[]":
                length = len(test_input.get(name) or [])
                assignments.append(f"char* {name}[] = {literal};")
                size_param = param.get("size_param")
                if size_param:
                    assignments.append(f"int {size_param} = {length};")
            else:
                assignments.append(f"{_c_decl(type_name)} {name} = {literal};")
            args.append(name)
        expected_decl = ""
        compare_expr = "false"
        if return_type in {"int", "bool"}:
            expected_decl = f"{_c_decl(return_type)} expected = {_c_literal(expected, return_type)};"
            compare_expr = "result == expected"
        elif return_type == "string":
            expected_decl = f"char* expected = {_c_literal(expected, 'string')};"
            compare_expr = "strcmp(result, expected) == 0"
        elif return_type == "int[]":
            expected_decl = f"int expected[] = {_c_literal(expected, 'int[]')}; int expectedSize = {len(expected or [])};"
            compare_expr = f"compare_int_array(result, {return_size_param}, expected, expectedSize, {str(unordered).lower()})"
        elif return_type == "string[]":
            expected_decl = f"char* expected[] = {_c_literal(expected, 'string[]')}; int expectedSize = {len(expected or [])};"
            compare_expr = f"compare_str_array(result, {return_size_param}, expected, expectedSize, {str(unordered).lower()})"

        call_args = ", ".join(args + ([f"&{return_size_param}"] if return_type in {"int*", "string*"} else []))
        result_decl = f"{_c_decl(return_type)} result = {func_name}({call_args});"
        if return_type in {"int[]", "string[]"}:
            result_decl = f"int {return_size_param} = 0; {_c_decl(return_type)} result = {func_name}({call_args});"

        test_blocks.append(
            "\n".join(assignments)
            + f"\n{expected_decl}\n{result_decl}\n"
            + f"if (!({compare_expr})) {{ printf(\"FAIL:{idx}\"); return 0; }}\n"
        )

    runner_source = (
        helper_code
        + "\n"
        + code
        + "\nint main() {\n"
        + "\n".join(test_blocks)
        + "\nprintf(\"PASS\");\nreturn 0;\n}\n"
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        runner_path = tmp_path / "runner.c"
        runner_path.write_text(runner_source, encoding="utf-8")

        compile_result = subprocess.run(
            ["gcc", "runner.c", "-std=c11", "-O2"],
            cwd=tmpdir,
            capture_output=True,
            text=True,
            timeout=15,
        )
        if compile_result.returncode != 0:
            return {"status": "error", "summary": "Compilation error."}

        try:
            run_result = subprocess.run(
                ["./a.out"],
                cwd=tmpdir,
                capture_output=True,
                text=True,
                timeout=8,
            )
        except subprocess.TimeoutExpired:
            return {"status": "error", "summary": "Execution timed out."}

        output = (run_result.stdout or "").splitlines()
        last_line = output[-1].strip() if output else ""
        if last_line.startswith("PASS"):
            return {"status": "pass", "summary": "All tests passed."}
        if last_line.startswith("FAIL"):
            return {"status": "fail", "summary": "Some test cases failed."}
        return {"status": "error", "summary": "Runtime error."}


def _c_decl(type_name: str) -> str:
    if type_name == "int":
        return "int"
    if type_name == "bool":
        return "bool"
    if type_name == "string":
        return "char*"
    if type_name == "int[]":
        return "int*"
    if type_name == "string[]":
        return "char**"
    return "void*"


def _clean_error(stderr: str) -> str:
    if not stderr:
        return ""
    lines = [line.strip() for line in stderr.splitlines() if line.strip()]
    return lines[-1] if lines else ""
