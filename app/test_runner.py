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

    if test_spec.get("kind") == "class":
        if language == "python":
            return _run_python_class(code, test_spec)
        if language == "java":
            return _run_java_class(code, test_spec)
        if language == "cpp":
            return _run_cpp_class(code, test_spec)
        return {"status": "error", "summary": f"Unsupported language: {language}"}

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
from collections import deque

def _ensure_support_types(module):
    if hasattr(module, "ListNode"):
        return
    class ListNode:
        def __init__(self, val=0, next=None):
            self.val = val
            self.next = next
    module.ListNode = ListNode

def main():
    with open("tests.json", "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    func_name = payload.get("function_name")
    signature = payload.get("signature", {})
    params = signature.get("parameters", [])
    return_type = signature.get("return_type", "")
    tests = payload.get("tests", [])

    spec = importlib.util.spec_from_file_location("solution", "solution.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _ensure_support_types(module)
    func = getattr(module, func_name)

    def is_listnode(type_name):
        return "listnode" in str(type_name).lower()

    def is_sea(type_name):
        return str(type_name).lower() == "sea"

    class Sea:
        def __init__(self, ships):
            self.ships = set(tuple(x) for x in ships or [])

        def hasShips(self, topRight, bottomLeft):
            x1, y1 = bottomLeft
            x2, y2 = topRight
            for x, y in self.ships:
                if x1 <= x <= x2 and y1 <= y <= y2:
                    return True
            return False

    def build_list(values):
        if values is None:
            return None
        if not isinstance(values, list):
            return values
        dummy = module.ListNode(0)
        cur = dummy
        for v in values:
            cur.next = module.ListNode(v)
            cur = cur.next
        return dummy.next

    def list_to_array(head):
        out = []
        while head is not None:
            out.append(head.val)
            head = head.next
        return out

    def coerce_arg(value, type_name, test_input):
        if is_listnode(type_name):
            return build_list(value)
        if is_sea(type_name):
            ships = test_input.get("ships", [])
            return Sea(ships)
        return value

    def normalize_output(value):
        if value is None:
            return None
        if isinstance(value, module.ListNode):
            return list_to_array(value)
        if isinstance(value, list):
            return [normalize_output(item) for item in value]
        return value

    outputs = []
    for test in tests:
        test_input = test.get("input", {})
        args = []
        for param in params:
            args.append(coerce_arg(test_input.get(param.get("name")), param.get("type"), test_input))
        result = func(*args)
        outputs.append(normalize_output(result))

    print(json.dumps({"outputs": outputs}))

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(json.dumps({"error": str(exc)}))
        sys.exit(1)
"""


def _python_class_runner_script() -> str:
    return """import importlib.util
import json
import sys

def main():
    with open("tests.json", "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    class_name = payload.get("class_name")
    tests = payload.get("tests", [])

    spec = importlib.util.spec_from_file_location("solution", "solution.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cls = getattr(module, class_name)

    outputs = []
    for test in tests:
        test_input = test.get("input", {})
        ops = test_input.get("operations", [])
        args = test_input.get("arguments", [])
        instance = None
        out = []
        for op, arg in zip(ops, args):
            if op == class_name:
                instance = cls(*arg)
                out.append(None)
            else:
                method = getattr(instance, op)
                result = method(*arg)
                out.append(result)
        outputs.append(out)

    print(json.dumps({"outputs": outputs}))

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(json.dumps({"error": str(exc)}))
        sys.exit(1)
"""


def _run_python_class(code: str, test_spec: Dict[str, Any]) -> Dict[str, str]:
    class_signature = (test_spec.get("class_signatures") or {}).get("python") or {}
    class_name = class_signature.get("class_name")
    if not class_name:
        return {"status": "error", "summary": "Missing Python class signature."}

    tests = test_spec.get("tests", [])
    comparison = test_spec.get("comparison", "exact")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        solution_path = tmp_path / "solution.py"
        runner_path = tmp_path / "runner.py"
        tests_path = tmp_path / "tests.json"

        solution_path.write_text(code, encoding="utf-8")
        tests_payload = {"tests": tests, "class_name": class_name}
        tests_path.write_text(json.dumps(tests_payload), encoding="utf-8")

        runner_path.write_text(_python_class_runner_script(), encoding="utf-8")

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


def _java_literal(value: Any, type_name: str) -> str:
    if type_name == "int":
        return str(int(value))
    if type_name == "long":
        return f"{int(value)}L"
    if type_name == "double":
        return f"{float(value)}"
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
    if type_name == "ListNode":
        return "buildList(" + _java_literal(value, "int[]") + ")"
    raise ValueError(f"Unsupported Java type: {type_name}")


def _java_decl_type(type_name: str) -> str:
    mapping = {
        "string": "String",
        "bool": "boolean",
        "int": "int",
        "long": "long",
        "double": "double",
        "int[]": "int[]",
        "string[]": "String[]",
        "int[][]": "int[][]",
        "string[][]": "String[][]",
        "ListNode": "ListNode",
        "Sea": "Sea",
    }
    return mapping.get(type_name, type_name)


def _java_compare_code(return_type: str, comparison: str) -> Tuple[str, str]:
    helpers = []
    compare_call = "result.equals(expected)"
    if return_type == "int":
        compare_call = "result == expected"
    elif return_type == "long":
        compare_call = "result == expected"
    elif return_type == "double":
        compare_call = "Math.abs(result - expected) < 1e-6"
    elif return_type == "bool":
        compare_call = "result == expected"
    elif return_type == "string":
        compare_call = "result.equals(expected)"
    elif return_type == "ListNode":
        helpers.append(
            "static ListNode buildList(int[] values) {"
            "if (values == null) return null;"
            "ListNode dummy = new ListNode(0); ListNode cur = dummy;"
            "for (int v : values) { cur.next = new ListNode(v); cur = cur.next; }"
            "return dummy.next; }"
        )
        helpers.append(
            "static boolean compareListNode(ListNode node, int[] expected) {"
            "int idx = 0; ListNode cur = node;"
            "while (cur != null && expected != null && idx < expected.length) {"
            "if (cur.val != expected[idx]) return false;"
            "cur = cur.next; idx++; }"
            "return cur == null && (expected == null || idx == expected.length); }"
        )
        compare_call = "compareListNode(result, expected)"
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
            "java.util.function.Function<Object, String> rowKey = row -> {"
            "if (row instanceof int[]) return java.util.Arrays.toString((int[]) row);"
            "if (row instanceof Object[]) return java.util.Arrays.deepToString((Object[]) row);"
            "return String.valueOf(row); };"
            "for (int i = 0; i < ac.length; i++) {"
            "Object row = ac[i];"
            "if (row instanceof int[]) { java.util.Arrays.sort((int[]) row); }"
            "else if (row instanceof Object[]) {"
            "java.util.Arrays.sort((Object[]) row, java.util.Comparator.comparing(Object::toString));"
            "}"
            "}"
            "for (int i = 0; i < bc.length; i++) {"
            "Object row = bc[i];"
            "if (row instanceof int[]) { java.util.Arrays.sort((int[]) row); }"
            "else if (row instanceof Object[]) {"
            "java.util.Arrays.sort((Object[]) row, java.util.Comparator.comparing(Object::toString));"
            "}"
            "}"
            "java.util.Arrays.sort(ac, java.util.Comparator.comparing(rowKey));"
            "java.util.Arrays.sort(bc, java.util.Comparator.comparing(rowKey));"
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

    needs_listnode = return_type == "ListNode" or any(
        isinstance(param, dict) and param.get("type") == "ListNode" for param in params
    )
    needs_sea = any(isinstance(param, dict) and param.get("type") == "Sea" for param in params)
    candidate_defines_sea_class = "class Sea" in code
    candidate_defines_sea_interface = "interface Sea" in code
    helper_code, compare_call = _java_compare_code(return_type, comparison)
    support_classes = []
    if needs_listnode and "class ListNode" not in code:
        support_classes.append(
            "class ListNode { int val; ListNode next; "
            "ListNode(int val) { this.val = val; } "
            "ListNode(int val, ListNode next) { this.val = val; this.next = next; } }"
        )
    if needs_sea and not (candidate_defines_sea_class or candidate_defines_sea_interface):
        support_classes.append(
            "interface Sea { boolean hasShips(int[] topRight, int[] bottomLeft); }"
        )
    if needs_sea and not candidate_defines_sea_class:
        support_classes.append(
            "class SeaImpl implements Sea { "
            "private final java.util.Set<String> ships = new java.util.HashSet<>();"
            "SeaImpl(int[][] shipsArr) { "
            "if (shipsArr != null) {"
            "for (int[] s : shipsArr) { ships.add(s[0] + \",\" + s[1]); }"
            "}}"
            "public boolean hasShips(int[] topRight, int[] bottomLeft) {"
            "int x1 = bottomLeft[0], y1 = bottomLeft[1];"
            "int x2 = topRight[0], y2 = topRight[1];"
            "for (String key : ships) {"
            "String[] parts = key.split(\",\");"
            "int x = Integer.parseInt(parts[0]);"
            "int y = Integer.parseInt(parts[1]);"
            "if (x1 <= x && x <= x2 && y1 <= y && y <= y2) return true;"
            "}"
            "return false; } }"
        )

    test_blocks = []
    for idx, test in enumerate(tests):
        test_input = test.get("input", {})
        expected = test.get("output")
        assignments = []
        args = []
        for param in params:
            name = param.get("name")
            type_name = param.get("type")
            decl_type = _java_decl_type(type_name)
            if type_name == "Sea":
                ships_literal = _java_literal(test_input.get("ships"), "int[][]")
                assignments.append(f"int[][] ships = {ships_literal};")
                if candidate_defines_sea_class:
                    assignments.append(f"{decl_type} sea = new Sea(ships);")
                else:
                    assignments.append(f"{decl_type} sea = new SeaImpl(ships);")
                args.append("sea")
            elif type_name == "ListNode":
                literal = _java_literal(test_input.get(name), "ListNode")
                assignments.append(f"{decl_type} {name} = {literal};")
                args.append(name)
            else:
                literal = _java_literal(test_input.get(name), type_name)
                assignments.append(f"{decl_type} {name} = {literal};")
                args.append(name)
        decl_return_type = _java_decl_type(return_type)
        if return_type == "ListNode":
            expected_literal = _java_literal(expected, "int[]")
            expected_decl = f"int[] expected = {expected_literal};"
        else:
            expected_literal = _java_literal(expected, return_type)
            expected_decl = f"{decl_return_type} expected = {expected_literal};"
        test_blocks.append(
            "{\n"
            + "\n".join(assignments)
            + f"\n{expected_decl}"
            + f"\n{decl_return_type} result = Solution.{method_name}({', '.join(args)});"
            + f"\nif (!({compare_call})) {{"
            + f"\n  System.out.println(\"FAIL:{idx}\");"
            + "\n  return;"
            + "\n}"
            + "\n}"
        )

    runner_source = (
        "import java.util.*;\n"
        + ("\n".join(support_classes) + "\n" if support_classes else "")
        + "public class Runner {\n"
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
        solution_path.write_text(_prepare_java_source(code), encoding="utf-8")
        runner_path.write_text(runner_source, encoding="utf-8")

        compile_result = subprocess.run(
            ["javac", "Solution.java", "Runner.java"],
            cwd=tmpdir,
            capture_output=True,
            text=True,
            timeout=15,
        )
        if compile_result.returncode != 0:
            return {
                "status": "error",
                "summary": _clean_error(compile_result.stderr) or "Compilation error.",
                "details": compile_result.stderr,
            }

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


def _run_java_class(code: str, test_spec: Dict[str, Any]) -> Dict[str, str]:
    class_signature = (test_spec.get("class_signatures") or {}).get("java") or {}
    class_name = class_signature.get("class_name")
    if not class_name:
        return {"status": "error", "summary": "Missing Java class signature."}

    tests = test_spec.get("tests", [])

    test_blocks = []
    for idx, test in enumerate(tests):
        test_input = test.get("input", {})
        expected = test.get("output", [])
        ops = test_input.get("operations", [])
        args = test_input.get("arguments", [])
        expected_literals = [
            "null" if value is None else f"\"{value}\"" for value in expected
        ]
        block_lines = [
            f"{class_name} obj = null;",
            "java.util.List<String> actual = new java.util.ArrayList<>();",
            f"String[] expected = new String[]{{{', '.join(expected_literals)}}};",
        ]
        for op, arg in zip(ops, args):
            if op == class_name:
                ctor_args = ", ".join(str(int(v)) for v in arg)
                block_lines.append(f"obj = new {class_name}({ctor_args});")
                block_lines.append("actual.add(null);")
            elif op == "put":
                put_args = ", ".join(str(int(v)) for v in arg)
                block_lines.append(f"obj.put({put_args});")
                block_lines.append("actual.add(null);")
            elif op == "get":
                get_args = ", ".join(str(int(v)) for v in arg)
                block_lines.append(f"actual.add(String.valueOf(obj.get({get_args})));" )
        block_lines.append(
            f"if (!compareList(actual, expected)) {{ System.out.println(\"FAIL:{idx}\"); return; }}"
        )
        test_blocks.append("{\n" + "\n".join(block_lines) + "\n}")

    runner_source = (
        "import java.util.*;\n"
        "public class Runner {\n"
        "static boolean compareList(List<String> actual, String[] expected) {"
        "if (actual.size() != expected.length) return false;"
        "for (int i = 0; i < expected.length; i++) {"
        "String a = actual.get(i); String b = expected[i];"
        "if (!Objects.equals(a, b)) return false;"
        "} return true; }\n"
        "public static void main(String[] args) {\n"
        "try {\n"
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
        solution_path.write_text(_prepare_java_source(code), encoding="utf-8")
        runner_path.write_text(runner_source, encoding="utf-8")

        compile_result = subprocess.run(
            ["javac", "Solution.java", "Runner.java"],
            cwd=tmpdir,
            capture_output=True,
            text=True,
            timeout=15,
        )
        if compile_result.returncode != 0:
            return {
                "status": "error",
                "summary": _clean_error(compile_result.stderr) or "Compilation error.",
                "details": compile_result.stderr,
            }

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
    if type_name == "long long":
        return f"{int(value)}LL"
    if type_name == "double":
        return f"{float(value)}"
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
    if type_name == "ListNode*":
        return "buildList(std::vector<int>{" + ", ".join(str(int(v)) for v in value) + "})"
    raise ValueError(f"Unsupported C++ type: {type_name}")


def _cpp_compare_code(return_type: str, comparison: str) -> Tuple[str, str]:
    helpers = []
    compare_call = "result == expected"
    unordered = "true" if comparison == "unordered" else "false"
    if return_type == "long long":
        compare_call = "result == expected"
    elif return_type == "double":
        helpers.append("#include <cmath>\n")
        compare_call = "std::fabs(result - expected) < 1e-6"
    elif return_type == "ListNode*":
        compare_call = "compareListNode(result, expected)"
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

    needs_listnode = return_type == "ListNode*" or any(
        isinstance(param, dict) and param.get("type") == "ListNode*" for param in params
    )
    needs_sea = any(isinstance(param, dict) and param.get("type") == "Sea" for param in params)
    helper_code, compare_call = _cpp_compare_code(return_type, comparison)

    listnode_struct = ""
    if needs_listnode and ("struct ListNode" not in code and "class ListNode" not in code):
        listnode_struct = (
            "struct ListNode { int val; ListNode* next; "
            "ListNode(int x) : val(x), next(nullptr) {} };\n"
        )
    sea_class = ""
    if needs_sea and ("class Sea" not in code and "struct Sea" not in code):
        sea_class = (
            "class Sea {"
            "public:"
            "std::unordered_set<long long> ships;"
            "Sea(const std::vector<std::vector<int>>& shipsVec) {"
            "for (const auto& s : shipsVec) {"
            "long long key = (static_cast<long long>(s[0]) << 32) | static_cast<unsigned int>(s[1]);"
            "ships.insert(key); }"
            "}"
            "bool hasShips(const std::vector<int>& topRight, const std::vector<int>& bottomLeft) {"
            "int x1 = bottomLeft[0], y1 = bottomLeft[1];"
            "int x2 = topRight[0], y2 = topRight[1];"
            "for (long long key : ships) {"
            "int x = static_cast<int>(key >> 32);"
            "int y = static_cast<int>(key & 0xffffffff);"
            "if (x1 <= x && x <= x2 && y1 <= y && y <= y2) return true;"
            "}"
            "return false; }"
            "};\n"
        )

    listnode_helpers = ""
    if needs_listnode:
        listnode_helpers = (
            "ListNode* buildList(const std::vector<int>& values) {"
            "ListNode dummy(0); ListNode* cur = &dummy;"
            "for (int v : values) { cur->next = new ListNode(v); cur = cur->next; }"
            "return dummy.next; }\n"
            "std::vector<int> listToVector(ListNode* node) {"
            "std::vector<int> out; while (node) { out.push_back(node->val); node = node->next; }"
            "return out; }\n"
            "bool compareListNode(ListNode* node, const std::vector<int>& expected) {"
            "return listToVector(node) == expected; }\n"
        )

    test_blocks = []
    for idx, test in enumerate(tests):
        test_input = test.get("input", {})
        expected = test.get("output")
        assignments = []
        args = []
        for param in params:
            name = param.get("name")
            type_name = param.get("type")
            if type_name == "Sea":
                ships_literal = _cpp_literal(test_input.get("ships"), "vector<vector<int>>")
                assignments.append(f"std::vector<std::vector<int>> ships = {ships_literal};")
                assignments.append("Sea sea(ships);")
                args.append("sea")
            elif type_name == "ListNode*":
                literal = _cpp_literal(test_input.get(name), "ListNode*")
                assignments.append(f"ListNode* {name} = {literal};")
                args.append(name)
            else:
                literal = _cpp_literal(test_input.get(name), type_name)
                assignments.append(f"{type_name} {name} = {literal};")
                args.append(name)
        if return_type == "ListNode*":
            expected_literal = _cpp_literal(expected, "vector<int>")
            expected_decl = f"std::vector<int> expected = {expected_literal};"
        else:
            expected_literal = _cpp_literal(expected, return_type)
            expected_decl = f"{return_type} expected = {expected_literal};"
        test_blocks.append(
            "{\n"
            + "\n".join(assignments)
            + f"\n{expected_decl}"
            + f"\n{return_type} result = {func_name}({', '.join(args)});"
            + f"\nif (!({compare_call})) {{ std::cout << \"FAIL:{idx}\"; return 0; }}\n"
            + "}\n"
        )

    runner_source = (
        "#include <bits/stdc++.h>\n"
        "using namespace std;\n"
        + helper_code
        + "\n"
        + listnode_struct
        + sea_class
        + "\n"
        + code
        + "\n"
        + listnode_helpers
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
            return {"status": "error", "summary": _clean_error(compile_result.stderr) or "Compilation error."}

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


def _run_cpp_class(code: str, test_spec: Dict[str, Any]) -> Dict[str, str]:
    class_signature = (test_spec.get("class_signatures") or {}).get("cpp") or {}
    class_name = class_signature.get("class_name")
    if not class_name:
        return {"status": "error", "summary": "Missing C++ class signature."}

    tests = test_spec.get("tests", [])

    test_blocks = []
    for idx, test in enumerate(tests):
        test_input = test.get("input", {})
        expected = test.get("output", [])
        ops = test_input.get("operations", [])
        args = test_input.get("arguments", [])
        expected_literals = []
        for value in expected:
            if value is None:
                expected_literals.append("\"null\"")
            else:
                expected_literals.append(f"\"{value}\"")
        block_lines = [
            "std::vector<std::string> actual;",
            f"std::vector<std::string> expected = {{{', '.join(expected_literals)}}};",
            f"{class_name}* obj = nullptr;",
        ]
        for op, arg in zip(ops, args):
            if op == class_name:
                ctor_args = ", ".join(str(int(v)) for v in arg)
                block_lines.append(f"obj = new {class_name}({ctor_args});")
                block_lines.append("actual.push_back(\"null\");")
            elif op == "put":
                put_args = ", ".join(str(int(v)) for v in arg)
                block_lines.append(f"obj->put({put_args});")
                block_lines.append("actual.push_back(\"null\");")
            elif op == "get":
                get_args = ", ".join(str(int(v)) for v in arg)
                block_lines.append(f"actual.push_back(std::to_string(obj->get({get_args})));")
        block_lines.append(
            f"if (!compareList(actual, expected)) {{ std::cout << \"FAIL:{idx}\"; return 0; }}"
        )
        test_blocks.append("{\n" + "\n".join(block_lines) + "\n}")

    runner_source = (
        "#include <bits/stdc++.h>\n"
        "using namespace std;\n"
        + code
        + "\n"
        + "bool compareList(const std::vector<std::string>& actual, const std::vector<std::string>& expected) {\n"
        + "return actual == expected;\n"
        + "}\n"
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
            return {"status": "error", "summary": _clean_error(compile_result.stderr) or "Compilation error."}

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
            "{\n"
            + "\n".join(assignments)
            + f"\n{expected_decl}\n{result_decl}\n"
            + f"if (!({compare_expr})) {{ printf(\"FAIL:{idx}\"); return 0; }}\n"
            + "}\n"
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


def _prepare_java_source(code: str) -> str:
    stripped = code.lstrip()
    if "import " in code:
        return code
    if stripped.startswith("package "):
        lines = code.splitlines()
        out = []
        inserted = False
        for line in lines:
            out.append(line)
            if line.strip().startswith("package ") and not inserted:
                out.append("import java.util.*;")
                inserted = True
        return "\n".join(out) + ("\n" if not code.endswith("\n") else "")
    return "import java.util.*;\n" + code


def _clean_error(stderr: str) -> str:
    if not stderr:
        return ""
    lines = [line.strip() for line in stderr.splitlines() if line.strip()]
    if not lines:
        return ""
    for line in lines:
        if "error:" in line.lower():
            return line
    return lines[-1]
