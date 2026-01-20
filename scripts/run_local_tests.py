import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.question_tests import get_cached_tests
from app.test_runner import run_code_tests


def main() -> int:
    parser = argparse.ArgumentParser(description="Run local interview tests against a solution file.")
    parser.add_argument("--title", help="Question title (uses local storage/question_tests)")
    parser.add_argument("--spec", help="Path to a test spec JSON file")
    parser.add_argument("--language", required=True, help="python|java|cpp")
    parser.add_argument("--code", required=True, help="Path to solution source file")
    args = parser.parse_args()

    if not args.title and not args.spec:
        raise SystemExit("Provide --title or --spec")

    if args.title:
        test_spec = get_cached_tests(args.title)
        if not test_spec:
            raise SystemExit(f"No local tests found for title: {args.title}")
    else:
        test_path = Path(args.spec)
        if not test_path.exists():
            raise SystemExit(f"Spec not found: {test_path}")
        test_spec = json.loads(test_path.read_text(encoding="utf-8"))

    code_path = Path(args.code)
    if not code_path.exists():
        raise SystemExit(f"Code not found: {code_path}")
    code = code_path.read_text(encoding="utf-8")

    result = run_code_tests(args.language, code, test_spec)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
