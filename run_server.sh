#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="${PROJECT_ROOT}/.venv"

if [[ ! -d "${VENV_PATH}" ]]; then
  python3 -m venv "${VENV_PATH}"
fi

# shellcheck disable=SC1090
source "${VENV_PATH}/bin/activate"

pip install --upgrade pip > /dev/null
pip install -r "./requirements.txt"

if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
  echo "Set DEEPSEEK_API_KEY in your environment before running the server." >&2
  exit 1
fi

export DEEPSEEK_MODEL="${DEEPSEEK_MODEL:-deepseek-chat}"
export DEEPSEEK_PHASE_MODEL="${DEEPSEEK_PHASE_MODEL:-deepseek-chat}"

if [[ -z "${FLASK_SECRET_KEY:-}" ]]; then
  FLASK_SECRET_KEY="$(python - <<'PY'
import secrets
print(secrets.token_hex(16))
PY
)"
  export FLASK_SECRET_KEY
fi

export FLASK_APP="app.app"

exec flask run --host "0.0.0.0" --port "1111"
