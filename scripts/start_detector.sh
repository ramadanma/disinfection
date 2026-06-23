#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PY="${PROJECT_ROOT}/.venv/bin/python"
CONFIG_PATH="${1:-${PROJECT_ROOT}/configs/config.yaml}"

PID_DIR="${PROJECT_ROOT}/run"
PID_FILE="${PID_DIR}/disinfection.pid"

LOG_DIR="${PROJECT_ROOT}/logs"
mkdir -p "${PID_DIR}" "${LOG_DIR}"
STARTUP_LOG="${LOG_DIR}/startup.log"

if [[ ! -x "${PY}" ]]; then
  echo "venv python not found or not executable: ${PY}"
  echo "please create venv: python3 -m venv .venv"
  exit 1
fi

if [[ -f "${PID_FILE}" ]]; then
  PID="$(cat "${PID_FILE}" || true)"
  if [[ -n "${PID}" ]] && kill -0 "${PID}" >/dev/null 2>&1; then
    echo "disinfection already running, pid=${PID}"
    exit 0
  fi
  rm -f "${PID_FILE}"
fi

cd "${PROJECT_ROOT}"

nohup "${PY}" -m disinfection.services.disinfection_service --config "${CONFIG_PATH}" \
  >>"${STARTUP_LOG}" 2>&1 &

echo $! > "${PID_FILE}"
echo "disinfection started, pid=$(cat "${PID_FILE}")"
