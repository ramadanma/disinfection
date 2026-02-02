#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PID_FILE="${PROJECT_ROOT}/run/disinfection.pid"

if [[ ! -f "${PID_FILE}" ]]; then
  echo "pid file not found: ${PID_FILE}"
  exit 0
fi

PID="$(cat "${PID_FILE}" || true)"
if [[ -z "${PID}" ]]; then
  echo "empty pid file"
  rm -f "${PID_FILE}"
  exit 0
fi

if ! kill -0 "${PID}" >/dev/null 2>&1; then
  echo "process not running, pid=${PID}"
  rm -f "${PID_FILE}"
  exit 0
fi

echo "stopping disinfection pid=${PID} ..."
kill -TERM "${PID}"

for i in {1..100}; do
  if ! kill -0 "${PID}" >/dev/null 2>&1; then
    rm -f "${PID_FILE}"
    echo "stopped"
    exit 0
  fi
  sleep 0.1
done

echo "not stopped in time, sending KILL ..."
kill -KILL "${PID}" || true
rm -f "${PID_FILE}"
echo "killed"
