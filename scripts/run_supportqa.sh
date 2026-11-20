#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "$0")/.."
export PYTHONDONTWRITEBYTECODE=1
root=runs/supportqa
batch_size=${1:-14}
workers=${2:-1}
[[ "$batch_size" =~ ^[1-9][0-9]*$ ]]
[[ "$workers" =~ ^[1-3]$ ]]
python=.venv/bin/python
module=ape_retrieval.supportqa

phase_completed() {
    "$python" -c 'import json, pathlib, sys
p = pathlib.Path(sys.argv[1]); phase = sys.argv[2]
state = json.loads(p.read_text()) if p.exists() else {}
raise SystemExit(state.get("phases", {}).get(phase, {}).get("status") != "COMPLETED")' "$root/status.json" "$1"
}
if phase_completed finalize; then
    cat "$root/report.md"
    exit 0
fi

mkdir -p "$root"
exec >>"$root/progress.log" 2>>"$root/errors.log"
exec 9>"$root/controller.lock"
flock -n 9
printf '%s\n' "$$" >"$root/controller.pid"

"$python" - "$root" "$$" "$batch_size" "$workers" <<'PY'
import json
import os
from pathlib import Path
import sys
import time

root, controller = Path(sys.argv[1]), int(sys.argv[2])
path = root / 'controller-attempts.json'
attempts = json.loads(path.read_text()) if path.exists() else []
attempts.append({'pid': controller, 'process_group': os.getpgid(controller),
                 'started_epoch': time.time(), 'command': ['bash', 'scripts/run_supportqa.sh', *sys.argv[3:]]})
temporary = path.with_suffix('.json.tmp')
temporary.write_text(json.dumps(attempts, indent=2) + '\n')
temporary.replace(path)
PY
annotator=
cleanup() {
    result=$?
    if [ -n "$annotator" ]; then
        kill -TERM "$annotator" 2>/dev/null || true
        wait "$annotator" 2>/dev/null || true
    fi
    rm -f -- "$root/controller.pid"
    if [ "$result" -ne 0 ] && [ -s "$root/selection.json" ]; then
        "$python" -u -m "$module" report 9>&- || true
        "$python" -c 'import json,sys
s=json.load(open(sys.argv[1])); status=s["status"]
print(f"FINAL STATUS {status}" if status != "RUNNING" else "CONTROLLER STOPPED. An independent phase is still active.")' "$root/status.json"
    fi
}
trap cleanup EXIT
trap 'exit 130' INT TERM

run_phase() {
    if ! phase_completed "$1"; then
        "$python" -u -m "$module" "$1" 9>&-
    fi
}

run_phase prepare
env -u OPENAI_API_KEY -u CODEX_API_KEY "$python" -u -m "$module" annotate --batch-size "$batch_size" --workers "$workers" 9>&- &
annotator=$!
# A previously detached CPU phase retains the common suite lock until it exits.
flock -x runs/benchmark-suite/suite.lock true 9>&- &
wait "$!"
run_phase decompose
for phase in acquire encode-simcse rank-simcse encode-contriever rank-contriever align support-simcse support-contriever; do
    run_phase "$phase"
done
# Annotation failure remains explicit in its receipt. Finalization also validates coverage.
wait "$annotator" || true
annotator=
run_phase finalize
