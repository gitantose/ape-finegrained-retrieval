#!/usr/bin/env bash
# Specific sequential benchmark commands; encoding and ranking stay separate.
set -euo pipefail
cd -- "$(dirname -- "$0")/.."
suite_config=${1:-configs/benchmarks.json}
suite_seconds=${2:?Supply the explicitly approved shared time allowance in seconds}
trap '.venv/bin/python -m ape_retrieval.benchmarks.report --config "$suite_config"' EXIT
trap 'exit 2' TERM INT
for benchmark in squad propsearch nq; do
    echo "Starting benchmark: $benchmark"
    .venv/bin/python -u -m ape_retrieval.benchmarks.encoding encode "$benchmark" --config "$suite_config" --budget-seconds "$suite_seconds"
    .venv/bin/python -u -m ape_retrieval.benchmarks.retrieval "$benchmark" --config "$suite_config"
done
