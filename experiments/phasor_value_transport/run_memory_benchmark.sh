#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
out=${1:?usage: run_memory_benchmark.sh RUN_DIR [original|closed|transition|recurrent ...]}
shift
if (( $# == 0 )); then set -- original closed transition recurrent; fi
export PHASOR_EVAL_BATCH_CAP=${PHASOR_EVAL_BATCH_CAP:-16}
bin=${PHASOR_BIN:-./target/release/phasor-value-transport}
for arm in "$@"; do
    for task in basis-parity20 gpt-probe-key-value-binding suite-lookup arithmetic-cubic-product; do
        for seed in 0 1 2; do
            "$bin" memory-bench "$arm" "$task" "$seed" 8000 "$out"
        done
    done
done
