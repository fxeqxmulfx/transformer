#!/usr/bin/env bash
# Run or independently replay all original task/seed cells and the NV11 control.
set -euo pipefail

if [[ $# -ne 2 || ( $1 != cell && $1 != replay ) ]]; then
    echo 'usage: run_stress.sh {cell|replay} RUN_DIR' >&2
    exit 2
fi
dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
bin=${PHASOR_BIN:-"$dir/target/release/phasor-value-transport"}
routes=$("$bin" routes)
names=$(awk '{print $1}' <<< "$routes")
for name in arithmetic-cubic-product $names; do
    for seed in 0 1 2; do
        "$bin" "$1" "$name" "$seed" "$2"
    done
done
