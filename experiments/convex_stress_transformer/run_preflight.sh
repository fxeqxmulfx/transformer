#!/usr/bin/env bash
# Data reconstruction and resource forecasts only; no model training or scoring.
set -euo pipefail
destination=${1:?usage: run_preflight.sh FRESH_OUTPUT_DIRECTORY}
binary=${TRANSFORMER_BINARY:-experiments/convex_stress_transformer/target/release/convex-stress-transformer}
test ! -e "$destination"
mkdir -p "$destination"
sha256sum "$binary" > "$destination/binary.sha256"
"$binary" tasks > "$destination/tasks.txt"
test "$(wc -l < "$destination/tasks.txt")" -eq 35
while IFS= read -r task; do
  /usr/bin/time -v -o "$destination/$task.resources" \
    bash experiments/convex_stress_hull/limited.sh 8 "$binary" preflight "$task" \
    > "$destination/$task.json"
done < "$destination/tasks.txt"
jq -s '{status:"preflight_complete",task_count:length,training_or_scoring:false,
  admitted:map(select(.memory.status=="admitted"))|length,
  rejected:map(select(.memory.status=="rejected"))|length,
  maximum_model_buffer_bytes:map(.memory.buffer_bytes_upper_bound // 0)|max,
  maximum_dataset_vector_bytes:map(.dataset_vector_bytes)|max,
  process_address_space_limit_bytes:8589934592}' "$destination"/*.json > "$destination/summary.tmp"
mv "$destination/summary.tmp" "$destination/summary.json"
