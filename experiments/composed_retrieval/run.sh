#!/usr/bin/env bash
# Train both independent frozen builds before allowing any held-out score.
set -euo pipefail
if (( $# < 4 || $# > 6 )); then
  echo 'usage: bash run.sh WORKSPACE STORAGE PRODUCER FRESH_OUTPUT [sparse|dense|logodds|{arithmetic|logodds-arithmetic|output-offsets|prefix-cohort|rewrite-cohort} SOURCE_INPUTS]' >&2
  exit 2
fi
if [[ -n "${BASH_SOURCE[0]:-}" ]]; then
  driver=$(git -C "$1" show "$3:experiments/composed_retrieval/run.sh")
  exec bash -c "$driver" "$0" "$@"
fi
workspace=$(realpath "$1")
storage=$(realpath "$2")
producer=$(git -C "$workspace" rev-parse --verify "$3^{commit}")
output=$(realpath -m "$4")
experiment=experiments/composed_retrieval
variant=${5:-sparse}
source_inputs=
features=()
protocols=(protocol.json)
case "$variant" in
  sparse) ;;
  dense) features=(--features dense-forward); protocols+=(dense/protocol.json) ;;
  logodds) features=(--features logodds-depth); protocols+=(dense/protocol.json logodds/protocol.json) ;;
  arithmetic|logodds-arithmetic|output-offsets|prefix-cohort|rewrite-cohort)
    (( $# == 6 ))
    source_inputs=$(realpath "$6")
    features=(--features arithmetic-cohort)
    protocols+=(dense/protocol.json arithmetic/protocol.json)
    if [[ "$variant" != arithmetic ]]; then
      features=(--features arithmetic-cohort,logodds-depth)
      protocols+=(logodds/protocol.json logodds/arithmetic/protocol.json)
    fi
    if [[ "$variant" == output-offsets || "$variant" == prefix-cohort || "$variant" == rewrite-cohort ]]; then
      features=(--features output-offsets)
      protocols+=(output_offsets/protocol.json)
    fi
    if [[ "$variant" == prefix-cohort ]]; then
      features=(--features prefix-cohort)
      protocols+=(prefix/protocol.json)
    fi
    if [[ "$variant" == rewrite-cohort ]]; then
      features=(--features rewrite-cohort)
      protocols+=(rewrite/protocol.json)
    fi
    ;;
  *) echo 'unknown coordinate schedule' >&2; exit 2 ;;
esac
[[ -n "${source_inputs:-}" || $# -le 5 ]]
workers=18
admit() {
  test "$(df -B1 --output=avail "$workspace" | tail -1)" -gt 2147483648
  test "$(df -B1 --output=avail "$storage" | tail -1)" -gt 17179869184
  ps -e -o rss= | awk -v reserved="$(( workers * 2147483648 ))" '{sum+=$1*1024} END {print "aggregate RSS",sum; if(sum+reserved>110000000000)exit 1}'
}
admit
mkdir -p "$(dirname "$output")"
mkdir "$output"
git -C "$workspace" archive -o "$output/source.tar" "$producer" -- "$experiment" \
  experiments/{learned_retrieval,learned_retrieval_transfer,convex_bilinear_sequence,convex_stress_minibatch,convex_stress} \
  experiments/convex_stress_catalog experiments/convex_stress_arithmetic_data experiments/convex_binding_diagnostic/src/counts.rs \
  experiments/convex_stress_{sequences,typed_data,remaining_synthetic_data,generative_data,calibrated,causal_pairs,counts,repair,tagged,adaptive,density,decode,relabel,budgets,schatten,moments,spectral,hull} \
  src/Transformer/Grokking/FlatSpectrum/ConvexChannel/ComposedRetrieval{,.lean} review/flat_spectrum.md
printf '%s\n' "$producer" > "$output/producer.commit"
rustc -Vv > "$output/rustc.txt"
cargo -V > "$output/cargo.txt"
for arm in original repeat; do
  src="$output/$arm/source"
  build_dir="$output/$arm/target"
  mkdir -p "$src"
  tar -xf "$output/source.tar" -C "$src"
  export RETRIEVAL_PRODUCER="$producer" CARGO_BUILD_JOBS=4
  export RUSTFLAGS="--remap-path-prefix=$src=/composed-source --remap-path-prefix=$build_dir=/composed-target"
  manifest="$src/$experiment/Cargo.toml"
  cargo test --locked --offline "${features[@]}" --manifest-path "$manifest" --target-dir "$build_dir" > "$output/$arm/tests.log" 2>&1
  cargo build --locked --offline --release "${features[@]}" --manifest-path "$manifest" --target-dir "$build_dir" > "$output/$arm/build.log" 2>&1
  cp "$build_dir/release/composed-retrieval" "$output/$arm/composed-retrieval"
done
(
  cd "$output"
  inputs=(source.tar producer.commit {original,repeat}/composed-retrieval)
  for file in "${protocols[@]}"; do inputs+=({original,repeat}/source/"$experiment/$file"); done
  sha256sum "${inputs[@]}" > frozen.sha256
  sha256sum -c frozen.sha256
)
protocol="$output/original/source/$experiment/protocol.json"
mapfile -t modes < <(jq -r '.modes[]' "$protocol")
mapfile -t seeds < <(jq -r '.seeds[]' "$protocol")
(( ${#modes[@]} == 2 && ${#seeds[@]} == 3 ))
invoke() {
  local arm=$1 action=$2
  shift 2
  local meter=()
  if [[ "$action" == fit && "$variant" != sparse ]]; then
    meter=(/usr/bin/time -f '%e %U %S %M' -o "$output/$arm/fit-$1-$2-$3.time")
  fi
  "${meter[@]}" prlimit --as=2147483648:2147483648 -- "$output/$arm/composed-retrieval" "$action" "$workspace" "$storage" "$output" "$arm" "$@"
}
export -f invoke
export output workspace storage variant
if [[ -n "${source_inputs:-}" ]]; then
  mkdir "$output/inputs"
  cp "$source_inputs/plan.json" "$output/inputs/"
  if [[ "$variant" != prefix-cohort && "$variant" != rewrite-cohort ]]; then
    cp "$source_inputs"/{arithmetic_seed1,grokking_task_inventory}.json "$output/inputs/"
  fi
  for arm in original repeat; do invoke "$arm" inventory "$output/inputs" >"$output/$arm/inventory.json"; done
  cmp "$output/original/inventory.json" "$output/repeat/inventory.json"
  inventory="$output/original/inventory.json"
  supported=48
  unavailable=6
  if [[ "$variant" == output-offsets ]]; then supported=51; fi
  if [[ "$variant" == prefix-cohort ]]; then supported=9; unavailable=0; fi
  if [[ "$variant" == rewrite-cohort ]]; then supported=7; unavailable=0; fi
  jq -e --argjson n "$supported" --argjson u "$unavailable" '(.cases|length)==($n+$u) and .fits==($n*12)' "$inventory" >/dev/null
  mapfile -t cases < <(jq -r '.cases|to_entries[]|select(.value.dimensions != null)|.key' "$inventory")
  (( ${#cases[@]} == supported ))
  required=$(jq -er '(.raw_bytes // .raw_bytes_upper_bound) | select(type=="number" and .>0) | .+17179869184' "$inventory")
  test "$(df -B1 --output=avail "$output" | tail -1)" -gt "$required"
else
  mapfile -t cases < <(jq -r '.cases|keys[]' "$protocol")
  (( ${#cases[@]} == 4 ))
fi
fit_cells() {
  for id in "${cases[@]}"; do for mode in "${modes[@]}"; do for seed in "${seeds[@]}"; do for arm in original repeat; do
    printf '%s\0' "$arm" "$id" "$mode" "$seed"
  done; done; done; done
}
admit
fit_cells | xargs -0 -r -n4 -P "$workers" bash -c 'invoke "$1" fit "$2" "$3" "$4" > "$output/$1/fit-$2-$3-$4.log" 2>&1' _
(cd "$output" && sha256sum -c frozen.sha256)
for arm in original repeat; do invoke "$arm" seal; done
for action in score replay; do
  admit
  (
    for id in "${cases[@]}"; do for arm in original repeat; do printf '%s\0' "$arm" "$id"; done; done
  ) | xargs -0 -r -n2 -P 4 bash -c 'invoke "$1" "$0" "$2" > "$output/$1/$0-$2.log" 2>&1' "$action"
done
for id in "${cases[@]}"; do diff -qr "$output/original/run/$id/scores" "$output/repeat/run/$id/scores"; done
echo "All $((2 * ${#cases[@]} * ${#modes[@]} * ${#seeds[@]})) training endpoints sealed. Models, optimizer traces, original scores and raw f64 logits independently matched and replayed."
