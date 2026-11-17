#!/usr/bin/env bash
# Train both independent frozen builds before allowing any held-out score.
set -euo pipefail
if (( $# != 4 )); then
  echo 'usage: bash run.sh WORKSPACE STORAGE PRODUCER FRESH_OUTPUT' >&2
  exit 2
fi
workspace=$(realpath "$1")
storage=$(realpath "$2")
producer=$(git -C "$workspace" rev-parse --verify "$3^{commit}")
output=$(realpath -m "$4")
experiment=experiments/composed_retrieval
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
  experiments/convex_stress_catalog experiments/convex_binding_diagnostic/src/counts.rs \
  experiments/convex_stress_sequences/src/{dataset,prefix,retrieval,wire}.rs \
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
  cargo test --locked --offline --manifest-path "$manifest" --target-dir "$build_dir" > "$output/$arm/tests.log" 2>&1
  cargo build --locked --offline --release --manifest-path "$manifest" --target-dir "$build_dir" > "$output/$arm/build.log" 2>&1
  cp "$build_dir/release/composed-retrieval" "$output/$arm/composed-retrieval"
done
(
  cd "$output"
  sha256sum source.tar producer.commit {original,repeat}/composed-retrieval {original,repeat}/source/"$experiment"/protocol.json > frozen.sha256
  sha256sum -c frozen.sha256
)
protocol="$output/original/source/$experiment/protocol.json"
mapfile -t cases < <(jq -r '.cases|keys[]' "$protocol")
mapfile -t modes < <(jq -r '.modes[]' "$protocol")
mapfile -t seeds < <(jq -r '.seeds[]' "$protocol")
(( ${#cases[@]} == 4 && ${#modes[@]} == 2 && ${#seeds[@]} == 3 ))
invoke() {
  local arm=$1 action=$2
  shift 2
  prlimit --as=2147483648:2147483648 -- "$output/$arm/composed-retrieval" "$action" "$workspace" "$storage" "$output" "$arm" "$@"
}
export -f invoke
export output workspace storage
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
echo 'All 48 training endpoints sealed. Models, optimizer traces, original scores and raw f64 logits independently matched and replayed.'
