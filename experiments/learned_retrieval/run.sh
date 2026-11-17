#!/usr/bin/env bash
# Build and train twice from the same tracked source before evaluating either cohort.
set -euo pipefail
if (( $# != 4 )); then
  echo 'usage: bash run.sh WORKSPACE STORAGE PRODUCER FRESH_OUTPUT' >&2
  exit 2
fi
workspace=$(realpath "$1")
storage=$(realpath "$2")
producer=$(git -C "$workspace" rev-parse --verify "$3^{commit}")
output=$(realpath -m "$4")
experiment=experiments/learned_retrieval
admit() {
  df -B1 "$workspace" "$storage"
  test "$(df -B1 --output=avail "$workspace" | tail -1)" -gt 2147483648
  test "$(df -B1 --output=avail "$storage" | tail -1)" -gt 17179869184
  ps -e -o rss= | awk '{sum+=$1*1024} END {print "aggregate RSS",sum; if(sum+25769803776>110000000000)exit 1}'
}
admit
mkdir -p "$(dirname "$output")"
mkdir "$output"
git -C "$workspace" archive -o "$output/source.tar" "$producer" -- \
  "$experiment" experiments/convex_bilinear_sequence experiments/convex_stress_minibatch \
  experiments/convex_stress experiments/convex_binding_diagnostic/src/counts.rs \
  review/flat_spectrum.md src/Transformer/Grokking/FlatSpectrum/ConvexChannel/LearnedRetrieval
printf '%s\n' "$producer" > "$output/producer.commit"
rustc -Vv > "$output/rustc.txt"
cargo -V > "$output/cargo.txt"
for arm in original repeat; do
  source_dir="$output/$arm/source"
  mkdir -p "$source_dir"
  tar -xf "$output/source.tar" -C "$source_dir"
  manifest="$source_dir/$experiment/Cargo.toml"
  target_dir="$output/$arm/target"
  export RETRIEVAL_PRODUCER="$producer" CARGO_BUILD_JOBS=4
  export RUSTFLAGS="--remap-path-prefix=$source_dir=/retrieval-source --remap-path-prefix=$target_dir=/retrieval-target"
  cargo test --locked --offline --manifest-path "$manifest" --target-dir "$target_dir" \
    > "$output/$arm/tests.log" 2>&1
  cargo build --release --locked --offline --manifest-path "$manifest" --target-dir "$target_dir" \
    > "$output/$arm/build.log" 2>&1
  cp "$target_dir/release/learned-retrieval" "$output/$arm/learned-retrieval"
done
(
  cd "$output"
  sha256sum source.tar producer.commit original/learned-retrieval repeat/learned-retrieval \
    "original/source/$experiment/protocol.json" "repeat/source/$experiment/protocol.json" > frozen.sha256
  sha256sum -c frozen.sha256
)
protocol="$output/original/source/$experiment/protocol.json"
mapfile -t seeds < <(jq -r '.seeds[]' "$protocol")
mapfile -t modes < <(jq -r '.modes[]' "$protocol")
admit
workers=()
for arm in original repeat; do
  for seed in "${seeds[@]}"; do
    for mode in "${modes[@]}"; do
      dir="$output/$arm"
      prlimit --as=2147483648:2147483648 -- "$dir/learned-retrieval" fit \
        "$workspace" "$storage" "$dir/run" "$mode" "$seed" > "$dir/fit-$mode-$seed.log" 2>&1 &
      workers+=("$!")
    done
  done
done
failed=0
for job in "${workers[@]}"; do
  if ! wait "$job"; then failed=1; fi
done
if ((failed)); then echo 'a training cell failed; see fit logs' >&2; exit 1; fi
(cd "$output" && sha256sum -c frozen.sha256)
for arm in original repeat; do
  dir="$output/$arm"
  "$dir/learned-retrieval" seal "$dir/run"
done
for seed in "${seeds[@]}"; do
  for mode in "${modes[@]}"; do
    cell="$mode-$seed"
    cmp "$output/original/run/$cell/model.json" "$output/repeat/run/$cell/model.json"
    diff <(jq -S 'del(.identity.binary_sha256)' "$output/original/run/$cell/receipt.json") \
         <(jq -S 'del(.identity.binary_sha256)' "$output/repeat/run/$cell/receipt.json")
  done
done
for arm in original repeat; do
  dir="$output/$arm"
  for action in score replay; do
    admit
    prlimit --as=2147483648:2147483648 -- "$dir/learned-retrieval" "$action" \
      "$workspace" "$storage" "$dir/run" > "$dir/$action.log" 2>&1
  done
done
diff -qr "$output/original/run/scores" "$output/repeat/run/scores"
echo 'All endpoints, training traces, original metrics and raw score files matched.'
