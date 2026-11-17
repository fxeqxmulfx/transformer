#!/usr/bin/env bash
# Replay the original six sparse-model training trajectories in two fresh builds.
set -euo pipefail
(( $# == 5 )) || { echo 'usage: bash run.sh WORKSPACE STORAGE PRODUCER SPARSE_OUTPUT FRESH_OUTPUT' >&2; exit 2; }
workspace=$(realpath "$1")
storage=$(realpath "$2")
producer=$(git -C "$workspace" rev-parse --verify "$3^{commit}")
baseline=$(realpath "$4")
output=$(realpath -m "$5")
experiment=experiments/composed_retrieval
test "$(df -B1 --output=avail "$workspace" | tail -1)" -gt 2147483648
test "$(df -B1 --output=avail "$storage" | tail -1)" -gt 17179869184
ps -e -o rss= | awk '{sum+=$1*1024} END {if(sum+12*2147483648>110000000000)exit 1}'
mkdir -p "$(dirname "$output")"
mkdir "$output"
git -C "$workspace" archive -o "$output/source.tar" "$producer" -- "$experiment" \
  experiments/{learned_retrieval,learned_retrieval_transfer,convex_bilinear_sequence,convex_stress_minibatch,convex_stress,convex_stress_catalog,convex_stress_arithmetic_data} \
  experiments/convex_binding_diagnostic/src/counts.rs experiments/convex_stress_sequences/src/{dataset,prefix,retrieval,wire}.rs
printf '%s\n' "$producer" >"$output/producer.commit"
for arm in original repeat; do
  src="$output/$arm/source"
  target="$output/$arm/target"
  mkdir -p "$src"
  tar -xf "$output/source.tar" -C "$src"
  export RUSTFLAGS="--remap-path-prefix=$src=/depth-source --remap-path-prefix=$target=/depth-target" CARGO_BUILD_JOBS=4
  flags=(--locked --offline --manifest-path "$src/$experiment/Cargo.toml" --target-dir "$target" --bin depth-signals)
  cargo test "${flags[@]}" >"$output/$arm/tests.log" 2>&1
  cargo build --release "${flags[@]}" >"$output/$arm/build.log" 2>&1
  cp "$target/release/depth-signals" "$output/$arm/depth-signals"
done
(
  cd "$output"
  sha256sum producer.commit source.tar {original,repeat}/depth-signals >frozen.sha256
  sha256sum -c frozen.sha256
)
export workspace storage baseline output
for mode in mini forward; do for seed in 0 1 2; do for arm in original repeat; do
  printf '%s\0' "$arm" "$mode" "$seed"
done; done; done | xargs -0 -n3 -P12 bash -c '
  dir="$output/$1"
  prlimit --as=2147483648:2147483648 -- "$dir/depth-signals" "$workspace" "$storage" "$baseline" "$1" \
    "$dir/results/$2-$3" "$2" "$3" >"$dir/$2-$3.log" 2>&1' _
diff -qr "$output/original/results" "$output/repeat/results"
echo 'Every executed estimate, original training receipt and model matched on all six trajectories in two fresh builds.'
