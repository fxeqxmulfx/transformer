#!/usr/bin/env bash
# Freeze the tracked source and every endpoint before the first held-out score.
set -euo pipefail
if (( $# < 4 || $# > 5 )); then
  echo 'usage: bash run.sh WORKSPACE STORAGE PRODUCER FRESH_OUTPUT [binding|transfer]' >&2
  exit 2
fi
workspace=$(realpath "$1")
storage=$(realpath "$2")
producer=$(git -C "$workspace" rev-parse --verify "$3^{commit}")
output=$(realpath -m "$4")
base=experiments/learned_retrieval
inputs=("$base" experiments/convex_bilinear_sequence experiments/convex_stress_minibatch
  experiments/convex_stress experiments/convex_binding_diagnostic/src/counts.rs
  review/flat_spectrum.md src/Transformer/Grokking/FlatSpectrum/ConvexChannel/LearnedRetrieval)
case "${5:-binding}" in
  binding) experiment=$base; binary=learned-retrieval ;;
  transfer)
    experiment=experiments/learned_retrieval_transfer
    binary=learned-retrieval-transfer
    inputs+=("$experiment" experiments/convex_stress_catalog/protocol.json
      experiments/convex_stress_sequences/src/{dataset,prefix,retrieval,wire}.rs)
    ;;
  *) echo 'unknown retrieval experiment' >&2; exit 2 ;;
esac

reserve=25769803776
admit() {
  df -B1 "$workspace" "$storage"
  test "$(df -B1 --output=avail "$workspace" | tail -1)" -gt 2147483648
  test "$(df -B1 --output=avail "$storage" | tail -1)" -gt 17179869184
  ps -e -o rss= | awk -v reserved="$reserve" '{sum+=$1*1024} END {print "aggregate RSS",sum; if(sum+reserved>110000000000)exit 1}'
}
admit
mkdir -p "$(dirname "$output")"
mkdir "$output"
git -C "$workspace" archive -o "$output/source.tar" "$producer" -- "${inputs[@]}"
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
  if [[ "$experiment" != "$base" ]]; then
    cargo test --locked --offline --manifest-path "$source_dir/$base/Cargo.toml" --target-dir "$target_dir" \
      > "$output/$arm/model-tests.log" 2>&1
  fi
  cargo test --locked --offline --manifest-path "$manifest" --target-dir "$target_dir" \
    > "$output/$arm/tests.log" 2>&1
  cargo build --release --locked --offline --manifest-path "$manifest" --target-dir "$target_dir" \
    > "$output/$arm/build.log" 2>&1
  cp "$target_dir/release/$binary" "$output/$arm/$binary"
done
(
  cd "$output"
  sha256sum source.tar producer.commit "original/$binary" "repeat/$binary" \
    "original/source/$experiment/protocol.json" "repeat/source/$experiment/protocol.json" > frozen.sha256
  sha256sum -c frozen.sha256
)
protocol="$output/original/source/$base/protocol.json"
mapfile -t seeds < <(jq -r '.seeds[]' "$protocol")
mapfile -t modes < <(jq -r '.modes[]' "$protocol")
cases=("")
if [[ "$experiment" != "$base" ]]; then
  mapfile -t cases < <(jq -r '.cases|keys[]' "$output/original/source/$experiment/protocol.json")
fi
(( ${#seeds[@]} && ${#modes[@]} && ${#cases[@]} ))
reserve=$(( 2 * ${#seeds[@]} * ${#modes[@]} * ${#cases[@]} * 2147483648 ))
admit
invoke() {
  local arm=$1 id=$2 action=$3
  shift 3
  local dest="$output/$arm/run${id:+/$id}"
  local args=()
  if [[ -n "$id" ]]; then args+=("$id"); fi
  args+=("$action")
  if [[ "$action" != seal ]]; then args+=("$workspace" "$storage"); fi
  prlimit --as=2147483648:2147483648 -- "$output/$arm/$binary" "${args[@]}" "$dest" "$@"
}
workers=()
for arm in original repeat; do
  for id in "${cases[@]}"; do
    for seed in "${seeds[@]}"; do
      for mode in "${modes[@]}"; do
        invoke "$arm" "$id" fit "$mode" "$seed" > "$output/$arm/fit-${id:+$id-}$mode-$seed.log" 2>&1 &
        workers+=("$!")
      done
    done
  done
done
failed=0
for job in "${workers[@]}"; do
  if ! wait "$job"; then failed=1; fi
done
if ((failed)); then echo 'a training cell failed; see fit logs' >&2; exit 1; fi
(cd "$output" && sha256sum -c frozen.sha256)
for id in "${cases[@]}"; do
  for arm in original repeat; do invoke "$arm" "$id" seal; done
  for seed in "${seeds[@]}"; do
    for mode in "${modes[@]}"; do
      cell="run${id:+/$id}/$mode-$seed"
      cmp "$output/original/$cell/model.json" "$output/repeat/$cell/model.json"
      diff <(jq -S 'del(.identity.binary_sha256)' "$output/original/$cell/receipt.json") \
           <(jq -S 'del(.identity.binary_sha256)' "$output/repeat/$cell/receipt.json")
    done
  done
done
for id in "${cases[@]}"; do
  for arm in original repeat; do
    for action in score replay; do
      admit
      invoke "$arm" "$id" "$action" > "$output/$arm/${id:+$id-}$action.log" 2>&1
    done
  done
  dest="run${id:+/$id}/scores"
  diff -qr "$output/original/$dest" "$output/repeat/$dest"
done
echo 'All endpoints, training traces, original metrics and raw score files matched.'
