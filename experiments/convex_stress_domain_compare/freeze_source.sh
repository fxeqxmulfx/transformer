#!/usr/bin/env bash
set -euo pipefail
[[ $# == 3 ]] || { echo 'usage: freeze_source.sh REPOSITORY PRODUCER_COMMIT NEW_DIRECTORY' >&2; exit 2; }
repository=$(realpath "$1")
producer=$2
destination=$(realpath -m "$3")
[[ "$producer" =~ ^[[:xdigit:]]{40}$ ]]
test ! -e "$destination"
test "$(git -C "$repository" rev-parse HEAD)" = "$producer"
test -z "$(git -C "$repository" status --porcelain)"
test -n "${CARGO_TARGET_DIR:-}"
export CARGO_BUILD_JOBS=2 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$destination"
cd "$repository"
manifest=experiments/convex_stress_domain_compare/Cargo.toml
printf '%s\n' "$producer" > "$destination/producer.commit"
prlimit --as=68719476736:68719476736 -- cargo metadata --offline --locked \
  --manifest-path "$manifest" --format-version 1 > "$destination/metadata.json"
{
  while IFS= read -r manifest_path; do
    package=$(realpath --relative-to="$repository" "$(dirname "$manifest_path")")
    [[ "$package" == experiments/* && "$package" != *..* ]]
    git ls-files -- "$package"
  done < <(jq -r '.packages[]|select(.source==null)|.manifest_path' "$destination/metadata.json")
  # The minibatch dependency borrows original modules outside Cargo's graph.
  git ls-files -- experiments/convex_stress
} | LC_ALL=C sort -u > "$destination/source.files"
test -s "$destination/source.files"
while IFS= read -r file; do
  test -f "$file"
  sha256sum "$file"
done < "$destination/source.files" > "$destination/source-inventory.sha256"
sha256sum "$destination/source-inventory.sha256" > "$destination/inventory.sha256"
tar -czf "$destination/source.tar.gz" -T "$destination/source.files"
sha256sum "$destination/source.tar.gz" > "$destination/archive.sha256"
prlimit --as=68719476736:68719476736 -- cargo test --offline --locked --no-run \
  --manifest-path "$manifest" --message-format=json > "$destination/build.jsonl" 2> "$destination/build.log"
jq -r 'select(.reason=="compiler-artifact" and .profile.test==true and .target.name=="convex_stress_domain_compare" and .executable!=null)|.executable' \
  "$destination/build.jsonl" > "$destination/test-executable.path"
test "$(wc -l < "$destination/test-executable.path")" = 1
cp "$(cat "$destination/test-executable.path")" "$destination/domain-compare-tests-v1"
sha256sum "$destination/domain-compare-tests-v1" > "$destination/test-binary.sha256"
prlimit --as=68719476736:68719476736 -- cargo build --offline --locked --release \
  --manifest-path "$manifest" > "$destination/release.log" 2>&1
cp "$CARGO_TARGET_DIR/release/convex-stress-domain-compare" "$destination/domain-compare-v1"
sha256sum "$destination/domain-compare-v1" > "$destination/binary.sha256"
prlimit --as=68719476736:68719476736 -- "$destination/domain-compare-v1" fingerprint > "$destination/source.fingerprint"
cp experiments/convex_stress_domain_compare/{protocol.json,freeze_original.sh,run_original.sh} "$destination/"
sha256sum "$destination/"{protocol.json,freeze_original.sh,run_original.sh} > "$destination/protocol-scripts.sha256"
rustc --version > "$destination/rustc.version"
cargo --version > "$destination/cargo.version"
test "$(git rev-parse HEAD)" = "$producer"
test -z "$(git status --porcelain)"
