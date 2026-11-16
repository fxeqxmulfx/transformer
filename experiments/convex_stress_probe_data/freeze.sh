#!/usr/bin/env bash
# Freeze the native source closure and the actual linked CPU runtime, without Python.
set -euo pipefail
test "$#" -eq 4
freeze=$1
binary=$2
helper=$3
torch_root=$(realpath "$4")
test ! -e "$freeze"
test -x "$binary"
test -x "$helper"
test -z "$(git status --porcelain)"
work=$(git rev-parse --show-toplevel)
cd "$work"
for disk in "$work" "$(dirname "$freeze")"; do
  available=$(df -B1 --output=avail -- "$disk" | tail -1)
  test "$available" -ge 4294967296
done
mkdir -p "$freeze/source" "$freeze/native-runtime/lib"
git rev-parse HEAD > "$freeze/producer.commit"
cargo metadata --offline --locked --manifest-path experiments/convex_stress_probe_data/Cargo.toml \
  --format-version 1 > "$freeze/cargo-metadata.json"
jq -r '.packages[] | select(.source == null) | .manifest_path | sub("/Cargo.toml$"; "")' \
  "$freeze/cargo-metadata.json" > "$freeze/package-paths.txt"
while IFS= read -r package; do
  git ls-files -- "${package#"$work/"}"
done < "$freeze/package-paths.txt" > "$freeze/unsorted-paths.txt"
git ls-files -- AGENTS.md EXPERIMENT_PLAN.md experiments/README.md review/flat_spectrum.md \
  experiments/convex_stress/suite.json \
  python/src/lab/infrastructure/benchmarks/synthetic/{generation.py,training.py,records.py,rows.py} \
  >> "$freeze/unsorted-paths.txt"
read -r -a dependencies < "$binary.d"
for dependency in "${dependencies[@]:1}"; do
  included=$(realpath -e "$dependency")
  if [[ "$included" == "$work/"* && -f "$included" ]]; then
    git ls-files --error-unmatch "${included#"$work/"}" >> "$freeze/unsorted-paths.txt"
  fi
done
sort -u "$freeze/unsorted-paths.txt" > "$freeze/source-paths.txt"
while IFS= read -r source; do
  mkdir -p "$freeze/source/$(dirname "$source")"
  cp "$source" "$freeze/source/$source"
  sha256sum "$source"
done < "$freeze/source-paths.txt" > "$freeze/source.sha256"
cp -a "$torch_root/include" "$freeze/native-runtime/"
ldd "$helper" > "$freeze/linked-libraries.txt"
if rg -i 'libpython|torch_python|not found' "$freeze/linked-libraries.txt"; then exit 1; fi
awk '/=> \// {print $3} /^[[:space:]]*\// {print $1}' "$freeze/linked-libraries.txt" > "$freeze/runtime-paths.txt"
while IFS= read -r library; do
  cp -L "$library" "$freeze/native-runtime/lib/$(basename "$library")"
done < "$freeze/runtime-paths.txt"
(
  cd "$freeze/native-runtime"
  rg --files -uuu include lib | sort > ../runtime-files.txt
  while IFS= read -r source; do sha256sum "$source"; done < ../runtime-files.txt
) > "$freeze/runtime.sha256"
cp "$binary" "$freeze/convex-stress-probe-data"
cp "$helper" "$freeze/probe-reference"
sha256sum "$freeze/convex-stress-probe-data" "$freeze/probe-reference" > "$freeze/binary.sha256"
rustc --version > "$freeze/rustc.version"
cargo --version > "$freeze/cargo.version"
g++ --version > "$freeze/compiler.version"
(
  cd "$freeze/source"
  sha256sum -c ../source.sha256 > ../source-check.log
)
chmod -R a-w "$freeze/source" "$freeze/native-runtime"
chmod a-w "$freeze/convex-stress-probe-data" "$freeze/probe-reference"
