#!/usr/bin/env bash
# Original and repeat both retain all six outcomes; refusals stay nonzero exits.
set -euo pipefail
test "$#" -eq 5
freeze=$(realpath "$1")
binary=$(realpath "$2")
workspace=$(realpath "$3")
output=$4
helper=$(realpath "$5")
test ! -e "$output"
test ! -e "$output.runtime-before.log"
test ! -e "$output.source-before.log"
(
  cd "$freeze/source"
  sha256sum -c ../source.sha256
) > "$output.source-before.log"
(
  cd "$freeze/native-runtime"
  sha256sum -c ../runtime.sha256
) > "$output.runtime-before.log"
sha256sum "$binary" "$helper" > "$output.executables.sha256"
export LD_LIBRARY_PATH="$freeze/native-runtime/lib"
export ATEN_CPU_CAPABILITY=avx2
producer=$(cat "$freeze/producer.commit")
set +e
prlimit --as=8589934592:8589934592 -- "$binary" prepare "$workspace" "$output" "$helper" "$producer"
status=$?
set -e
sha256sum -c "$output.executables.sha256" > "$output.executables-after.log"
(
  cd "$freeze/native-runtime"
  sha256sum -c ../runtime.sha256
) > "$output.runtime-after.log"
printf '%s\n' "$status" > "$output.exit"
exit "$status"
