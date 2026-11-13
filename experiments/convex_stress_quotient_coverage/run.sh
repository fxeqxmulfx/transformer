#!/usr/bin/env bash
# Configuration-only coverage test of the reviewed native CM30 component.
set -euo pipefail
[[ $# == 4 && $4 =~ ^[[:xdigit:]]{40}$ ]] || {
  echo 'usage: run.sh SOURCE_CONFIGURATION FROZEN_BINARY FRESH_DESTINATION COMMIT' >&2
  exit 2
}
source_configuration=$(realpath "$1")
binary=$(realpath "$2")
destination=$(realpath -m "$3")
producer=$4
protocol=$(realpath "$(dirname "$0")/protocol.json")
test ! -e "$destination"
test -d "$(dirname "$destination")"
test "$(sha256sum "$source_configuration" | cut -d ' ' -f 1)" = \
  "$(jq -r '.source_configuration_sha256' "$protocol")"
test "$(sha256sum "$binary" | cut -d ' ' -f 1)" = \
  "$(jq -r '.binary_sha256' "$protocol")"
limit=$(jq -er '.memory_limit_bytes' "$protocol")
test "$(prlimit --as="$limit:$limit" -- "$binary" fingerprint)" = \
  "$(jq -r '.source_fingerprint' "$protocol")"
reserve() {
  local available required
  available=$(df -Pk "$(dirname "$destination")" | awk 'NR==2 {print $4}')
  required=$(jq -er '.minimum_disk_reserve_kib' "$protocol")
  test "$available" -ge "$required"
}
reserve
mkdir "$destination"
trap 'printf "%s\n" "$?" > "$destination/process.exit"' EXIT
cp "$protocol" "$destination/protocol.json"
cp "$0" "$destination/runner.sh"
cp "$source_configuration" "$destination/source-configuration.json"
old_limit=$(jq -er '.old_equation_limit' "$protocol")
new_limit=$(jq -er '.new_equation_limit' "$protocol")
jq -e --argjson old "$old_limit" '.input.policy.equations == $old' \
  "$source_configuration" > "$destination/source-check.json"
jq --argjson limit "$new_limit" '.input.policy.equations = $limit' \
  "$source_configuration" > "$destination/configuration.json"
jq -e --slurpfile source "$source_configuration" \
  'del(.input.policy.equations) == ($source[0] | del(.input.policy.equations))' \
  "$destination/configuration.json" > "$destination/unchanged-fields.json"
jq '.input as $i | $i.observations as $all |
  $all[:$i.policy.equations] as $prefix |
  {observations:($all|length),equation_limit:$i.policy.equations,
   prefix_missing:{
    rows:(([$all[][0]%$i.n]|unique)-([$prefix[][0]%$i.n]|unique)),
    columns:(([$all[][0]/$i.n|floor]|unique)-([$prefix[][0]/$i.n|floor]|unique)),
    classes:(([$all[][1]]|unique)-([$prefix[][1]]|unique))}}' \
  "$source_configuration" > "$destination/prefix-coverage.json"
printf '%s\n' "$producer" > "$destination/producer.commit"
sha256sum "$binary" "$destination/protocol.json" "$destination/runner.sh" \
  "$destination/source-configuration.json" "$destination/configuration.json" \
  > "$destination/frozen.sha256"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
ulimit -c 0
status=0
/usr/bin/time -v -o "$destination/run.resources" \
  prlimit --as="$limit:$limit" -- "$binary" run "$destination/configuration.json" \
  "$destination/catalogue" "$producer" > "$destination/run.log" 2>&1 || status=$?
printf '%s\n' "$status" > "$destination/run.exit"
test "$status" = 0
reserve
status=0
/usr/bin/time -v -o "$destination/replay.resources" \
  prlimit --as="$limit:$limit" -- "$binary" replay "$destination/catalogue" \
  > "$destination/replay.json" 2> "$destination/replay.log" || status=$?
printf '%s\n' "$status" > "$destination/replay.exit"
test "$status" = 0
sha256sum -c "$destination/frozen.sha256" > "$destination/frozen-check.log"
