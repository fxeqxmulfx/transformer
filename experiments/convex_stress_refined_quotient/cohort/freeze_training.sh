#!/usr/bin/env bash
set -euo pipefail
[[ $# == 5 ]] || { echo 'usage: freeze_training.sh REPOSITORY_ROOT PRODUCER_COMMIT SOURCE_ROOT SOURCE_BINARY NEW_REQUEST' >&2; exit 2; }
repository=$(realpath "$1")
producer=$2
source_root=$(realpath "$3")
source_binary=$(realpath "$4")
destination=$5
test ! -e "$destination"
[[ "$producer" =~ ^[[:xdigit:]]{40}$ ]]
for task in 241 242; do agent-kanban show "$task" | sed -n '1p' | rg -q "^#$task [^ ]+ done "; done
digest() { sha256sum "$1" | cut -d ' ' -f 1; }
jq -n --arg producer "$producer" --arg suite "$(digest "$repository/experiments/convex_stress/suite.json")" \
  --arg root "$source_root" --arg binary "$source_binary" --arg binary_hash "$(digest "$source_binary")" \
  --arg inventory "$(digest "$source_root/inventory.json")" --slurpfile source "$source_root/inventory.json" \
  '{producer_commit:$producer,suite_sha256:$suite,discovery:{root:$root,binary:$binary,
    inventory_sha256:$inventory,source_fingerprint:$source[0].source_fingerprint,
    binary_sha256:$binary_hash,producer_commit:$source[0].producer_commit}}' > "$destination"
