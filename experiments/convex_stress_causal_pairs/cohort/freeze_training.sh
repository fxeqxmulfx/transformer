#!/usr/bin/env bash
set -euo pipefail
[[ $# == 3 ]] || { echo 'usage: freeze_training.sh REPOSITORY_ROOT PRODUCER_COMMIT NEW_REQUEST' >&2; exit 2; }
repository=$(realpath "$1")
producer=$2
destination=$3
test ! -e "$destination"
[[ "$producer" =~ ^[[:xdigit:]]{40}$ ]]
for task in 233 235 236; do agent-kanban show "$task" | sed -n '1p' | rg -q "^#$task [^ ]+ done "; done
digest() { sha256sum "$1" | cut -d ' ' -f 1; }
jq -n --arg producer "$producer" --arg suite "$(digest "$repository/experiments/convex_stress/suite.json")" \
  '{producer_commit:$producer,suite_sha256:$suite}' > "$destination"
