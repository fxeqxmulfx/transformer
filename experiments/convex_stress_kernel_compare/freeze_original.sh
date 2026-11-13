#!/usr/bin/env bash
set -euo pipefail
[[ $# == 9 ]] || { echo 'usage: freeze_original.sh REPO PRODUCER PARENT_ROOT PARENT_BINARY PARENT_REVIEW STAGE_ROOT STAGE_BINARY STAGE_REVIEW NEW_REQUEST' >&2; exit 2; }
repository=$(realpath "$1")
producer=$2
parent_root=$(realpath "$3")
parent_binary=$(realpath "$4")
parent_review=$(realpath "$5")
stage_root=$(realpath "$6")
stage_binary=$(realpath "$7")
stage_review=$(realpath "$8")
destination=$9
test ! -e "$destination"
[[ "$producer" =~ ^[[:xdigit:]]{40}$ ]]
for task in 232 234 237 243 244 245 246 250 251 252 253; do agent-kanban show "$task" | sed -n '1p' | rg -q "^#$task [^ ]+ done "; done
digest() { sha256sum "$1" | cut -d ' ' -f 1; }
jq -n --arg producer "$producer" --arg suite "$(digest "$repository/experiments/convex_stress/suite.json")" \
  --arg pr "$parent_root" --arg pb "$parent_binary" --arg pv "$parent_review" \
  --arg ps "$(digest "$parent_root/summary.json")" --arg pi "$(digest "$parent_root/inventory.json")" \
  --arg pe "$(digest "$parent_review/process.exit")" --arg pc "$(digest "$parent_review/reviewed.commit")" \
  --arg sr "$stage_root" --arg sb "$stage_binary" --arg sv "$stage_review" \
  --arg ss "$(digest "$stage_root/summary.json")" --arg si "$(digest "$stage_root/inventory.json")" \
  --arg se "$(digest "$stage_review/process.exit")" --arg sc "$(digest "$stage_review/reviewed.commit")" \
  'def pin($root;$binary;$summary;$inventory;$review;$exit_hash;$commit_hash): {root:$root,binary:$binary,summary_sha256:$summary,inventory_sha256:$inventory,review:{exit:($review+"/process.exit"),exit_sha256:$exit_hash,commit:($review+"/reviewed.commit"),commit_sha256:$commit_hash,summary:($review+"/summary.json")}};
   {producer_commit:$producer,suite_sha256:$suite,parent:pin($pr;$pb;$ps;$pi;$pv;$pe;$pc),staging:pin($sr;$sb;$ss;$si;$sv;$se;$sc)}' > "$destination"
