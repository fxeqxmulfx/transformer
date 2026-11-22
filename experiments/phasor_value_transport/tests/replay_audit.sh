#!/usr/bin/env bash
# Exercise the complete and partial cohort audits without training checkpoints.
set -eu
dir=$(cd -- "$(dirname -- "$0")/.." && pwd)
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
cd "$tmp"
jq -n '{model:"model",tasks:[{case:"case",seeds:[range(3) |
  {seed:.,status:"pass",step:(.+1)*10,correct:99,total:100,
   weights_sha256:("a"*64),timing_environment:{kernite_version:"0.6.0"}}]}]}' > report
jq -c '.model as $model | .tasks[] as $t | $t.seeds[] |
   {case:$t.case,seed,correct,total,model:$model,weights_sha256,kernite_version:"0.6.0",
    scored_with_kernite:"0.6.0",observation:{case:$t.case,seed,step,correct,total,pass:true}}' report > all
audit() { jq -es "$@" --slurpfile report report -f "$dir/verify_replays.jq" records; }
reject() { if audit "$@" >/dev/null 2>&1; then echo 'accepted invalid replay cohort' >&2; exit 1; fi; }
cp all records
audit | jq -e '.verified==3 and (has("pending")|not)' >/dev/null
jq '(.tasks[0].seeds[1].status)="pending" | (.tasks[0].seeds[2].status)="budget miss"' report > partial
mv partial report
head -1 all > records
reject
audit --argjson passing_only true | jq -e '.verified==1 and .pending==1 and .budget_misses==1' >/dev/null
valid=$(cat records)
for expression in '.observation.step+=1' '.observation.correct-=1' '.observation.pass=false' '.weights_sha256="b"*64'; do
    jq -c "$expression" <<< "$valid" > records
    reject --argjson passing_only true
done
cp all records
reject --argjson passing_only true
: > records
reject --argjson passing_only true
jq '(.tasks[0].seeds[0].status)="pending"' report > pending
mv pending report
audit --argjson passing_only true | jq -e '.verified==0 and .pending==2 and .budget_misses==1' >/dev/null
reject --arg passing_only yes
