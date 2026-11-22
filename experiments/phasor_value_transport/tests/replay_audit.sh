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

# The optional NV11 gate binds all exact scores to the cohort by digest.
jq -n '{model:"model",tasks:[{case:"arithmetic-cubic-product",tier:"NV11",seeds:[range(3) |
  {seed:.,status:"pass",step:250,correct:4705,total:4705,
   weights_sha256:(tostring*64),timing_environment:{kernite_version:"0.6.0"}}]}]}' > report
jq -c '.model as $m | .tasks[] as $t | $t.seeds[] |
  {case:$t.case,seed,correct,total,model:$m,weights_sha256,kernite_version:"0.6.0",
   scored_with_kernite:"0.6.0",observation:{case:$t.case,seed,step,correct,total,pass:true}}' report > records
jq -sc 'reverse[] | {scored_with_model:.model,weights_sha256,scored_with_kernite,
  training:{correct:4704,examples:4704},held_out:{correct:4705,examples:4705}}' records > exact
audit --slurpfile nv11 exact | jq -e '.verified==3 and .exact_nv11_verified==3' >/dev/null
changes=('.[0].scored_with_model="other"' '.[0].scored_with_kernite="other"'
         '.[0].training.correct=4703' '.[0].held_out.correct=4704'
         '.[0].weights_sha256=.[1].weights_sha256' 'del(.[0])')
for change in "${changes[@]}"; do
    jq -sc "$change | .[]" exact > invalid
    reject --slurpfile nv11 invalid
done
