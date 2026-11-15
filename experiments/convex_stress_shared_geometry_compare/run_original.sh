#!/usr/bin/env bash
# Run this immutable frozen driver. Attempt every original case and retain failures.
set -euo pipefail
test "$#" -eq 3
binary=$(realpath "$1")
root=$2
request=$3
test ! -e "$root"
mkdir -p "$(dirname "$root")"
/usr/bin/time -v -o "$root.prepare.resources" prlimit --as=68719476736:68719476736 \
    -- "$binary" prepare "$root" "$request" > "$root.prepare.log" 2>&1
status=0
while IFS= read -r task; do
    set +e
    /usr/bin/time -v -o "$root/$task.run.resources" prlimit --as=68719476736:68719476736 \
        -- "$binary" run-case "$root" "$task" > "$root/$task.run.log" 2>&1
    result=$?
    set -e
    printf '%s\n' "$result" > "$root/$task.run.exit"
    if [[ "$result" -eq 0 ]]; then
        set +e
        /usr/bin/time -v -o "$root/$task.replay.resources" prlimit --as=68719476736:68719476736 \
            -- "$binary" replay-case "$root" "$task" > "$root/$task.replay.json" 2> "$root/$task.replay.log"
        result=$?
        set -e
        printf '%s\n' "$result" > "$root/$task.replay.exit"
    else
        printf 'not_run\n' > "$root/$task.replay.exit"
    fi
    if [[ "$result" -ne 0 ]]; then status=1; fi
done < <(jq -r '.cases[].task' "$root/inventory.json")
set +e
/usr/bin/time -v -o "$root/summary.resources" prlimit --as=68719476736:68719476736 \
    -- "$binary" summarize "$root" > "$root/summary.json" 2> "$root/summary.log"
result=$?
set -e
printf '%s\n' "$result" > "$root/summary.exit"
if [[ "$result" -ne 0 ]]; then status=1; fi
printf '%s\n' "$status" > "$root/process.exit"
exit "$status"
