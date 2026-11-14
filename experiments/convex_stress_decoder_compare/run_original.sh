#!/usr/bin/env bash
set -u
exe=$1
request=$2
out=$3
test ! -e "$out" || exit 1
mkdir -p "$(dirname "$out")"
/usr/bin/time -v -o "$out.prepare.resources" prlimit --as=68719476736:68719476736 -- "$exe" prepare "$out" "$request" > "$out.prepare.log" 2>&1
prepared=$?
printf '%s\n' "$prepared" > "$out.prepare.exit"
test "$prepared" -eq 0 || exit "$prepared"
while IFS= read -r task; do
  /usr/bin/time -v -o "$out/$task.run.resources" prlimit --as=68719476736:68719476736 -- "$exe" run-case "$out" "$task" > "$out/$task.log" 2>&1
  rc=$?
  printf '%s\n' "$rc" > "$out/$task.run.exit"
  if test "$rc" -eq 0; then
    /usr/bin/time -v -o "$out/$task.replay.resources" prlimit --as=68719476736:68719476736 -- "$exe" replay-case "$out" "$task" > "$out/$task.replay.json" 2> "$out/$task.replay.log"
    printf '%s\n' "$?" > "$out/$task.replay.exit"
  fi
done < <(jq -r '.tasks[]' "$out/inventory.json")
/usr/bin/time -v -o "$out/summary.resources" prlimit --as=68719476736:68719476736 -- "$exe" summarize "$out" > "$out/summary.json" 2> "$out/summary.log"
rc=$?
printf '%s\n' "$rc" > "$out/summary.exit"
exit "$rc"
