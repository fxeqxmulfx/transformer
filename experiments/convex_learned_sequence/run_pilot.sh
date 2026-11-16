#!/usr/bin/env bash
# Run only from a committed, frozen producer and pinned inputs. Output must
# not exist; failures are retained and are never overwritten by a restart.
set -euo pipefail
if test "$#" -ne 5; then
  printf '%s\n' 'usage: run_pilot.sh BINARY FROZEN_SOURCE OUT REVISION SOURCE_SHA' >&2
  exit 2
fi
task_binary=$1
task_source=$2
task_out=$3
task_revision=$4
task_source_sha=$5
test ! -e "$task_out"
test ! -e "$task_out.exit"
trap 'printf "%s\n" "$?" > "$task_out.exit"' EXIT
prlimit --as=4294967296:4294967296 -- "$task_binary" train \
  "$task_source/experiments/convex_learned_sequence/fixtures/pilot_train.json" \
  "$task_out" "$task_revision" "$task_source_sha"
prlimit --as=4294967296:4294967296 -- "$task_binary" score "$task_out" \
  "$task_source/experiments/convex_learned_sequence/fixtures/pilot_eval.json"
