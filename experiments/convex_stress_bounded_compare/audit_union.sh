#!/usr/bin/env bash
# Read-only named-case reconciliation; generated JSON goes to stdout.
set -euo pipefail
if test "$#" -ne 7; then
  printf '%s\n' 'usage: audit_union.sh RETAINED344 ORIGINAL35 CATALOG77 HISTORICAL77 BOUNDED_REPORT BOUNDED_VERIFIED PROBES_VERIFIED' >&2
  exit 2
fi
audit_source_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
digest() { sha256sum -- "$1" | cut -d ' ' -f 1; }
jq -ne \
  --slurpfile retained "$1" --arg retained_sha256 "$(digest "$1")" \
  --slurpfile original35 "$2" --arg original35_sha256 "$(digest "$2")" \
  --slurpfile catalog77 "$3" --arg catalog77_sha256 "$(digest "$3")" \
  --slurpfile historical77 "$4" --arg historical77_sha256 "$(digest "$4")" \
  --slurpfile bounded "$5" --arg bounded_sha256 "$(digest "$5")" \
  --slurpfile verified "$6" --arg verified_sha256 "$(digest "$6")" \
  --slurpfile probes "$7" --arg probes_sha256 "$(digest "$7")" \
  -f "$audit_source_dir/audit_union.jq"
