#!/usr/bin/env bash
# New CM18 runs: the user's 2026-11-12 limit is 64 GiB, including allocator arenas.
set -euo pipefail
if [[ "$#" == 0 ]]; then
  echo 'usage: limited.sh COMMAND [ARGS...]' >&2
  exit 2
fi
ulimit -c 0
exec prlimit --as=68719476736:68719476736 -- "$@"
