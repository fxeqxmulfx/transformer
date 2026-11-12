#!/usr/bin/env bash
# Bound native address space, including allocator arenas; keep core dumps off.
set -euo pipefail
gib=${1:?usage: limited.sh MEMORY_GIB COMMAND [ARGS...]}
shift
if [[ ! "$gib" =~ ^([1-9]|1[0-6])$ ]] || [ "$#" = 0 ]; then
  echo 'Native jobs require a 1-16 GiB limit and a command.' >&2
  exit 2
fi
ulimit -c 0
bytes=$((gib * 1024 * 1024 * 1024))
exec prlimit --as="$bytes:$bytes" -- "$@"
