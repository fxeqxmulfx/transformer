#!/usr/bin/env bash
# User's 2026-11-12 limit for every new native process, including library buffers.
set -euo pipefail
if [[ $# == 0 ]]; then echo 'usage: limited.sh COMMAND [ARGS...]' >&2; exit 2; fi
ulimit -c 0
exec prlimit --as=68719476736:68719476736 -- "$@"
