#!/usr/bin/env bash
# Preserve frozen numerical producers; create a canonical view of saved parents.
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
ulimit -c 0
ulimit -v 67108864
check_paths() {
  local saved=$1 rebuilt=$2 left right
  jq -e 'type=="object"' "$saved" > /dev/null
  jq -e 'type=="object"' "$rebuilt" > /dev/null
  # Every non-path field, including metrics, choices, hashes and work, is exact.
  cmp <(jq -S 'walk(if type=="object" then del(.model_path) else . end)' "$saved") \
      <(jq -S 'walk(if type=="object" then del(.model_path) else . end)' "$rebuilt")
  # Changed spellings must resolve to the very same existing artifact, not
  # merely to a different file with coincidentally equal metrics or bytes.
  while IFS=$'\t' read -r left right; do
    test "$(realpath -e -- "$left")" = "$(realpath -e -- "$right")"
  done < <(jq -sr '.[0] as $a | .[1] as $b | $a | paths(scalars) as $p |
      select($p[-1]=="model_path") | [getpath($p),($b|getpath($p))] | @tsv' "$saved" "$rebuilt")
}
case ${1:-} in
  check)
    test "$#" -eq 3
    check_paths "$2" "$3"
    ;;
  view)
    test "$#" -eq 4
    source_root=$(realpath -e -- "$2")
    view_root=$(realpath -m -- "$3")
    phase_binary=$(realpath -e -- "$4")
    test ! -e "$view_root"
    test -s "$source_root/summary.json"
    mkdir -- "$view_root"
    # All original artifacts remain untouched. Only the regenerated summary
    # belongs to this view; model, input and replay paths resolve to originals.
    for original in "$source_root"/*; do
      name=${original##*/}
      if [[ $name != summary.json ]]; then ln -s -- "$original" "$view_root/$name"; fi
    done
    /usr/bin/time -v -o "$view_root/recovery.resources" prlimit --as=68719476736:68719476736 -- \
      "$phase_binary" summarize "$view_root" > "$view_root/summary.pending.json"
    check_paths "$source_root/summary.json" "$view_root/summary.pending.json"
    mv -- "$view_root/summary.pending.json" "$view_root/summary.json"
    prlimit --as=68719476736:68719476736 -- "$phase_binary" summarize "$view_root" > "$view_root/summary.replay.json"
    cmp "$view_root/summary.json" "$view_root/summary.replay.json"
    sha256sum "$source_root/summary.json" "$view_root/summary.json" "$phase_binary" > "$view_root/recovery.sha256"
    ;;
  *) printf '%s\n' 'usage: recover_paths.sh check SAVED REBUILT | view ORIGINAL_PHASE FRESH_VIEW FROZEN_PHASE_BINARY' >&2; exit 2 ;;
esac
