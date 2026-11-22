#!/usr/bin/env bash
# Fetch the selected official eight-needle CSV at its pinned object generation.
set -euo pipefail
if [[ $# != 2 ]]; then echo 'usage: fetch.sh {upto-128k|at-1m} OUTPUT_DIR' >&2; exit 2; fi
registry=$(dirname -- "$0")/sources.tsv
while IFS=$'\t' read -r split object generation bytes sha; do
    [[ $split == "$1" ]] || continue
    mkdir -p -- "$2"
    dest=$2/$split.csv
    matches() { [[ $(wc -c < "$1") == "$bytes" && $(sha256sum < "$1") == "$sha  -" ]]; }
    if [[ -f $dest ]] && matches "$dest"; then exit 0; fi
    tmp=$(mktemp -- "$dest.XXXXXXXX")
    trap 'rm -f -- "$tmp"' EXIT
    curl --globoff --fail --silent --show-error --location --retry 3 \
        "https://storage.googleapis.com/mrcr_v2/$object?generation=$generation" --output "$tmp"
    matches "$tmp" || { echo "source digest mismatch for $split" >&2; exit 1; }
    mv -- "$tmp" "$dest"
    exit 0
done < "$registry"
echo "unknown MRCR split: $1" >&2
exit 2
