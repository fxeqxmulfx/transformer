#!/usr/bin/env bash
# Copy the exact source texts and historical reference, without executing them.
set -euo pipefail
if (( $# != 3 )); then
  echo 'usage: bash freeze-sources.sh TRANSFORMER_REPO DIFF_GPT_REPO FRESH_DIRECTORY' >&2
  exit 2
fi
sources=$(dirname "$0")/sources.json
mkdir "$3"
jq -r '.commits as $c | .files|to_entries[]|[.key,.value[0],$c[.value[0]],.value[1],.value[2]]|@tsv' "$sources" |
while IFS=$'\t' read -r name repo commit path expected; do
  case "$repo" in
    transformer) checkout=$1 ;;
    diff-gpt) checkout=$2 ;;
    *) exit 2 ;;
  esac
  git -C "$checkout" show "$commit:$path" > "$3/$name"
  (cd "$3" && printf '%s  %s\n' "$expected" "$name" | sha256sum -c -)
done
