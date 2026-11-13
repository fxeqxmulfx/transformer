#!/usr/bin/env bash
set -euo pipefail
source_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
temporary=$(mktemp -d)
trap 'rm -rf -- "$temporary"' EXIT
cd "$temporary"
printf '%s\n' 'retained model' > model.json
ln -s model.json alias.json
jq -n '{selected:{model_path:"model.json",score:0.25},cost:4}' > a.json
jq -n --arg path "$temporary/alias.json" '{selected:{model_path:$path,score:0.25},cost:4}' > b.json
bash "$source_dir/recover_paths.sh" check a.json b.json
jq '.selected.score=0.5' b.json > wrong.json
if bash "$source_dir/recover_paths.sh" check a.json wrong.json > /dev/null 2>&1; then exit 1; fi
jq '.cost=5' b.json > wrong.json
if bash "$source_dir/recover_paths.sh" check a.json wrong.json > /dev/null 2>&1; then exit 1; fi
cp model.json different.json
jq '.selected.model_path="different.json"' b.json > wrong.json
if bash "$source_dir/recover_paths.sh" check a.json wrong.json > /dev/null 2>&1; then exit 1; fi
jq '.selected.model_path="missing.json"' b.json > wrong.json
if bash "$source_dir/recover_paths.sh" check a.json wrong.json > /dev/null 2>&1; then exit 1; fi
printf '%s\n' 'Canonical aliases accepted; changed score/work, different same-byte files and missing files rejected.'
if bash "$source_dir/recover_paths.sh" check absent.json absent.json > /dev/null 2>&1; then exit 1; fi
