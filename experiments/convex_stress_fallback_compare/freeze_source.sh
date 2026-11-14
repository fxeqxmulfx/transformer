#!/usr/bin/env bash
set -euo pipefail
[[ $# == 3 ]] || { echo 'usage: freeze_source.sh REPOSITORY PRODUCER_COMMIT NEW_DIRECTORY' >&2; exit 2; }
repository=$(realpath "$1")
producer=$2
destination=$(realpath -m "$3")
[[ "$producer" =~ ^[[:xdigit:]]{40}$ ]]
test ! -e "$destination"
test "$(git -C "$repository" rev-parse HEAD)" = "$producer"
test -z "$(git -C "$repository" status --porcelain)"
mkdir -p "$destination"
cd "$repository"
printf '%s\n' "$producer" > "$destination/producer.commit"
prlimit --as=68719476736:68719476736 -- cargo metadata --offline --locked \
  --manifest-path experiments/convex_stress_fallback_compare/Cargo.toml --format-version 1 > "$destination/metadata.json"
{
  while IFS= read -r manifest; do
    package=$(realpath --relative-to="$repository" "$(dirname "$manifest")")
    [[ "$package" == experiments/* && "$package" != *..* ]]
    git ls-files -- "$package"
  done < <(jq -r '.packages[]|select(.source==null)|.manifest_path' "$destination/metadata.json")
  # The minibatch dependency borrows modules with #[path] from this original
  # package. These files are outside Cargo's declared package dependency graph.
  git ls-files -- experiments/convex_stress
} | LC_ALL=C sort -u > "$destination/source.files"
test -s "$destination/source.files"
while IFS= read -r file; do
  test -f "$file"
  sha256sum "$file"
done < "$destination/source.files" > "$destination/source-inventory.sha256"
sha256sum "$destination/source-inventory.sha256" > "$destination/inventory.sha256"
tar -czf "$destination/source.tar.gz" -T "$destination/source.files"
sha256sum "$destination/source.tar.gz" > "$destination/archive.sha256"
rustc --version > "$destination/rustc.version"
cargo --version > "$destination/cargo.version"
test "$(git rev-parse HEAD)" = "$producer"
test -z "$(git status --porcelain)"
