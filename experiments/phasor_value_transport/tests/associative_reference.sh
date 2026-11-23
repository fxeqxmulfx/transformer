#!/usr/bin/env bash
set -euo pipefail

# Use NumPy's standalone C PCG64 and distributions. NUMPY_SOURCE points
# to the numpy directory inside the source distribution.
: "${NUMPY_SOURCE:?set NUMPY_SOURCE to the NumPy source numpy directory}"
: "${NUMPY_RANDOM_ARCHIVE:?set NUMPY_RANDOM_ARCHIVE to libnpyrandom.a}"
cd "$(dirname "$0")/.."
tmp=$(mktemp -d)
trap 'rm -rf -- "$tmp"' EXIT
pcg="$NUMPY_SOURCE/random/src/pcg64"
cc -std=c11 -O2 -c "$pcg/pcg64.c" -o "$tmp/pcg.o"
c++ -std=c++20 -O2 -Wall -Wextra -Werror -I"$pcg" -I"$NUMPY_SOURCE/_core/include" \
    tests/reference/associative_numpy.cpp "$tmp/pcg.o" "$NUMPY_RANDOM_ARCHIVE" -lm -o "$tmp/source"
for config in '2 0 0' '2 0 1' '8 0 0' '8 4 0'; do
    read -r loads extra random <<< "$config"
    for split in 0 1; do
        "$tmp/source" "$loads" "$extra" "$split" "$random" > "$tmp/frames-$loads-$extra-$split-$random.i16"
    done
done
c++ -std=c++20 -O2 -Wall -Wextra -Werror -I"$pcg" -I"$NUMPY_SOURCE/_core/include" \
    tests/reference/register_program_numpy.cpp "$tmp/pcg.o" "$NUMPY_RANDOM_ARCHIVE" -lm -o "$tmp/programs"
for steps in 2 4; do
    for split in 0 1; do
        "$tmp/programs" "$steps" "$split" > "$tmp/program-$steps-$split.i16"
    done
done
PHASOR_EQUATION_REFERENCE="$tmp" cargo +stable test --locked --release --offline \
    --test associative_equations -- --include-ignored
