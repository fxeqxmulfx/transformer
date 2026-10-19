# transformer

Lean in `src/` formalizes the manuscripts in `papers/`; the lab in
[`python/`](python/README.md) trains the experiments in
[`experiments/`](experiments/README.md). `./run.py` runs every task;
`./run.py -h` lists them.

## Requirements

- [uv](https://docs.astral.sh/uv/). It installs Python 3.14 and the locked
  packages, among them torch 2.14.1 built for CUDA 12.6, which also runs
  without a GPU.
- A C++ compiler (g++). TorchInductor compiles the kernels of a CPU run
  with it.
- A GPU, only for runs that replay CUDA graphs.
- For the Lean tree, elan. It installs the toolchain that `lean-toolchain`
  names.

Before committing, `./run.py verify` builds Lean, audits axioms, regenerates
the index, checks forbidden constructs, and runs the Python tests. Strict
lint is optional following the user's instruction: `./run.py lint` runs
`lake --wfail lint --builtin-lint --linters=linter.all --code-quality` manually.
`./run.py kernel` is an optional kernel replay; the 39 existing unfinished
statements are retained and are not claimed to pass that checker.

## The minimal run

```sh
./run.py setup                                          # install the locked environment
./run.py run experiments/basis easy-small-depth-seed0   # train one run of the benchmark
./run.py report experiments/basis easy-small-depth-seed0
```

The run trains the two-layer model of width 64 on one core until it passes
E_2, in about 20 s. The first time, compiling its kernels adds 40 to 80 s;
TorchInductor then keeps them in its cache on disk. The run writes into
`experiments/basis/runs/easy-small-depth-seed0/`. `report` prints what the
run recorded as JSON, and its `stop` says whether it passed or reached its
budget.

## The benchmark

A run of the basis is labeled `<mode>-<model>-<task>-seed<seed>`. The mode
names the tasks and the model names the network: `easy-large-recall-seed0`
is the large model on the easy recall. Both models are GPTMini with four
heads: `small` has width 64 and two layers, `large` width 128 and six. The
two modes:

| Task | easy | hard |
| --- | --- | --- |
| depth | E_2, judged at the trained lengths, 32 to 64 | E_4, judged at length 128 |
| recall | 8 keys, each bound once | 16 writes to 8 keys, each key queried for its latest value |
| parity | 1 to 16 bits | the same |

The easy mode is set for the small model and the hard mode for the large
one. At bd63e50, from seeds 0, 1 and 2:

| | small | large |
| --- | --- | --- |
| easy | passes every task from every seed | passes depth and parity; recall from seed 2 alone |
| hard | fails depth and recall from every seed, as intended; passes parity | passes every task from every seed |

The smallest complete version takes one mode on one model from one seed.
Its three runs train side by side in about four minutes:

```sh
./run.py run experiments/basis easy-small-{depth,recall,parity}-seed0
```

All 15 runs of the small model, both modes from three seeds, train side by
side on 15 cores in about nine minutes:

```sh
./run.py run experiments/basis easy-small-{depth,recall,parity}-seed{0,1,2} \
                                hard-small-{depth,recall}-seed{0,1,2}
```

These runs screen an idea before the large model takes it. In the easy
mode, where the small model passes everything, a failure counts against an
idea. In the hard mode, where it fails depth and recall, a pass counts for
it. The recipes were set for GPTMini, so before dropping an idea, try a
rate or two around its recipe. Compare seed by seed: on the small model a
seed moved a pass by up to twice the updates, and on the large model's hard
depth by seven times (800 against 5,800).

A model of its own takes the benchmark in an experiment file whose runs are
`basis(model, mode, seed)`. Labeled as here, the same commands train it.

The whole benchmark has 30 runs: each mode on both models, from seeds 0, 1
and 2. A hard parity run is its easy one, so the hard mode lists only depth
and recall:

```sh
./run.py run experiments/basis easy-{small,large}-{depth,recall,parity}-seed{0,1,2} \
                                hard-{small,large}-{depth,recall}-seed{0,1,2}
```

Each run trains on the physical cores it asks for: four for the large
model's recall, two for its depth and parity, one for every task of the
small model. On 28 cores the 30 runs took 19 min 52 s. Without labels, `run`
trains all 180 runs of `experiments/basis`, including the 150 GPU runs that
set the recipes. `./run.py check experiments/basis` lists every label.
[`experiments/basis/README.md`](experiments/basis/README.md) says what the
benchmark asks and what it found.

## Repeating a recorded result

A result holds at the commit that recorded it. An experiment's README names
that commit; for the basis, it is the table under Found. Each run also
records its commit and the hashes of the lab's sources in
`runs/<label>/experiment.json`. To repeat a result, check its commit out in
a fresh clone, where `runs/` is empty, and train the same labels:

```sh
git checkout bd63e50   # the CPU benchmark of the basis
./make.py run experiments/basis <labels>
```

That historical revision predates the `run.py` rename.

A run repeats its own losses on the same machine, with the same build of
torch and the same number of threads. A different number of threads rounds
differently.
