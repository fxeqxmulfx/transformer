# transformer

Lean in `src/` formalizes the manuscripts in `papers/`; the lab in
[`python/`](python/README.md) trains the experiments in
[`experiments/`](experiments/README.md). `./make.py` runs every task;
`./make.py -h` lists them.

## Requirements

- [uv](https://docs.astral.sh/uv/). It installs Python 3.14 and the locked
  packages, among them torch 2.14.1 built for CUDA 12.6, which also runs
  without a GPU.
- A C++ compiler (g++). TorchInductor compiles the kernels of a CPU run
  with it.
- A GPU, only for runs that replay CUDA graphs.
- For the Lean tree, elan. It installs the toolchain that `lean-toolchain`
  names.

## The minimal run

```sh
./make.py setup                                          # install the locked environment
./make.py run experiments/basis easy-small-depth-seed0   # train one run of the benchmark
./make.py report experiments/basis easy-small-depth-seed0
```

The run trains the two-layer model of width 64 on one core until it passes
E_2, in about 20 s. The first time, compiling its kernels adds 40 to 80 s;
TorchInductor then keeps them in its cache on disk. The run writes into
`experiments/basis/runs/easy-small-depth-seed0/`. `report` prints what the
run recorded as JSON, and its `stop` says whether it passed or reached its
budget.

## The benchmark

The smallest complete version takes one mode on one model from one seed.
Its three runs train side by side in about four minutes:

```sh
./make.py run experiments/basis easy-small-{depth,recall,parity}-seed0
```

All 15 runs of the small model, both modes from three seeds, train side by
side on 15 cores in about nine minutes:

```sh
./make.py run experiments/basis easy-small-{depth,recall,parity}-seed{0,1,2} \
                                hard-small-{depth,recall}-seed{0,1,2}
```

These runs screen an idea before the large model takes it. At bd63e50 the
small model passed the easy mode from every seed. It failed the hard depth
and recall from every seed, as intended, since the hard mode is set for the
large model. So a failure in the easy mode counts against an idea, and a
pass in the hard mode counts for it. The recipes were set for this model:
before dropping an idea, try a rate or two around its recipe. Compare
seed by seed: on the small model a seed moved a pass by up to twice the
updates, and on the large model's hard depth by seven times (800 against
5,800). On the large model, the easy recall missed from seeds 0 and 1 too.

A model of its own takes the benchmark in an experiment file whose runs are
`basis(model, mode, seed)`. Labeled as here, the same commands train it.

The whole benchmark has 30 runs: each mode on both models, from seeds 0, 1
and 2. A hard parity run is its easy one, so the hard mode lists only depth
and recall:

```sh
./make.py run experiments/basis easy-{small,large}-{depth,recall,parity}-seed{0,1,2} \
                                hard-{small,large}-{depth,recall}-seed{0,1,2}
```

Each run trains on the physical cores it asks for: four for the large
model's recall, two for its depth and parity, one for every task of the
small model. On 28 cores the 30 runs took 19 min 52 s. Without labels, `run`
trains all 180 runs of `experiments/basis`, including the 150 GPU runs that
set the recipes. `./make.py check experiments/basis` lists every label.
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

A run repeats its own losses on the same machine, with the same build of
torch and the same number of threads. A different number of threads rounds
differently.
