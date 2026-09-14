# The released artefacts

`plan.yaml` is the schedule the released model was built from: seven layers,
`d_model = 38`, the assignment of every gate of the machine graph to a phase.
It is an *input* — the one file of the release that is not a build product —
and 8 KB, so it is here, and `alm-compile` compiles it in
(`alm_compile::release::PLAN`).  `alm-schedule` reproduces it from nothing but
the graph; this copy is what that reproduction is checked against.

`sha256sums` is what the release's build products hash to: `model.bin` and the
eighteen files of `transformer_vm/data/`.  Those are ten megabytes and are not
in this repository.  The tests build each of them — `alm-compile` from
`plan.yaml`, the programs from the C in `programs/` — and compare the digest,
which is byte identity by another name and costs 1.6 KB instead of 10 MB.

Both come from the upstream release (Apache-2.0, see `../programs/LICENSE`),
with `patches/reproducible-build.patch` applied: without it the Python picks a
different permutation of layer 5's FFN passthrough neurons on most runs and
`model.bin` is not a function of `plan.yaml` (`todo3.md` section 9).

To regenerate, with the release checked out at `transformer-vm/`:

    cp transformer-vm/plan.yaml vm-rs/reference/plan.yaml
    ( cd transformer-vm && sha256sum model.bin &&
      cd transformer_vm && sha256sum data/*.txt ) > vm-rs/reference/sha256sums
