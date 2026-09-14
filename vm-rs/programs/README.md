# programs — the six reference programs, and the runtime they compile against

Copied verbatim from the Percepta `transformer-vm` release: the `.c` and
`manifest.yaml` from `transformer_vm/examples/`, and `runtime.h` from
`transformer_vm/compilation/`.  They are Apache-2.0, and `LICENSE` is the
release's own copy of that licence.

They are here because they are the machine's *input*, not part of it, and
without them the port cannot build a program at all: the compiler is ported,
the programs were not.  With them, `alm-cc` needs no vendored checkout —

```
cd vm-rs
cargo run --release --bin alm-cc -- --all --out data
```

`manifest.yaml` names the six programs the release ships and the argument
string each is compiled with.  `lowering_test.c` is not one of them; it is the
release's own exercise of every hard WebAssembly operation, and it is what
`the_released_programs_lower_to_the_dispatch_table` lowers along with the six.

`runtime.h` is the only header any of them includes — it is `#include`-free
itself — so clang needs nothing but this directory.

What is *not* here, and still wants the vendored release, is everything these
files are checked against: the released `data/*.txt`, `*_spec.txt` and
`*_ref.txt`, `plan.yaml` and `model.bin`.  Those are build products, they are
large, and reproducing them is the whole point of the port — a copy in this
repository would be a copy of the answer.
