# Historical references and floating-point portability

The `legacy_*.json` files preserve measurements from the commits named in
their `source` fields. They are evidence, not a requirement that a newer
PyTorch wheel or another CPU produce an old wheel's floating-point bytes.

Numerical model and training tests replay those immutable implementations
with the current test interpreter and PyTorch. `reference_fixtures.py`
exports Python sources from the pinned Git commits to a temporary directory;
the helpers in `../references/` measure the original models and trainers in
an isolated subprocess. A reference importing `lab` fails. Initialization is
compared exactly to that independent replay, and scientific configurations
remain checked. The comparisons of the replay's outputs, gradients,
trajectories and checkpoints remain exact where the two implementations use
the same arithmetic. Optimizers with different operation orders are checked
against the frozen optimizers on 40 shared gradient updates in float32 and
float64, within 64 machine epsilons. Resetting the input parameters at each
update prevents rounding differences from feeding back into later gradients;
moment buffers and MAGMA masks continue across all updates. Full trajectories
also agree exactly when the historical first-moment arithmetic is selected.
Source code and archived measurements are never regenerated from the
implementation under test.

Historical run names and outcomes are not invariants across numerical
backends. A large-rate run previously labeled `nonfinite`, for example, may
now stop for divergence. The lab must make the same observations and decisions
as the independent replay. Runtime errors in a reference fail the test. The
dedicated nonfinite-handling test exercises its failure condition explicitly.
Archived CUDA measurements are compared only on their recorded GPU and PyTorch
build; current CUDA execution is covered separately by the engine and kernel tests.

These tests need the repository's historical Git objects, as well as the
locked Python environment. Use a full clone when running them; missing
history fails visibly and never skips a regression. Replays require no
network access and create no worktrees or persistent fixtures.

The frozen text trainers call CUDA timing hooks even for CPU runs. The replay
replaces only those timing hooks with no-ops in the trainer's local `torch`
view, so CPU tests need no NVIDIA driver. Model operations, gradients,
optimizer arithmetic and the actual PyTorch module remain unchanged.
