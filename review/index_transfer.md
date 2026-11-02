# Index-transfer investigation: task159

Python3.15's parent-owned sampling profiler found63–74% of successful
main-thread wall samples at the blocking index upload in the arithmetic
cohort. Those eight10-second windows have roughly5–6% sampling errors,
recorded in `experiments/adaptive_residual/profiles/`. A blocking upload
can wait for earlier GPU work, so this is not evidence that CPU data
preprocessing consumes that fraction of computation.

`scripts/index_transfer_profile.py` compares the same2,048-byte index
upload with and without preceding queued matrix multiplies. CUDA events
separate the preceding work from the interval containing the copy; both
intervals can contain host enqueue gaps. These are not claims of pure
memcpy-kernel duration or bandwidth.

The execution candidate copies a sampler permutation/window tensor once
to the GPU and takes batch views there. Its bounded cache detects both
source replacement and PyTorch in-place version changes. It keeps sampler
state on CPU, consumes no randomness, preserves epoch remainders and is
reconstructed on resume. A batch crossing permutations concatenates views
of the corresponding copies. CPU execution retains its old path. This
changes no architecture, optimizer equation, loss, dataset or stopping rule.

The control is the compiled per-batch `gather(parts)` input path at0f457bac.
Both timing arms use the same actual model instance and compiled graphs,
restored to the same complete100-update state before each trial. Twenty
updates warm the chosen input path, then200 updates are timed with device
synchronization at both boundaries. Six repetitions alternate arm order.
Every measured gradient and complete320-update model, optimizer and sampler
state must agree exactly. Modular P97 and large hard Basis recall supply
the actual arithmetic and longer-sequence workloads. No final test is read.

The acceptance rule, declared before measurement, requires every equality
check, median arithmetic speedup at least1.05, and median hard-recall speedup
at least1.0. Report every repetition and peak allocated memory. A failed
check or a missing measured improvement prevents retaining the candidate.
The profiling command refuses to run during the active Basis GPU cohort.

CPU preparation ran four checks in18.22 seconds: three passed and the CUDA
check was skipped. These cover exact epoch/window/browser-sampler draws and restores, cache
invalidation and preservation of prior batches, and exact compiled model,
optimizer, gradient and validation histories through an epoch remainder
and resumed run. The guard was exercised during the active cohort and
refused before GPU initialization or writing a timing result. Full CPU
verification passes502 lab tests in797.48 seconds with14 skips and104
optimizer tests in12.25 seconds with14 skips. The subsequent CUDA checks and timing attempt are recorded below.
The candidate was not retained; the live compiled input path is unchanged.

This assistant performs the review; no delegated reviewer was used.


## Measured outcome

The20 order-balanced pairs copy the same2,048 index bytes. Median host-call
time is0.02391ms when idle and6.74493ms after queued matrix multiplies.
The median preceding CUDA-event interval is6.90702ms; the post-compute
interval is0.03482ms. CUDA intervals include host enqueue gaps, so this is
not a pure-copy bandwidth measurement. The result supports queued GPU work
as the dominant explanation of the sampled upload wait, rather than proving
that CPU index preprocessing consumed63–74% of the total computation.

The CUDA trajectory test fails: two optimizer-base entries differ, with
maximum absolute difference9.31e-10. A separate five-run diagnostic also
finds differences between unchanged per-batch controls; one control repeat
has model differences up to3.73e-9. The candidate is therefore not uniquely
implicated by that failure. The strict exactness criterion nevertheless did
not pass and was not relaxed.

The declared actual-model profile then fails on its first P97 candidate
comparison: the first measured gradient norm differs by2.38e-7. Only one
control trial completed; no paired median speedup or hard-recall speedup
was obtained. That incomplete trial is not an accepted throughput improvement.
The GPU was free of learning workloads. CPU correctness checks briefly ran
during compilation and ended before the measured P97 control trial; no
performance conclusion is drawn from the failed training comparison.

## Retention and review

`index_transfer/candidate.patch` preserves the rejected implementation,
its tests and profiling command against ef1ec800. Applying it was checked
before restoring the task's changes. `manifest.json` hashes every candidate
source and evidence file. Raw CPU/CUDA test logs, the failed profile and the
unchanged-control repeatability records are retained alongside it. The lab,
optimizer, sampler and production GPT source remain unchanged by task159.

Task163 records the unresolved repeatability question. It must characterize
the unchanged control and declare a new protocol before reconsidering caching;
these results do not justify retroactively weakening task159's acceptance rule.
