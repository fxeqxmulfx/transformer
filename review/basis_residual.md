# Basis comparison: same-assistant review

Task160 is the user's requested broader check of the conditional-gain candidate.
Task158 recorded three earlier arithmetic crossings and one tie; its frozen
all-four-earlier promotion rule failed, so production GPT remains `df1e9f7`.
Basis results must not be inferred from those arithmetic results.

The24 recipes are the original small/easy and large/hard Basis settings, each
with canonical/adaptive GPT and model seeds0/1. All six full problem/settings
have exactly the original split fingerprints, including every training row,
validation/test row, length distribution and original label. The generators and
their oracles are unchanged. The optimizer deviation is explicit: only
Schedule-Free AdamW, with the original Basis rates/batches/warmup. Compilation
uses CUDA fullgraph max-autotune and math SDPA, with evaluation batch64.

The source adapter gathers supervised logits from the actual GPT's full output.
Independent full-position cross-entropy and every parameter gradient agree for
depth, recall and parity. Original free-generation scoring is inherited for
parity; token accuracy cannot replace sequence accuracy. Hard depth stops on
length128 validation. Tests exercise first99% stopping without opening test or
selecting a best state. Compiled continuation is exact across the final partial
batch of an epoch, including model, optimizer, sampler and gradient trace.

Full Python verification initially ran495 lab tests in825.09 seconds with13
CUDA skips and one error: the new first-crossing test used `Eager()`'s default
CUDA device despite being intended for CPU. This was a test-fixture error, not
a model or engine failure. Explicit `Eager(device="cpu")` fixes it; all six
focused Basis tests pass with CUDA hidden in15.52 seconds. The separately
completed optimizer phase passes104 tests with14 CUDA skips in12.86 seconds.
The existing CUDA ScheduleFreeMuon allocation failures remain task157 and are
not claimed fixed by running CPU checks. This cohort uses Schedule-Free AdamW.

Original Basis is sampled with replacement. All depth and recall validation
examples are novel against training, but only157/512 parity validation examples
are novel; its final sampled test contains184/512 novel examples. No predictions
or test accuracies were consulted to obtain those input-overlap counts. The
report preserves the original score and adds a separate novel-input score using
the generator's input key, which is the prompt for generated answers.

The source/data/runtime protocol is frozen before learning. GPU preparation
checks take zero training updates and compare the complete model/optimizer frame
before and afterward. Their timings are preparation measurements, not learning
or isolated throughput results. The large-depth check has passed; remaining
preparation checks and the scored learning cohort are still in progress.

The same assistant performs this review; no delegated reviewer was used.
