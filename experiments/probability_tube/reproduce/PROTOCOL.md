# Original-source Rotate90 reproduction

This bootstrap implements the parent `../PROTOCOL.md` after exact source
recovery on 2026-11-09. Commit the completed runner before training. It
imports trainer b70811e with concatenated digest
`14cff3b553d3afb1f9bf90bb6b50cfd7a879058ec2876f859a198e07a09a83da`,
Kernite a1b6bc2c and the four already frozen GPT source digests.

Reuse the original initializer, Data and Plan, four-normalization plain
complex-product graph, full backward, ownership contraction, clipping,
coordinate AdamW and schedules. The main-optimizer betas are (0.9,0.95),
decay 0.1 and epsilon 1e-8. The original standalone trainer's (0.9,0.98)
optimizer is not the selected main-optimizer cohort. Every shared scientific
source is included in the executable's content fingerprint.

Build the unchanged schedule variants with default features (.001 WSD),
`lr01` (.01 WSD), and `settle` (early settling). Run each in a fresh
directory. Each retains all eight seeds [7,11,1,3,5,13,17,19], 10000 updates,
the original 31 geometry probes, metrics every 100 updates, the complete
embedding trace and every update's finite-component accounting. Preserve
full parameter/moment states at every anchor and declared horizon endpoint,
including endpoints outside the old probe cadence. No new baseline report
is fabricated to satisfy the historical runner's old report hashes.

Each cohort has a 7200-second cap including failures and verification, with
compilation excluded. Charge a sixty-second preparation reserve to each
cohort for standalone source/optimizer preflight checks, including the initial
optimizer-only integration check (measured 1.23 seconds).
Every recorded prior attempt under the same cohort root also consumes the
budget. Completed outcomes, errors, capped seeds and pending checkpoints
remain in the inventory. A new attempt never overwrites an old one.

The cohort preflight checks the unchanged optimizer against its independent
FP32 reference and the source forward/readout/full-embedding derivatives
against the existing FP64 oracle. Every geometry probe keeps the existing
feature/logit allowances and all ranks. At each applicable probe update,
compare instrumented versus ordinary backward/update bytes and check the
independent optimizer reference. Keep the existing numerical tolerances.

Record GPU, driver, CUDA capability, Rust version, source/executable/PTX
fingerprints, source-input hashes, initialization/split/sampler hashes,
wall time and all checkpoint hashes. The exact source is recovered; the new
hardware execution is a reproduction with its own provenance. Synthetic
preflight checks and new runs do not replace any historical measurement.

The later probability constructor receives checkpoint-only inputs. Freeze
its forecast payload and hash before a separate verifier reads later states.
Retain all 144 anchors, declared horizons and 4704 possible training inputs
under the parent protocol. This bootstrap does not assert any useful bound,
CE nonincrease for the full update, convergence or recovery.
