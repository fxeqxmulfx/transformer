# Review of the archived mod-193 final checkpoints

This archives step 5's preliminary CPU inspection after reviewing its
source and independently reproducing its observations. It concerns the
historical 300,000-update checkpoints, not the fresh lab runs requested by
[EXPERIMENT_PLAN.md](../../../../../../EXPERIMENT_PLAN.md).

`inspector_441f47b.py` is the exact source from commit `441f47b`, formerly at
`experiments/archive/synthetic_trainers/protocols/adamw_stability_20261002/inspect_sparsemax_generalization.py`.
Its SHA256 matches `observations.json`, copied exactly from the untracked
`runs/adamw_stability_20261002/bootstrap/sparsemax-generalization-inspection-20261003/`
directory. The historical inspector imports removed trainers and refers
to their former paths, so replaying it needs that historical checkout and
its frozen inputs. Its `all_original_frozen_sources_unchanged` assertion
checks sources at startup; it is not a continuous filesystem monitor.

`verify_observations.py` independently recomputes the inspection using the
lab at `d03f594`. The source and its output, `independent_review.json`, are
exact snapshots. The embedded lab source hashes have been checked against
that git commit, and the current frozen checkpoint bytes against both
recorded hashes. `manifest.json` records all bundle file hashes and input
provenance. The later optional QKNorm initialization field was absent from
this review's pinned source.

The independent program rebuilds the complete 37,056-equation corpus and
checks its train/held-out fingerprints. It renames checkpoint parameter
keys without changing their values, loads all 436,104 parameters strictly,
and computes all four checkpoint/forward-normalizer combinations on CPU.
It observes the actual lab modules, sorts scores independently for gaps,
and computes float64 entropy with `xlogy`. Both numeric and EOS routing
groups are selected by numeric-answer correctness, as in the original
inspection; entropy uses natural logs and is not normalized by prefix
length. All reported counts and floating-point fields match exactly:
the rounding-differences list is empty.

| Trained checkpoint | Forward normalizer | Train complete-answer accuracy | Held-out complete-answer accuracy |
| --- | --- | ---: | ---: |
| sparsemax | sparsemax | 100% | 9,465/27,792 = 34.0566% |
| sparsemax | softmax | 208/9,264 = 2.2453% | 235/27,792 = 0.8456% |
| softmax | softmax | 100% | 100% |
| softmax | sparsemax | 0% | 0% |

Diagonal derivative probes use the first 512 equations of each frozen
partition. Inactive score gradients are exactly zero, and instrumented
and ordinary logits, losses and every parameter gradient agree bit for
bit. Parameters remain unchanged. These are local derivative checks, not
exhaustive derivative claims about every equation. No optimizer update
was performed and CUDA was not initialized. Changing a forward map at
fixed final weights does not isolate the mechanism of a training failure
and is not the trained forward/backward intervention of step 3.

`confirmation_metrics.py` implements step 5's canonical-observation
policy, independently of the lab's default two-observation transition.
`confirmation_archive_check.json` checks it against both complete archived
1,201-point histories: softmax memorizes over updates 4,500--27,250,
confirms jointly over 95,750--100,500 and fails 2 of the 201 final-window
checks; sparsemax memorizes over 49,750--255,750, never confirms and fails
all 201. This validates the policy against the historical table, not a
new repair's result. Confirmation requires 20 consecutive canonical
evaluations with both accuracies at least 99%, and a final accuracy of
100% on both splits. Persistence, its failed checks and worst held-out
accuracy are reported separately; the archived softmax reference itself
does not satisfy strict persistence.

The scripts preserve their original source and workspace paths for
provenance. To replay the independent inspection, use the pinned lab
source, retain the ignored frozen inputs at those paths, and run the
verifier with plain `uv run --locked python` from `python/`. A new output
should be compared with the archived record rather than replacing it.
