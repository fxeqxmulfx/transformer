# Compare the general bank with all earlier pair candidates

This native integration step belongs to task #209. It leaves the frozen training
producer unchanged. The nine tasks and three modes retain every completed
general-bank decoder, every refusal or failed process, and the complete earlier
comparisons. The eight shared tasks use the CM18 certified comparison, including
its paired, decoded, phase, graph and earlier parents. The added long blind task
uses the recovered full-suite hull baseline. No partial parent is substituted.

`protocol.json` fixes candidate order: new decoders in declaration order, then
the previous selected model. Choose the first within `1e-8` of the common ideal
CE, otherwise the smallest training CE. A later candidate's better held-out
accuracy cannot change this choice. All candidates use the original vocabulary
and unit-scale CE on these tasks. The full 105-choice baseline remains attached,
but this comparison updates only 27 choices: it does not integrate every depth
follow-up across all 35 tasks or a fresh transformer comparison.

The wrapper pins the frozen native producers and recovery verifier. It reruns
the general summary's artifact checks, the certified integrator's recursive
parent checks, and the recovery verifier's complete 105-choice native replay.
It then checks exact identity sets, preserves terminal failures without scores,
and records hashes, subprocess exits and resources. Every child runs under the
64 GiB limit. Its fresh output directory retains failed checks too. A missing
complete predecessor stops before creating output or running the expensive
checks. Presence checks alone are not an audit.

Run the metadata-only tests from the repository root:

```sh
bash -n experiments/convex_stress_general_bank/comparison/run.sh experiments/convex_stress_general_bank/comparison/test.sh
bash experiments/convex_stress_general_bank/comparison/test.sh
```

The tests cover all 27 choices and the retained 105-choice baseline, training-CE
ties and improvements, held-out invariance, all-process-failure and missing-model
cases, 17 malformed-input mutations, and the missing-parent gate. They construct
no original samples and run no learner.

After the decoded parent and recovered baseline are complete, use a committed
copy of this directory kept outside a temporary worktree:

```sh
bash experiments/convex_stress_general_bank/comparison/run.sh preflight \
  experiments/convex_stress_general_bank/runs/cohort-v1 \
  experiments/convex_stress_certified/runs/cohort-v1 \
  experiments/convex_stress_paired/runs/cohort-v1 \
  experiments/convex_stress_decode/runs/cohort-v1 \
  experiments/convex_stress_hull/runs/recovery-v1
bash experiments/convex_stress_general_bank/comparison/run.sh compare \
  experiments/convex_stress_general_bank/runs/cohort-v1 \
  experiments/convex_stress_certified/runs/cohort-v1 \
  experiments/convex_stress_paired/runs/cohort-v1 \
  experiments/convex_stress_decode/runs/cohort-v1 \
  experiments/convex_stress_hull/runs/recovery-v1 \
  experiments/convex_stress_general_bank/runs/comparison-v1
```

Generated evidence remains ignored. The separate protocol is an adaptive
integration of the already frozen cohort, not a new unseen-data experiment.
Integration is still pending the unchanged live parent queues.
