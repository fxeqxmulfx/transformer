# Early settling and active-answer certificates

The user asks to close the discovery/persistence and certificate gaps after
the eight-seed .01 study. The [frozen protocol](PROTOCOL.md) separates two
changes: reduce the large learning rate before the late measurement window,
and fit the rank-limited readout basis to observed answer classes while still
bounding every rival class. All other training and numerical controls remain.

| Intervention | Changed | Fixed comparison |
| --- | --- | --- |
| Early settling | .01 to .0001 during updates 1000–4000, then hold until 8000 and cool to 1e-8 | Same eight seeds, 10000 updates and all-class certificates against .001 and .01 |
| Active-answer basis | Readout rows used to choose reconstruction directions | Same saved states, training folds, rank budgets and all 97 rival margins |

Implementation and outcomes are pending. Generated evidence is ignored and
preserved outside temporary worktrees. This experiment does not assert a
global training or convergence theorem.
