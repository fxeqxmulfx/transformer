# Original results with raw-margin constraints

The corrected run retains 35/35 primary and auxiliary task successes in all
three modes and both initializer policies. Fibonacci validation cross-entropy
falls modestly while every raw prediction stays equal to the stronger task296
reference. All 35 original cases and 35 replays finish with exit0. This improves
CE relative to that raw reference; the earlier task298 five-gate model still
has lower CE and more raw errors. Complete independent reproduction follows
this authored findings commit before task301 approval.

| Fibonacci mode, 4704 validation records | Reference raw errors | Guarded raw errors | Reference CE | Guarded CE | Five-gate CE / errors |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full batch | 101 | 101 | 0.1726145666 | 0.1708308624 | 0.1336592553 / 105 |
| Minibatch | 100 | 100 | 0.1751058800 | 0.1737371214 | 0.1322171895 / 107 |
| Forward only | 100 | 100 | 0.1772798206 | 0.1762888293 | 0.1301328768 / 105 |

Raw correct counts remain 4603, 4604 and 4604. Original field accuracy stays
4701/4704, or 99.9362%, in every mode. Observed CE remains
9.626594307e-9. Each policy changes the same three Fibonacci choices and
inherits 102 reviewed choices. No primary, auxiliary, raw-accuracy or CE
regression occurs against the actual task296 reference. The complete task298
CE/raw tradeoff remains an independent retained reference, never a held-out
selector. Earlier suite results motivated the architecture, so this is not
an untouched architecture-selection holdout.

The exact CM46 reference and a half-scale copy precede the existing context
bank. Five fixed source-excluded count gates use radius and margin retention
0.5. Every accepted proposal recomputes the actual complex products and keeps
at least half of every reference raw margin, with the metric's exact argmax
convention. Each mode certifies 29,894 prior and 30,782 supplement inputs against
all 256 classes. Observed and unsupported inputs keep the parent path. All final
score hashes reproduce through public inference, and fresh reference metrics
match the reviewed parent exactly on every original split.

Prior components retain their initialization in all modes. Their diagnostic
CEs are 0.0399514114 full, 0.0439758225 mini and 0.0479323749 forward. The full
supplement retains step4; mini retains initialization after its eight epochs.
The guard therefore limits useful movement, especially in nearly tied regions.
This result demonstrates decision stability with a small CE gain, not discovery
of the remaining missing answers. Preserving the reference also preserves its
100-101 wrong raw labels.

Across the six prior/supplement optimizer trajectories, accounting records
288 gradient batches, 164 accepted optimizer updates and 770 local-row updates.
Full contributes 128 batches, four updates and nine row updates; mini contributes
160 batches, 160 updates and 761 row updates; forward contributes zero gradients
and optimizer updates. Initialization remains separately recorded. There are
677376 gradient-record visits, 11943104 scoring-record visits and 1249 rejected
proposals. Constraint work includes 6570482688 cap-class visits, 8711620 query
visits, 2230174720 guard-class visits and 108595133696 actual head terms. Every
earlier constructor and optimizer cost remains explicit. Reconstruction and
replay repeat these trajectories; these counters are not an aggregate timing
measurement over all reconstruction passes.

All 35 conventional transformer endpoints and 336 depth outcomes are retained
without retraining. The frozen conventional Fibonacci endpoint has 95.1318%
raw accuracy and CE0.234544, compared with the guarded model's 97.85-97.87%
and CE0.171-0.176. Historical DiffGPT's recorded 35/35 on three seeds remains
a separate result. This is not a matched-compute or new DiffGPT experiment.

Models are constructed one at a time, sealed to disk and released. All three
complete endpoints are sealed before any new metrics; scoring reconstructs
one endpoint at a time and checks its seal. Replay also loads each saved model
through full recursive reconstruction. Preparation takes 40.14 seconds, all
cases total 624.73 seconds, replays total 973.97 seconds, and summary generation
takes 3.11 seconds. Peak RSS is 8396104 KiB, about 8.01 GiB. The eligible case
forecasts 42241052032 buffer bytes and 124050941208432 scalar-work units,
including three construction passes. The 64 GiB process limit and 56 GiB
buffer forecast limit remain in force. Timing describes this run only.

The earlier v1 attempt remains a failed attempt. All 35 cases were attempted;
34 runs and 34 replays completed, while Fibonacci exceeded the 8 trillion work
cap after sealing its forward model and before complete selection or new
held-out scoring. Its summary exited1 and peak RSS was 6.53 GiB. The corrected
component permits 16 trillion work units without changing the numerical model
or memory limits. The complete failed manifest, every process status, missing
replay reason and partial sealed model are bound into this run. The frozen
forward trajectory matches except for the protocol digest. No failed record
is removed or reinterpreted as success.

Reproduce producer `e971b72276b9d22eb6dbf776f16f991a9c911131`, reviewed budget
component `26af04ec`, numerical component `70fb09bb`, and Lean CM47 `85aa4a2e`.
The frozen source fingerprint is
`e6bf387ee1120942d1bca019ece75378258aebb089eceee8cf34fe28ba81b6f4`;
the executable hash is
`c3058847e4f1409283587aafc10dd25e34288b3a77b1ffd3861bc0f6c0231d13`.
The inventory hash is
`1430419d0f6ab8beca42bfcb155c41a1c566f16429a244cc206a8998a99aca2a`;
the summary hash is
`1b32220b8e35940c0e5fb3528411daa5484f181c3349bf3a400a2c3fcdb53c45`;
the report hash is
`76bfd620893e05438a4ef9046a68d74fa88f616d428062edd4fb42409ad9f2f9`.
The completed-manifest hash is
`46b13de2fc88a78be26ae703503322875e27d49739507585b7a7cbd5ca70c87b`.

Evidence stays on storage under `transformer-runs/margin-budget-compare/`;
the source, executable and request are frozen under
`transformer-runs/freezes/margin-budget-compare-v2/`. The complete manifest
contains 334 files. Commands in the parent README regenerate the comparison
and report. Exact-commit review must freshly prepare all inputs, run all 35
cases and replays, reproduce all 117 JSON artifacts and the report byte for
byte, and pass native/Lean/index/axiom and frozen-source checks.
