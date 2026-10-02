# Lower-rate AdamW: first confirmed delayed-generalization phase

The frozen mod-193 / 25% calibration at learning rate 0.0003 observes a
qualifying memorization plateau, then twenty consecutive joint 99% scores.
The whole 150,000-update budget remains unfinished. A later held-out failure
at 105,250 already refutes final-window persistence; this prefix preserves the
valid phase before that failure. Independent confirmation and architecture
comparison remain closed.

The [portable prefix](first-long-confirmation-lower-rate/summary.json) retains
every canonical observation from 0 through 100,500: 403 exhaustive train/
held-out evaluations and all 100,500 pre-update gradient records. Its complete
frozen plan, configuration, criterion and source fingerprints are included.
The [offline validation](first-long-confirmation-lower-rate-validation.json)
uses the unchanged semantic verifier without PyTorch and checks exact original
history/gradient lines, every batch/rate/support, all 31 core fingerprints and
both pinned launcher/worker hashes.

| Observation | Updates | Train / held-out accuracy |
| --- | --- | --- |
| First canonical train 99% | 1,250 | 99.7949% / 0.8492% |
| Longest qualifying memorization plateau | 4,500–27,250; 92 points | Minimum train 99.0285%; maximum held-out 0.8995% |
| First canonical held-out 99% | 95,750 | 100% / 99.0141% |
| First twenty joint target points | 95,750–100,500 | Last point 100% / 99.7697% |

The first canonical train-to-held-out target gap is 94,500 updates. It is a
sampled timing difference, not uninterrupted train-target retention throughout
that interval. The long confirmation costs 2,765.91 training / 3,375.93 wall
seconds. First held-out target costs 2,637.32 / 3,218.69 seconds. These are
observed event costs in one incomplete calibration, not eligible stable-target
times or a causal speed comparison.

The largest prefix gradient norm is 93.51931 at update 63,261, after a full
512-example batch. There is no exhaustive accuracy observation at that update;
the norm alone does not identify a trigger or prove optimizer stability.
The prefix contains only three final-window points, 100,000–100,500, all
jointly passing; minimum held-out is 99.7625%. All 201 canonical points in
100,000–150,000 would have to pass for persistence. The later canonical point
105,250 has train / held-out 100% / 97.4813%, with EOS 100%, so the full-window
criterion is already refuted. That separate validated point is recorded beside
the positive-prefix verification; finish the full budget and retain every failure.

Standalone [PNG](lower-rate-phase-curves/phase-prefix.png) and
[PDF](lower-rate-phase-curves/phase-prefix.pdf) show every canonical point with
no smoothing. Both were visually reviewed, including a raster of the actual
PDF. Provenance pins the exact prefix and renderer, and the caption explicitly
states the partial scope, rate and actual seeds. The shared standalone renderer
now labels rate/seeds from the configuration; historical quarter-split figures
and their original provenance remain byte-for-byte intact.

This is an explicit *Convexifying Transformers*, Section 4 adaptation with the
unchanged stronger phase/persistence criterion. It measures fixed-length novel
operand pairs, not length transfer or an internal algorithm. Continue the
[frozen full-budget recipe](lower-rate-protocol.md), retain any later failure,
then archive its complete negative persistence result. The unchanged six-case
gate requires a separate full passing primary calibration with fresh model/data seeds.
