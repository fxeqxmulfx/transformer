# Coverage of possible fraction controls

This is a data inspection on calibration data seed 0, not a new training plan
or a selected recipe. All four existing learning-rate/sampling budgets remain
unchanged and must finish before choosing a follow-up.

| Requested train fraction | Train / exhaustive held-out | Numerators / denominators / answers seen | Minimum training answer count |
| --- | ---: | ---: | ---: |
| 0.5 | 4,656 / 4,656 | 97 / 96 / 97 | 38 |
| 0.2 | 1,862 / 7,450 | 97 / 96 / 97 | 12 |
| 0.1 | 931 / 8,381 | 97 / 96 / 97 | 2 |
| 0.05 | 466 / 8,846 | 97 / 96 / 97 | 1 |

The generator shuffles the same 9,312 mod-97 equations and takes a rounded
training prefix. These pools therefore remain nested under the same data seed,
with different exhaustive complements. Every inspected pool contains all
numeric classes in each legal role. This finite coverage says nothing about
learnability, memorization phases, persistent convergence, or an internal
algorithm. It does not certify coverage on independent confirmation splits.

The local [*Convexifying Transformers* manuscript](../../../../papers/arXiv-2211.11052v1/arxiv.tex),
Section 4 (`arxiv.tex`, line 487), specifies the mod-97 task and reports training at about 1,000
iterations followed by generalization beyond 100,000; it does not give the
train fraction or regularization used there. Any fraction change remains an
explicit calibration adaptation.

The archived [20% author-reference control](../../baselines/mod97_fraction20_reference_20261002)
used two layers, width 128, AdamW learning rate 0.001 and decay 1.0, on the
same data seed 0 corpus. It completed 150,000 updates and ended at 100% train /
1.7852% held-out accuracy without confirmed generalization. Its source model,
optimizer and decay differ from the current GPTMini/raw AMSGradW controls;
that negative result cannot predict their outcomes. It must remain retained.

[Measured counts and corpus/source fingerprints](candidate-corpus-coverage.json)
include each answer class frequency and the checked historical corpus match.
Counts, disjoint exhaustive complements, nested training prefixes, and modular
inverse answers were recomputed. All 19 frozen calibration source hashes remain
unchanged; no additional training was launched for this inspection.
If the completed current grid fails its phase/persistence gate, these covered
smaller pools are possible single-fraction interventions. Choosing one still
requires the complete current comparison, a justified fresh frozen plan, and
unchanged success criteria.
