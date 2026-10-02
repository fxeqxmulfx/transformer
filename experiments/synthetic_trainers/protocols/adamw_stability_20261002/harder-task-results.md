# Harder-task AdamW calibration results

Reducing the mod-193 training fraction to 25% produced the required
memorization-then-generalization phase in one calibration. Increasing the
modulus alone delayed fitting and generalization together. Every completed
recipe still fails final-window persistence, despite final train and held-out
accuracy of 100%.

The [offline verification](harder-task-result-validation.json) checks all three
complete archives without PyTorch: 450,000 updates and 1,803 canonical
observations. Each recipe finishes 150,000 updates, with exhaustive evaluation
every 250 updates and all gradients/neighbors retained in its original archive.
All use GPTMini width 128, two layers, four heads, native AdamW, seeds 0/0,
learning rate 0.001, decay 0.1, warmup 10 and short-final batches of 512.
The 15 training-source files, environment and success criterion are identical.

| Modulus / train fraction | First train 99% | First held-out 99% | Qualifying memorization plateau | Failed final-window observations / 201 | Lowest final-window held-out |
| --- | ---: | ---: | --- | ---: | ---: |
| 97 / 50% | 1,000 | 1,250 | None | 7 | 43.9003% |
| 193 / 50% | 39,500 | 40,000 | None | 7 | 79.8953% |
| 193 / 25% | 1,000 | 20,000 | 4,500–11,500; 29 observations | 3 | 48.9206% |

The first 20 consecutive joint 99% observations span 1,250–6,000,
52,250–57,000 and 20,000–24,750 respectively. Their observed training costs
are 181.81, 1,436.21 and 678.96 seconds. Eligible stable-target timing support
is zero for every case, so no speed ratio follows from these measurements.
Full training / wall costs are 4,022.04 / 4,284.18, 3,750.27 / 4,648.78 and
4,193.88 / 5,106.13 seconds. These sequential calibration costs are descriptive;
CPU preparation and verification overlapped parts of later training.

The adjacent configuration changes are exactly `prime` and then
`train_fraction`, with derived changes kept explicit. Mod 97 has 4,656/4,656
train/held-out pairs and 423,816 parameters. Mod 193 at 50% has 18,528/18,528
pairs; at 25%, 9,264/27,792 pairs. Both mod-193 models have 436,104 parameters.
Full-budget example exposures are 69,840,000, 75,113,536 and 73,137,184.
Changing the modulus also changes vocabulary, initial weights and epoch tails;
changing the fraction changes sampling/exposure and exhaustive evaluation cost.
The mod-193 fraction comparison verifies identical initial CPU model states.
Two later analysis-only repairs are documented in the individual result;
historical fingerprints and measurement bytes are preserved.

The frozen stable-grokking criterion requires train ≥99% / held-out ≤10% for
at least five consecutive observations spanning 1,000 updates before the first
held-out 99%, followed by 20 joint ≥99% observations and joint ≥99% at every
canonical point in updates 100,000–150,000. The quarter split observes the
phase but fails persistence at 115,000, 119,250 and 139,000. This establishes
a delayed-generalization observation in one calibration, not a repeatable
stable benchmark. It tests novel operand pairs at fixed length, not length
transfer or discovery of an internal algorithm.

Complete individual results and standalone curves remain in
[mod-97 AdamW](adamw-result.md), [mod-193 / 50%](larger-modulus-result.md) and
[mod-193 / 25%](fraction25-result.md). These are explicit adaptations of
*Convexifying Transformers*, Section 4, rather than its numerical timing claim.

The [lower-rate calibration](lower-rate-protocol.md) is prospectively frozen
and running with only learning rate changed to 0.0003. Its result is pending;
the same phase, full budget and persistence requirements apply. Only a complete
passing primary calibration permits the separately frozen six fresh cases
model seeds 4/5/6 crossed with data seeds 2/3. Architecture selection and
scientific complementary-task comparisons remain outstanding.
