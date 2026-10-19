# Dependency and verification review

The maintenance requested on 2026-10-19 precedes the active research cycle in
`landscape_grokking_plan.md`. The full build, axiom audit, index and Python
suite passed. The mandatory strict lint remains enabled and has not passed;
its reproducible configuration blocker is recorded below.

The task entry point is now `./run.py`. Lean and Mathlib are updated from
4.34.1 to stable 4.35.0. The Python lock refresh updates NumPy to 2.5.4,
filelock to 4.2.4 and cuda-bindings to 12.9.10; PyTorch remains at the current
2.14.1 CUDA 12.6 build. The build backend is uv_build 0.13.1. Archived
measurements retain their original versions and provenance.

Completed maintenance commits on `main`:

- `cdad6727`: portable numerical regression references.
- `b3c3a288`: Python dependencies and build backend.
- `1a3158d9`: `run.py`, active workflows and the mandatory strict lint gate.

`./run.py test` passed all 320 tests in 1810.085 seconds, with three archived
CUDA comparisons skipped because their recorded GPU differs. The current
experiment-loading command `./run.py check experiments/mod97_scalar_mlp`
also passed. No archived measurement JSON was changed.

The final `lake build` completed all 12,771 jobs with the 39 retained `sorry`
warnings and zero other warnings. `./run.py audit` reported
`39 sorry · 0 resting on a sorry · 0 extra axioms`. A temporary theorem that
merely reused one unfinished theorem was correctly classified as an indirect
rest; the audit then returned exit 1. That temporary declaration was never
added to `src/` or the index.

`./run.py index` generated 3,674 modules, 18,875 theorems, 39 `sorry`, zero
vacuous statements and zero placeholders. `./run.py forbidden` and whitespace
checks passed. The axiom and sorry counts have not been reduced by hiding or
removing an unfinished result.

`./run.py verify` retains the exact mandatory lint command:

```sh
lake --wfail lint --builtin-lint --linters=linter.all --code-quality
```

No linter is disabled. The user explicitly removed kernel replay from the
mandatory checks and retained the 39 unfinished statements. `./run.py kernel`
remains available for an optional `lake check`; it is expected to reject
`sorryAx` while these declarations are unfinished. Neither `--paranoid` nor
`lake check` is run by `verify`. The axiom audit still requires zero proved
declarations resting on unfinished inputs and zero nonstandard extra axioms.

## Strict lint configuration blocker

The command remains enabled exactly as requested. It has not passed. A full
attempt also enables Mathlib's diagnostic import tracing and tactic-search
profiling, which can spend many minutes on a small module. That attempt was
stopped after a separate, bounded reproduction established a configuration
conflict; no result from it is reported as a successful check.

The pinned `Mathlib/Tactic/Linter/MinImports.lean` intentionally emits a lint
warning whenever a command increases its cumulative set of required imports.
This reports information about import growth even when the imports and proof
are correct. Lake's `--wfail` makes such a build warning fatal before the
code-quality collection phase. Correcting redundant imports cannot eliminate
the first increase.

The following temporary module has no unfinished proof:

```lean
module

public import Mathlib.Data.Fintype.Card

/-! A minimal check of the required all-linters configuration. -/

@[expose] public section

/-- The finite type with two elements has cardinality two. -/
theorem allLintersProbe : Fintype.card (Fin 2) = 2 := Fintype.card_fin 2
```

With that source temporarily at `src/Transformer/LintConfigurationProbe.lean`,
the command

```sh
lake --wfail lint --builtin-lint --linters=linter.all --code-quality Transformer.LintConfigurationProbe
```

returned exit 1 on 2026-10-19, including this diagnostic. The import is used
by the proof; there is no redundant-import diagnostic in this reproduction.

```text
warning: src/Transformer/LintConfigurationProbe.lean:9:0: Imports increased by 1468 to
[Mathlib.Data.Fintype.Card]
```

The temporary source was then removed. The retained `sorry` declarations also
produce build warnings; keeping them does not imply that a warning-fatal lint
can pass. No linter option has been disabled, no exceptions have been recorded,
and `verify` continues to expose a lint failure rather than treating it as a
pass. The build, axiom audit and Python regressions are checked independently
while this policy conflict is reviewed.

## Lean 4.35 compatibility

The source tree now uses Lean's module headers, public imports and exposed
public sections to preserve its existing exported definitions. File summaries
are module docstrings after the imports. The compatibility fixes add direct
imports for APIs separated by Mathlib, update dependent product evaluation,
and make helper declarations public where they occur in exposed definitions
or public statements. The mathematical formulas are retained.

A source comparison covered all 18,875 existing theorem and lemma statements.
The only textual statement change is the canonical spelling
`QuasiMeasurePreserving` in `Causal/Section5_ConditionalVolume.lean`, replacing
the deprecated `Measure.QuasiMeasurePreserving` alias. Inspection of the pinned
library confirms the alias is exactly the same predicate. Normalizing that
one rename leaves zero changed statements. Private helper visibility changes
do not change hypotheses or conclusions.

The axiom and direct-sorry inspection scripts intentionally use the legacy
frontend to read transitive implementation bodies. Importing only the new
module interfaces would misclassify all 39 direct holes as indirect rests and
erase their positions from the index. The audit still distinguishes the two;
it now exits nonzero on any indirect rest or nonstandard extra axiom.

## Numerical regression tests

The original tests compared current CPU floating-point bytes with archived
PyTorch 2.14.0 measurements. Numerical expectations now come from independent
replays of the original implementations at their immutable Git commits,
using the current test interpreter and backend. The reference subprocess
rejects any import of `lab`. Initialization hashes, configurations and the
archived JSON files are preserved. The tests retain exact comparisons where
arithmetic is shared. Different optimizer operation orders are compared on
40 shared gradient updates in both float32 and float64, within 64 machine
epsilons, instead of using a backend-specific tolerance for a diverging
training trajectory. Details are in
`python/tests/fixtures/README.md`.

## Retained proof debt and relevance to the research goal

All 39 declarations below remain in `src/`. None has been deleted or
presented as proved. The Landscape framework itself has no proof holes.

The closest possible conceptual inputs are the two remaining Perceptron
atomicity extensions and Normalization's Gaussian clustering alternative.
The Perceptron holes concern degenerate ReLU cases; its analytic strict
critical-point results are already proved. Normalization concerns token
dynamics with identity attention parameters. Applying either development to
training the browser's tied Rotate90 classifier would require a new bridge
with the actual loss and optimizer. They currently supply no generalization
or gradient-accessibility theorem for that model.

The three Interpolation holes concern control and expressivity, so they may
help the later question about representations, but do not establish gradient
learning. Clustering, MeanField, Metastability and the GPTMini capstone
concern forward or depth dynamics of particular attention systems. The eight
Homogenized holes concern stochastic depth limits, SDEs and approximation
errors. The three Kinetic holes concern ALiBi token dynamics and correlations;
the ten Modes holes concern Gaussian kernel density modes. No current proof
of the Rotate90 analytic loss or of the generic Landscape convergence scheme
depends on these unfinished declarations.

| Declaration | Source module | Paper |
| --- | --- | --- |
| `subspace_cluster` | [src/Transformer/Causal/MainTheorem.lean](../src/Transformer/Causal/MainTheorem.lean) | 2411.04990v2 |
| `idCase_tendsto_zero_or_frontier_generic` | [src/Transformer/Clusters/Section12_Generic.lean](../src/Transformer/Clusters/Section12_Generic.lean) | 2305.05465v6 |
| `multiplicity_dist_tendsto_zero_generic` | [src/Transformer/Clusters/Section12_Generic.lean](../src/Transformer/Clusters/Section12_Generic.lean) | 2305.05465v6 |
| `boolean_tendsto_isBooleanRows_ae` | [src/Transformer/Clusters/Section2_LowRank.lean](../src/Transformer/Clusters/Section2_LowRank.lean) | 2305.05465v6 |
| `layer_clustering` | [src/Transformer/GPTMini/ClusteringTheorem.lean](../src/Transformer/GPTMini/ClusteringTheorem.lean) | 2411.04990v2 |
| `weak_error_centered` | [src/Transformer/Homogenized/GaussianInit.lean](../src/Transformer/Homogenized/GaussianInit.lean) | 2604.01978v1 |
| `large_beta_logistic_limit` | [src/Transformer/Homogenized/LogisticLimit.lean](../src/Transformer/Homogenized/LogisticLimit.lean) | 2604.01978v1 |
| `large_beta_metastability` | [src/Transformer/Homogenized/Metastability.lean](../src/Transformer/Homogenized/Metastability.lean) | 2604.01978v1 |
| `ballistic_regime` | [src/Transformer/Homogenized/Regimes.lean](../src/Transformer/Homogenized/Regimes.lean) | 2604.01978v1 |
| `modified_regime` | [src/Transformer/Homogenized/Regimes.lean](../src/Transformer/Homogenized/Regimes.lean) | 2604.01978v1 |
| `existence_SDE_sphere` | [src/Transformer/Homogenized/SphereSde.lean](../src/Transformer/Homogenized/SphereSde.lean) | 2604.01978v1 |
| `weak_error_clean` | [src/Transformer/Homogenized/WeakError.lean](../src/Transformer/Homogenized/WeakError.lean) | 2604.01978v1 |
| `exists_mckeanVlasovSolution` | [src/Transformer/Homogenized/WellPosed.lean](../src/Transformer/Homogenized/WellPosed.lean) | 2604.01978v1 |
| `compression` | [src/Transformer/Interpolation/Clustering.lean](../src/Transformer/Interpolation/Clustering.lean) | 2411.04551v3 |
| `main_result` | [src/Transformer/Interpolation/Main.lean](../src/Transformer/Interpolation/Main.lean) | 2411.04551v3 |
| `univ_approx` | [src/Transformer/Interpolation/Main.lean](../src/Transformer/Interpolation/Main.lean) | 2411.04551v3 |
| `lost_correlations` | [src/Transformer/Kinetic/Correlations.lean](../src/Transformer/Kinetic/Correlations.lean) | 2605.09213v1 |
| `mean_field_limit` | [src/Transformer/Kinetic/MeanField.lean](../src/Transformer/Kinetic/MeanField.lean) | 2605.09213v1 |
| `soft_accuracy_expansion` | [src/Transformer/Kinetic/SoftAccuracy.lean](../src/Transformer/Kinetic/SoftAccuracy.lean) | 2605.09213v1 |
| `global_clustering` | [src/Transformer/MeanField/Clustering.lean](../src/Transformer/MeanField/Clustering.lean) | 2512.01868v4 |
| `meanField_exponential_rate` | [src/Transformer/MeanField/GlobalRate.lean](../src/Transformer/MeanField/GlobalRate.lean) | 2512.01868v4 |
| `metastability_mf` | [src/Transformer/Metastability/MeanFieldMetastability.lean](../src/Transformer/Metastability/MeanFieldMetastability.lean) | 2410.06833v1 |
| `saddle_to_saddle_gradient_reparam` | [src/Transformer/Metastability/Section6_Profile.lean](../src/Transformer/Metastability/Section6_Profile.lean) | 2410.06833v1 |
| `staircase_profile` | [src/Transformer/Metastability/StaircaseProfile.lean](../src/Transformer/Metastability/StaircaseProfile.lean) | 2410.06833v1 |
| `mainResult_expectedModes` | [src/Transformer/Modes/Section1_Main.lean](../src/Transformer/Modes/Section1_Main.lean) | 2412.09080v3 |
| `mainResult_expectedModes_compl_belt` | [src/Transformer/Modes/Section1_Main.lean](../src/Transformer/Modes/Section1_Main.lean) | 2412.09080v3 |
| `mammen_crit` | [src/Transformer/Modes/Section1_Mammen.lean](../src/Transformer/Modes/Section1_Mammen.lean) | 2412.09080v3 |
| `mammen_mid` | [src/Transformer/Modes/Section1_Mammen.lean](../src/Transformer/Modes/Section1_Mammen.lean) | 2412.09080v3 |
| `mammen_gt` | [src/Transformer/Modes/Section1_Mammen.lean](../src/Transformer/Modes/Section1_Mammen.lean) | 2412.09080v3 |
| `prop_main_int_T` | [src/Transformer/Modes/Section1_Sketch.lean](../src/Transformer/Modes/Section1_Sketch.lean) | 2412.09080v3 |
| `prop_main_int_T'` | [src/Transformer/Modes/Section1_Sketch.lean](../src/Transformer/Modes/Section1_Sketch.lean) | 2412.09080v3 |
| `prop_main_tail` | [src/Transformer/Modes/Section1_Sketch.lean](../src/Transformer/Modes/Section1_Sketch.lean) | 2412.09080v3 |
| `error_higher_two_KR_T'` | [src/Transformer/Modes/Section3_ErrorKR.lean](../src/Transformer/Modes/Section3_ErrorKR.lean) | 2412.09080v3 |
| `error_higher_three_KR_T'` | [src/Transformer/Modes/Section3_ErrorKR.lean](../src/Transformer/Modes/Section3_ErrorKR.lean) | 2412.09080v3 |
| `clusters_or_stalls_from_gaussian` | [src/Transformer/Normalization/Clustering.lean](../src/Transformer/Normalization/Clustering.lean) | 2510.22026v2 |
| `ext_circle_isFinitelyAtomic` | [src/Transformer/Perceptron/GeneralAtomicity.lean](../src/Transformer/Perceptron/GeneralAtomicity.lean) | 2601.21366v2 |
| `ext_any_d_measure_support_eq_zero` | [src/Transformer/Perceptron/GeneralAtomicity.lean](../src/Transformer/Perceptron/GeneralAtomicity.lean) | 2601.21366v2 |
| `clustering_probability_small_beta` | [src/Transformer/Perspective/Section3_SmallBeta.lean](../src/Transformer/Perspective/Section3_SmallBeta.lean) | 2312.10794v5 |
| `sset0_probability_tendsto_one` | [src/Transformer/Perspective/Section3_SmallBeta.lean](../src/Transformer/Perspective/Section3_SmallBeta.lean) | 2312.10794v5 |
