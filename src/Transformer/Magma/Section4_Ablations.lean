/-
# Exact audits of the ablation tables

Formalization of arXiv:2602.15322v1, Appendix C, Tables
table:masking-component and table:masking-granularity. Printed decimal
scores are rationals. These are finite comparisons of the reported
observations; no causal or universal performance claim is inferred.
-/

import Transformer.Magma.Section4_C4Results

open scoped BigOperators

namespace Transformer.Magma

/-- Mask targets in Appendix C, Table table:masking-component. -/
inductive MaskTarget | baseline | attention | attentionMLP | allParameters
  deriving DecidableEq

/-- Reported target-specific validation perplexity. Source:
arXiv:2602.15322v1, Appendix C, Table table:masking-component. -/
def targetPerplexity : MaskTarget → ℚ
  | .baseline => 2264 / 100
  | .attention => 2192 / 100
  | .attentionMLP => 2165 / 100
  | .allParameters => 2194 / 100

/-- Targeting attention and MLP wins this finite ablation. Source:
arXiv:2602.15322v1, Appendix C, Masking Component. -/
theorem attention_mlp_best (target : MaskTarget) (h : target ≠ .attentionMLP) :
    targetPerplexity .attentionMLP < targetPerplexity target := by
  cases target <;> norm_num [targetPerplexity] at *

/-- The target-comparison hypothesis has a measured witness.
Source: arXiv:2602.15322v1, Appendix C, Masking Component. -/
example : MaskTarget.attention ≠ .attentionMLP := by decide

/-- Rows of the sampling/damping ablation in Appendix C,
Table table:masking-granularity. -/
inductive MaskVariant
  | uniform | dampingOnly | uniformDamping | alignment | alignmentDamping
  deriving DecidableEq

/-- Reported perplexity; indices 0--3 mean element, row, column, block.
Source: arXiv:2602.15322v1, Appendix C, Table table:masking-granularity. -/
def granularityPerplexity (variant : MaskVariant) (granularity : Fin 4) : ℚ :=
  (match variant with
  | .uniform => ![2173, 2176, 2178, 2181]
  | .dampingOnly => ![2197, 2195, 2191, 2192]
  | .uniformDamping => ![2158, 2162, 2161, 2165]
  | .alignment => ![2177, 2178, 2175, 2178]
  | .alignmentDamping => ![2163, 2160, 2161, 2165]) granularity / 100

/-- Every ablation cell improves the measured 22.64 RMSProp baseline.
Source: arXiv:2602.15322v1, Appendix C, Masking Granularity. -/
theorem all_ablation_cells_improve (variant : MaskVariant) (granularity : Fin 4) :
    granularityPerplexity variant granularity < 2264 / 100 := by
  cases variant <;> fin_cases granularity <;> norm_num [granularityPerplexity]

/-- Uniform sampling scores stay within the narrow printed interval;
this formalizes the observed near-equivalence across granularities,
without turning the proposed explanation about preconditioning into a
universal theorem. Source: arXiv:2602.15322v1, Section 2, Impacts of
structured masking, and Appendix C, Robustness Across Granularities. -/
theorem uniform_granularity_range (granularity : Fin 4) :
    2173 / 100 ≤ granularityPerplexity .uniform granularity ∧
      granularityPerplexity .uniform granularity ≤ 2181 / 100 := by
  fin_cases granularity <;> norm_num [granularityPerplexity]

/-- Damping alone is worse than uniform sampling, while adding it to
either sampling scheme improves every measured granularity. Source:
arXiv:2602.15322v1, Appendix C, Effectiveness of Damping and Sampling. -/
theorem sampling_damping_comparisons (granularity : Fin 4) :
    granularityPerplexity .uniform granularity < granularityPerplexity .dampingOnly granularity ∧
    granularityPerplexity .uniformDamping granularity < granularityPerplexity .uniform granularity ∧
    granularityPerplexity .alignmentDamping granularity < granularityPerplexity .alignment granularity := by
  fin_cases granularity <;> norm_num [granularityPerplexity]

/-- The smallest value in the twenty-cell ablation is 21.58 at element
granularity with uniform sampling plus damping. Source:
arXiv:2602.15322v1, Appendix C, Table table:masking-granularity. -/
theorem uniform_damping_element_minimum (variant : MaskVariant) (granularity : Fin 4) :
    granularityPerplexity .uniformDamping 0 ≤ granularityPerplexity variant granularity := by
  cases variant <;> fin_cases granularity <;> norm_num [granularityPerplexity]

/-- Its improvement over element-level uniform sampling is exactly 0.15
on the rounded table. Source: arXiv:2602.15322v1, Appendix C,
Effectiveness of Damping and Sampling. -/
theorem uniform_damping_element_gain :
    granularityPerplexity .uniform 0 - granularityPerplexity .uniformDamping 0 = 15 / 100 := by
  norm_num [granularityPerplexity]

/-- Alignment-based sampling has the same four-granularity arithmetic
mean as uniform sampling; column and block scores are actually better,
so the prose "does not outperform" cannot mean pointwise dominance.
Source: arXiv:2602.15322v1, Appendix C, final Masking Granularity paragraph. -/
theorem alignment_sampling_qualification :
    (∑ granularity : Fin 4, granularityPerplexity .uniform granularity) / 4 =
      (∑ granularity : Fin 4, granularityPerplexity .alignment granularity) / 4 ∧
    granularityPerplexity .alignment 2 < granularityPerplexity .uniform 2 ∧
    granularityPerplexity .alignment 3 < granularityPerplexity .uniform 3 := by
  norm_num [granularityPerplexity, Fin.sum_univ_succ]

/-- The paper also reports a setting without improvement: 94.46% for
the AdamW baseline versus 93.82% for Magma on ResNet-50/CIFAR-10.
This is only a comparison of that reported pair. Source:
arXiv:2602.15322v1, Section 4, Magma on Heterogeneous Quadratics. -/
theorem resnet_reported_accuracy_drop : (9382 / 100 : ℚ) < 9446 / 100 := by norm_num

end Transformer.Magma
