/-
# Exact arithmetic audits of the C4 table

Formalization of arXiv:2602.15322v1, Section 4,
Table table:benchmark_c4. Values are the printed, rounded perplexities,
represented as rationals. These theorems concern the reported table,
not a reproduction of training or universal optimizer superiority.
-/

import Mathlib.Tactic

namespace Transformer.Magma

/-- The eleven methods with results at all four model sizes.
The incomplete RMSProp row is represented separately. Source:
arXiv:2602.15322v1, Section 4, Table table:benchmark_c4. -/
inductive C4CompleteMethod
  | adam | cautiousAdam | adamSGG | adamMagma | laprop | lapropMagma
  | adafactor | apollo | apolloSGG | muon | rmspropMagma
  deriving DecidableEq

/-- Reported validation perplexity; indices 0--3 are 60M, 130M, 350M,
and 1B. Source: arXiv:2602.15322v1, Table table:benchmark_c4. -/
def c4Perplexity (method : C4CompleteMethod) (scale : Fin 4) : ℚ :=
  (match method with
  | .adam => ![3079, 2477, 1842, 1635]
  | .cautiousAdam => ![2970, 2359, 1858, 1592]
  | .adamSGG => ![3031, 2218, 1728, 1430]
  | .adamMagma => ![2909, 2208, 1641, 1371]
  | .laprop => ![2998, 2307, 1856, 1638]
  | .lapropMagma => ![2905, 2216, 1637, 1382]
  | .adafactor => ![3257, 2398, 1774, 1519]
  | .apollo => ![3155, 2294, 1685, 1420]
  | .apolloSGG => ![3018, 2252, 1654, 1395]
  | .muon => ![2893, 2234, 1709, 1452]
  | .rmspropMagma => ![2855, 2166, 1616, 1319]) scale / 100

/-- RMSProp's diverged 1B entry is absent, not a fictitious large value.
Source: arXiv:2602.15322v1, Table table:benchmark_c4 and its caption. -/
def c4RMSProp (scale : Fin 4) : Option ℚ :=
  ![some (2929 / 100), some (2264 / 100), some (1747 / 100), none] scale

/-- RMSProp+Magma is strictly best among every complete measured row at
each model scale. Source: arXiv:2602.15322v1, Section 4,
Table table:benchmark_c4. This proves a finite table comparison only. -/
theorem c4_rmsprop_magma_best (method : C4CompleteMethod) (scale : Fin 4)
    (hmethod : method ≠ .rmspropMagma) :
    c4Perplexity .rmspropMagma scale < c4Perplexity method scale := by
  cases method <;> fin_cases scale <;> norm_num [c4Perplexity] at *

/-- An alternative complete method satisfies the ranking hypothesis.
Source: arXiv:2602.15322v1, Table table:benchmark_c4. -/
example : C4CompleteMethod.adam ≠ .rmspropMagma := by decide

/-- RMSProp+Magma also improves every reported RMSProp entry.
Source: arXiv:2602.15322v1, Table table:benchmark_c4. -/
theorem c4_rmsprop_magma_improves (scale : Fin 4) (value : ℚ)
    (hreported : c4RMSProp scale = some value) :
    c4Perplexity .rmspropMagma scale < value := by
  fin_cases scale <;> norm_num [c4RMSProp, c4Perplexity] at * <;> linarith

/-- The incomplete row has genuine reported entries.
Source: arXiv:2602.15322v1, Table table:benchmark_c4. -/
example : c4RMSProp 0 = some (2929 / 100) := by norm_num [c4RMSProp]

/-- Magma improves Adam and LaProp at each reported scale. Source:
arXiv:2602.15322v1, Section 4, Table table:benchmark_c4. -/
theorem c4_adam_laprop_improve (scale : Fin 4) :
    c4Perplexity .adamMagma scale < c4Perplexity .adam scale ∧
      c4Perplexity .lapropMagma scale < c4Perplexity .laprop scale := by
  fin_cases scale <;> norm_num [c4Perplexity]

/-- Among Adam's reported enhancers, Magma beats Cautious Adam and SGG
at all four scales. Source: arXiv:2602.15322v1, Section 4,
Table table:benchmark_c4 and the paragraph discussing enhancers. -/
theorem c4_adam_magma_beats_enhancers (scale : Fin 4) :
    c4Perplexity .adamMagma scale < c4Perplexity .cautiousAdam scale ∧
      c4Perplexity .adamMagma scale < c4Perplexity .adamSGG scale := by
  fin_cases scale <;> norm_num [c4Perplexity]

/-- The reported relative reductions of Adam and LaProp increase with
model scale. This is a finite-table fact, not a general scaling law.
Source: arXiv:2602.15322v1, Section 4, favorable-scaling paragraph. -/
theorem c4_reported_relative_gain_increases (small large : Fin 4) (h : small < large) :
    (c4Perplexity .adam small - c4Perplexity .adamMagma small) / c4Perplexity .adam small <
      (c4Perplexity .adam large - c4Perplexity .adamMagma large) / c4Perplexity .adam large ∧
    (c4Perplexity .laprop small - c4Perplexity .lapropMagma small) / c4Perplexity .laprop small <
      (c4Perplexity .laprop large - c4Perplexity .lapropMagma large) / c4Perplexity .laprop large := by
  fin_cases small <;> fin_cases large <;> norm_num [c4Perplexity] at *

/-- Distinct ordered model scales occur in the reported table.
Source: arXiv:2602.15322v1, Table table:benchmark_c4. -/
example : (0 : Fin 4) < 1 := by decide

/-- The headline reductions at 1B exceed 19% relative to Adam and 9%
relative to Muon on the rounded table. Source: arXiv:2602.15322v1,
abstract and Section 4, Table table:benchmark_c4. -/
theorem c4_headline_improvements :
    (c4Perplexity .adam 3 - c4Perplexity .rmspropMagma 3) /
      c4Perplexity .adam 3 > 19 / 100 ∧
    (c4Perplexity .muon 3 - c4Perplexity .rmspropMagma 3) /
      c4Perplexity .muon 3 > 9 / 100 := by
  norm_num [c4Perplexity]

/-- The missing 1B entry stays missing. Source: arXiv:2602.15322v1,
Table table:benchmark_c4, divergence within the learning-rate search. -/
theorem c4_rmsprop_1B_missing : c4RMSProp 3 = none := rfl

/-- The prose says Adam+Magma at 1B is 13.81, while the table says 13.71.
The formalized numerical inventory follows the table and records this
discrepancy rather than silently changing either source value. Source:
arXiv:2602.15322v1, Section 4, paragraph after Table table:benchmark_c4. -/
theorem c4_adam_magma_prose_discrepancy :
    c4Perplexity .adamMagma 3 = 1371 / 100 ∧
      c4Perplexity .adamMagma 3 ≠ 1381 / 100 := by
  norm_num [c4Perplexity]

end Transformer.Magma
