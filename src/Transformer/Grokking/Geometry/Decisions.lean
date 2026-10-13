import Transformer.Grokking.Geometry.Cells
import Transformer.Grokking.NaiveLoss.Section4_LogitScaling

/-!
# When a structured reference actually certifies decisions

Source: Nanda et al., arXiv:2301.05217v1, section 5.1, evaluates restricted
loss rather than assuming that a structural component is correct.
Deviation: these are finite-class margin certificates for the fixed-orbit
observer at 4436290, using actual logits and their computed residual.

A correct reference with a quantitative margin survives sufficiently small
error. Row shifts do not affect the result; class-dependent global bias
must be included in the reference instead of discarded. The error bound
is on absolute squared energy, not only its fraction of total energy.

The margin premise is an observable condition involving correct labels.
No learning dynamics, correct margin, or future cleanup is inferred from
projection geometry alone. Strict correctness excludes an argmax tie.
-/

namespace Transformer.Grokking.Geometry

open scoped BigOperators
open Transformer.Grokking.NaiveLoss

variable {C : Type*}

/-- Quantitative target margin of a reference logit vector. Source:
restricted-logit correctness in arXiv:2301.05217v1, section 5.1;
this adds an explicit quantitative condition to the orbit observer. -/
def MarginAtLeast (z : C → ℝ) (y : C) (margin : ℝ) : Prop :=
  ∀ k, k ≠ y → margin ≤ z y - z k

/-- A common row shift preserves strict predictions. Source: the
centering in the orbit observer at 4436290, adapting restricted logits
from arXiv:2301.05217v1, section 5.1. -/
theorem strictCorrect_shift_iff (z : C → ℝ) (y : C) (b : ℝ) :
    StrictCorrect (fun k => z k + b) y ↔ StrictCorrect z y := by
  constructor
  · intro h k hk
    have he := h k hk
    linarith
  · intro h k hk
    have he := h k hk
    linarith

/-- A reference margin larger than twice the uniform logit error certifies
the actual prediction. Source: correctness must be measured alongside
restricted logits in arXiv:2301.05217v1, section 5.1. -/
theorem strictCorrect_of_margin_and_uniform_error (z p : C → ℝ) (y : C)
    (margin error : ℝ) (hm : MarginAtLeast p y margin)
    (he : ∀ k, |z k - p k| ≤ error) (hg : 2 * error < margin) : StrictCorrect z y := by
  intro k hk
  have hmargin := hm k hk
  have hy := abs_le.mp (he y)
  have hother := abs_le.mp (he k)
  linarith

example : MarginAtLeast (fun b : Bool => if b then (3 : ℝ) else 0) true 2 ∧
    (∀ b : Bool, |(if b then (3 : ℝ) else 0) - (if b then (3 : ℝ) else 0)| ≤ 1 / 2) ∧
    2 * (1 / 2 : ℝ) < 2 := by
  refine ⟨?_, ?_, by norm_num⟩
  · intro b hb
    cases b
    · norm_num
    · exact False.elim (hb rfl)
  · intro b
    norm_num

/-- Two different class errors are both included in the full energy.
Source: all-vocabulary errors of the orbit observer at 4436290,
adapting arXiv:2301.05217v1, section 5.1. -/
theorem error_pair_le_energy [Fintype C] (r : C → ℝ) (y k : C) (hk : k ≠ y) :
    (r y) ^ 2 + (r k) ^ 2 ≤ energyOver Finset.univ r := by
  classical
  have hsub : ({y, k} : Finset C) ⊆ Finset.univ := by
    intro c hc
    exact Finset.mem_univ c
  have hle := Finset.sum_le_sum_of_subset_of_nonneg hsub
    (fun c hc hout => sq_nonneg (r c))
  have hy : y ∉ ({k} : Finset C) := by
    intro hmem
    have heq := Finset.mem_singleton.mp hmem
    exact hk heq.symm
  rw [Finset.sum_insert hy, Finset.sum_singleton] at hle
  exact hle

example : (false : Bool) ≠ true := by decide

/-- Absolute squared error below half the squared reference margin
certifies every competing-class comparison. Source: the restricted-logit
correctness check of arXiv:2301.05217v1, section 5.1; this derives a
quantitative cleanup criterion for the orbit observer at 4436290. -/
theorem strictCorrect_of_margin_and_energy [Fintype C] (z p : C → ℝ) (y : C)
    (margin : ℝ) (hm : MarginAtLeast p y margin) (hp : 0 < margin)
    (he : 2 * energyOver Finset.univ (fun k => z k - p k) < margin ^ 2) :
    StrictCorrect z y := by
  intro k hk
  by_contra hwrong
  have hmarg := hm k hk
  have hpair := error_pair_le_energy (fun j => z j - p j) y k hk
  have hsmall : 0 ≤ (z k - p k) - (z y - p y) - margin := by linarith
  have hlarge : 0 ≤ (z k - p k) - (z y - p y) + margin := by linarith
  have hprod : 0 ≤ ((z k - p k) - (z y - p y) - margin) *
      ((z k - p k) - (z y - p y) + margin) := by positivity
  nlinarith [sq_nonneg ((z k - p k) + (z y - p y))]

example : MarginAtLeast (fun b : Bool => if b then (3 : ℝ) else 0) true 2 ∧
    0 < (2 : ℝ) ∧ 2 * energyOver Finset.univ
      (fun b : Bool => (if b then (3 : ℝ) else 0) - (if b then (3 : ℝ) else 0)) < 2 ^ 2 := by
  refine ⟨?_, by norm_num, ?_⟩
  · intro b hb
    cases b
    · norm_num
    · exact False.elim (hb rfl)
  · norm_num [energyOver]

/-- A failed strict prediction against a positive-margin reference needs
at least half its squared margin in residual energy. Source: the cleanup
interpretation of restricted logits in arXiv:2301.05217v1, section 5.1. -/
theorem failed_decision_requires_energy [Fintype C] (z p : C → ℝ) (y : C)
    (margin : ℝ) (hm : MarginAtLeast p y margin) (hp : 0 < margin)
    (hw : ¬StrictCorrect z y) :
    margin ^ 2 ≤ 2 * energyOver Finset.univ (fun k => z k - p k) := by
  by_contra h
  have he : 2 * energyOver Finset.univ (fun k => z k - p k) < margin ^ 2 := by linarith
  exact hw (strictCorrect_of_margin_and_energy z p y margin hm hp he)

example : MarginAtLeast (fun b : Bool => if b then (3 : ℝ) else 0) true 2 ∧
    0 < (2 : ℝ) ∧ ¬StrictCorrect (fun b : Bool => if b then (0 : ℝ) else 3) true := by
  refine ⟨?_, by norm_num, ?_⟩
  · intro b hb
    cases b
    · norm_num
    · exact False.elim (hb rfl)
  · intro h
    have he := h false (by decide)
    norm_num at he

/-- The strict energy inequality is sharp: equality can give tied actual
logits against a margin-two reference. Source: the correctness condition
needed alongside restricted logits in arXiv:2301.05217v1, section 5.1;
this counterexample rejects replacing the cleanup criterion's `<` by `≤`. -/
theorem margin_energy_boundary_allows_tie :
    MarginAtLeast (fun b : Bool => if b then (1 : ℝ) else -1) true 2 ∧
    2 * energyOver Finset.univ (fun b : Bool => 0 - (if b then (1 : ℝ) else -1)) = 2 ^ 2 ∧
    ¬StrictCorrect (fun _ : Bool => (0 : ℝ)) true := by
  refine ⟨?_, ?_, ?_⟩
  · intro b hb
    cases b
    · norm_num
    · exact False.elim (hb rfl)
  · norm_num [energyOver, Fintype.sum_bool]
  · intro h
    have he := h false (by decide)
    norm_num at he

end Transformer.Grokking.Geometry
