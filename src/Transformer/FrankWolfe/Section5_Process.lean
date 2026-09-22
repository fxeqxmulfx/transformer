/-
# Attention's forward pass and Frank-Wolfe — finite `β`

Formalization of Alcalde, Geshkovski, Ruiz-Balet — arXiv:2508.09628v1,
*Attention's forward pass and Frank-Wolfe*, §5, up to the Gumbel trick.

At finite `β` the hardmax step is replaced by a softmax one, `eq: softmax.ODE`,
whose weights are the attention scores (`attWeight`); the Gumbel trick
rewrites the same step as a Markov chain `eq: softmax.process`, whose jump
probabilities are those scores.  That `eq: softmax.ODE` itself collapses to a
single point (`prop: origin`) is `Section5_Collapse`.
-/

import Transformer.FrankWolfe.Section1_Models
import Mathlib.MeasureTheory.Constructions.Pi
import Mathlib.MeasureTheory.Integral.Bochner.Basic
import Mathlib.MeasureTheory.Measure.Haar.InnerProductSpace

open scoped BigOperators
open Real MeasureTheory

namespace Transformer
namespace FrankWolfe

variable {d n : ℕ}

/-- The attention score particle `i` gives particle `j` at inverse temperature
`β`, for the key-query matrix `B = I_d`:
`e^{β⟨x_i, x_j⟩} / Σ_k e^{β⟨x_i, x_k⟩}`.

Source: arXiv:2508.09628v1, §5, `eq: softmax.process`. -/
noncomputable def attWeight (β : ℝ) (X : Idx n → EucSpace d) (i j : Idx n) : ℝ :=
  Real.exp (β * inner (𝕜 := ℝ) (X i) (X j)) /
    ∑ k : Idx n, Real.exp (β * inner (𝕜 := ℝ) (X i) (X k))

theorem attWeight_pos (β : ℝ) (X : Idx n → EucSpace d) (i j : Idx n) :
    0 < attWeight β X i j :=
  div_pos (Real.exp_pos _) (Finset.sum_pos (fun _ _ => Real.exp_pos _) ⟨i, Finset.mem_univ i⟩)

/-- The scores of one particle are a probability vector. -/
theorem sum_attWeight (β : ℝ) (X : Idx n → EucSpace d) (i : Idx n) :
    ∑ j : Idx n, attWeight β X i j = 1 := by
  unfold attWeight
  rw [← Finset.sum_div]
  exact div_self (Finset.sum_pos (fun _ _ => Real.exp_pos _) ⟨i, Finset.mem_univ i⟩).ne'

theorem attWeight_le_one (β : ℝ) (X : Idx n → EucSpace d) (i j : Idx n) :
    attWeight β X i j ≤ 1 :=
  (Finset.single_le_sum (f := attWeight β X i) (fun k _ => (attWeight_pos β X i k).le)
    (Finset.mem_univ j)).trans_eq (sum_attWeight β X i)

/-- **The self-attention process `eq: softmax.process` (SA_ℙ).**

`P` is the law, on path space `(ℝ^d)^{n × ℕ}`, of the Markov chain that starts
at `X₀` and, at every step and for every particle `i`, replaces `x_i^t` by
`(1-γ) x_i^t + γ x_j^t` with probability `attWeight β (x^t) i j`.

The Markov property and the transition law are recorded as one condition: for
every event `S` depending only on the trajectory up to time `t`, and every
measurable `A`,

  `ℙ[S ∩ {x_i^{t+1} ∈ A}] = ∫_S Σ_j attWeight β (x^t) i j · 𝟙_A((1-γ)x_i^t + γ x_j^t) dℙ`.

Taking `S` to be the whole space and `A` a point gives the displayed
transition probability of the source; taking `S` in the past says that the
jump depends on the past only through `x^t`, which is the source's "clearly
this process is a Markov chain".  The law of `x_i^{t+1}` is written as a
kernel, not event by event: when two particles `j ≠ j'` sit at the same point,
the events `{x_i^{t+1} = (1-γ)x_i^t + γ x_j^t}` for `j` and `j'` coincide and
carry `attWeight … j + attWeight … j'`, which a condition per `j` would get
wrong.  The source constrains each particle separately and says nothing of the
joint law of one step, so neither is anything said here.

This is a predicate of `β`, `γ`, `X₀` and `P`, not a claim: it says which
measures on path space are the process, and it is `P` that theorems quantify
over.

Source: arXiv:2508.09628v1, §5, `eq: softmax.process`. -/
def IsSAProcess (β γ : ℝ) (X₀ : Idx n → EucSpace d)
    (P : Measure (ℕ → Idx n → EucSpace d)) : Prop :=
  IsProbabilityMeasure P ∧
  (∀ᵐ x ∂P, x 0 = X₀) ∧
  ∀ (t : ℕ) (i : Idx n) (A : Set (EucSpace d)) (S : Set (ℕ → Idx n → EucSpace d)),
    MeasurableSet A → MeasurableSet S →
    (∀ x y : ℕ → Idx n → EucSpace d, (∀ s ≤ t, x s = y s) → (x ∈ S ↔ y ∈ S)) →
    (P (S ∩ {x | x (t + 1) i ∈ A})).toReal
      = ∫ x in S, ∑ j : Idx n,
          attWeight β (x t) i j * A.indicator (fun _ => (1 : ℝ)) ((1 - γ) • x t i + γ • x t j) ∂P

/-- A single particle attends only to itself, so it never moves, and the
process is the point mass at the constant path. -/
theorem isSAProcess_single (β γ : ℝ) (X₀ : Idx 1 → EucSpace d) :
    IsSAProcess β γ X₀ (Measure.dirac fun _ => X₀) := by
  classical
  refine ⟨inferInstance, ?_, fun t i A S hA hS _ => ?_⟩
  · exact (ae_dirac_iff (measurableSet_eq_fun (measurable_pi_apply 0) measurable_const)).mpr rfl
  have hmeas : MeasurableSet {x : ℕ → Idx 1 → EucSpace d | x (t + 1) i ∈ A} :=
    hA.preimage ((measurable_pi_apply i).comp (measurable_pi_apply (t + 1)))
  have hw : attWeight β X₀ i i = 1 := by
    unfold attWeight
    rw [Fintype.sum_subsingleton _ i]
    exact div_self (Real.exp_pos _).ne'
  rw [Measure.dirac_apply' _ (hS.inter hmeas), restrict_dirac' hS]
  by_cases hX : (fun _ : ℕ => X₀) ∈ S
  · rw [ite_eq_left hX, integral_dirac, Fintype.sum_subsingleton _ i, hw, ← add_smul, sub_add_cancel,
      one_smul, one_mul]
    by_cases hXA : X₀ i ∈ A <;> simp [Set.indicator, hX, hXA]
  · rw [ite_eq_right hX]
    simp [Set.indicator, hX]

end FrankWolfe
end Transformer
