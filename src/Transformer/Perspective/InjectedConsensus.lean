/-
# Softmax attention with injection does not collapse

**Not a statement of any paper.**  `IsInjectedFlow.spread` for `injectedODE`:
if two of the injections, `z_k` and `z_l`, are linearly independent, the
tokens cannot stay within `δ` of each other for a time `T`, where `δ` and `T`
depend on `z_k` and `z_l` only — not on `β`, not on `Q(t)`, `K(t)`, not on the
initial tokens (`injectedODE_spread`).  So they reach no consensus
(`injectedODE_not_consensus`), and in particular do not converge to a point at
an exponential rate (`not_expConvergent_injectedODE`), which is what they do
at `z = 0` from any open hemisphere (`expConvergent_injectedODE_zero`: cone
collapse, `lem: hemisphere.clustering` of arXiv:2312.10794v5, §6.1).  The
independence cannot be dropped: parallel injections `z_i = c_i x` let all
tokens rest at `x` (`injectedODE_consensus`).
-/

import Transformer.Perspective.InjectedAttention

open scoped BigOperators
open Filter Topology

namespace Transformer
namespace Perspective

variable {d n : ℕ}

/-- **Theorem (no collapse under injection, for softmax attention).**  Let
`z_k`, `z_l` be linearly independent.  There are `δ, T > 0` such that for every
`β`, all `Q(t)`, `K(t)` and every solution of `injectedODE`, every time window
of length `T` contains a time at which two tokens are more than `δ` apart.

Source: none — posed here; `IsInjectedFlow.spread` at `A = 1`, the row sum of
the attention matrix `eq:P` (arXiv:2312.10794v5, §2.2). -/
theorem injectedODE_spread {z : Idx n → EucSpace d} {k l : Idx n}
    (hz : LinearIndependent ℝ ![z k, z l]) :
    ∃ δ T : ℝ, 0 < δ ∧ 0 < T ∧
      ∀ (β : ℝ) (Q K : TimeParam d) (X : ℝ → SphereTuple d n), injectedODE d n β Q K z X →
        ∀ t₀ : ℝ, ∃ t ∈ Set.Icc t₀ (t₀ + T), ∃ i j : Idx n,
          δ < ‖(X t i : EucSpace d) - X t j‖ := by
  obtain ⟨δ, T, hδ, hT, h⟩ := IsInjectedFlow.spread hz zero_le_one
  exact ⟨δ, T, hδ, hT, fun β Q K X hX =>
    h X _ (fun t i => (sum_abs_attention β Q K X t i).le) hX.isInjectedFlow⟩

/-- **No consensus under injection.**  If two injections are linearly
independent, the distances between the tokens of a solution of `injectedODE`
do not all tend to `0`.

Source: none — posed here; the negation, for `injectedODE`, of the consensus
that `lem: hemisphere.clustering` (arXiv:2312.10794v5, §6.1) proves at
`z = 0`. -/
theorem injectedODE_not_consensus {z : Idx n → EucSpace d} {k l : Idx n}
    (hz : LinearIndependent ℝ ![z k, z l]) {β : ℝ} {Q K : TimeParam d}
    {X : ℝ → SphereTuple d n} (hX : injectedODE d n β Q K z X) :
    ¬ ∀ i j : Idx n, Tendsto (fun t => ‖(X t i : EucSpace d) - X t j‖) atTop (𝓝 0) := by
  intro h
  obtain ⟨δ, T, hδ, -, hs⟩ := injectedODE_spread hz
  have hev : ∀ᶠ t in atTop, ∀ i j : Idx n, ‖(X t i : EucSpace d) - X t j‖ < δ :=
    eventually_all.2 fun i => eventually_all.2 fun j => (h i j).eventually (gt_mem_nhds hδ)
  obtain ⟨t₁, ht₁⟩ := eventually_atTop.1 hev
  obtain ⟨t, ht, i, j, hij⟩ := hs β Q K X hX t₁
  linarith [ht₁ t ht.1 i j]

/-- **No exponential collapse under injection.**  If two injections are
linearly independent, the conclusion of `lem: hemisphere.clustering` fails for
`injectedODE` from every initial configuration it has a solution from — among
them those in an open hemisphere, from which the same dynamics at `z = 0`
collapses (`expConvergent_injectedODE_zero`).

Source: none — posed here; the negation, for `injectedODE`, of the conclusion
of `lem: hemisphere.clustering` (arXiv:2312.10794v5, §6.1). -/
theorem not_expConvergent_injectedODE {z : Idx n → EucSpace d} {k l : Idx n}
    (hz : LinearIndependent ℝ ![z k, z l]) {β : ℝ} {Q K : TimeParam d}
    {X : ℝ → SphereTuple d n} (hX : injectedODE d n β Q K z X) :
    ¬ ExpConvergent d n (injectedODE d n β Q K z) (X 0) := by
  rintro ⟨x_star, C, lam, -, hlam, hconv⟩
  refine injectedODE_not_consensus hz hX fun i j => ?_
  have hlim : Tendsto (fun t => 2 * C * Real.exp (-(lam * t))) atTop (𝓝 0) := by
    simpa using (Real.tendsto_exp_neg_atTop_nhds_zero.comp
      (tendsto_id.const_mul_atTop hlam)).const_mul (2 * C)
  refine squeeze_zero' (Eventually.of_forall fun t => norm_nonneg _)
    (eventually_atTop.2 ⟨0, fun t ht => ?_⟩) hlim
  have h1 := hconv X rfl hX i t ht
  have h2 := hconv X rfl hX j t ht
  calc ‖(X t i : EucSpace d) - X t j‖
      = ‖((X t i : EucSpace d) - x_star) - ((X t j : EucSpace d) - x_star)‖ := by
        rw [sub_sub_sub_cancel_right]
    _ ≤ ‖(X t i : EucSpace d) - x_star‖ + ‖(X t j : EucSpace d) - x_star‖ := norm_sub_le _ _
    _ ≤ 2 * C * Real.exp (-(lam * t)) := by linarith

/-- The hypotheses of `injectedODE_spread`, `injectedODE_not_consensus` and
`not_expConvergent_injectedODE` are satisfiable together: at `β = 0` the unit
vectors `e₀`, `e₁` of `ℝ²` stay at rest under the independent injections
`z₀ = -e₁/2`, `z₁ = -e₀/2`, since each drive `(e₀ + e₁)/2 + z_i = e_i/2` is
normal to the sphere at `e_i`. -/
example : ∃ (z : Idx 2 → EucSpace 2) (X : ℝ → SphereTuple 2 2),
    LinearIndependent ℝ ![z 0, z 1] ∧ ∀ Q K : TimeParam 2, injectedODE 2 2 0 Q K z X := by
  refine ⟨![-(1 / 2 : ℝ) • EuclideanSpace.single 1 1, -(1 / 2 : ℝ) • EuclideanSpace.single 0 1],
    fun _ => ![basePoint 1, ⟨EuclideanSpace.single 1 1, by simp [PiLp.norm_single]⟩], ?_,
    fun Q K => injectedODE_const fun t i => ⟨1 / 2, ?_⟩⟩
  · refine linearIndependent_of_ne_zero_of_inner_eq_zero (fun i => ?_) fun i j hij => ?_
    · fin_cases i <;> simp
    · fin_cases i <;> fin_cases j <;>
        simp_all [inner_smul_left, inner_smul_right, EuclideanSpace.inner_single_left]
  · simp only [partitionQKV, zero_mul, Real.exp_zero, one_smul, Fin.sum_univ_two]
    fin_cases i <;> simp [basePoint] <;> module

end Perspective
end Transformer
