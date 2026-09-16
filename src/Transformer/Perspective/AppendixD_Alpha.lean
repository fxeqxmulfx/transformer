/-
# Appendix D — the smallest coordinate `α(1/n)`

Geshkovski, Letrouit, Polyanskiy, Rigollet — arXiv:2312.10794v5,
*A mathematical perspective on Transformers*.

`e:1/n`, the estimate that starts the second half of Appendix D: at time `1/n`
the configuration is already within half of the orthogonal angle `γ_β(1/n)` of
one of its own particles.  It is proved here, and so is the fact that it holds
of *that* particle only and not of an arbitrary direction.

The stability estimates it rests on are in
`Perspective.AppendixD_PhaseTransition`, and what the survey builds on top of
it is in `Perspective.AppendixD_Assembly`.
-/

import Transformer.Perspective.AppendixD_PhaseTransition

open scoped BigOperators
open Real

namespace Transformer
namespace Perspective

variable (d n : ℕ)

/-- **Equation (e:1/n).**

  `α(1/n) ≥ (1/2) γ_β(1/n)`,  for a suitable `x⋆`.

At time `1/n` the configuration is already within half of the orthogonal angle
`γ_β(1/n)` of a direction `x⋆`, and a direction that works is one of the
particles itself, `x⋆ = x_j(1/n)`.

The hypotheses are the two halves of the survey's argument evaluated at the
single time `t = 1/n`: `hpair` is `e:ineqfirstpart` there, where the stability
factor `c(β)^{n · (1/n)}` is `c(β)`, and `hlarge` is `eq: d.large`, which is
exactly what makes the error `2 c(β) √(log d / d)` smaller than
`γ_β(1/n)/2`.  `hγle` records that `γ_β` is a cosine, so that the diagonal
`⟨x_j(1/n), x_j(1/n)⟩ = 1` clears the bound as well.

The survey states this for `n ≥ 2`; only `n ≥ 1` is needed, and only so that
there is a particle to point at.

Reading `x⋆` as universally quantified — which the survey's notation
`α(t) = min_i ⟨x_i(t), x⋆⟩` invites, since it never says which `x⋆` — makes
the statement false: `not_alpha_at_one_over_n_of_free` measures a consensus
configuration against its own antipode, where `α ≡ -1` while `γ_β(1/n) > 0`.

Source: arXiv:2312.10794v5, Appendix D, `e:1/n`. -/
theorem alpha_at_one_over_n
    (hn : 0 < n) (hd : 2 ≤ d) (β : ℝ) (γ : ℝ → ℝ) (X : ℝ → SphereTuple d n)
    (hγpos : 0 < γ ((n : ℝ)⁻¹)) (hγle : γ ((n : ℝ)⁻¹) ≤ 2)
    (hlarge : 16 * (cBeta β) ^ 2 / (γ ((n : ℝ)⁻¹)) ^ 2 ≤ (d : ℝ) / Real.log d)
    (hpair : ∀ i j : Idx n, i ≠ j →
      |inner (𝕜 := ℝ) ((X ((n : ℝ)⁻¹) i : EucSpace d))
            ((X ((n : ℝ)⁻¹) j : EucSpace d)) - γ ((n : ℝ)⁻¹)|
        ≤ 2 * cBeta β * Real.sqrt (Real.log d / d)) :
    ∃ (x_star : SSphere d) (α : ℝ → ℝ),
      IsMinInner d n X x_star α ∧ (1/2 : ℝ) * γ ((n : ℝ)⁻¹) ≤ α ((n : ℝ)⁻¹) := by
  have hne : (Finset.univ : Finset (Idx n)).Nonempty := ⟨⟨0, hn⟩, Finset.mem_univ _⟩
  refine ⟨X ((n : ℝ)⁻¹) ⟨0, hn⟩,
    fun s => Finset.univ.inf' hne (fun i =>
      inner (𝕜 := ℝ) ((X s i : EucSpace d)) ((X ((n : ℝ)⁻¹) ⟨0, hn⟩ : EucSpace d))),
    fun s => ⟨fun i => Finset.inf'_le _ (Finset.mem_univ i), ?_⟩, ?_⟩
  · obtain ⟨i, _, hi⟩ := Finset.exists_mem_eq_inf' hne (fun i =>
      inner (𝕜 := ℝ) ((X s i : EucSpace d)) ((X ((n : ℝ)⁻¹) ⟨0, hn⟩ : EucSpace d)))
    exact ⟨i, hi⟩
  · have hc1 : 1 ≤ cBeta β := one_le_cBeta β
    have hd2 : (2 : ℝ) ≤ (d : ℝ) := by exact_mod_cast hd
    have hdpos : (0 : ℝ) < (d : ℝ) := by linarith
    have hlog : 0 < Real.log d := Real.log_pos (by linarith)
    have hcross : 16 * cBeta β ^ 2 * Real.log d ≤ (d : ℝ) * γ ((n : ℝ)⁻¹) ^ 2 :=
      (div_le_div_iff₀ (by positivity) hlog).mp hlarge
    have hratio : Real.log d / d ≤ (γ ((n : ℝ)⁻¹) / (4 * cBeta β)) ^ 2 := by
      rw [div_pow, div_le_div_iff₀ hdpos (by positivity)]
      nlinarith
    have hsqrt : Real.sqrt (Real.log d / d) ≤ γ ((n : ℝ)⁻¹) / (4 * cBeta β) :=
      le_trans (Real.sqrt_le_sqrt hratio) (le_of_eq (Real.sqrt_sq (by positivity)))
    have herr : 2 * cBeta β * Real.sqrt (Real.log d / d) ≤ γ ((n : ℝ)⁻¹) / 2 := by
      have hmul := mul_le_mul_of_nonneg_left hsqrt
        (by positivity : (0 : ℝ) ≤ 2 * cBeta β)
      have hval : 2 * cBeta β * (γ ((n : ℝ)⁻¹) / (4 * cBeta β)) = γ ((n : ℝ)⁻¹) / 2 := by
        field_simp
        ring
      linarith [hval ▸ hmul]
    refine Finset.le_inf' hne _ fun i _ => ?_
    by_cases hij : i = ⟨0, hn⟩
    · rw [hij, real_inner_self_eq_norm_mul_norm,
        mem_sphere_zero_iff_norm.mp (X ((n : ℝ)⁻¹) ⟨0, hn⟩).2]
      linarith
    · linarith [(abs_le.mp (hpair i ⟨0, hn⟩ hij)).1]

/-- The hypotheses of `alpha_at_one_over_n` are satisfiable, and the estimate
it yields is not vacuous: one particle, where `hpair` quantifies over an empty
range, `γ ≡ 1`, and any dimension past the threshold `exists_le_div_log`
produces for `eq: d.large`. -/
example (β : ℝ) :
    ∃ m : ℕ, ∃ (x_star : SSphere (m + 2)) (α : ℝ → ℝ),
      IsMinInner (m + 2) 1 (fun _ _ => basePoint (m + 1)) x_star α ∧
        (1/2 : ℝ) * 1 ≤ α (((1 : ℕ) : ℝ)⁻¹) := by
  obtain ⟨D, hD⟩ := exists_le_div_log (16 * (cBeta β) ^ 2 / (1 : ℝ) ^ 2)
  exact ⟨D, alpha_at_one_over_n (D + 2) 1 one_pos (by omega) β (fun _ => 1)
    (fun _ _ => basePoint (D + 1)) one_pos one_le_two (hD _ (by omega))
    (fun i j hij => absurd (Subsingleton.elim i j) hij)⟩

/-- **`e:1/n` is a statement about one `x⋆`, not about every one.**

Read with `x⋆` quantified universally, the estimate fails at the simplest
configuration there is: a single particle standing still, measured against its
own antipode.  There `α ≡ -1`, while the closed-form solution of `eq: ybeta`
at `n = 1`, `β = 0` has `γ_β(1) = 1 - e^{-2} > 0`.

Source: arXiv:2312.10794v5, Appendix D, `e:1/n`. -/
theorem not_alpha_at_one_over_n_of_free :
    ¬ ∀ (d n : ℕ) (β : ℝ) (γ α : ℝ → ℝ) (X : ℝ → SphereTuple d n)
        (x_star : SSphere d),
        SA d n β X → ybetaODE_SA n β γ → IsMinInner d n X x_star α →
          (1/2 : ℝ) * γ ((n : ℝ)⁻¹) ≤ α ((n : ℝ)⁻¹) := by
  intro h
  have hx : ‖((basePoint 0 : SSphere 1) : EucSpace 1)‖ = 1 :=
    mem_sphere_zero_iff_norm.mp (basePoint 0).2
  have hxx : inner (𝕜 := ℝ) (((basePoint 0 : SSphere 1)) : EucSpace 1)
      (((basePoint 0 : SSphere 1)) : EucSpace 1) = 1 := by
    rw [real_inner_self_eq_norm_mul_norm, hx]; ring
  have hmin : IsMinInner 1 1 (fun _ _ => basePoint 0) (antipode 1 (basePoint 0))
      (fun _ => -1) := by
    have hval : inner (𝕜 := ℝ) (((basePoint 0 : SSphere 1)) : EucSpace 1)
        ((antipode 1 (basePoint 0) : EucSpace 1)) = -1 := by
      show inner (𝕜 := ℝ) (((basePoint 0 : SSphere 1)) : EucSpace 1)
        (-(((basePoint 0 : SSphere 1)) : EucSpace 1)) = -1
      rw [inner_neg_right, hxx]
    exact fun _ => ⟨fun _ => le_of_eq hval.symm, ⟨0, hval.symm⟩⟩
  have hineq := h 1 1 0 (fun t => 1 - Real.exp (-2 * t)) (fun _ => -1)
    (fun _ _ => basePoint 0) (antipode 1 (basePoint 0))
    (SA_const_consensus 1 1 one_pos 0 (basePoint 0)) ybetaODE_SA_one_zero hmin
  have hlt : Real.exp (-2 : ℝ) < 1 := Real.exp_lt_one_iff.mpr (by norm_num)
  norm_num at hineq
  linarith

end Perspective
end Transformer
