/-
# Appendix D — the smallest coordinate `α(1/n)`

Geshkovski, Letrouit, Polyanskiy, Rigollet — arXiv:2312.10794v5,
*A mathematical perspective on Transformers*.

`e:1/n`, the estimate that starts the second half of Appendix D: at time `1/n`
every particle is already within half of the orthogonal angle `γ_β(1/n)` of
the limit `x⋆` of the dynamics.  It is proved here, from the cone of step 2 of
`lem: hemisphere.clustering` (`Perspective.limit_mem_cone`).

The stability estimates it rests on are in
`Perspective.AppendixD_PhaseTransition`, and what the survey builds on top of
it is in `Perspective.AppendixD_Assembly`.
-/

import Transformer.Perspective.AppendixD_PhaseTransition
import Transformer.Perspective.Section5_HemisphereCone

open scoped BigOperators
open Real

namespace Transformer
namespace Perspective

variable (d n : ℕ)

/-- **Equation (e:1/n).**

  `α(1/n) ≥ (1/2) γ_β(1/n)`,  `α(t) = min_i ⟨x_i(t), x⋆⟩`,

where `x⋆` is the common limit of the particles, which exists by
`lem: hemisphere.clustering`.

The hypotheses are the survey's setting and the two halves of its argument
evaluated at the single time `t = 1/n`.  `hw` and `hlim` say that `x⋆` is the
limit of `lem: hemisphere.clustering`: the particles start in the open
hemisphere around `w` and converge to `x⋆`.  `hpair` is `e:ineqfirstpart` at
`t = 1/n`, where the stability factor `c(β)^{n · (1/n)}` is `c(β)`, and
`hlarge` is `eq: d.large`, which is exactly what makes the error
`2 c(β) √(log d / d)` smaller than `γ_β(1/n)/2`.  `hγle` records that `γ_β` is
a cosine, so that the diagonal `⟨x_j(1/n), x_j(1/n)⟩ = 1` clears the bound as
well.

The proof is the survey's: `x⋆` lies in the cone of the particles at time
`1/n` (`limit_mem_cone`), so `⟨x_i(1/n), x⋆⟩` is at least the smallest
`⟨x_i(1/n), x_j(1/n)⟩`, once that is non-negative.

The survey states this for `n ≥ 2`; only `n ≥ 1` is needed, and only so that
there is a particle.

Source: arXiv:2312.10794v5, Appendix D, `e:1/n`. -/
theorem alpha_at_one_over_n
    (hn : 0 < n) (hd : 2 ≤ d) (β : ℝ) (γ : ℝ → ℝ) (X : ℝ → SphereTuple d n)
    (hX : SA d n β X) (w : SSphere d)
    (hw : ∀ i : Idx n, 0 < inner (𝕜 := ℝ) ((X 0 i : EucSpace d)) ((w : EucSpace d)))
    (x_star : SSphere d)
    (hlim : ∀ i : Idx n, Filter.Tendsto (fun s => (X s i : EucSpace d)) Filter.atTop
      (nhds (x_star : EucSpace d)))
    (α : ℝ → ℝ) (hα : IsMinInner d n X x_star α)
    (hγpos : 0 < γ ((n : ℝ)⁻¹)) (hγle : γ ((n : ℝ)⁻¹) ≤ 2)
    (hlarge : 16 * (cBeta β) ^ 2 / (γ ((n : ℝ)⁻¹)) ^ 2 ≤ (d : ℝ) / Real.log d)
    (hpair : ∀ i j : Idx n, i ≠ j →
      |inner (𝕜 := ℝ) ((X ((n : ℝ)⁻¹) i : EucSpace d))
            ((X ((n : ℝ)⁻¹) j : EucSpace d)) - γ ((n : ℝ)⁻¹)|
        ≤ 2 * cBeta β * Real.sqrt (Real.log d / d)) :
    (1/2 : ℝ) * γ ((n : ℝ)⁻¹) ≤ α ((n : ℝ)⁻¹) := by
  have hc1 : 1 ≤ cBeta β := one_le_cBeta β
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
  -- every inner product at time `1/n` is at least `γ_β(1/n)/2`
  have hall : ∀ i j : Idx n, γ ((n : ℝ)⁻¹) / 2 ≤
      inner (𝕜 := ℝ) ((X ((n : ℝ)⁻¹) i : EucSpace d)) ((X ((n : ℝ)⁻¹) j : EucSpace d)) := by
    intro i j
    by_cases hij : i = j
    · rw [hij, real_inner_self_eq_norm_mul_norm,
        mem_sphere_zero_iff_norm.mp (X ((n : ℝ)⁻¹) j).2]
      linarith
    · linarith [(abs_le.mp (hpair i j hij)).1]
  -- the cone of step 2 at time `1/n`, and `e:mineqalpha` there
  obtain ⟨η, ⟨hη0, hη1⟩, hcone⟩ :=
    limit_mem_cone d n hn β w X hX hw x_star hlim _ (inv_pos.mpr (Nat.cast_pos.mpr hn)).le
  obtain ⟨i, hi⟩ := (hα ((n : ℝ)⁻¹)).2
  obtain ⟨j, hj⟩ := exists_inner_le_of_mem_convexHull d n hn (X ((n : ℝ)⁻¹)) _ hcone i
  rw [real_inner_smul_right] at hj
  have := hall i j
  rw [hi]
  nlinarith

/-- The hypotheses of `alpha_at_one_over_n` are satisfiable, and the estimate
it yields is not vacuous: one particle at rest, which is its own limit and its
own hemisphere, where `hpair` quantifies over an empty range, `γ ≡ 1`, and any
dimension past the threshold `exists_le_div_log` produces for `eq: d.large`. -/
example (β : ℝ) :
    ∃ D : ℕ, 16 * (cBeta β) ^ 2 / (1 : ℝ) ^ 2 ≤ ((D + 2 : ℕ) : ℝ) / Real.log ((D + 2 : ℕ) : ℝ) ∧
      (1/2 : ℝ) * 1 ≤ (fun _ : ℝ => (1 : ℝ)) (((1 : ℕ) : ℝ)⁻¹) := by
  obtain ⟨D, hD⟩ := exists_le_div_log (16 * (cBeta β) ^ 2 / (1 : ℝ) ^ 2)
  have hx : ‖((basePoint (D + 1) : SSphere (D + 2)) : EucSpace (D + 2))‖ = 1 :=
    mem_sphere_zero_iff_norm.mp (basePoint (D + 1)).2
  have hxx : inner (𝕜 := ℝ) (((basePoint (D + 1) : SSphere (D + 2))) : EucSpace (D + 2))
      (((basePoint (D + 1) : SSphere (D + 2))) : EucSpace (D + 2)) = 1 := by
    rw [real_inner_self_eq_norm_mul_norm, hx]; ring
  exact ⟨D, hD _ (by omega), alpha_at_one_over_n (D + 2) 1 one_pos (by omega) β (fun _ => 1)
    (fun _ _ => basePoint (D + 1)) (SA_const_consensus (D + 2) 1 one_pos β _)
    (basePoint (D + 1)) (fun _ => by rw [hxx]; norm_num) (basePoint (D + 1))
    (fun _ => tendsto_const_nhds) (fun _ => 1)
    (fun _ => ⟨fun _ => le_of_eq hxx.symm, ⟨0, hxx.symm⟩⟩)
    one_pos one_le_two (hD _ (by omega))
    (fun i j hij => absurd (Subsingleton.elim i j) hij)⟩

end Perspective
end Transformer
