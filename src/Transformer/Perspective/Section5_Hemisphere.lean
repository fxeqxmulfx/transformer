/-
# §6.1, step 1 — the minimum along `w` does not decrease

Geshkovski, Letrouit, Polyanskiy, Rigollet — arXiv:2312.10794v5,
*A mathematical perspective on Transformers*, §6.1, step 1 of
`lem: hemisphere.clustering`.

If all the particles start in the open hemisphere `⟨x, w⟩ > 0`, none of them
ever leaves it: `r(t) = min_i ⟨x_i(t), w⟩` is non-decreasing on `ℝ_{≥0}`.

The proof is the one the survey sketches, in two halves.  `step1_deriv_nonneg`
is the algebraic half: at an index `i` attaining the minimum, every bracket of
`eq: therighthandside` is non-negative, because `⟨x_j, w⟩ ≥ ⟨x_i, w⟩ ≥ 0` and
`⟨x_i, x_j⟩ ≤ 1` by Cauchy–Schwarz on the sphere — this is exactly where the
projection `Proj_{x_i}` pays off, and where the hemisphere hypothesis is used.
`le_min_curve_of_deriv_nonneg` (in `Perspective.MinCurve`) is the analytic
half: a minimum of finitely many differentiable curves is only one-sidedly
differentiable, so the conclusion is drawn from a fencing argument rather than
from `ṙ ≥ 0`.

Step 2 of the same lemma opens with `e:decompox*.step2`, the decomposition of
`x⋆` along the particles.  It is stated in the survey for the limit `x⋆` of
step 1, and there it is justified by `x⋆` lying in the convex hull of the
configuration; written for an arbitrary point of the sphere it is false, which
is what `not_step2_decomposition` records, so `step2_decomposition` carries
that hull membership as a hypothesis.
-/

import Mathlib.Analysis.Convex.Combination
import Transformer.Perspective.MinCurve
import Transformer.Perspective.Section5_HighD

open scoped BigOperators

namespace Transformer
namespace Perspective

variable (d n : ℕ)

/-- **At a minimising index the right-hand side of `eq: therighthandside` is
non-negative**, as soon as that minimum is itself non-negative.

`h0` and `hmin` are the two facts available at a time when `i` attains
`min_j ⟨x_j, w⟩ ≥ 0`.  Each bracket is then
`⟨x_j, w⟩ - ⟨x_i, x_j⟩ ⟨x_i, w⟩ ≥ ⟨x_i, w⟩ (1 - ⟨x_i, x_j⟩) ≥ 0`,
and the prefactor `Z_{β,i}⁻¹` and the weights `e^{β ⟨x_i,x_j⟩}` are positive.

Source: arXiv:2312.10794v5, §6.1, `lem: hemisphere.clustering`, step 1. -/
theorem step1_deriv_nonneg
    (β : ℝ) (X : ℝ → SphereTuple d n) (w : SSphere d) (t : ℝ) (i : Idx n)
    (h0 : 0 ≤ inner (𝕜 := ℝ) ((X t i : EucSpace d)) ((w : EucSpace d)))
    (hmin : ∀ j : Idx n,
      inner (𝕜 := ℝ) ((X t i : EucSpace d)) ((w : EucSpace d))
        ≤ inner (𝕜 := ℝ) ((X t j : EucSpace d)) ((w : EucSpace d))) :
    0 ≤ (partitionSA d n β X t i)⁻¹ *
      ∑ j : Idx n,
        Real.exp (β * inner (𝕜 := ℝ) ((X t i : EucSpace d)) ((X t j : EucSpace d))) *
          (inner (𝕜 := ℝ) ((X t j : EucSpace d)) ((w : EucSpace d))
            - inner (𝕜 := ℝ) ((X t i : EucSpace d)) ((X t j : EucSpace d))
                * inner (𝕜 := ℝ) ((X t i : EucSpace d)) ((w : EucSpace d))) := by
  have hZ : 0 ≤ (partitionSA d n β X t i)⁻¹ :=
    inv_nonneg.mpr (Finset.sum_nonneg fun _ _ => Real.exp_nonneg _)
  refine mul_nonneg hZ (Finset.sum_nonneg fun j _ => ?_)
  refine mul_nonneg (Real.exp_nonneg _) ?_
  have hij : inner (𝕜 := ℝ) ((X t i : EucSpace d)) ((X t j : EucSpace d)) ≤ 1 := by
    have h := real_inner_le_norm ((X t i : EucSpace d)) ((X t j : EucSpace d))
    rwa [mem_sphere_zero_iff_norm.mp (X t i).2, mem_sphere_zero_iff_norm.mp (X t j).2,
      one_mul] at h
  nlinarith [hmin j, mul_nonneg h0 (sub_nonneg.mpr hij)]

/-- *Step 1 inequalities in the proof of `lem: hemisphere.clustering`:*

`r(t) := min_i ⟨x_i(t), w⟩` is non-decreasing on `ℝ_{≥0}`.

The hemisphere hypothesis `hinit` enters twice: it makes `r(0)` positive, and
the conclusion feeds it back, since `r` stays above `r(0) > 0` and so
`step1_deriv_nonneg` applies at every later time.  Without it the statement is
false — a pair of antipodal particles has `r ≡ -1 < 0` at one of them only
because it is at the other, and the minimum of two curves crossing at a time
`t` is not monotone there.

Source: arXiv:2312.10794v5, §6.1, `lem: hemisphere.clustering`, step 1. -/
theorem hemisphere_step1_monotone
    (β : ℝ) (w : SSphere d) (X : ℝ → SphereTuple d n) (r : ℝ → ℝ)
    (hX : Perspective.SA d n β X) (hr : IsMinInner d n X w r)
    (hinit : ∀ i : Idx n,
              0 < inner (𝕜 := ℝ) ((X 0 i : EucSpace d)) ((w : EucSpace d))) :
    MonotoneOn r (Set.Ici (0 : ℝ)) := by
  obtain ⟨i₀, hi₀⟩ := (hr 0).2
  have : Nonempty (Idx n) := ⟨i₀⟩
  have hr0 : 0 < r 0 := by rw [hi₀]; exact hinit i₀
  have hmono : ∀ s : ℝ, 0 < r s → ∀ t : ℝ, s ≤ t → r s ≤ r t := by
    intro s hs t hst
    refine le_min_curve_of_deriv_nonneg
      (f := fun (i : Idx n) (s : ℝ) => inner (𝕜 := ℝ) ((X s i : EucSpace d)) ((w : EucSpace d)))
      (f' := fun (i : Idx n) (s : ℝ) => (partitionSA d n β X s i)⁻¹ *
        ∑ j : Idx n,
          Real.exp (β * inner (𝕜 := ℝ) ((X s i : EucSpace d)) ((X s j : EucSpace d))) *
            (inner (𝕜 := ℝ) ((X s j : EucSpace d)) ((w : EucSpace d))
              - inner (𝕜 := ℝ) ((X s i : EucSpace d)) ((X s j : EucSpace d))
                  * inner (𝕜 := ℝ) ((X s i : EucSpace d)) ((w : EucSpace d))))
      (fun i u => step1_rhs d n β X hX ((w : EucSpace d)) u i)
      (fun u i => (hr u).1 i) (fun u => (hr u).2) ?_ hs hst
    intro u i hu hcontact
    refine step1_deriv_nonneg d n β X w u i (by rw [hcontact]; exact hu)
      (fun j => by rw [hcontact]; exact (hr u).1 j)
  intro a ha b _ hab
  exact hmono a (lt_of_lt_of_le hr0 (hmono 0 hr0 a ha)) b hab

/-- **Equation (e:decompox*.step2).**

  `x⋆ = Σ_k θ_k(t) x_k(t)`,  with `θ_k(t) ≥ 0` and `Σ_k θ_k(t) ≥ 1`.

The survey states the decomposition for the limit `x⋆` of step 1 and reads it
off the convex hull of the configuration, which is where the coefficients come
from; that membership is the hypothesis `hhull` here.  It is also the whole
content of the step: once `x⋆` is a convex combination of the particles the
dynamics plays no further role, which is why neither `β` nor the equation
`SA` appears — and, as `not_step2_decomposition` shows, without `hhull` the
statement is false.

Source: arXiv:2312.10794v5, §6.1, `e:decompox*.step2`. -/
theorem step2_decomposition
    (X : ℝ → SphereTuple d n) (x_star : SSphere d) (t : ℝ)
    (hhull : ((x_star : EucSpace d)) ∈
      convexHull ℝ (Set.range fun k : Idx n => ((X t k : EucSpace d)))) :
    ∃ θ : Idx n → ℝ,
      (∀ k, 0 ≤ θ k) ∧ (1 ≤ ∑ k : Idx n, θ k) ∧
        ((x_star : EucSpace d) = ∑ k : Idx n, (θ k) • ((X t k : EucSpace d))) := by
  classical
  rw [convexHull_range_eq_exists_affineCombination] at hhull
  obtain ⟨s, w, hw0, hw1, hx⟩ := hhull
  refine ⟨fun k => if k ∈ s then w k else 0, fun k => ?_, ?_, ?_⟩
  · by_cases hk : k ∈ s
    · simp [hk, hw0 k hk]
    · simp [hk]
  · rw [Finset.sum_ite_mem, Finset.univ_inter, hw1]
  · rw [← hx, Finset.affineCombination_eq_linear_combination s _ w hw1,
      ← Finset.sum_subset (Finset.subset_univ s) fun k _ hk => by simp [hk]]
    exact Finset.sum_congr rfl fun k hk => by simp [hk]

/-- The hypothesis of `step2_decomposition` is satisfiable: a point of the
configuration is in the convex hull of the configuration. -/
example :
    (((basePoint 0 : SSphere 1)) : EucSpace 1) ∈
      convexHull ℝ (Set.range fun _ : Idx 1 => (((basePoint 0 : SSphere 1)) : EucSpace 1)) :=
  subset_convexHull ℝ _ ⟨0, rfl⟩

/-- **Without the hull membership the decomposition fails.**

One particle sitting at `x` is a solution of `SA`, and the antipode `-x` is
not a non-negative combination of it: the coefficient would have to be `-1`.
The survey's `x⋆` is the limit of the very configuration it is decomposed
along, and that is what excludes this. -/
theorem not_step2_decomposition :
    ¬ ∀ (β : ℝ) (X : ℝ → SphereTuple 1 1), Perspective.SA 1 1 β X →
        ∀ (x_star : SSphere 1) (t : ℝ), 0 < t →
          ∃ θ : Idx 1 → ℝ, (∀ k, 0 ≤ θ k) ∧ (1 ≤ ∑ k : Idx 1, θ k) ∧
            ((x_star : EucSpace 1) = ∑ k : Idx 1, (θ k) • ((X t k : EucSpace 1))) := by
  intro h
  obtain ⟨θ, hθ0, -, hθ⟩ :=
    h 0 (fun _ _ => basePoint 0) (Perspective.SA_const_consensus 1 1 one_pos 0 (basePoint 0))
      (antipode 1 (basePoint 0)) 1 one_pos
  have hx : ‖((basePoint 0 : SSphere 1) : EucSpace 1)‖ = 1 :=
    mem_sphere_zero_iff_norm.mp (basePoint 0).2
  have hxx : inner (𝕜 := ℝ) (((basePoint 0 : SSphere 1)) : EucSpace 1)
      (((basePoint 0 : SSphere 1)) : EucSpace 1) = 1 := by
    rw [real_inner_self_eq_norm_mul_norm, hx]; ring
  rw [Fin.sum_univ_one] at hθ
  have hinner := congrArg
    (fun v : EucSpace 1 => inner (𝕜 := ℝ) v (((basePoint 0 : SSphere 1)) : EucSpace 1)) hθ
  simp only [antipode, real_inner_smul_left, inner_neg_left, hxx] at hinner
  have := hθ0 0
  linarith

/-- The hypotheses of `hemisphere_step1_monotone` are satisfiable: the
consensus solution, with `w` the common position and `r ≡ ⟨x, x⟩ = 1`. -/
example :
    Perspective.SA 1 1 0 (fun _ _ => basePoint 0) ∧
      IsMinInner 1 1 (fun _ _ => basePoint 0) (basePoint 0) (fun _ => 1) ∧
      ∀ i : Idx 1,
        0 < inner (𝕜 := ℝ)
              (((fun _ _ => basePoint 0 : ℝ → SphereTuple 1 1) 0 i : EucSpace 1))
              (((basePoint 0 : SSphere 1)) : EucSpace 1) := by
  have hx : ‖((basePoint 0 : SSphere 1) : EucSpace 1)‖ = 1 :=
    mem_sphere_zero_iff_norm.mp (basePoint 0).2
  have hxx : inner (𝕜 := ℝ) (((basePoint 0 : SSphere 1)) : EucSpace 1)
      (((basePoint 0 : SSphere 1)) : EucSpace 1) = 1 := by
    rw [real_inner_self_eq_norm_mul_norm, hx]; ring
  exact ⟨Perspective.SA_const_consensus 1 1 one_pos 0 (basePoint 0),
    fun _ => ⟨fun _ => le_of_eq hxx.symm, ⟨0, hxx.symm⟩⟩, fun _ => by norm_num [hxx]⟩

end Perspective
end Transformer
