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
`x⋆` along the particles, which the survey reads off `η x⋆` being a convex
combination of them for some `η ∈ (0,1]`: that deduction is
`step2_decomposition`.  That the limit `x⋆` of step 1 satisfies the cone
condition is proved in `Perspective.Section5_HemisphereCone`.
`exists_inner_le_of_mem_convexHull` is the one consequence of the hull
Appendix D uses, `e:mineqalpha`.
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

/-- **The smallest inner product of `x_i` with the configuration is at most its
inner product with any point of the configuration's convex hull.**

This is `e:mineqalpha` without the coefficients: the survey reads it off the
decomposition `e:decompox*.step2`, `x⋆ = Σ_k θ_k x_k` with `θ_k ≥ 0` summing
to one, and the version below is the same fact stated through the hull, where
the convexity of a half-space replaces the computation with the `θ_k`.

Source: arXiv:2312.10794v5, Appendix D, `e:mineqalpha`. -/
theorem exists_inner_le_of_mem_convexHull (hn : 0 < n)
    (Y : SphereTuple d n) (x_star : SSphere d)
    (hhull : ((x_star : EucSpace d)) ∈
      convexHull ℝ (Set.range fun k : Idx n => ((Y k : EucSpace d))))
    (i : Idx n) :
    ∃ j : Idx n,
      inner (𝕜 := ℝ) ((Y i : EucSpace d)) ((Y j : EucSpace d))
        ≤ inner (𝕜 := ℝ) ((Y i : EucSpace d)) ((x_star : EucSpace d)) := by
  have : Nonempty (Idx n) := ⟨⟨0, hn⟩⟩
  obtain ⟨j, -, hj⟩ := Finset.exists_min_image (Finset.univ : Finset (Idx n))
    (fun k => inner (𝕜 := ℝ) ((Y i : EucSpace d)) ((Y k : EucSpace d)))
    Finset.univ_nonempty
  refine ⟨j, ?_⟩
  have hlin : IsLinearMap ℝ
      (fun v : EucSpace d => inner (𝕜 := ℝ) ((Y i : EucSpace d)) v) :=
    ⟨fun a b => inner_add_right _ _ _, fun c a => real_inner_smul_right _ _ _⟩
  have hsub : (Set.range fun k : Idx n => ((Y k : EucSpace d)))
      ⊆ { v : EucSpace d |
          inner (𝕜 := ℝ) ((Y i : EucSpace d)) ((Y j : EucSpace d))
            ≤ inner (𝕜 := ℝ) ((Y i : EucSpace d)) v } := by
    rintro v ⟨k, rfl⟩
    exact hj k (Finset.mem_univ k)
  exact convexHull_min hsub (convex_halfSpace_ge hlin _) hhull

/-- The hypothesis of `exists_inner_le_of_mem_convexHull` is satisfiable: a
particle of the configuration lies in its own convex hull. -/
example :
    (((basePoint 0 : SSphere 1)) : EucSpace 1) ∈
      convexHull ℝ (Set.range fun _ : Idx 1 => (((basePoint 0 : SSphere 1)) : EucSpace 1)) :=
  subset_convexHull ℝ _ ⟨0, rfl⟩

/-- **Equation (e:decompox*.step2).**

  `x⋆ = Σ_k θ_k(t) x_k(t)`,  with `θ_k(t) ≥ 0` and `Σ_k θ_k(t) ≥ 1`.

The survey derives it from `x⋆` lying in the convex cone of the particles:
"there exists some `η ∈ (0,1]` such that `η x⋆` is a convex combination of the
points `x_1(t), …, x_n(t)`, which implies" the decomposition.  That sentence is
the hypothesis `hcone` here, and dividing the convex weights by `η` is the
proof.  The cone condition itself holds for the limit `x⋆` of step 1:
`Perspective.limit_mem_cone`, and the two together are
`Perspective.step2_decomposition_of_limit`.

Source: arXiv:2312.10794v5, §6.1, `e:decompox*.step2`. -/
theorem step2_decomposition
    (X : ℝ → SphereTuple d n) (x_star : SSphere d) (t : ℝ)
    (hcone : ∃ η : ℝ, η ∈ Set.Ioc (0 : ℝ) 1 ∧ η • ((x_star : EucSpace d)) ∈
      convexHull ℝ (Set.range fun k : Idx n => ((X t k : EucSpace d)))) :
    ∃ θ : Idx n → ℝ,
      (∀ k, 0 ≤ θ k) ∧ (1 ≤ ∑ k : Idx n, θ k) ∧
        ((x_star : EucSpace d) = ∑ k : Idx n, (θ k) • ((X t k : EucSpace d))) := by
  classical
  obtain ⟨η, ⟨hη0, hη1⟩, hhull⟩ := hcone
  rw [convexHull_range_eq_exists_affineCombination] at hhull
  obtain ⟨s, w, hw0, hw1, hx⟩ := hhull
  refine ⟨fun k => if k ∈ s then w k / η else 0, fun k => ?_, ?_, ?_⟩
  · by_cases hk : k ∈ s
    · simp only [hk, ite_true]; exact div_nonneg (hw0 k hk) hη0.le
    · simp [hk]
  · rw [Finset.sum_ite_mem, Finset.univ_inter, ← Finset.sum_div, hw1]
    exact (one_le_div hη0).mpr hη1
  · have hx' : η • ((x_star : EucSpace d)) = ∑ k ∈ s, w k • ((X t k : EucSpace d)) := by
      rw [← hx, Finset.affineCombination_eq_linear_combination s _ w hw1]
    rw [← Finset.sum_subset (Finset.subset_univ s) fun k _ hk => by simp [hk]]
    calc ((x_star : EucSpace d)) = η⁻¹ • η • ((x_star : EucSpace d)) := by
          rw [smul_smul, inv_mul_cancel₀ hη0.ne', one_smul]
      _ = _ := by
          rw [hx', Finset.smul_sum]
          exact Finset.sum_congr rfl fun k hk => by
            simp only [hk, ite_true, smul_smul, div_eq_inv_mul]

/-- The hypothesis of `step2_decomposition` is satisfiable: a point of the
configuration is in the convex hull of the configuration, with `η = 1`. -/
example : ∃ η : ℝ, η ∈ Set.Ioc (0 : ℝ) 1 ∧ η • (((basePoint 0 : SSphere 1)) : EucSpace 1) ∈
    convexHull ℝ (Set.range fun _ : Idx 1 => (((basePoint 0 : SSphere 1)) : EucSpace 1)) :=
  ⟨1, ⟨one_pos, le_rfl⟩, by rw [one_smul]; exact subset_convexHull ℝ _ ⟨0, rfl⟩⟩

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
