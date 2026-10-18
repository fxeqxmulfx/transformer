/-
# Saddle avoidance — smooth cutoffs near a compact set

The avoidance theorems ask for a `C²` function and a bounded, Lipschitz field on
the whole space.  A function and a field given only on an open set `U` meet them
after a cutoff.  For a compact `K ⊆ U` of a finite-dimensional space, a smooth
`χ` with values in `[0, 1]` and compact support inside `U` is `1` on an open
neighborhood of `K` (`exists_cutoff`), and a product `χ • φ` is smooth wherever
`φ` is smooth on a set containing the support of `χ`
(`contDiff_smul_of_tsupport_subset`).  A strict saddle depends only on the germ
of the function (`IsStrictSaddle.congr_of_eventuallyEq`), so the cut-off function
has the same strict saddles near `K`.  An open set is covered by countably many
compact subsets, one of them containing any given compact subset
(`exists_isCompact_cover`).

The cutoff of `Perspective.SphereCutoffFlow`, for a general open set.
-/

import Transformer.SaddleAvoidance.Defs
import Mathlib.Geometry.Manifold.PartitionOfUnity
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Deriv

open Filter Metric Set
open scoped Topology

namespace Transformer.SaddleAvoidance

/-- **A cut-off product is smooth where the other factor is:** if `χ` is `Cᵏ` and `φ`
is `Cᵏ` at every point of a set containing the support of `χ`, then `χ • φ` is `Cᵏ`
everywhere, being `0` near every other point. -/
theorem contDiff_smul_of_tsupport_subset {E F : Type*} [NormedAddCommGroup E]
    [NormedSpace ℝ E] [NormedAddCommGroup F] [NormedSpace ℝ F] {k : WithTop ℕ∞}
    {χ : E → ℝ} {φ : E → F} {U : Set E} (hχ : ContDiff ℝ k χ) (hχU : tsupport χ ⊆ U)
    (hφ : ∀ x ∈ U, ContDiffAt ℝ k φ x) : ContDiff ℝ k fun x => χ x • φ x := by
  refine contDiff_iff_contDiffAt.2 fun x => ?_
  by_cases hx : x ∈ U
  · exact hχ.contDiffAt.smul (hφ x hx)
  · refine (contDiffAt_const (c := (0 : F))).congr_of_eventuallyEq ?_
    filter_upwards [notMem_tsupport_iff_eventuallyEq.1 fun h => hx (hχU h)] with y hy
    simp [hy]

/-- The hypotheses of `contDiff_smul_of_tsupport_subset` hold for a bump `χ` around `1`
supported in `[1/2, 3/2]`, `φ x = x⁻¹` and `U = (0, ∞)`: `χ x • x⁻¹` is smooth on the
whole line, although `x⁻¹` is not at `0`. -/
example : ContDiff ℝ 1 fun x : ℝ =>
    (⟨1 / 4, 1 / 2, by norm_num, by norm_num⟩ : ContDiffBump (1 : ℝ)) x • x⁻¹ := by
  refine contDiff_smul_of_tsupport_subset (U := Ioi 0) (ContDiffBump.contDiff _) ?_
    fun x hx => contDiffAt_inv ℝ (ne_of_gt hx)
  rw [ContDiffBump.tsupport_eq]
  intro y hy
  rw [mem_closedBall, Real.dist_eq] at hy
  have := (abs_le.1 hy).1
  rw [mem_Ioi]
  linarith

section Cutoff

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E] [FiniteDimensional ℝ E]

/-- **A smooth cutoff near a compact set.**  For a compact `K` inside an open `U`, some
`Cⁿ` function `χ` with values in `[0, 1]` and compact support inside `U` is `1` on an
open neighborhood `V` of `K`: take `χ` with support the `δ/2`-thickening of `K`, equal
to `1` on its closed `δ/4`-thickening, for a `δ`-thickening inside `U`. -/
theorem exists_cutoff {K U : Set E} (hK : IsCompact K) (hU : IsOpen U) (hKU : K ⊆ U)
    (n : ℕ∞) : ∃ χ : E → ℝ, ContDiff ℝ n χ ∧ (∀ x, 0 ≤ χ x ∧ χ x ≤ 1) ∧
      HasCompactSupport χ ∧ tsupport χ ⊆ U ∧ ∃ V, IsOpen V ∧ K ⊆ V ∧ ∀ x ∈ V, χ x = 1 := by
  obtain ⟨δ, hδ, hδU⟩ := hK.exists_thickening_subset_open hU hKU
  obtain ⟨χ, hχ, hrange, hsupp, hone⟩ := exists_contDiff_support_eq_eq_one_iff (n := n)
    (isOpen_thickening (δ := δ / 2) (E := K)) (isClosed_cthickening (δ := δ / 4) (E := K))
    (cthickening_subset_thickening' (by positivity) (by linarith) K)
  have hts : tsupport χ ⊆ cthickening (δ / 2) K := by
    rw [tsupport, hsupp]
    exact closure_thickening_subset_cthickening _ _
  refine ⟨χ, hχ, fun x => hrange ⟨x, rfl⟩, (hK.cthickening).of_isClosed_subset
    (isClosed_tsupport χ) hts, hts.trans ((cthickening_subset_thickening' hδ (by linarith)
    K).trans hδU), thickening (δ / 4) K, isOpen_thickening, self_subset_thickening
    (by positivity) K, fun x hx => (hone x).1 (thickening_subset_cthickening _ _ hx)⟩

/-- The hypotheses of `exists_cutoff` hold for the point `0` of the line inside the
line. -/
example : ∃ χ : ℝ → ℝ, ContDiff ℝ 2 χ ∧ (∀ x, 0 ≤ χ x ∧ χ x ≤ 1) ∧ HasCompactSupport χ ∧
    tsupport χ ⊆ univ ∧ ∃ V, IsOpen V ∧ {0} ⊆ V ∧ ∀ x ∈ V, χ x = 1 :=
  exists_cutoff isCompact_singleton isOpen_univ (subset_univ _) 2

/-- **An open set is a countable union of compact sets catching every compact
subset**: `K_k = B̄(0, k) ∖ (1/(k+1))-thickening of Uᶜ`.  A compact `C ⊆ U` lies in a
ball, and some `δ`-thickening of it lies in `U`, so `C ⊆ K_k` once `k ≥ 1/δ`. -/
theorem exists_isCompact_cover {U : Set E} (hU : IsOpen U) :
    ∃ K : ℕ → Set E, (∀ k, IsCompact (K k)) ∧ (∀ k, K k ⊆ U) ∧
      ∀ C, IsCompact C → C ⊆ U → ∃ k, C ⊆ K k := by
  refine ⟨fun k => closedBall 0 k \ thickening (1 / ((k : ℝ) + 1)) Uᶜ,
    fun k => (isCompact_closedBall 0 _).diff isOpen_thickening, fun k x hx => ?_,
    fun C hC hCU => ?_⟩
  · by_contra hxU
    exact hx.2 (self_subset_thickening (by positivity) Uᶜ hxU)
  obtain ⟨δ, hδ, hδU⟩ := hC.exists_thickening_subset_open hU hCU
  obtain ⟨r, hr⟩ := hC.isBounded.subset_closedBall 0
  obtain ⟨k, hk⟩ := exists_nat_gt (max r (1 / δ))
  have hkδ : 1 / ((k : ℝ) + 1) < δ :=
    (one_div_lt (by positivity) hδ).2 (by linarith [le_max_right r (1 / δ)])
  refine ⟨k, fun x hx => ⟨closedBall_subset_closedBall (by linarith [le_max_left r (1 / δ)])
    (hr hx), fun hxt => ?_⟩⟩
  obtain ⟨w, hw, hxw⟩ := mem_thickening_iff.1 hxt
  exact hw (hδU (mem_thickening_iff.2 ⟨x, hx, by rw [dist_comm]; linarith⟩))

/-- The hypothesis of `exists_isCompact_cover` holds for the half-line
`(0, ∞)`. -/
example : ∃ K : ℕ → Set ℝ, (∀ k, IsCompact (K k)) ∧ (∀ k, K k ⊆ Ioi 0) ∧
    ∀ C, IsCompact C → C ⊆ Ioi 0 → ∃ k, C ⊆ K k :=
  exists_isCompact_cover isOpen_Ioi

end Cutoff

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [CompleteSpace E]

/-- **A strict saddle depends only on the germ of the function**: if `f₁ = f` near
`z`, their gradients agree near `z`, and so do the derivatives of the gradients at
`z`. -/
theorem IsStrictSaddle.congr_of_eventuallyEq {f₁ f : E → ℝ} {z : E}
    (hz : IsStrictSaddle f z) (h : f₁ =ᶠ[𝓝 z] f) : IsStrictSaddle f₁ z := by
  obtain ⟨h0, μ, hμ, hev⟩ := hz
  refine ⟨by rw [h.gradient_eq, h0], μ, hμ, ?_⟩
  rwa [h.gradient.fderiv_eq]

/-- The hypotheses of `IsStrictSaddle.congr_of_eventuallyEq` hold for the strict
saddle `0` of `cos`, a maximum, and `min (cos x) 1 = cos x`. -/
example : IsStrictSaddle (fun x : ℝ => min (Real.cos x) 1) 0 := by
  have hgrad : gradient Real.cos = fun x => -Real.sin x :=
    funext fun x => (Real.hasDerivAt_cos x).hasGradientAt'.gradient
  have hH : fderiv ℝ (gradient Real.cos) 0 =
      ContinuousLinearMap.smulRight (1 : ℝ →L[ℝ] ℝ) (-Real.cos 0) := by
    rw [hgrad]
    exact (Real.hasDerivAt_sin 0).neg.hasFDerivAt.fderiv
  have hcos : IsStrictSaddle Real.cos 0 := ⟨by simp [hgrad], -1, by norm_num,
    Module.End.hasEigenvalue_of_hasEigenvector (x := 1) ⟨by simp [hH], one_ne_zero⟩⟩
  exact hcos.congr_of_eventuallyEq
    (Eventually.of_forall fun x => min_eq_left (Real.cos_le_one x))

end Transformer.SaddleAvoidance
