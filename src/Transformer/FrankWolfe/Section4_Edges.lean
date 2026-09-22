/-
# Attention's forward pass and Frank-Wolfe — a vertex above its edges

The fact behind `prop: polytope.condition`: a linear functional that is
strictly smaller at the far end of every edge leaving a vertex `p` of a
polytope `𝒦 = conv V` is strictly maximized over `𝒦` at `p` alone
(`lt_of_forall_isExtreme_segment`).  The source leaves it implicit
("`eq: cond.hyperplane` is like being a local maximizer of `𝖩`"); it is the
local-to-global principle of the simplex method.

The proof is the vertex figure.  A functional `g` separates `p` from the other
points of `V`; the directions `(u - p)/(g(p) - g(u))`, `u ∈ V \ {p}`, span a
polytope `B` in the hyperplane `{g = -1}`, and `𝒦 ⊆ p + ℝ₊ B`.  An extreme
point `e` of `B` is the direction of an edge: the segment `[p, u]` along it is
an extreme subset of `𝒦`, because `e` is extreme in `B` and `u` is extreme in
`𝒦`.  So the functional is negative at the extreme points of `B`, hence on `B`
(Krein–Milman), hence below its value at `p` on the rest of `𝒦`.

Source: arXiv:2508.09628v1, §4, `prop: polytope.condition`.
-/

import Transformer.FrankWolfe.Section1_Models
import Mathlib.Analysis.Convex.Between
import Mathlib.Analysis.Convex.KreinMilman
import Mathlib.Analysis.LocallyConvex.Separation

open Set

namespace Transformer
namespace FrankWolfe

variable {d : ℕ}

/-- The cone with apex `p` over a convex set `B`, `{p + t b : t ≥ 0, b ∈ B}`, is
convex. -/
theorem convex_apexCone {B : Set (EucSpace d)} (hB : Convex ℝ B) (p : EucSpace d) :
    Convex ℝ {x | ∃ t : ℝ, 0 ≤ t ∧ ∃ b ∈ B, x = p + t • b} := by
  rintro _ ⟨t₁, ht₁, b₁, hb₁, rfl⟩ _ ⟨t₂, ht₂, b₂, hb₂, rfl⟩ a₁ a₂ ha₁ ha₂ hab
  have h₁ := mul_nonneg ha₁ ht₁
  have h₂ := mul_nonneg ha₂ ht₂
  have key : a₁ • (p + t₁ • b₁) + a₂ • (p + t₂ • b₂) =
      p + ((a₁ * t₁) • b₁ + (a₂ * t₂) • b₂) := by
    rw [smul_add, smul_add, smul_smul, smul_smul, add_add_add_comm, ← add_smul, hab, one_smul]
  rcases (add_nonneg h₁ h₂).eq_or_lt with h | h
  · refine ⟨0, le_rfl, b₁, hb₁, ?_⟩
    rw [key, show a₁ * t₁ = 0 by linarith, show a₂ * t₂ = 0 by linarith]
    simp
  · refine ⟨a₁ * t₁ + a₂ * t₂, h.le,
      (a₁ * t₁ / (a₁ * t₁ + a₂ * t₂)) • b₁ + (a₂ * t₂ / (a₁ * t₁ + a₂ * t₂)) • b₂,
      hB hb₁ hb₂ (div_nonneg h₁ h.le) (div_nonneg h₂ h.le) (by rw [← add_div, div_self h.ne']),
      ?_⟩
    rw [key, smul_add, smul_smul, smul_smul, mul_div_cancel₀ _ h.ne', mul_div_cancel₀ _ h.ne']

/-- The hypothesis of `convex_apexCone` is satisfiable: `B = univ`. -/
example : Convex ℝ (univ : Set (EucSpace 1)) := convex_univ

/-- **A vertex above its edges is the unique maximizer.**  Let `𝒦 = conv V`,
with `V` finite and made of extreme points of `𝒦`, and let `p ∈ V`.  If a
linear functional `ℓ` is strictly smaller than at `p` at the far end of every
edge of `𝒦` leaving `p` — every `w ∈ V \ {p}` with `[p, w]` an extreme subset
of `𝒦` — then it is strictly smaller than at `p` on all of `𝒦 \ {p}`.

Source: arXiv:2508.09628v1, §4, `prop: polytope.condition` ("like being a
local maximizer"). -/
theorem lt_of_forall_isExtreme_segment {V : Set (EucSpace d)} (hVf : V.Finite)
    (hV : V ⊆ (convexHull ℝ V).extremePoints ℝ) {p : EucSpace d} (hp : p ∈ V)
    (ℓ : EucSpace d →L[ℝ] ℝ)
    (h : ∀ w ∈ V, w ≠ p → IsExtreme ℝ (convexHull ℝ V) (segment ℝ p w) → ℓ w < ℓ p) :
    ∀ x ∈ convexHull ℝ V, x ≠ p → ℓ x < ℓ p := by
  intro x hx hxp
  have hpP : p ∈ convexHull ℝ V := subset_convexHull ℝ V hp
  -- a functional `g` separating `p` from the other points of `V`
  obtain ⟨g, hg⟩ : ∃ g : EucSpace d →L[ℝ] ℝ, ∀ u ∈ V, u ≠ p → g u < g p := by
    have hpC : p ∉ convexHull ℝ (V \ {p}) := fun hmem =>
      (convexHull_min (sdiff_subset_sdiff_left (subset_convexHull ℝ V))
        ((convex_convexHull ℝ V).mem_extremePoints_iff_convex_sdiff.mp (hV hp)).2 hmem).2
        (mem_singleton p)
    obtain ⟨f, c, hf, hfp⟩ := geometric_hahn_banach_closed_point (convex_convexHull ℝ _)
      (hVf.sdiff.isCompact_convexHull (𝕜 := ℝ)).isClosed hpC
    exact ⟨f, fun u hu hup => (hf u (subset_convexHull ℝ _ ⟨hu, hup⟩)).trans hfp⟩
  -- the directions of the vertex figure `B = conv G`, in the hyperplane `{g = -1}`
  obtain ⟨G, hGf, hGV, hVG⟩ : ∃ G : Set (EucSpace d), G.Finite ∧
      (∀ e ∈ G, ∃ u ∈ V, u ≠ p ∧ e = (g p - g u)⁻¹ • (u - p)) ∧
      ∀ u ∈ V, u ≠ p → (g p - g u)⁻¹ • (u - p) ∈ G :=
    ⟨(fun u => (g p - g u)⁻¹ • (u - p)) '' (V \ {p}), hVf.sdiff.image _,
      fun e ⟨u, ⟨hu, hup⟩, he⟩ => ⟨u, hu, hup, he.symm⟩, fun u hu hup => ⟨u, ⟨hu, hup⟩, rfl⟩⟩
  have hgB : convexHull ℝ G ⊆ {b | g b = -1} := by
    refine convexHull_min (fun e he => ?_) (convex_hyperplane g.toLinearMap.isLinear (-1))
    obtain ⟨u, hu, hup, rfl⟩ := hGV e he
    have hs : g p - g u ≠ 0 := (sub_pos.mpr (hg u hu hup)).ne'
    show g ((g p - g u)⁻¹ • (u - p)) = -1
    simp only [map_smul, map_sub, smul_eq_mul]
    field_simp
    ring
  -- every point of `𝒦` is `p + t b` with `t ≥ 0` and `b ∈ B`
  have hD : ∀ y ∈ convexHull ℝ V, ∃ t : ℝ, 0 ≤ t ∧ ∃ b ∈ convexHull ℝ G, y = p + t • b := by
    obtain ⟨u₀, hu₀, hu₀p⟩ : ∃ u ∈ V, u ≠ p := by
      by_contra hne
      exact hxp (convexHull_min (fun u hu => not_not.mp fun hup => hne ⟨u, hu, hup⟩)
        (convex_singleton p) hx)
    refine fun y hy => convexHull_min (fun u hu => ?_)
      (convex_apexCone (convex_convexHull ℝ G) p) hy
    by_cases hup : u = p
    · exact ⟨0, le_rfl, _, subset_convexHull ℝ G (hVG u₀ hu₀ hu₀p),
        by rw [zero_smul, add_zero, hup]⟩
    · have hs : 0 < g p - g u := sub_pos.mpr (hg u hu hup)
      refine ⟨g p - g u, hs.le, _, subset_convexHull ℝ G (hVG u hu hup), ?_⟩
      rw [smul_smul, mul_inv_cancel₀ hs.ne', one_smul, add_sub_cancel]
  -- `ℓ < 0` at every extreme point `e` of `B`: `e` is the direction of an edge
  have hext : ∀ e ∈ (convexHull ℝ G).extremePoints ℝ, ℓ e < 0 := by
    intro e he
    obtain ⟨u, hu, hup, he_def⟩ := hGV e (extremePoints_convexHull_subset he)
    have hs : 0 < g p - g u := sub_pos.mpr (hg u hu hup)
    obtain ⟨s, hs_def⟩ : ∃ s, g p - g u = s := ⟨_, rfl⟩
    rw [hs_def] at he_def hs
    have hue : u - p = s • e := by rw [he_def, smul_smul, mul_inv_cancel₀ hs.ne', one_smul]
    -- the points of `𝒦` on the ray from `p` through `u` lie on `[p, u]`
    have hray : ∀ t : ℝ, 0 ≤ t → p + t • e ∈ convexHull ℝ V → p + t • e ∈ segment ℝ p u := by
      intro t ht htP
      have hu' : s • e +ᵥ p = u := by rw [vadd_eq_add, ← hue, sub_add_cancel]
      have ht' : t • e +ᵥ p = p + t • e := by rw [vadd_eq_add, add_comm]
      rcases wbtw_or_wbtw_smul_vadd_of_nonneg p e ht hs.le with hw | hw
      · rw [hu', ht'] at hw
        exact mem_segment_iff_wbtw.mpr hw
      · rw [hu', ht'] at hw
        rcases (mem_extremePoints_iff_forall_segment.mp (hV hu)).2 p hpP _ htP
          (mem_segment_iff_wbtw.mpr hw) with h1 | h1
        · exact absurd h1.symm hup
        · rw [h1]
          exact right_mem_segment ℝ p u
    -- `[p, u]` is an edge
    have hedge : IsExtreme ℝ (convexHull ℝ V) (segment ℝ p u) := by
      refine ⟨(convex_convexHull ℝ V).segment_subset hpP (subset_convexHull ℝ V hu), ?_⟩
      intro x₁ hx₁ x₂ hx₂ z hz hzo
      obtain ⟨t₁, ht₁, b₁, hb₁, rfl⟩ := hD x₁ hx₁
      obtain ⟨t₂, ht₂, b₂, hb₂, rfl⟩ := hD x₂ hx₂
      rw [segment_eq_image'] at hz
      obtain ⟨θ, ⟨hθ0, -⟩, rfl⟩ := hz
      obtain ⟨a₁, a₂, ha₁, ha₂, hab, hz⟩ := hzo
      beta_reduce at hz
      rw [hue] at hz
      have hsum : (a₁ * t₁) • b₁ + (a₂ * t₂) • b₂ = (θ * s) • e := by
        rw [smul_add, smul_add, smul_smul, smul_smul, add_add_add_comm, ← add_smul, hab, one_smul,
          smul_smul] at hz
        exact add_left_cancel hz
      have hσ : a₁ * t₁ + a₂ * t₂ = θ * s := by
        have := congrArg g hsum
        rw [map_add, map_smul, map_smul, map_smul, hgB hb₁, hgB hb₂, hgB he.1] at this
        simp only [smul_eq_mul] at this
        linarith
      rcases ht₁.eq_or_lt with h0 | hpos
      · rw [← h0, zero_smul, add_zero]
        exact left_mem_segment ℝ p u
      have hb₁e : b₁ = e := by
        rcases (mul_nonneg ha₂.le ht₂).eq_or_lt with h2 | h2
        · rw [← h2, zero_smul, add_zero] at hsum
          rw [← h2, add_zero] at hσ
          rw [← hσ] at hsum
          exact smul_right_injective (EucSpace d) (mul_pos ha₁ hpos).ne' hsum
        · have hT : 0 < θ * s := by linarith [mul_pos ha₁ hpos]
          refine ((mem_extremePoints.mp he).2 b₁ hb₁ b₂ hb₂ ⟨a₁ * t₁ / (θ * s), a₂ * t₂ / (θ * s),
            by positivity, by positivity, by rw [← add_div, hσ, div_self hT.ne'], ?_⟩).1
          have hc : (θ * s)⁻¹ • ((a₁ * t₁) • b₁ + (a₂ * t₂) • b₂) = e := by
            rw [hsum, smul_smul, inv_mul_cancel₀ hT.ne', one_smul]
          rw [smul_add, smul_smul, smul_smul] at hc
          rw [div_eq_inv_mul, div_eq_inv_mul]
          exact hc
      rw [hb₁e] at hx₁ ⊢
      exact hray t₁ ht₁ hx₁
    have hlt := h u hu hup hedge
    have hu_eq : u = p + s • e := by rw [← hue, add_sub_cancel]
    rw [hu_eq, map_add, map_smul, smul_eq_mul] at hlt
    nlinarith
  -- so `ℓ < 0` on `B`, the convex hull of its extreme points
  have hB : ∀ b ∈ convexHull ℝ G, ℓ b < 0 := by
    have hK := closure_convexHull_extremePoints (hGf.isCompact_convexHull (𝕜 := ℝ))
      (convex_convexHull ℝ G)
    rw [((hGf.subset extremePoints_convexHull_subset).isCompact_convexHull
      (𝕜 := ℝ)).isClosed.closure_eq] at hK
    intro b hb
    rw [← hK] at hb
    exact convexHull_min hext (convex_halfSpace_lt ℓ.toLinearMap.isLinear 0) hb
  obtain ⟨t, ht, b, hb, rfl⟩ := hD x hx
  have htpos : 0 < t := ht.lt_of_ne fun h0 => hxp (by rw [← h0, zero_smul, add_zero])
  rw [map_add, map_smul, smul_eq_mul]
  linarith [mul_neg_of_pos_of_neg htpos (hB b hb)]

/-- The hypotheses of `lt_of_forall_isExtreme_segment` are satisfiable: a single
point `V = {0}`, with `p = 0` and `ℓ = 0`. -/
example : ({0} : Set (EucSpace 1)).Finite ∧
    ({0} : Set (EucSpace 1)) ⊆ (convexHull ℝ {0}).extremePoints ℝ ∧
    ∀ w ∈ ({0} : Set (EucSpace 1)), w ≠ 0 →
      IsExtreme ℝ (convexHull ℝ {0}) (segment ℝ 0 w) →
        (0 : EucSpace 1 →L[ℝ] ℝ) w < (0 : EucSpace 1 →L[ℝ] ℝ) 0 := by
  refine ⟨finite_singleton 0, ?_, fun w hw hw0 => absurd hw hw0⟩
  rw [convexHull_singleton, extremePoints_singleton]

end FrankWolfe
end Transformer
