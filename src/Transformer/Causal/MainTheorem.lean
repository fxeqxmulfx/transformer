/-
# Causal attention — Main theorem for `V = I_d` (§4 of 2411.04990v2)

* `Theorem thm1`        — almost-everywhere convergence to a single cluster
                          for `V = I_d` and *arbitrary* `Q, K`,
* `Conjecture thm1.5`   — analogue for `V` diagonalizable with `λ_max > 0`
                          of multiplicity 1,
* `Conjecture thm2`     — analogue for `V` with `λ_max > 0` of multiplicity
                          `≥ 2`,
* `Lemma lemma:scalar`  — the scalar inequality the proof of `thm1` opens
                          with, and its equality cases.

All three statements quantify over "almost every initial configuration", which
is relative to a reference measure on `(𝕊^{d-1})^n`; the paper's measure is the
uniform one, and this development does not construct it.  So each is a
`Prop`-valued definition carrying that measure as a parameter, and none of them
is proved here — `thm1.5` and `thm2` are conjectures even in the paper.
-/

import Transformer.Basic
import Transformer.Causal.Basic
import Transformer.Perspective.Section2_FlowMap
import Mathlib.Analysis.InnerProductSpace.Projection.Basic
import Mathlib.Basic.Real.Sign

open scoped BigOperators
open Real MeasureTheory

namespace Transformer
namespace Causal

open Causal Perspective

variable (d n : ℕ)

/-- **Theorem (thm1).** *Single-cluster convergence with `V = I_d`.*

For arbitrary `Q, K`, `β ≥ 0` and `V = I_d`, for almost any
`(x_1(0),…,x_n(0)) ∈ (𝕊^{d-1})^n` the CSA dynamics satisfy

  `∀ k ∈ [n],  lim_{t → ∞} x_k(t) = x_1(0)`.

The leader `x_1` is a fixed point of `eq: csa`: its own row of the causal mask
sees only itself, so the whole cloud is dragged onto where it started.
Source: arXiv:2411.04990v2, §4. -/
def SingleCluster
    (σ : Measure (SphereTuple d n)) (β : ℝ) (Q K : ParamMatrix d) (hn : 1 ≤ n) : Prop :=
  0 ≤ β →
  ∀ᵐ X₀ ∂σ, ∀ X : ℝ → SphereTuple d n, X 0 = X₀ →
    Causal.CSA d n β Q K (ContinuousLinearMap.id ℝ (EucSpace d)) X →
      ∀ k : Idx n,
        Filter.Tendsto (fun t : ℝ => X t k) Filter.atTop (nhds (X₀ ⟨0, hn⟩))

/-- **Conjecture (thm1.5).**  *Two-cluster convergence (`λ_max > 0`, mult 1).*

Let `λ > 0` be the largest real eigenvalue of `V` — every real eigenvalue is
`≤ λ` — and let it be simple, its eigenspace being the line spanned by
`ξ ∈ 𝕊^{d-1}`.  Then for arbitrary `Q, K` and almost any initialization the
CSA dynamics satisfy

  `∀ k ∈ [n], lim_{t → ∞} x_k(t) ∈ {ξ, -ξ}`.

The two limits are genuinely both possible: `V` fixes the line `ℝ ξ`, not a
ray, so a token starting in the half-space `⟨x, ξ⟩ < 0` is pulled to `-ξ`.
Source: arXiv:2411.04990v2, §4. -/
def TwoCluster
    (σ : Measure (SphereTuple d n)) (β : ℝ) (Q K V : ParamMatrix d)
    (lam : ℝ) (ξ : SSphere d) : Prop :=
  0 ≤ β → 0 < lam →
  V (ξ : EucSpace d) = lam • (ξ : EucSpace d) →
  (∀ (v : EucSpace d) (c : ℝ), v ≠ 0 → V v = c • v → c ≤ lam) →
  (∀ v : EucSpace d, V v = lam • v → ∃ c : ℝ, v = c • (ξ : EucSpace d)) →
  ∀ᵐ X₀ ∂σ, ∀ X : ℝ → SphereTuple d n, X 0 = X₀ → Causal.CSA d n β Q K V X →
    ∀ k : Idx n,
      Filter.Tendsto (fun t : ℝ => (X t k : EucSpace d)) Filter.atTop
          (nhds (ξ : EucSpace d)) ∨
        Filter.Tendsto (fun t : ℝ => (X t k : EucSpace d)) Filter.atTop
          (nhds (-(ξ : EucSpace d)))

/-- **Conjecture (thm2).**  *Single-cluster convergence with `λ_max > 0`,
`dim L ≥ 2`.*

Let `λ > 0`, let `L` be an eigenspace of `V` for `λ` with `dim L ≥ 2`, let `V`
preserve `L^⊥`, and let `⟨V z, z⟩ < λ ‖z‖²` for every `z ∈ L^⊥ \ {0}` — so `λ`
really is the top eigenvalue and everything transverse to `L` decays relative
to it.  Then almost every initialization yields convergence of every token to
the normalized `L`-component of `x_1(0)`, which is defined as soon as that
component is nonzero.

Source: arXiv:2411.04990v2, §4. -/
def SubspaceCluster
    (σ : Measure (SphereTuple d n)) (β : ℝ) (Q K V : ParamMatrix d)
    (lam : ℝ) (L : Submodule ℝ (EucSpace d)) (hn : 1 ≤ n) : Prop :=
  0 ≤ β → 0 < lam →
  (∀ v ∈ L, V v = lam • v) →
  2 ≤ Module.finrank ℝ L →
  (∀ z ∈ Lᗮ, V z ∈ Lᗮ) →
  (∀ z ∈ Lᗮ, z ≠ 0 → inner (𝕜 := ℝ) (V z) z < lam * ‖z‖ ^ 2) →
  ∀ᵐ X₀ ∂σ, ∀ X : ℝ → SphereTuple d n, X 0 = X₀ → Causal.CSA d n β Q K V X →
    L.starProjection ((X₀ ⟨0, hn⟩ : SSphere d) : EucSpace d) ≠ 0 →
      ∀ k : Idx n,
        Filter.Tendsto (fun t : ℝ => (X t k : EucSpace d)) Filter.atTop
          (nhds (‖L.starProjection ((X₀ ⟨0, hn⟩ : SSphere d) : EucSpace d)‖⁻¹ •
            L.starProjection ((X₀ ⟨0, hn⟩ : SSphere d) : EucSpace d)))

/-- **Lemma (lemma:scalar).** *A scalar inequality for unit vectors.*

For `‖x‖ = ‖y‖ = 1` and `|⟨y, z⟩| ≤ ⟨x, z⟩`,

  `⟨x, z⟩ ≥ ⟨x, y⟩ ⟨y, z⟩`.

The chain is `⟨x,y⟩⟨y,z⟩ ≤ |⟨x,y⟩| |⟨y,z⟩| ≤ |⟨x,y⟩| ⟨x,z⟩ ≤ ⟨x,z⟩`, the last
step because `|⟨x,y⟩| ≤ 1` for unit vectors and `⟨x,z⟩ ≥ 0` by hypothesis.

Source: arXiv:2411.04990v2, §A, `lemma:scalar`. -/
theorem inner_mul_le_inner_of_abs_le
    {d : ℕ} (x y z : EucSpace d) (hx : ‖x‖ = 1) (hy : ‖y‖ = 1)
    (h : |inner (𝕜 := ℝ) y z| ≤ inner (𝕜 := ℝ) x z) :
    inner (𝕜 := ℝ) x y * inner (𝕜 := ℝ) y z ≤ inner (𝕜 := ℝ) x z := by
  have hxz : 0 ≤ inner (𝕜 := ℝ) x z := le_trans (abs_nonneg _) h
  have hxy : |inner (𝕜 := ℝ) x y| ≤ 1 := by
    simpa [hx, hy] using abs_real_inner_le_norm x y
  calc inner (𝕜 := ℝ) x y * inner (𝕜 := ℝ) y z
      ≤ |inner (𝕜 := ℝ) x y| * |inner (𝕜 := ℝ) y z| := by
        rw [← abs_mul]; exact le_abs_self _
    _ ≤ |inner (𝕜 := ℝ) x y| * inner (𝕜 := ℝ) x z :=
        mul_le_mul_of_nonneg_left h (abs_nonneg _)
    _ ≤ 1 * inner (𝕜 := ℝ) x z := mul_le_mul_of_nonneg_right hxy hxz
    _ = inner (𝕜 := ℝ) x z := one_mul _

/-- The hypotheses of `inner_mul_le_inner_of_abs_le` are satisfiable: two
copies of the first standard basis vector, and `z = 0`. -/
example :
    ‖EuclideanSpace.single (0 : Fin 1) (1 : ℝ)‖ = 1 ∧
      |inner (𝕜 := ℝ) (EuclideanSpace.single (0 : Fin 1) (1 : ℝ))
          (0 : EucSpace 1)|
        ≤ inner (𝕜 := ℝ) (EuclideanSpace.single (0 : Fin 1) (1 : ℝ))
            (0 : EucSpace 1) := by
  refine ⟨by simp, ?_⟩
  simp

/-- **Lemma (lemma:scalar), the equality cases.**

Equality `⟨x, z⟩ = ⟨x, y⟩ ⟨y, z⟩` holds if and only if either `⟨x, z⟩ = 0`, or
`|⟨y, z⟩| = ⟨x, z⟩` together with `⟨x, y⟩ = sign⟨y, z⟩`.

The three inequalities of `inner_mul_le_inner_of_abs_le` are `⟨x,y⟩⟨y,z⟩ ≤
|⟨x,y⟩||⟨y,z⟩| ≤ |⟨x,y⟩|⟨x,z⟩ ≤ ⟨x,z⟩`, and equality throughout forces
`|⟨y,z⟩| = ⟨x,z⟩` and — once `⟨x,z⟩ > 0` makes `⟨y,z⟩` nonzero — `|⟨x,y⟩| = 1`
with the sign of `⟨y,z⟩`.

Source: arXiv:2411.04990v2, §A, `lemma:scalar`. -/
theorem inner_mul_eq_inner_iff
    {d : ℕ} (x y z : EucSpace d) (hx : ‖x‖ = 1) (hy : ‖y‖ = 1)
    (h : |inner (𝕜 := ℝ) y z| ≤ inner (𝕜 := ℝ) x z) :
    inner (𝕜 := ℝ) x y * inner (𝕜 := ℝ) y z = inner (𝕜 := ℝ) x z ↔
      inner (𝕜 := ℝ) x z = 0 ∨
        (|inner (𝕜 := ℝ) y z| = inner (𝕜 := ℝ) x z ∧
          inner (𝕜 := ℝ) x y = Real.sign (inner (𝕜 := ℝ) y z)) := by
  set a : ℝ := inner (𝕜 := ℝ) x y with ha
  set b : ℝ := inner (𝕜 := ℝ) y z with hb
  set c : ℝ := inner (𝕜 := ℝ) x z with hc
  have hab : |a| ≤ 1 := by
    rw [ha]; simpa [hx, hy] using abs_real_inner_le_norm x y
  have hb0 : 0 ≤ |b| := abs_nonneg b
  have hc0 : 0 ≤ c := hb0.trans h
  constructor
  · intro heq
    rcases eq_or_lt_of_le hc0 with h0 | hpos
    · exact Or.inl h0.symm
    · refine Or.inr ?_
      have hbne : b ≠ 0 := by
        intro hbz
        rw [hbz, mul_zero] at heq
        exact absurd heq.symm hpos.ne'
      have hbabs : 0 < |b| := abs_pos.mpr hbne
      have habs : |a| * |b| = c := by rw [← abs_mul, heq, abs_of_nonneg hc0]
      have hbc : |b| = c := le_antisymm h (by nlinarith)
      rw [hbc] at habs
      have ha1 : |a| = 1 := mul_right_cancel₀ hpos.ne' (by rw [one_mul]; exact habs)
      refine ⟨hbc, ?_⟩
      rcases (abs_eq (by norm_num : (0 : ℝ) ≤ 1)).mp ha1 with h1 | h1
      · have hbpos : 0 < b := by
          rcases lt_or_gt_of_ne hbne with hneg | hpos'
          · rw [h1, one_mul] at heq
            rw [← heq, abs_of_neg hneg] at hbc
            linarith
          · exact hpos'
        rw [h1, Real.sign_of_pos hbpos]
      · have hbneg : b < 0 := by
          rcases lt_or_gt_of_ne hbne with hneg | hpos'
          · exact hneg
          · rw [h1] at heq
            rw [abs_of_pos hpos'] at hbc
            linarith
        rw [h1, Real.sign_of_neg hbneg]
  · rintro (h0 | ⟨h1, h2⟩)
    · have hbz : b = 0 := abs_eq_zero.mp (le_antisymm (h0 ▸ h) hb0)
      rw [hbz, mul_zero, h0]
    · rcases lt_trichotomy b 0 with hneg | hzero | hpos'
      · rw [h2, Real.sign_of_neg hneg, ← h1, abs_of_neg hneg]; ring
      · rw [h2, hzero, ← h1, hzero]; simp
      · rw [h2, Real.sign_of_pos hpos', ← h1, abs_of_pos hpos']; ring

/-- The hypotheses of `inner_mul_eq_inner_iff` are satisfiable away from the
degenerate case as well: three copies of the first standard basis vector make
`⟨x,z⟩ = 1` rather than `0`, so the second disjunct is the live one. -/
example :
    ‖EuclideanSpace.single (0 : Fin 1) (1 : ℝ)‖ = 1 ∧
      |inner (𝕜 := ℝ) (EuclideanSpace.single (0 : Fin 1) (1 : ℝ))
          (EuclideanSpace.single (0 : Fin 1) (1 : ℝ))|
        ≤ inner (𝕜 := ℝ) (EuclideanSpace.single (0 : Fin 1) (1 : ℝ))
            (EuclideanSpace.single (0 : Fin 1) (1 : ℝ)) := by
  refine ⟨by simp, ?_⟩
  simp

end Causal
end Transformer
