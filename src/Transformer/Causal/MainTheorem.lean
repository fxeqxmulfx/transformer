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

All three statements quantify over "almost every initial configuration" with
respect to the volume measure on `(𝕊^{d-1})^n`.  Its null sets are those of
the uniform law, which is read as every `σ` with `Perspective.UniformTuple d n σ`
— it holds for exactly one `σ`.  None of the three is proved here — `thm1.5`
and `thm2` are conjectures even in the paper.
-/

import Transformer.Basic
import Transformer.Causal.Basic
import Transformer.Perspective.Section2_FlowMap
import Transformer.Perspective.Section3_SmallBeta
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

*`d ≥ 2`.*  The source puts no bound on `d`, and at `d = 1` the statement is
false: `𝕊^0 = {±1}` has no tangent directions, so no token moves, and the
configuration `(1, -1)` has mass `1/4` under the uniform law and never reaches
`x_1(0)`.  `2 ≤ d` is added.

Not proved here.

Source: arXiv:2411.04990v2, §4, `thm1`. -/
theorem single_cluster (hd : 2 ≤ d) (β : ℝ) (Q K : ParamMatrix d) (hn : 1 ≤ n)
    (hβ : 0 ≤ β) :
    ∀ σ : Measure (SphereTuple d n), Perspective.UniformTuple d n σ →
    ∀ᵐ X₀ ∂σ, ∀ X : ℝ → SphereTuple d n, X 0 = X₀ →
      Causal.CSA d n β Q K (ContinuousLinearMap.id ℝ (EucSpace d)) X →
        ∀ k : Idx n,
          Filter.Tendsto (fun t : ℝ => X t k) Filter.atTop (nhds (X₀ ⟨0, hn⟩)) := by
  sorry

/-- The hypotheses of `single_cluster` are satisfiable: the circle, one token
and `β = 0`. -/
example : 2 ≤ 2 ∧ (1 : ℕ) ≤ 1 ∧ (0 : ℝ) ≤ 0 := ⟨le_rfl, le_rfl, le_rfl⟩

/-- **Conjecture (thm1.5).**  *Two-cluster convergence.*

Let `V` be diagonalizable with `d` distinct positive real eigenvalues, the
largest `λ_max` with unit eigenvector `ξ`.  Then for arbitrary `Q, K` and
almost any initialization the CSA dynamics satisfy

  `∀ k ∈ [n], lim_{t → ∞} x_k(t) ∈ {ξ, -ξ}`.

Diagonalizability is an eigenbasis `e` with eigenvalues `μ`: `d` linearly
independent eigenvectors in `ℝ^d`.  `λ_max` is an eigenvalue because `ξ` is
an eigenvector for it, and it is the largest because it bounds every `μ_i`.

Not proved here.

Source: arXiv:2411.04990v2, §4, `thm1.5`. -/
theorem two_cluster (β : ℝ) (Q K V : ParamMatrix d)
    (lam : ℝ) (ξ : SSphere d) (hβ : 0 ≤ β)
    (hdiag : ∃ (e : Idx d → EucSpace d) (μ : Idx d → ℝ), LinearIndependent ℝ e ∧
      (∀ i, V (e i) = μ i • e i) ∧ Function.Injective μ ∧ (∀ i, 0 < μ i) ∧
      ∀ i, μ i ≤ lam)
    (heig : V (ξ : EucSpace d) = lam • (ξ : EucSpace d)) :
    ∀ σ : Measure (SphereTuple d n), Perspective.UniformTuple d n σ →
    ∀ᵐ X₀ ∂σ, ∀ X : ℝ → SphereTuple d n, X 0 = X₀ → Causal.CSA d n β Q K V X →
      ∀ k : Idx n,
        Filter.Tendsto (fun t : ℝ => (X t k : EucSpace d)) Filter.atTop
            (nhds (ξ : EucSpace d)) ∨
          Filter.Tendsto (fun t : ℝ => (X t k : EucSpace d)) Filter.atTop
            (nhds (-(ξ : EucSpace d))) := by
  sorry

/-- The hypotheses of `two_cluster` are satisfiable: on the line, `V = I_1` is
diagonal with the single eigenvalue `1`, and its eigenvector is the basis
vector. -/
example :
    (0 : ℝ) ≤ 0 ∧
      (∃ (e : Idx 1 → EucSpace 1) (μ : Idx 1 → ℝ), LinearIndependent ℝ e ∧
        (∀ i, (ContinuousLinearMap.id ℝ (EucSpace 1)) (e i) = μ i • e i) ∧
        Function.Injective μ ∧ (∀ i, 0 < μ i) ∧ ∀ i, μ i ≤ 1) ∧
      (ContinuousLinearMap.id ℝ (EucSpace 1)) (EuclideanSpace.single (0 : Fin 1) (1 : ℝ))
        = (1 : ℝ) • EuclideanSpace.single (0 : Fin 1) (1 : ℝ) := by
  refine ⟨le_rfl, ⟨fun _ => EuclideanSpace.single (0 : Fin 1) (1 : ℝ), fun _ => 1,
    linearIndependent_unique_iff.mpr (by simp), fun _ => by simp,
    fun i j _ => Subsingleton.elim i j, fun _ => one_pos, fun _ => le_rfl⟩, by simp⟩

/-- **Conjecture (thm2).**  *Single-cluster convergence with `λ_max > 0`,
`dim L ≥ 2`.*

Let `λ > 0`, let `L` be an eigenspace of `V` for `λ` with `dim L ≥ 2`, let `V`
preserve `L^⊥`, and let `⟨V z, z⟩ < λ ‖z‖²` for every `z ∈ L^⊥ \ {0}` — so `λ`
really is the top eigenvalue and everything transverse to `L` decays relative
to it.  Then almost every initialization yields convergence of every token to
the normalized `L`-component of `x_1(0)`, which is defined as soon as that
component is nonzero.

The source defines `ξ` in words as "the normalized `L`-component of `x_1(0)`"
and then in symbols as `y_1 := P_{L^⊥}(x_1(0))`, `ξ := y_1/|y_1|`.  The
symbols contradict the words and the surrounding discussion (tokens are
attracted *into* `L`); the words are taken, `ξ` is the projection on `L`.

Not proved here.

Source: arXiv:2411.04990v2, §4, `thm2`. -/
theorem subspace_cluster (β : ℝ) (Q K V : ParamMatrix d)
    (lam : ℝ) (L : Submodule ℝ (EucSpace d)) (hn : 1 ≤ n)
    (hβ : 0 ≤ β) (hlam : 0 < lam)
    (heig : ∀ v ∈ L, V v = lam • v)
    (hdim : 2 ≤ Module.finrank ℝ L)
    (hinv : ∀ z ∈ Lᗮ, V z ∈ Lᗮ)
    (hdecay : ∀ z ∈ Lᗮ, z ≠ 0 → inner (𝕜 := ℝ) (V z) z < lam * ‖z‖ ^ 2) :
    ∀ σ : Measure (SphereTuple d n), Perspective.UniformTuple d n σ →
    ∀ᵐ X₀ ∂σ, ∀ X : ℝ → SphereTuple d n, X 0 = X₀ → Causal.CSA d n β Q K V X →
      L.starProjection ((X₀ ⟨0, hn⟩ : SSphere d) : EucSpace d) ≠ 0 →
        ∀ k : Idx n,
          Filter.Tendsto (fun t : ℝ => (X t k : EucSpace d)) Filter.atTop
            (nhds (‖L.starProjection ((X₀ ⟨0, hn⟩ : SSphere d) : EucSpace d)‖⁻¹ •
              L.starProjection ((X₀ ⟨0, hn⟩ : SSphere d) : EucSpace d))) := by
  sorry

/-- The hypotheses of `subspace_cluster` are satisfiable: in the plane, with
`L = ⊤`, `V = I_2` and `λ = 1`.  The whole plane has rank `2`, is invariant, and
`⊤ᗮ = ⊥`, so the decay condition holds on an empty set of `z`. -/
example :
    (0 : ℝ) ≤ 0 ∧ (0 : ℝ) < 1 ∧
      (∀ v ∈ (⊤ : Submodule ℝ (EucSpace 2)),
        (ContinuousLinearMap.id ℝ (EucSpace 2)) v = (1 : ℝ) • v) ∧
      2 ≤ Module.finrank ℝ (⊤ : Submodule ℝ (EucSpace 2)) ∧
      (∀ z ∈ (⊤ : Submodule ℝ (EucSpace 2))ᗮ,
        (ContinuousLinearMap.id ℝ (EucSpace 2)) z ∈ (⊤ : Submodule ℝ (EucSpace 2))ᗮ) ∧
      (∀ z ∈ (⊤ : Submodule ℝ (EucSpace 2))ᗮ, z ≠ 0 →
        inner (𝕜 := ℝ) ((ContinuousLinearMap.id ℝ (EucSpace 2)) z) z < 1 * ‖z‖ ^ 2) := by
  refine ⟨le_rfl, one_pos, fun v _ => by simp, ?_, fun z hz => hz, ?_⟩
  · rw [finrank_top ℝ (EucSpace 2), finrank_euclideanSpace_fin]
  · intro z hz hz0
    rw [Submodule.top_orthogonal_eq_bot, Submodule.mem_bot] at hz
    exact absurd hz hz0

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
