import Transformer.AMSGrad.Section1_AMSGrad
import Mathlib.LinearAlgebra.Matrix.PosDef
import Mathlib.Analysis.Convex.Basic
import Mathlib.Algebra.Order.Star.Real

/-
# AMSGrad — projections are non-expansive

§2 of arXiv:1904.03590v4, Lemma 2.7 (`McM&Str`, Lemma 3 of McMahan and
Streeter, as quoted by Reddi et al.): the projection onto a convex set in the
norm `‖Q^{1/2} ·‖` is non-expansive in that norm.

**What the source says and what is carried here.**

* The source writes `u = min_{x∈F} ‖Q^{1/2}(x - z)‖`, meaning the minimizer.
  `u` is taken as a point of `F` minimizing `(x - z)ᵀQ(x - z)` over `F`, and
  `‖Q^{1/2}v‖` is `√(vᵀQv)`.

* The proof is the variational inequality `(u - z)ᵀQ(x - u) ≥ 0` for
  `x ∈ F`, `proj_variational`, added twice.

Source: arXiv:1904.03590v4, §2, Lemma 2.7.
-/

open Matrix

namespace Transformer
namespace AMSGrad

variable {d : ℕ}

/-- A positive definite real matrix gives a symmetric bilinear form. -/
theorem posDef_comm {Q : Matrix (Fin d) (Fin d) ℝ} (hQ : Q.PosDef)
    (v w : Vec d) : v ⬝ᵥ Q *ᵥ w = w ⬝ᵥ Q *ᵥ v := by
  have hT : Qᵀ = Q := by
    have := hQ.isHermitian.eq
    rwa [conjTranspose_eq_transpose_of_trivial] at this
  rw [dotProduct_mulVec, dotProduct_comm, ← vecMul_transpose, hT]

/-- A positive definite real matrix gives a non-negative quadratic form. -/
theorem posDef_nonneg {Q : Matrix (Fin d) (Fin d) ℝ} (hQ : Q.PosDef)
    (v : Vec d) : 0 ≤ v ⬝ᵥ Q *ᵥ v := by
  simpa using hQ.posSemidef.dotProduct_mulVec_nonneg v

/-- **The variational inequality.**  If `u ∈ F` minimizes `(x - z)ᵀQ(x - z)`
over a convex `F`, then `(u - z)ᵀQ(x - u) ≥ 0` for every `x ∈ F`.
arXiv:1904.03590v4, §2, the proof of Lemma 2.7. -/
theorem proj_variational {Q : Matrix (Fin d) (Fin d) ℝ} (hQ : Q.PosDef) {F : Set (Vec d)}
    (hF : Convex ℝ F) {z u : Vec d} (hu : u ∈ F)
    (hmin : ∀ x ∈ F, (u - z) ⬝ᵥ Q *ᵥ (u - z) ≤ (x - z) ⬝ᵥ Q *ᵥ (x - z))
    {x : Vec d} (hx : x ∈ F) : 0 ≤ (u - z) ⬝ᵥ Q *ᵥ (x - u) := by
  set a := (u - z) ⬝ᵥ Q *ᵥ (x - u)
  set b := (x - u) ⬝ᵥ Q *ᵥ (x - u)
  have hb : 0 ≤ b := posDef_nonneg hQ _
  have hs : ∀ s : ℝ, 0 < s → s ≤ 1 → 0 ≤ 2 * a + s * b := by
    intro s hs0 hs1
    have hw : u + s • (x - u) ∈ F := by
      have := hF hu hx (by linarith : 0 ≤ 1 - s) hs0.le (by ring)
      convert this using 1
      rw [smul_sub, sub_smul, one_smul]; abel
    have h := hmin _ hw
    have he : u + s • (x - u) - z = (u - z) + s • (x - u) := by abel
    rw [he] at h
    simp only [mulVec_add, mulVec_smul, dotProduct_add, add_dotProduct, dotProduct_smul,
      smul_dotProduct, smul_eq_mul] at h
    have hc := posDef_comm hQ (x - u) (u - z)
    have : s * 0 ≤ s * (2 * a + s * b) := by nlinarith
    exact le_of_mul_le_mul_left this hs0
  by_contra ha
  replace ha := not_le.mp ha
  set s := min 1 (-a / (b + 1))
  have hs0 : 0 < s := lt_min one_pos (div_pos (by linarith) (by linarith))
  have hsb : s * b ≤ -a / (b + 1) * b := mul_le_mul_of_nonneg_right (min_le_right _ _) hb
  have hlt : -a / (b + 1) * b ≤ -a := by
    rw [div_mul_eq_mul_div, div_le_iff₀ (by linarith)]; nlinarith
  linarith [hs s hs0 (min_le_left _ _)]

/-- **Lemma 2.7 (McMahan and Streeter).**  For `Q` positive definite and `F`
convex, if `u₁, u₂ ∈ F` minimize `‖Q^{1/2}(x - z₁)‖`, `‖Q^{1/2}(x - z₂)‖` over
`F`, then `‖Q^{1/2}(u₁ - u₂)‖ ≤ ‖Q^{1/2}(z₁ - z₂)‖`.

Source: arXiv:1904.03590v4, §2, Lemma 2.7. -/
theorem mcm_str {Q : Matrix (Fin d) (Fin d) ℝ} (hQ : Q.PosDef) {F : Set (Vec d)}
    (hF : Convex ℝ F) {z₁ z₂ u₁ u₂ : Vec d} (hu₁ : u₁ ∈ F) (hu₂ : u₂ ∈ F)
    (hmin₁ : ∀ x ∈ F, (u₁ - z₁) ⬝ᵥ Q *ᵥ (u₁ - z₁) ≤ (x - z₁) ⬝ᵥ Q *ᵥ (x - z₁))
    (hmin₂ : ∀ x ∈ F, (u₂ - z₂) ⬝ᵥ Q *ᵥ (u₂ - z₂) ≤ (x - z₂) ⬝ᵥ Q *ᵥ (x - z₂)) :
    Real.sqrt ((u₁ - u₂) ⬝ᵥ Q *ᵥ (u₁ - u₂)) ≤ Real.sqrt ((z₁ - z₂) ⬝ᵥ Q *ᵥ (z₁ - z₂)) := by
  refine Real.sqrt_le_sqrt ?_
  have h₁ := proj_variational hQ hF hu₁ hmin₁ hu₂
  have h₂ := proj_variational hQ hF hu₂ hmin₂ hu₁
  have h₃ := posDef_nonneg hQ ((z₁ - z₂) - (u₁ - u₂))
  have c := posDef_comm hQ
  simp only [mulVec_sub, dotProduct_sub, sub_dotProduct] at h₁ h₂ h₃ ⊢
  linarith [c u₁ u₂, c u₁ z₁, c u₁ z₂, c u₂ z₁, c u₂ z₂, c z₁ z₂]

/-- The hypotheses of `mcm_str` are satisfiable: `Q = 1`, `F = ℝ^d`, `uᵢ = zᵢ`. -/
example {d : ℕ} (z : Vec d) :
    (1 : Matrix (Fin d) (Fin d) ℝ).PosDef ∧ Convex ℝ (Set.univ : Set (Vec d)) ∧
      ∀ x ∈ (Set.univ : Set (Vec d)),
        (z - z) ⬝ᵥ (1 : Matrix (Fin d) (Fin d) ℝ) *ᵥ (z - z) ≤
          (x - z) ⬝ᵥ (1 : Matrix (Fin d) (Fin d) ℝ) *ᵥ (x - z) :=
  ⟨Matrix.PosDef.one, convex_univ, fun x _ => by
    simpa using posDef_nonneg Matrix.PosDef.one (x - z)⟩

end AMSGrad
end Transformer
