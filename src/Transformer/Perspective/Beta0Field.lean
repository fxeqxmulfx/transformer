/-
# §4 — the `β = 0` drift as a Lipschitz vector field

Geshkovski, Letrouit, Polyanskiy, Rigollet — arXiv:2312.10794v5,
*A mathematical perspective on Transformers*.

`e:Snonres0` reads `ẋ_i = Proj_{x_i}( (1/n) Σ_j x_j )`, and the right-hand
side is a vector field on `(ℝ^d)^n` that does not depend on time.  This file
names it (`beta0Field`) and proves the one property the Grönwall comparison of
`e:approxsphere` needs: on tuples of unit vectors it is Lipschitz with constant
`3` for the sup-norm.

The constant is the survey's: the `e^{3t}` of `e:approxsphere` is exactly
`e^{Kt}` for this `K`.  It comes out of the splitting

  `Proj_a m - Proj_b m' = Proj_a (m - m') - ⟨a, m'⟩ (a - b) - ⟨a - b, m'⟩ b`,

whose three summands are bounded by `‖m - m'‖`, `‖m'‖ ‖a - b‖` and
`‖a - b‖ ‖m'‖ ‖b‖`; on unit vectors `‖m'‖ ≤ 1` and `‖m - m'‖ ≤ sup_j ‖a_j - b_j‖`.
The naive split `Proj_a m - Proj_b m' = (m - m') - (⟨a,m⟩a - ⟨b,m'⟩b)` gives `4`
instead, which would turn `e^{3t}` into `e^{4t}`.
-/

import Transformer.Perspective.Section3_SmallBeta
import Mathlib.Topology.MetricSpace.Pseudo.Pi

open scoped BigOperators NNReal
open Real

namespace Transformer
namespace Perspective

variable (d n : ℕ)

/-- The tuples all of whose entries are unit vectors: the image of
`SphereTuple d n` inside `(ℝ^d)^n`, written as a set so that the Grönwall
comparison can be run in the flat space `Idx n → EucSpace d`. -/
def unitTuples : Set (Idx n → EucSpace d) := { Y | ∀ i : Idx n, ‖Y i‖ = 1 }

/-- The empirical mean `(1/n) Σ_j y_j` of a tuple. -/
noncomputable def meanTuple (Y : Idx n → EucSpace d) : EucSpace d :=
  ((n : ℝ)⁻¹) • ∑ j : Idx n, Y j

/-- **The vector field of `e:Snonres0`:** `(β0 Y)_i = Proj_{y_i}( (1/n) Σ_j y_j )`.

Source: arXiv:2312.10794v5, §4, `e:Snonres0`. -/
noncomputable def beta0Field (Y : Idx n → EucSpace d) : Idx n → EucSpace d :=
  fun i => proj d (Y i) (meanTuple d n Y)

/-- A path solves `e:Snonres0` exactly when it is an integral curve of
`beta0Field`. -/
theorem beta0Dynamics_iff (X : ℝ → SphereTuple d n) :
    beta0Dynamics d n X ↔
      ∀ t : ℝ, ∀ i : Idx n,
        HasDerivAt (fun s => (X s i : EucSpace d))
          (beta0Field d n (fun j => ((X t j : EucSpace d))) i) t :=
  Iff.rfl

/-- The mean of a tuple whose entries are all shorter than `r` is shorter
than `r`. -/
theorem norm_meanTuple_le {Y : Idx n → EucSpace d} {r : ℝ} (hn : 0 < n)
    (h : ∀ i : Idx n, ‖Y i‖ ≤ r) : ‖meanTuple d n Y‖ ≤ r := by
  have hn' : (0 : ℝ) < (n : ℝ) := Nat.cast_pos.mpr hn
  have hsum : ‖∑ j : Idx n, Y j‖ ≤ (n : ℝ) * r := by
    refine (norm_sum_le _ _).trans ?_
    calc ∑ j : Idx n, ‖Y j‖ ≤ ∑ _j : Idx n, r := Finset.sum_le_sum fun j _ => h j
      _ = (n : ℝ) * r := by simp [mul_comm]
  have hnorm : ‖((n : ℝ)⁻¹)‖ = (n : ℝ)⁻¹ := by
    rw [Real.norm_eq_abs, abs_of_nonneg (by positivity)]
  rw [meanTuple, norm_smul, hnorm]
  calc (n : ℝ)⁻¹ * ‖∑ j : Idx n, Y j‖ ≤ (n : ℝ)⁻¹ * ((n : ℝ) * r) := by
        gcongr
    _ = r := by field_simp

/-- The satisfiability of the hypotheses of `norm_meanTuple_le`: one token at
the base point. -/
example : (0 : ℕ) < 1 ∧ ∀ _i : Idx 1, ‖((basePoint 0 : SSphere 1) : EucSpace 1)‖ ≤ 1 :=
  ⟨one_pos, fun _ => le_of_eq (mem_sphere_zero_iff_norm.mp (basePoint 0).2)⟩

/-- Averaging commutes with differences. -/
theorem meanTuple_sub (Y Z : Idx n → EucSpace d) :
    meanTuple d n Y - meanTuple d n Z = meanTuple d n (fun j => Y j - Z j) := by
  simp only [meanTuple, ← smul_sub, Finset.sum_sub_distrib]

/-- **The `β = 0` drift is `3`-Lipschitz on tuples of unit vectors.**

This is the constant that produces the `e^{3t}` of `e:approxsphere`.

Source: arXiv:2312.10794v5, §4, the Grönwall step before `e:approxsphere`. -/
theorem lipschitzOnWith_beta0Field :
    LipschitzOnWith 3 (beta0Field d n) (unitTuples d n) := by
  rw [lipschitzOnWith_iff_dist_le_mul]
  intro Y hY Z hZ
  have hd0 : (0 : ℝ) ≤ dist Y Z := dist_nonneg
  have h3 : ((3 : ℝ≥0) : ℝ) = 3 := by norm_num
  rw [h3, dist_pi_le_iff (by positivity)]
  intro i
  have hn : 0 < n := i.pos
  set δ := dist Y Z with hδ
  have hstep : ∀ j : Idx n, ‖Y j - Z j‖ ≤ δ := by
    intro j
    rw [← dist_eq_norm]
    exact dist_le_pi_dist Y Z j
  have hYi : ‖Y i‖ = 1 := hY i
  have hZi : ‖Z i‖ = 1 := hZ i
  have hmZ : ‖meanTuple d n Z‖ ≤ 1 :=
    norm_meanTuple_le d n hn fun j => le_of_eq (hZ j)
  have hmsub : ‖meanTuple d n Y - meanTuple d n Z‖ ≤ δ := by
    rw [meanTuple_sub]
    exact norm_meanTuple_le d n hn hstep
  have hdecomp : beta0Field d n Y i - beta0Field d n Z i
      = proj d (Y i) (meanTuple d n Y - meanTuple d n Z)
        - ((inner (𝕜 := ℝ) (Y i) (meanTuple d n Z)) • (Y i - Z i)
            + (inner (𝕜 := ℝ) (Y i - Z i) (meanTuple d n Z)) • Z i) := by
    simp only [beta0Field, proj, inner_sub_right, inner_sub_left, sub_smul]
    module
  have hA : ‖proj d (Y i) (meanTuple d n Y - meanTuple d n Z)‖ ≤ δ :=
    (norm_proj_le hYi _).trans hmsub
  have hB : ‖(inner (𝕜 := ℝ) (Y i) (meanTuple d n Z)) • (Y i - Z i)‖ ≤ δ := by
    rw [norm_smul, Real.norm_eq_abs]
    have h1 : |inner (𝕜 := ℝ) (Y i) (meanTuple d n Z)| ≤ 1 := by
      have := abs_real_inner_le_norm (Y i) (meanTuple d n Z)
      rw [hYi, one_mul] at this
      exact this.trans hmZ
    calc |inner (𝕜 := ℝ) (Y i) (meanTuple d n Z)| * ‖Y i - Z i‖
        ≤ 1 * δ := by
          exact mul_le_mul h1 (hstep i) (norm_nonneg _) zero_le_one
      _ = δ := one_mul δ
  have hC : ‖(inner (𝕜 := ℝ) (Y i - Z i) (meanTuple d n Z)) • Z i‖ ≤ δ := by
    rw [norm_smul, Real.norm_eq_abs, hZi, mul_one]
    have := abs_real_inner_le_norm (Y i - Z i) (meanTuple d n Z)
    refine this.trans ?_
    calc ‖Y i - Z i‖ * ‖meanTuple d n Z‖ ≤ δ * 1 := by
          exact mul_le_mul (hstep i) hmZ (norm_nonneg _) hd0
      _ = δ := mul_one δ
  rw [dist_eq_norm, hdecomp]
  calc ‖proj d (Y i) (meanTuple d n Y - meanTuple d n Z)
          - ((inner (𝕜 := ℝ) (Y i) (meanTuple d n Z)) • (Y i - Z i)
              + (inner (𝕜 := ℝ) (Y i - Z i) (meanTuple d n Z)) • Z i)‖
      ≤ ‖proj d (Y i) (meanTuple d n Y - meanTuple d n Z)‖
          + ‖(inner (𝕜 := ℝ) (Y i) (meanTuple d n Z)) • (Y i - Z i)
              + (inner (𝕜 := ℝ) (Y i - Z i) (meanTuple d n Z)) • Z i‖ := norm_sub_le _ _
    _ ≤ ‖proj d (Y i) (meanTuple d n Y - meanTuple d n Z)‖
          + (‖(inner (𝕜 := ℝ) (Y i) (meanTuple d n Z)) • (Y i - Z i)‖
              + ‖(inner (𝕜 := ℝ) (Y i - Z i) (meanTuple d n Z)) • Z i‖) := by
          gcongr
          exact norm_add_le _ _
    _ ≤ δ + (δ + δ) := by gcongr
    _ = 3 * δ := by ring

end Perspective
end Transformer
