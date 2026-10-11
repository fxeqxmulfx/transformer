/-
# A smooth ambient extension of the spherical attention field

ArXiv:2312.10794v5, §2, `eq: SA`, and arXiv:2411.04551v3, §5,
`prop:generic-discrete`. Global existence of the actual spherical
flow is needed before the latter's conditional cluster map can be used.

The ambient polynomial projection grows faster than linearly away from
the sphere. Multiplying it by `(n+1)/(1+Σ_j ‖y_j‖²)` gives a smooth
field of linear growth. This factor is exactly one on unit tuples.
The following flow module proves sphere invariance from the resulting
ODE, so the extension supplies solutions of the original dynamics.

The softmax, scores and projected average remain the actual formulas.
No convergence, invariance or well-posedness is built into a definition.
The manuscript does not prescribe this ambient extension; it is an
auxiliary construction whose spherical restriction is the source's ODE.
-/

import Transformer.Perspective.SAField
import Mathlib.Analysis.Calculus.ContDiff.RCLike
import Mathlib.Analysis.InnerProductSpace.Calculus

open scoped BigOperators
open Real

namespace Transformer.Perspective

variable {d n : ℕ}

/-- The smooth growth-control factor, equal to one on unit tuples.
Source: the ambient extension used to justify `eq: SA` in §2 of
arXiv:2312.10794v5; its restriction is verified below. -/
noncomputable def saExtensionScale (Y : Idx n → EucSpace d) : ℝ :=
  ((n : ℝ) + 1) / (1 + ∑ j : Idx n, inner (𝕜 := ℝ) (Y j) (Y j))

/-- A smooth field of linear growth agreeing with the actual attention
field on unit tuples. Source: arXiv:2312.10794v5, §2, `eq: SA`.
The scalar factor is a proof device only outside the spherical state space. -/
noncomputable def saGlobalField (β : ℝ) (Y : Idx n → EucSpace d) : Idx n → EucSpace d :=
  saExtensionScale Y • saField d n β Y

/-- The extension's denominator is strictly positive at every tuple.
Source: arXiv:2312.10794v5, §2, the smooth ambient extension of `eq: SA`. -/
theorem saExtensionDenom_pos (Y : Idx n → EucSpace d) :
    0 < 1 + ∑ j : Idx n, inner (𝕜 := ℝ) (Y j) (Y j) := by
  have hs : 0 ≤ ∑ j : Idx n, inner (𝕜 := ℝ) (Y j) (Y j) :=
    Finset.sum_nonneg fun j _ => by rw [real_inner_self_eq_norm_sq]; positivity
  linarith

/-- The extension scales by a positive number, preserving the field's sign.
Source: arXiv:2312.10794v5, §2, the extension of `eq: SA`. -/
theorem saExtensionScale_pos (Y : Idx n → EucSpace d) : 0 < saExtensionScale Y :=
  div_pos (by positivity) (saExtensionDenom_pos Y)

/-- The extension agrees exactly with the original dynamics on unit tuples.
Source: arXiv:2312.10794v5, §2, `eq: SA`. -/
theorem saExtensionScale_eq_one (Y : Idx n → EucSpace d) (hY : ∀ j, ‖Y j‖ = 1) :
    saExtensionScale Y = 1 := by
  have hs : ∀ j, inner (𝕜 := ℝ) (Y j) (Y j) = 1 := by
    intro j
    rw [real_inner_self_eq_norm_sq, hY j, one_pow]
  simp only [saExtensionScale, hs, Finset.sum_const, Finset.card_univ,
    Fintype.card_fin, nsmul_eq_mul, mul_one]
  rw [add_comm, div_self (by positivity)]

example : ∀ j : Idx 1, ‖(fun _ => (basePoint 0 : EucSpace 1)) j‖ = 1 := by
  intro j
  simp [basePoint, PiLp.norm_single]

/-- The actual attention average is smooth for every temperature.
Source: arXiv:2312.10794v5, §2, the exponential weights in `eq: SA`. -/
theorem contDiff_saAvg (β : ℝ) (i : Idx n) : ContDiff ℝ (⊤ : ℕ∞)
    (fun Y : Idx n → EucSpace d =>
      ∑ j : Idx n, softmaxWeight (saScore d n β Y i) j • Y j) := by
  have hn : 0 < n := Fin.pos_iff_nonempty.mpr ⟨i⟩
  have hcoord (j : Idx n) : ContDiff ℝ (⊤ : ℕ∞)
      (fun Y : Idx n → EucSpace d => Y j) := by fun_prop
  have he (j : Idx n) : ContDiff ℝ (⊤ : ℕ∞)
      (fun Y : Idx n → EucSpace d => Real.exp (β * inner (𝕜 := ℝ) (Y i) (Y j))) :=
    (contDiff_const.mul ((hcoord i).inner ℝ (hcoord j))).exp
  have hw (j : Idx n) : ContDiff ℝ (⊤ : ℕ∞)
      (fun Y : Idx n → EucSpace d => softmaxWeight (saScore d n β Y i) j) := by
    unfold softmaxWeight saScore
    exact ContDiff.div (he j) (ContDiff.sum fun k _ => he k)
      (fun Y => (softmaxPartition_pos hn (saScore d n β Y i)).ne')
  exact ContDiff.sum fun j _ => (hw j).smul (hcoord j)

/-- The actual ambient attention field is smooth for every temperature.
Source: arXiv:2312.10794v5, §2, the projection in `eq: SA`. -/
theorem contDiff_saField (β : ℝ) : ContDiff ℝ (⊤ : ℕ∞) (saField d n β) := by
  apply contDiff_pi.mpr
  intro i
  have hcoord : ContDiff ℝ (⊤ : ℕ∞) (fun Y : Idx n → EucSpace d => Y i) := by fun_prop
  exact (contDiff_saAvg β i).sub ((hcoord.inner ℝ (contDiff_saAvg β i)).smul hcoord)

/-- The extension factor is smooth because its denominator never vanishes.
Source: arXiv:2312.10794v5, §2, the ambient extension of `eq: SA`. -/
theorem contDiff_saExtensionScale :
    ContDiff ℝ (⊤ : ℕ∞) (saExtensionScale (d := d) (n := n)) := by
  unfold saExtensionScale
  have hden : ContDiff ℝ (⊤ : ℕ∞) (fun Y : Idx n → EucSpace d =>
      1 + ∑ j : Idx n, inner (𝕜 := ℝ) (Y j) (Y j)) := by
    apply contDiff_const.add (ContDiff.sum fun j _ => ?_)
    exact (show ContDiff ℝ (⊤ : ℕ∞) (fun Y : Idx n → EucSpace d => Y j) by fun_prop).inner
      ℝ (by fun_prop)
  exact ContDiff.div contDiff_const hden (fun Y => (saExtensionDenom_pos Y).ne')

/-- The extended field is smooth on the whole ambient product space.
Source: arXiv:2312.10794v5, §2, `eq: SA`. -/
theorem contDiff_saGlobalField (β : ℝ) : ContDiff ℝ (⊤ : ℕ∞) (saGlobalField (d := d) (n := n) β) :=
  contDiff_saExtensionScale.smul (contDiff_saField β)

/-- A softmax average is bounded by the maximum norm of its input tuple.
Source: arXiv:2312.10794v5, §2, the normalized average in `eq: SA`. -/
theorem norm_saAvg_le_norm (β : ℝ) (Y : Idx n → EucSpace d) (i : Idx n) :
    ‖∑ j : Idx n, softmaxWeight (saScore d n β Y i) j • Y j‖ ≤ ‖Y‖ := by
  have hn : 0 < n := Fin.pos_iff_nonempty.mpr ⟨i⟩
  refine (norm_sum_le _ _).trans ?_
  calc ∑ j : Idx n, ‖softmaxWeight (saScore d n β Y i) j • Y j‖
      ≤ ∑ j : Idx n, softmaxWeight (saScore d n β Y i) j * ‖Y‖ := by
        apply Finset.sum_le_sum
        intro j _
        rw [norm_smul, Real.norm_of_nonneg (softmaxWeight_nonneg _ j)]
        exact mul_le_mul_of_nonneg_left (norm_le_pi_norm Y j) (softmaxWeight_nonneg _ j)
    _ = ‖Y‖ := by rw [← Finset.sum_mul, sum_softmaxWeight hn, one_mul]

/-- **The extended field has linear growth.** The denominator absorbs the
quadratic norm factor of the ambient projection.
Source: the global-existence justification for `eq: SA` in §2 of
arXiv:2312.10794v5 and `prop:generic-discrete` in §5 of arXiv:2411.04551v3. -/
theorem norm_saGlobalField_le (β : ℝ) (Y : Idx n → EucSpace d) :
    ‖saGlobalField β Y‖ ≤ ((n : ℝ) + 1) * ‖Y‖ := by
  apply (pi_norm_le_iff_of_nonneg (by positivity)).mpr
  intro i
  let v := ∑ j : Idx n, softmaxWeight (saScore d n β Y i) j • Y j
  have hv : ‖v‖ ≤ ‖Y‖ := norm_saAvg_le_norm β Y i
  have hinner : |inner (𝕜 := ℝ) (Y i) v| ≤ ‖Y i‖ * ‖Y‖ :=
    (abs_real_inner_le_norm _ _).trans (mul_le_mul_of_nonneg_left hv (norm_nonneg _))
  have hproj : ‖saField d n β Y i‖ ≤ (1 + ‖Y i‖ ^ 2) * ‖Y‖ := by
    calc ‖saField d n β Y i‖ ≤ ‖v‖ + ‖inner (𝕜 := ℝ) (Y i) v • Y i‖ := norm_sub_le _ _
      _ = ‖v‖ + |inner (𝕜 := ℝ) (Y i) v| * ‖Y i‖ := by rw [norm_smul, Real.norm_eq_abs]
      _ ≤ ‖Y‖ + (‖Y i‖ * ‖Y‖) * ‖Y i‖ :=
        add_le_add hv (mul_le_mul_of_nonneg_right hinner (norm_nonneg _))
      _ = (1 + ‖Y i‖ ^ 2) * ‖Y‖ := by ring
  have hsum : ‖Y i‖ ^ 2 ≤ ∑ j : Idx n, inner (𝕜 := ℝ) (Y j) (Y j) := by
    rw [← real_inner_self_eq_norm_sq]
    apply Finset.single_le_sum _ (Finset.mem_univ i)
    intro j _
    rw [real_inner_self_eq_norm_sq]
    positivity
  have hscale : saExtensionScale Y * (1 + ‖Y i‖ ^ 2) ≤ (n : ℝ) + 1 := by
    unfold saExtensionScale
    rw [div_mul_eq_mul_div]
    apply (div_le_iff₀ (saExtensionDenom_pos Y)).mpr
    exact mul_le_mul_of_nonneg_left (add_le_add le_rfl hsum) (by positivity)
  change ‖saExtensionScale Y • saField d n β Y i‖ ≤ _
  rw [norm_smul, Real.norm_of_nonneg (saExtensionScale_pos Y).le]
  calc saExtensionScale Y * ‖saField d n β Y i‖
      ≤ saExtensionScale Y * ((1 + ‖Y i‖ ^ 2) * ‖Y‖) :=
        mul_le_mul_of_nonneg_left hproj (saExtensionScale_pos Y).le
    _ = (saExtensionScale Y * (1 + ‖Y i‖ ^ 2)) * ‖Y‖ := by ring
    _ ≤ ((n : ℝ) + 1) * ‖Y‖ := mul_le_mul_of_nonneg_right hscale (norm_nonneg _)

end Transformer.Perspective
