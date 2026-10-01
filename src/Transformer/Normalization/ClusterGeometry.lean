/-
# Geometry of a normalized token cluster

The mean and variance identities used in Appendix C, proof of Theorem 4.3,
of arXiv:2510.22026v2. These are finite-dimensional identities; no convergence
or probabilistic premise is used.
-/

import Transformer.Normalization.Velocities
import Mathlib.Analysis.InnerProductSpace.Calculus
import Mathlib.Analysis.Calculus.Deriv.Add
import Mathlib.Analysis.Calculus.Deriv.Mul

open scoped BigOperators

namespace Transformer.Normalization

variable {d n : ℕ}

/-- The empirical mean of the token directions in Appendix C of
arXiv:2510.22026v2, proof of Theorem 4.3. -/
noncomputable def tokenMean (Θ : Idx n → EucSpace d) : EucSpace d :=
  (n : ℝ)⁻¹ • ∑ j, Θ j

/-- The centered directions sum to zero. Source: arXiv:2510.22026v2,
Appendix C, the cancellation in the derivative of the variance. -/
theorem sum_sub_tokenMean (hn : 0 < n) (Θ : Idx n → EucSpace d) :
    ∑ j, (Θ j - tokenMean Θ) = 0 := by
  have hn' : (n : ℝ) ≠ 0 := by exact_mod_cast hn.ne'
  simp only [Finset.sum_sub_distrib, Finset.sum_const, Finset.card_univ,
    Fintype.card_fin, tokenMean]
  rw [← Nat.cast_smul_eq_nsmul ℝ, smul_smul]
  rw [mul_inv_cancel₀ hn', one_smul, sub_self]

/-- A nonempty cluster is available already with one token. -/
example : 0 < 1 := Nat.one_pos

/-- The sum of the inner products with the mean is `n ‖mean‖².
Source: arXiv:2510.22026v2, Appendix C, proof of Theorem 4.3. -/
theorem sum_inner_tokenMean (hn : 0 < n) (Θ : Idx n → EucSpace d) :
    ∑ j, inner (𝕜 := ℝ) (Θ j) (tokenMean Θ) = (n : ℝ) * ‖tokenMean Θ‖ ^ 2 := by
  have hn' : (n : ℝ) ≠ 0 := by exact_mod_cast hn.ne'
  have hsum : ∑ j, Θ j = (n : ℝ) • tokenMean Θ := by
    rw [tokenMean, smul_smul, mul_inv_cancel₀ hn', one_smul]
  rw [← sum_inner, hsum, real_inner_smul_left, real_inner_self_eq_norm_sq]

/-- One token satisfies the nonemptiness condition. -/
example : 0 < 1 := Nat.one_pos

/-- For unit directions, the empirical variance is `1 - ‖mean‖².
Source: arXiv:2510.22026v2, Appendix C, first display in the proof of
Theorem 4.3. -/
theorem intraClusterVar_eq_one_sub_norm_mean_sq (hn : 0 < n)
    (θ : ℝ → Idx n → EucSpace d) (t : ℝ) (hθ : ∀ j, ‖θ t j‖ = 1) :
    intraClusterVar d n θ t = 1 - ‖tokenMean (θ t)‖ ^ 2 := by
  have hn' : (n : ℝ) ≠ 0 := by exact_mod_cast hn.ne'
  have hs : ∑ j, ‖θ t j - tokenMean (θ t)‖ ^ 2 =
      (n : ℝ) - (n : ℝ) * ‖tokenMean (θ t)‖ ^ 2 := by
    simp_rw [norm_sub_sq_real, hθ, one_pow]
    simp only [Finset.sum_add_distrib, Finset.sum_sub_distrib,
      Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul,
      ← Finset.mul_sum, sum_inner_tokenMean hn]
    ring
  change (n : ℝ)⁻¹ * (∑ j, ‖θ t j - tokenMean (θ t)‖ ^ 2) = _
  rw [hs]
  field_simp

/-- A constant unit direction witnesses the variance identity's hypotheses. -/
example : 0 < 1 ∧ ∀ j : Idx 1,
    ‖(fun _ : ℝ => fun _ : Idx 1 => EuclideanSpace.single (0 : Fin 1) (1 : ℝ)) 0 j‖ = 1 :=
  ⟨Nat.one_pos, fun _ => by simp⟩

/-- Every mean projection lies in the same local cone as the tokens.
Source: arXiv:2510.22026v2, Appendix C, the estimate `Var ≤ δ`. -/
theorem inner_tokenMean_ge_of_localCone (hn : 0 < n) (δ : ℝ)
    (Θ : Idx n → EucSpace d)
    (hcone : ∀ j k, 1 - δ ≤ inner (𝕜 := ℝ) (Θ j) (Θ k)) (j : Idx n) :
    1 - δ ≤ inner (𝕜 := ℝ) (Θ j) (tokenMean Θ) := by
  have hn' : (0 : ℝ) < n := by exact_mod_cast hn
  rw [tokenMean, real_inner_smul_right, inner_sum]
  have hs : (n : ℝ) * (1 - δ) ≤ ∑ k, inner (𝕜 := ℝ) (Θ j) (Θ k) := by
    have h := Finset.sum_le_sum (s := Finset.univ) (fun k _ => hcone j k)
    simpa only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul] using h
  have h := mul_le_mul_of_nonneg_left hs (inv_nonneg.mpr hn'.le)
  simpa [← mul_assoc, inv_mul_cancel₀ hn'.ne'] using h

/-- Copies of a unit direction form a local cone with `δ = 0`. -/
example : 0 < 1 ∧ ∀ j k : Idx 1, 1 - (0 : ℝ) ≤
    inner (𝕜 := ℝ) ((fun _ : Idx 1 => EuclideanSpace.single (0 : Fin 1) (1 : ℝ)) j)
      ((fun _ : Idx 1 => EuclideanSpace.single (0 : Fin 1) (1 : ℝ)) k) :=
  ⟨Nat.one_pos, fun _ _ => by simp⟩

/-- The variance has the centered derivative stated in Appendix C of
arXiv:2510.22026v2, proof of Theorem 4.3. -/
theorem hasDerivWithinAt_intraClusterVar (hn : 0 < n)
    (θ : ℝ → Idx n → EucSpace d) (v : Idx n → EucSpace d) (s : Set ℝ) (t : ℝ)
    (hθ : ∀ j, HasDerivWithinAt (fun u => θ u j) (v j) s t) :
    HasDerivWithinAt (intraClusterVar d n θ)
      (2 / (n : ℝ) * ∑ j, inner (𝕜 := ℝ) (θ t j - tokenMean (θ t)) (v j)) s t := by
  have hm : HasDerivWithinAt (fun u => tokenMean (θ u)) (tokenMean v) s t :=
    HasDerivWithinAt.fun_const_smul (n : ℝ)⁻¹
      (HasDerivWithinAt.fun_sum (u := Finset.univ) fun j _ => hθ j)
  have hd := HasDerivWithinAt.const_mul (n : ℝ)⁻¹
    (HasDerivWithinAt.fun_sum (u := Finset.univ) fun j _ => ((hθ j).sub hm).norm_sq)
  have hcancel : ∑ j, inner (𝕜 := ℝ) (θ t j - tokenMean (θ t)) (tokenMean v) = 0 := by
    rw [← sum_inner, sum_sub_tokenMean hn, inner_zero_left]
  convert hd using 1
  · rfl
  · simp only [Pi.sub_apply, inner_sub_right, Finset.sum_sub_distrib,
      ← Finset.mul_sum, hcancel, sub_zero]
    ring

/-- A constant one-token path has the required derivatives. -/
example : 0 < 1 ∧ ∀ j : Idx 1,
    HasDerivWithinAt (fun _ : ℝ => (0 : Idx 1 → EucSpace 1) j) 0 (Set.Ici 0) 0 :=
  ⟨Nat.one_pos, fun _ => (hasDerivAt_const 0 (0 : EucSpace 1)).hasDerivWithinAt⟩

/-- Two unit directions in the same cap of height `1-δ` have mutual inner
product at least `1-4δ`. Source: arXiv:2510.22026v2, Appendix C, the local
cone used in the proof of Theorem 4.3. -/
theorem inner_ge_of_common_cap (δ : ℝ) (x y w : EucSpace d)
    (hx : ‖x‖ = 1) (hy : ‖y‖ = 1) (hw : ‖w‖ = 1)
    (hxw : 1 - δ ≤ inner (𝕜 := ℝ) w x) (hyw : 1 - δ ≤ inner (𝕜 := ℝ) w y) :
    1 - 4 * δ ≤ inner (𝕜 := ℝ) x y := by
  have ha : ‖x - w‖ ^ 2 ≤ 2 * δ := by
    rw [norm_sub_sq_real, hx, hw]
    rw [real_inner_comm x w] at hxw
    nlinarith
  have hb : ‖w - y‖ ^ 2 ≤ 2 * δ := by
    rw [norm_sub_sq_real, hw, hy]
    nlinarith
  have htri := norm_sub_le_norm_sub_add_norm_sub x w y
  have hsum : (‖x - w‖ + ‖w - y‖) ^ 2 ≤ 8 * δ := by
    nlinarith [sq_nonneg (‖x - w‖ - ‖w - y‖)]
  have hc : ‖x - y‖ ^ 2 ≤ 8 * δ := by
    nlinarith [norm_nonneg (x - y), norm_nonneg (x - w), norm_nonneg (w - y)]
  rw [norm_sub_sq_real, hx, hy] at hc
  nlinarith

/-- A single unit vector lies in its own cap with `δ = 0`. -/
example : let u := EuclideanSpace.single (0 : Fin 1) (1 : ℝ)
    ‖u‖ = 1 ∧ 1 - (0 : ℝ) ≤ inner (𝕜 := ℝ) u u := by simp

end Transformer.Normalization
