module

public import Transformer.SequenceModel.Attention.Adaptive.Norms

/-!+# Algebraic checks for operation-dependent propagation

Source: `review/flat_spectrum.md`, *Trace-dependent execution bounds*, M2.
Squared norm and matrix checks use exact stored coordinates. Every operation
keeps its own residual; no late residual is charged at an earlier operation.
-/

@[expose] public section
noncomputable section
namespace Transformer.SequenceModel.Attention.Adaptive
open GPTMini Stability Execution
open scoped BigOperators

/-- M2: a rational squared-norm check certifies a recorded vector bound. -/
theorem norm_checked {d : ℕ} (x : EucSpace d) (R : ℝ) (hR : 0 ≤ R)
    (hx : ∑ c, x c ^ 2 ≤ R ^ 2) : ‖x‖ ≤ R := by
  rw [← EuclideanSpace.real_norm_sq_eq] at hx
  nlinarith [norm_nonneg x]

/-- M2: the squared Frobenius norm bounds the exact stored linear map. -/
theorem matrix_checked {d e : ℕ} (A : EucSpace d →L[ℝ] EucSpace e)
    (M : Matrix (Fin e) (Fin d) ℝ) (L : ℝ) (hL : 0 ≤ L)
    (hA : ∀ x c, A x c = ∑ j, M c j * x j)
    (hM : ∑ c, ∑ j, M c j ^ 2 ≤ L ^ 2) : ‖A‖ ≤ L := by
  apply A.opNorm_le_bound hL
  intro x
  have hsq : ‖A x‖ ^ 2 ≤ L ^ 2 * ‖x‖ ^ 2 := by
    rw [EuclideanSpace.real_norm_sq_eq, EuclideanSpace.real_norm_sq_eq]
    calc
      _ ≤ ∑ c, (∑ j, M c j ^ 2) * ∑ j, x j ^ 2 := by
        apply Finset.sum_le_sum
        intro c hc
        rw [hA]
        exact Finset.sum_mul_sq_le_sq_mul_sq Finset.univ (M c) (fun j => x j)
      _ = (∑ c, ∑ j, M c j ^ 2) * ∑ j, x j ^ 2 := by rw [Finset.sum_mul]
      _ ≤ _ := mul_le_mul_of_nonneg_right hM (Finset.sum_nonneg (by intros; positivity))
  nlinarith [norm_nonneg (A x), norm_nonneg x, mul_nonneg hL (norm_nonneg x)]

/-- M2: a learned matrix propagates the input radius and its own residual. -/
theorem linear_step {d e : ℕ} (A : EucSpace d →L[ℝ] EucSpace e)
    (x y : EucSpace d) (z : EucSpace e) (L E δ : ℝ)
    (hA : ‖A‖ ≤ L) (hx : ‖x - y‖ ≤ E) (hz : ‖z - A x‖ ≤ δ) :
    ‖z - A y‖ ≤ L * E + δ := by
  have hE := (norm_nonneg _).trans hx
  exact (linear_error A y x z E δ hx hz).trans
    (add_le_add (mul_le_mul_of_nonneg_right hA hE) le_rfl)

/-- M2: both residual additions keep their independent arithmetic residual. -/
theorem add_step {d : ℕ} (x y x₀ y₀ z : EucSpace d) (E F δ : ℝ)
    (hx : ‖x - x₀‖ ≤ E) (hy : ‖y - y₀‖ ≤ F) (hz : ‖z - (x + y)‖ ≤ δ) :
    ‖z - (x₀ + y₀)‖ ≤ E + F + δ := by
  have h : ‖(x + y) - (x₀ + y₀)‖ ≤ E + F := by
    rw [show (x + y) - (x₀ + y₀) = (x - x₀) + (y - y₀) by module]
    exact (norm_add_le _ _).trans (add_le_add hx hy)
  exact (norm_sub_le_norm_sub_add_norm_sub z (x + y) _).trans
    ((add_le_add hz h).trans_eq (by ring))

/-- M2: an RMS gain is certified by one polynomial inequality at its input. -/
theorem rms_step {d : ℕ} (hd : 0 < d) {eps : ℝ} (heps : 0 < eps)
    (x y z : EucSpace d) (L E δ : ℝ) (hL : 0 ≤ L)
    (hden : 4 * (d : ℝ) ≤ L ^ 2 * (‖x‖ ^ 2 + (d : ℝ) * eps))
    (hx : ‖x - y‖ ≤ E) (hz : ‖z - rmsNormEps eps x‖ ≤ δ) :
    ‖z - rmsNormEps eps y‖ ≤ L * E + δ := by
  have hd' : (0 : ℝ) < d := by exact_mod_cast hd
  have hp : 0 < ‖x‖ ^ 2 + (d : ℝ) * eps := by positivity
  have hs : 0 < Real.sqrt (‖x‖ ^ 2 + (d : ℝ) * eps) := Real.sqrt_pos.mpr hp
  have hg : 2 * Real.sqrt (d : ℝ) / Real.sqrt (‖x‖ ^ 2 + (d : ℝ) * eps) ≤ L := by
    rw [div_le_iff₀ hs]
    have hsq := Real.sq_sqrt hp.le
    have hsd := Real.sq_sqrt hd'.le
    have hn := mul_nonneg hL hs.le
    nlinarith [Real.sqrt_nonneg (d : ℝ)]
  have hE := (norm_nonneg _).trans hx
  have hm := (rms_secant eps heps hd x y).trans (mul_le_mul hg hx (norm_nonneg _) hL)
  exact (norm_sub_le_norm_sub_add_norm_sub z (rmsNormEps eps x) _).trans
    ((add_le_add hz hm).trans_eq (by ring))

/-- M2: a recorded L2 denominator suffices even if the exact input is clipped. -/
theorem l2_step {d : ℕ} {eps : ℝ} (heps : 0 < eps)
    (x y z : EucSpace d) (L E δ : ℝ) (hL : 0 ≤ L)
    (hden : 2 ≤ L * max ‖x‖ eps)
    (hx : ‖x - y‖ ≤ E) (hz : ‖z - normL2 eps x‖ ≤ δ) :
    ‖z - normL2 eps y‖ ≤ L * E + δ := by
  have hg : 2 / max ‖x‖ eps ≤ L := (div_le_iff₀ (lt_max_of_lt_right heps)).mpr hden
  have hm := (l2_secant eps heps x y).trans (mul_le_mul hg hx (norm_nonneg _) hL)
  exact (norm_sub_le_norm_sub_add_norm_sub z (normL2 eps x) _).trans
    ((add_le_add hz hm).trans_eq (by ring))

example := norm_checked (EuclideanSpace.single 0 1 : EucSpace 4) 1 (by norm_num) (by simp)
example := matrix_checked (0 : EucSpace 4 →L[ℝ] EucSpace 4) 0 0 (by norm_num)
  (by intros; simp) (by simp)
example := linear_step (0 : EucSpace 4 →L[ℝ] EucSpace 4) 0 0 0 0 0 0
  (by simp) (by simp) (by simp)
example := add_step (0 : EucSpace 4) 0 0 0 0 0 0 0 (by simp) (by simp) (by simp)
example := rms_step (d := 4) (by decide) (eps := 1) (by norm_num) 0 0 0 2 0 0
  (by norm_num) (by norm_num) (by simp) (by simp [rmsNormEps])
example := l2_step (d := 4) (eps := 1) (by norm_num) 0 0 0 2 0 0
  (by norm_num) (by norm_num) (by simp) (by simp [normL2])

end Transformer.SequenceModel.Attention.Adaptive
