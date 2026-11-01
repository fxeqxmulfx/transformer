module

public import Transformer.SequenceModel.Attention.Execution.Trace

/-!
# Composition of operation errors

Source: `review/flat_spectrum.md`, *Primitive execution certificates*, C4.
Each estimate separates movement of the operation's inputs from error in
evaluating it. In particular, an approximate XSA direction may have norm
greater than one; the earlier unit-direction bound cannot be reused directly.
-/

@[expose] public section
noncomputable section

namespace Transformer.SequenceModel.Attention.Execution

open GPTMini Stability

/-- C4: local matrix error and propagated input error add with the actual operator norm. -/
theorem linear_error {d e : ℕ} (A : EucSpace d →L[ℝ] EucSpace e)
    (x y : EucSpace d) (z : EucSpace e) (E δ : ℝ)
    (hx : ‖y - x‖ ≤ E) (hz : ‖z - A y‖ ≤ δ) :
    ‖z - A x‖ ≤ ‖A‖ * E + δ := by
  have hp : ‖A y - A x‖ ≤ ‖A‖ * E := by
    rw [← map_sub]
    exact (A.le_opNorm _).trans (mul_le_mul_of_nonneg_left hx (norm_nonneg _))
  exact (norm_sub_le_norm_sub_add_norm_sub z (A y) (A x)).trans
    ((add_le_add hz hp).trans_eq (by ring))

/-- C4: an approximate normalizer retains its own error as well as input sensitivity. -/
theorem rms_error {d : ℕ} (hd : 0 < d) {eps : ℝ} (heps : 0 < eps)
    (x y z : EucSpace d) (E δ : ℝ) (hx : ‖y - x‖ ≤ E)
    (hz : ‖z - rmsNormEps eps y‖ ≤ δ) :
    ‖z - rmsNormEps eps x‖ ≤ rmsLip eps * E + δ := by
  have hp := (rmsNormEps_lipschitz eps heps hd y x).trans
    (mul_le_mul_of_nonneg_left hx (by positivity))
  exact (norm_sub_le_norm_sub_add_norm_sub z (rmsNormEps eps y) _).trans
    ((add_le_add hz hp).trans_eq (by unfold rmsLip; ring))

/-- C4: an error ball around a bounded reference also bounds the recorded value. -/
theorem norm_of_error {d : ℕ} (x y : EucSpace d) (E B : ℝ)
    (he : ‖x - y‖ ≤ E) (hy : ‖y‖ ≤ B) : ‖x‖ ≤ B + E := by
  have h := norm_add_le (x - y) y
  rw [sub_add_cancel] at h
  linarith

/-- C4: a bilinear dot product includes the cross term from two perturbed inputs. -/
theorem inner_pair_error {d : ℕ} (x y x₀ y₀ : EucSpace d) (E B : ℝ)
    (hx : ‖x - x₀‖ ≤ E) (hy : ‖y - y₀‖ ≤ E) (hx₀ : ‖x₀‖ ≤ B) (hy₀ : ‖y₀‖ ≤ B) :
    |inner ℝ x y - inner ℝ x₀ y₀| ≤ (2 * B + E) * E := by
  have hE : 0 ≤ E := (norm_nonneg _).trans hx
  have hB : 0 ≤ B := (norm_nonneg _).trans hx₀
  have hyn := norm_of_error y y₀ E B hy hy₀
  have heq : inner ℝ x y - inner ℝ x₀ y₀ =
      inner ℝ (x - x₀) y + inner ℝ x₀ (y - y₀) := by
    rw [inner_sub_left, inner_sub_right]
    ring
  rw [heq]
  refine (abs_add_le _ _).trans ((add_le_add (abs_real_inner_le_norm _ _)
    (abs_real_inner_le_norm _ _)).trans ?_)
  have h1 := mul_le_mul hx hyn (norm_nonneg _) hE
  have h2 := mul_le_mul hx₀ hy (norm_nonneg _) hB
  exact (add_le_add h1 h2).trans_eq (by ring)

/-- C4: XSA error when the computed direction is only close to a unit-or-shorter
one. Its excess norm, both moving inputs, and their interaction are retained. -/
theorem xsa_pair_error {d : ℕ} (a b u v : EucSpace d) (A D M : ℝ)
    (ha : ‖a - b‖ ≤ A) (hu : ‖u - v‖ ≤ D) (hb : ‖b‖ ≤ M) (hv : ‖v‖ ≤ 1) :
    ‖(a - inner ℝ a u • u) - (b - inner ℝ b v • v)‖ ≤
      (1 + (1 + D) ^ 2) * A + M * (2 + D) * D := by
  have hA : 0 ≤ A := (norm_nonneg _).trans ha
  have hD : 0 ≤ D := (norm_nonneg _).trans hu
  have hM : 0 ≤ M := (norm_nonneg _).trans hb
  have hun := norm_of_error u v D 1 hu hv
  have heq : (a - inner ℝ a u • u) - (b - inner ℝ b v • v) =
      (a - b) - inner ℝ (a - b) u • u - inner ℝ b (u - v) • u - inner ℝ b v • (u - v) := by
    rw [inner_sub_left, inner_sub_right]
    module
  have h1 : ‖inner ℝ (a - b) u • u‖ ≤ A * (1 + D) ^ 2 := by
    rw [norm_smul, Real.norm_eq_abs]
    calc
      _ ≤ (‖a - b‖ * ‖u‖) * ‖u‖ :=
        mul_le_mul_of_nonneg_right (abs_real_inner_le_norm _ _) (norm_nonneg _)
      _ ≤ (A * (1 + D)) * (1 + D) := by gcongr
      _ = _ := by ring
  have h2 : ‖inner ℝ b (u - v) • u‖ ≤ M * D * (1 + D) := by
    rw [norm_smul, Real.norm_eq_abs]
    calc
      _ ≤ (‖b‖ * ‖u - v‖) * ‖u‖ :=
        mul_le_mul_of_nonneg_right (abs_real_inner_le_norm _ _) (norm_nonneg _)
      _ ≤ M * D * (1 + D) := by gcongr
  have h3 : ‖inner ℝ b v • (u - v)‖ ≤ M * D := by
    rw [norm_smul, Real.norm_eq_abs]
    calc
      _ ≤ (‖b‖ * ‖v‖) * ‖u - v‖ :=
        mul_le_mul_of_nonneg_right (abs_real_inner_le_norm _ _) (norm_nonneg _)
      _ ≤ (M * 1) * D := by gcongr
      _ = _ := by ring
  rw [heq]
  have hn1 := norm_sub_le (a - b) (inner ℝ (a - b) u • u)
  have hn2 := norm_sub_le ((a - b) - inner ℝ (a - b) u • u) (inner ℝ b (u - v) • u)
  have hn3 := norm_sub_le ((a - b) - inner ℝ (a - b) u • u - inner ℝ b (u - v) • u)
    (inner ℝ b v • (u - v))
  nlinarith

example := linear_error (0 : EucSpace 2 →L[ℝ] EucSpace 2) 0 0 0 0 0 (by simp) (by simp)
example := rms_error (d := 2) (by decide) (eps := 1 / 8388608) (by norm_num)
  0 0 0 0 0 (by simp) (by simp [rmsNormEps])
example := norm_of_error (0 : EucSpace 2) 0 0 1 (by simp) (by simp)
example := inner_pair_error (0 : EucSpace 2) 0 0 0 0 1 (by simp) (by simp) (by simp) (by simp)
example := xsa_pair_error (0 : EucSpace 2) 0 (WithLp.toLp 2 ![2, 0]) (WithLp.toLp 2 ![1, 0])
  0 1 1 (by simp)
  (by norm_num [EuclideanSpace.norm_eq, Fin.sum_univ_succ, PiLp.sub_apply]) (by simp)
  (by norm_num [EuclideanSpace.norm_eq, Fin.sum_univ_succ])

end Transformer.SequenceModel.Attention.Execution
