module

public import Transformer.SequenceModel.Attention.Adaptive.Products

/-!
# Stored RoPE, scores and causal averages with local bounds

Source: `review/flat_spectrum.md`, *Trace-dependent execution bounds*, M2.
The exact phases are arbitrary. Captured Q/K and value magnitudes replace
worst-case normalized-ball bounds, while table error remains explicit.
-/

@[expose] public section
noncomputable section
namespace Transformer.SequenceModel.Attention.Adaptive
open GPTMini Stability Execution
open scoped BigOperators

/-- M2: table error multiplies the actual recorded input norm. -/
theorem rotate_step {d : ℕ} (a c s : Fin (d / 2) → ℝ) (x y z : EucSpace d)
    (R E δ τ : ℝ) (hτ : 0 ≤ τ) (hn : ‖x‖ ≤ R) (hx : ‖x - y‖ ≤ E)
    (hz : ‖z - tableRotate d c s x‖ ≤ δ)
    (hc : ∀ k, |c k - Real.cos (a k)| ≤ τ)
    (hs : ∀ k, |s k - Real.sin (a k)| ≤ τ) :
    ‖z - rotateHead d a y‖ ≤ 2 * τ * R + E + δ := by
  have ht := (tableRotate_error d c s a x τ hτ hc hs).trans
    (mul_le_mul_of_nonneg_left hn (by positivity))
  have hm : ‖rotateHead d a x - rotateHead d a y‖ ≤ E := by rw [rotation_dist]; exact hx
  exact (norm_sub_le_norm_sub_add_norm_sub z (tableRotate d c s x) _).trans
    ((add_le_add hz ((norm_sub_le_norm_sub_add_norm_sub _ (rotateHead d a x) _).trans
      (add_le_add ht hm))).trans_eq (by ring))

/-- M2: score errors use recorded query/key bounds, including the cross term. -/
theorem score_step {d : ℕ} (x y x₀ y₀ : EucSpace d) (z R E δ : ℝ)
    (hx : ‖x - x₀‖ ≤ E) (hy : ‖y - y₀‖ ≤ E) (hnx : ‖x‖ ≤ R) (hny : ‖y‖ ≤ R)
    (hz : |z - inner ℝ x y / Real.sqrt (d : ℝ)| ≤ δ) :
    |z - inner ℝ x₀ y₀ / Real.sqrt (d : ℝ)| ≤ (2 * R + E) * E / Real.sqrt (d : ℝ) + δ := by
  have hi := inner_pair_error x₀ y₀ x y E R
    (by rwa [norm_sub_rev]) (by rwa [norm_sub_rev]) hnx hny
  have hm : |inner ℝ x y / Real.sqrt (d : ℝ) - inner ℝ x₀ y₀ / Real.sqrt (d : ℝ)| ≤
      (2 * R + E) * E / Real.sqrt (d : ℝ) := by
    rw [← sub_div, abs_div, abs_of_nonneg (Real.sqrt_nonneg _), abs_sub_comm]
    exact div_le_div_of_nonneg_right hi (Real.sqrt_nonneg _)
  exact (abs_sub_le z (inner ℝ x y / Real.sqrt (d : ℝ)) _).trans
    ((add_le_add hz hm).trans_eq (by ring))

/-- M2: the average keeps distinct score, value and SDPA residual allowances. -/
theorem average_step {T d : ℕ} (s t : Fin T → Fin T → ℝ)
    (v v₀ : Fin T → EucSpace d) (z : EucSpace d) (i : Fin T) (R E F δ : ℝ)
    (hs : ∀ j ≤ i, |s i j - t i j| ≤ E)
    (hv : ∀ j, ‖v j - v₀ j‖ ≤ F) (hn : ∀ j, ‖v j‖ ≤ R)
    (hz : ‖z - ∑ j, causalWeight s i j • v j‖ ≤ δ) :
    ‖z - ∑ j, causalWeight t i j • v₀ j‖ ≤ 8 * E * R + F + δ := by
  have hm := average_joint_error t s v₀ v i E R F
    (by intro j hj; rw [abs_sub_comm]; exact hs j hj)
    (by intro j; rw [norm_sub_rev]; exact hv j) hn
  rw [norm_sub_rev] at hm
  exact (norm_sub_le_norm_sub_add_norm_sub z (∑ j, causalWeight s i j • v j) _).trans
    ((add_le_add hz hm).trans_eq (by ring))

example := rotate_step (d := 4) (fun _ => 0) (fun _ => 1) (fun _ => 0)
  0 0 0 0 0 0 0 (by norm_num) (by simp) (by simp) (by rw [tableRotate_zero]; simp)
  (by intro c; norm_num) (by intro c; norm_num)
example := score_step (d := 4) 0 0 0 0 0 0 0 0
  (by simp) (by simp) (by simp) (by simp) (by simp)
example := average_step (fun _ _ : Fin 2 => (0 : ℝ)) (fun _ _ => 0)
  (fun _ => (0 : EucSpace 4)) (fun _ => 0) 0 1 0 0 0 0
  (by intro j hj; simp) (by intro j; simp) (by intro j; simp) (by simp)

end Transformer.SequenceModel.Attention.Adaptive
