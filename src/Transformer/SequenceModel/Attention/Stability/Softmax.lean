module

public import Transformer.SequenceModel.RotaryStability
public import Transformer.SequenceModel.Attention.Concentration
public import Transformer.GPTMini.AttentionLipschitz

/-!
# Phase errors through causal softmax and XSA

Source: `review/flat_spectrum.md`, *Quantitative RoPE approximation*, A2;
`diff_gpt/domain/model/mha.py` at `b98c665b`. The softmax argument adapts
`GPTMini.SoftmaxStability` to the source's arbitrary score matrices and mask.
Unlike that older model, these source logits have RMS Q/K and no temperature.
-/

@[expose] public section
noncomputable section

namespace Transformer.SequenceModel.Attention.Stability

open GPTMini
open scoped BigOperators

/-- A2: source causal weights are nonnegative, including masked positions. -/
theorem weight_nonneg {T : ℕ} (s : Fin T → Fin T → ℝ) (i j : Fin T) :
    0 ≤ causalWeight s i j := by
  unfold causalWeight
  split_ifs
  · exact (div_pos (Real.exp_pos _) (denominator_pos s i)).le
  · exact le_rfl

/-- A2: a logit perturbation r changes each unmasked softmax probability
by a factor at most exp(2r). Scores need only be close inside the causal row. -/
theorem weight_le_mul {T : ℕ} (s t : Fin T → Fin T → ℝ) (i : Fin T) (r : ℝ)
    (h : ∀ j ≤ i, |s i j - t i j| ≤ r) (j : Fin T) :
    causalWeight s i j ≤ Real.exp (2 * r) * causalWeight t i j := by
  have hst (j : Fin T) (hj : j ≤ i) :
      Real.exp (s i j) ≤ Real.exp r * Real.exp (t i j) := by
    rw [← Real.exp_add]
    exact Real.exp_le_exp.mpr (by linarith [(abs_le.mp (h j hj)).2])
  have hts (j : Fin T) (hj : j ≤ i) :
      Real.exp (t i j) ≤ Real.exp r * Real.exp (s i j) := by
    rw [← Real.exp_add]
    exact Real.exp_le_exp.mpr (by linarith [(abs_le.mp (h j hj)).1])
  have hsum : (∑ k ∈ Finset.Iic i, Real.exp (t i k)) ≤
      Real.exp r * ∑ k ∈ Finset.Iic i, Real.exp (s i k) := by
    rw [Finset.mul_sum]
    exact Finset.sum_le_sum (fun k hk => hts k (Finset.mem_Iic.mp hk))
  unfold causalWeight
  split_ifs with hj
  · rw [mul_div_assoc', div_le_div_iff₀ (denominator_pos s i) (denominator_pos t i)]
    calc
      _ ≤ (Real.exp r * Real.exp (t i j)) *
          (Real.exp r * ∑ k ∈ Finset.Iic i, Real.exp (s i k)) :=
        mul_le_mul (hst j hj) hsum (denominator_pos t i).le (by positivity)
      _ = _ := by
        rw [show (2 : ℝ) * r = r + r by ring, Real.exp_add]
        ring
  · simp

/-- A2: the whole row has L1 error at most 8r, independently of how many
variables are attended to. The constant is conservative, not claimed sharp. -/
theorem weight_l1_error {T : ℕ} (s t : Fin T → Fin T → ℝ) (i : Fin T) (r : ℝ)
    (h : ∀ j ≤ i, |s i j - t i j| ≤ r) :
    ∑ j, |causalWeight s i j - causalWeight t i j| ≤ 8 * r :=
  l1_le_of_le_exp _ _ r ((abs_nonneg _).trans (h i le_rfl))
    (weight_nonneg s i) (weight_nonneg t i) (weight_le_mul s t i r h)
    (causalWeight_sum s i) (causalWeight_sum t i)

/-- A2: identical value vectors turn the softmax L1 error into an output
error. B bounds their actual Euclidean norms, not their normalized directions. -/
theorem average_error {T d : ℕ} (s t : Fin T → Fin T → ℝ)
    (v : Fin T → EucSpace d) (i : Fin T) (r B : ℝ)
    (h : ∀ j ≤ i, |s i j - t i j| ≤ r) (hB : ∀ j, ‖v j‖ ≤ B) :
    ‖(∑ j, causalWeight s i j • v j) - ∑ j, causalWeight t i j • v j‖ ≤ 8 * r * B := by
  have hB0 := (norm_nonneg _).trans (hB i)
  rw [← Finset.sum_sub_distrib]
  simp only [← sub_smul]
  calc
    _ ≤ ∑ j, ‖(causalWeight s i j - causalWeight t i j) • v j‖ := norm_sum_le _ _
    _ ≤ ∑ j, |causalWeight s i j - causalWeight t i j| * B := by
      apply Finset.sum_le_sum
      intro j hj
      rw [norm_smul, Real.norm_eq_abs]
      exact mul_le_mul_of_nonneg_left (hB j) (abs_nonneg _)
    _ = (∑ j, |causalWeight s i j - causalWeight t i j|) * B := by rw [Finset.sum_mul]
    _ ≤ 8 * r * B := mul_le_mul_of_nonneg_right (weight_l1_error s t i r h) hB0

/-- A2: the actual source head, including clipped XSA and first-position
zero, has phase error at most 32 δ sqrt(d) B. Q/K and values are held fixed;
later blocks require propagation of their changing residuals as well. -/
theorem headOutput_phase_error {T d : ℕ} (hd : 0 < d) (eps xeps : ℝ)
    (heps : 0 < eps) (hxeps : 0 ≤ xeps)
    (a b : Fin T → Fin (d / 2) → ℝ) (q k v : Fin T → EucSpace d)
    (i : Fin T) (δ B : ℝ) (hδ : 0 ≤ δ)
    (h : ∀ j c, |a j c - b j c| ≤ δ) (hB : ∀ j, ‖v j‖ ≤ B) :
    ‖headOutput eps xeps a q k v i - headOutput eps xeps b q k v i‖ ≤
      32 * δ * Real.sqrt (d : ℝ) * B := by
  have hB0 := (norm_nonneg _).trans (hB i)
  let s := fun i j => headScore d eps (a i) (a j) (q i) (k j)
  let t := fun i j => headScore d eps (b i) (b j) (q i) (k j)
  have hc : ∀ j ≤ i, |s i j - t i j| ≤ (δ + δ) * Real.sqrt (d : ℝ) := by
    intro j hj
    exact headScore_phase_error d hd eps heps _ _ _ _ _ _ δ δ hδ hδ (h i) (h j)
  have hav := average_error s t v i _ B hc hB
  have hp := GPTMini.norm_proj_sub_proj_le (∑ j, causalWeight s i j • v j)
    (∑ j, causalWeight t i j • v j) (normL2 xeps (v i)) (normL2 xeps (v i))
    (normL2_norm_le xeps hxeps _) (normL2_norm_le xeps hxeps _)
  simp only [sub_self, norm_zero, mul_zero, add_zero] at hp
  unfold headOutput
  split_ifs
  · simpa only [sub_self, norm_zero] using
      (show (0 : ℝ) ≤ 32 * δ * Real.sqrt (d : ℝ) * B by positivity)
  · exact hp.trans (by nlinarith [hav])

example := weight_le_mul (fun i j : Fin 3 => (i.val + j.val : ℝ))
  (fun i j => (i.val + j.val : ℝ) + 1 / 10) 2 (1 / 10)
  (by intro j hj; ring_nf; norm_num) 0
example := weight_l1_error (fun _ _ : Fin 3 => (0 : ℝ)) (fun _ _ => 1 / 10) 2 (1 / 10)
  (by intro j hj; norm_num)
example := average_error (fun _ _ : Fin 3 => (0 : ℝ)) (fun _ _ => 1 / 10)
  (fun _ => (0 : EucSpace 4)) 2 (1 / 10) 1 (by intro j hj; norm_num)
  (by intro j; simp)
example := headOutput_phase_error (T := 3) (d := 4) (by decide)
  (1 / 8388608) (1 / 1000000000000) (by norm_num) (by norm_num)
  (fun j _ => (j.val : ℝ) / 7) (fun j _ => (j.val : ℝ) / 7 + 1 / 100)
  (fun _ => WithLp.toLp 2 ![1, 2, 3, 4]) (fun _ => WithLp.toLp 2 ![2, 1, 4, 3])
  (fun _ => WithLp.toLp 2 ![1, 0, 0, 0]) 2 (1 / 100) 1 (by norm_num)
  (by intro j c; ring_nf; norm_num)
  (by intro j; norm_num [EuclideanSpace.norm_eq, Fin.sum_univ_succ])

end Transformer.SequenceModel.Attention.Stability
