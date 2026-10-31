module

public import Transformer.SequenceModel.Attention.Normalization
public import Mathlib.Analysis.SpecialFunctions.Trigonometric.Bounds

/-!
# Quantitative perturbations of the source RoPE

Source: `review/flat_spectrum.md`, *Quantitative RoPE approximation*, A1;
`diff_gpt/domain/model/{rope,mha}.py` at `b98c665b`. These bounds compare
actual half-split rotations and RMS-normalized logits. They allow arbitrary
real phases and do not replace them with periodic position tags.
-/

@[expose] public section
noncomputable section

namespace Transformer.SequenceModel

open GPTMini Landscape.Neural
open scoped BigOperators

/-- A1: the chord between two unit-circle phases is no longer than the
angle difference. This is a bound on both sine and cosine together. -/
theorem phase_chord_sq_le (a b : ℝ) :
    (Real.cos a - Real.cos b) ^ 2 + (Real.sin a - Real.sin b) ^ 2 ≤ (a - b) ^ 2 := by
  have h := Real.one_sub_sq_div_two_le_cos (x := a - b)
  rw [Real.cos_sub] at h
  nlinarith [Real.sin_sq_add_cos_sq a, Real.sin_sq_add_cos_sq b]

/-- A1: one rotated pair has error at most the phase error times its
length, in squared form without a square-root loss. -/
theorem rotate_pair_error {K : Type*} (a b : K → ℝ) (x : Channels K → ℝ)
    (k : K) (δ : ℝ) (hδ : 0 ≤ δ) (h : |a k - b k| ≤ δ) :
    (rotate a x (k, 0) - rotate b x (k, 0)) ^ 2 +
      (rotate a x (k, 1) - rotate b x (k, 1)) ^ 2 ≤
        δ ^ 2 * (x (k, 0) ^ 2 + x (k, 1) ^ 2) := by
  have hs : (a k - b k) ^ 2 ≤ δ ^ 2 := by
    simpa only [sq_abs] using (sq_le_sq₀ (abs_nonneg _) hδ).mpr h
  calc
    _ = ((Real.cos (a k) - Real.cos (b k)) ^ 2 +
        (Real.sin (a k) - Real.sin (b k)) ^ 2) *
          (x (k, 0) ^ 2 + x (k, 1) ^ 2) := by
      simp only [rotate, Fin.isValue, one_ne_zero, ↓reduceIte]
      ring
    _ ≤ δ ^ 2 * (x (k, 0) ^ 2 + x (k, 1) ^ 2) :=
      mul_le_mul_of_nonneg_right ((phase_chord_sq_le _ _).trans hs) (by positivity)

/-- A1: the source half-split rotation is Lipschitz in its phase table.
The constant is independent of width; an odd leftover coordinate is fixed. -/
theorem rotateHead_phase_error (d : ℕ) (a b : Fin (d / 2) → ℝ)
    (x : EucSpace d) (δ : ℝ) (hδ : 0 ≤ δ) (h : ∀ k, |a k - b k| ≤ δ) :
    ‖rotateHead d a x - rotateHead d b x‖ ≤ δ * ‖x‖ := by
  apply (sq_le_sq₀ (norm_nonneg _) (mul_nonneg hδ (norm_nonneg _))).mp
  rw [mul_pow, EuclideanSpace.real_norm_sq_eq, EuclideanSpace.real_norm_sq_eq]
  simp only [PiLp.sub_apply]
  rw [sum_split d (fun i => (rotateHead d a x i - rotateHead d b x i) ^ 2),
    sum_split d (fun i => x i ^ 2)]
  have hf (c : Fin (d / 2) → ℝ) (k : Fin (d / 2)) :
      rotateHead d c x (ropeFst d k) = rotate c (splitPairs d x) (k, 0) := by
    simp only [rotateHead, ropeFst, PiLp.toLp_apply, Equiv.apply_symm_apply]
  have hs (c : Fin (d / 2) → ℝ) (k : Fin (d / 2)) :
      rotateHead d c x (ropeSnd d k) = rotate c (splitPairs d x) (k, 1) := by
    simp only [rotateHead, ropeSnd, PiLp.toLp_apply, Equiv.apply_symm_apply]
  have ht (c : Fin (d / 2) → ℝ) (j : Fin (d - 2 * (d / 2))) :
      rotateHead d c x ((ropeSplit d).symm (.inr j)) = x ((ropeSplit d).symm (.inr j)) := by
    simp only [rotateHead, PiLp.toLp_apply, Equiv.apply_symm_apply]
  simp only [hf, hs, ht, sub_self, ne_eq, OfNat.ofNat_ne_zero, not_false_eq_true,
    zero_pow, Finset.sum_const_zero, add_zero]
  rw [← Finset.sum_add_distrib, ← Finset.sum_add_distrib, mul_add, Finset.mul_sum]
  have hp := Finset.sum_le_sum (fun k (_ : k ∈ Finset.univ) =>
    rotate_pair_error a b (splitPairs d x) k δ hδ (h k))
  simp only [splitPairs, Fin.isValue, one_ne_zero, ↓reduceIte] at hp
  exact hp.trans (le_add_of_nonneg_right (by positivity))

/-- A1: a frequency error η produces at most |position|η phase error.
No uniform-in-position bound follows from a fixed nonzero η. -/
theorem frequency_phase_error {K : Type*} (a b : K → ℝ) (position η : ℝ)
    (h : ∀ k, |a k - b k| ≤ η) (k : K) :
    |position * a k - position * b k| ≤ |position| * η := by
  rw [← mul_sub, abs_mul]
  exact mul_le_mul_of_nonneg_left (h k) (abs_nonneg _)

namespace Attention

/-- A1: Q/K RMSNorm and 1/sqrt(d) attention scaling bound logit movement
by sqrt(d) times the sum of query and key phase errors. -/
theorem headScore_phase_error (d : ℕ) (hd : 0 < d) (eps : ℝ) (heps : 0 < eps)
    (a b a' b' : Fin (d / 2) → ℝ) (q k : EucSpace d) (δq δk : ℝ)
    (hδq : 0 ≤ δq) (hδk : 0 ≤ δk)
    (ha : ∀ c, |a c - a' c| ≤ δq) (hb : ∀ c, |b c - b' c| ≤ δk) :
    |headScore d eps a b q k - headScore d eps a' b' q k| ≤
      (δq + δk) * Real.sqrt (d : ℝ) := by
  let u := rmsNormEps eps q
  let v := rmsNormEps eps k
  have hu := rmsNormEps_norm_le eps heps hd q
  have hv := rmsNormEps_norm_le eps heps hd k
  have hq := rotateHead_phase_error d a a' u δq hδq ha
  have hk := rotateHead_phase_error d b b' v δk hδk hb
  have hsplit : inner ℝ (rotateHead d a u) (rotateHead d b v) -
      inner ℝ (rotateHead d a' u) (rotateHead d b' v) =
      inner ℝ (rotateHead d a u - rotateHead d a' u) (rotateHead d b v) +
      inner ℝ (rotateHead d a' u) (rotateHead d b v - rotateHead d b' v) := by
    rw [inner_sub_left, inner_sub_right]
    ring
  have hnorm : |inner ℝ (rotateHead d a u) (rotateHead d b v) -
      inner ℝ (rotateHead d a' u) (rotateHead d b' v)| ≤
        (δq + δk) * Real.sqrt (d : ℝ) * Real.sqrt (d : ℝ) := by
    rw [hsplit]
    refine (abs_add_le _ _).trans ((add_le_add (abs_real_inner_le_norm _ _)
      (abs_real_inner_le_norm _ _)).trans ?_)
    rw [rotateHead_isometry, rotateHead_isometry]
    have h1 := mul_le_mul_of_nonneg_right hq (norm_nonneg v)
    have h2 := mul_le_mul_of_nonneg_left hk (norm_nonneg u)
    have huv : ‖u‖ * ‖v‖ ≤ Real.sqrt (d : ℝ) * Real.sqrt (d : ℝ) :=
      mul_le_mul hu hv (norm_nonneg v) (Real.sqrt_nonneg _)
    nlinarith [mul_le_mul_of_nonneg_left huv (add_nonneg hδq hδk)]
  have hs : 0 < Real.sqrt (d : ℝ) := Real.sqrt_pos.mpr (Nat.cast_pos.mpr hd)
  unfold headScore
  rw [← sub_div, abs_div, abs_of_pos hs, div_le_iff₀ hs]
  exact hnorm

end Attention

example := rotate_pair_error (fun _ : Fin 1 => (1 : ℝ) / 7) (fun _ => 0)
  (fun i => (i.2.val + 1 : ℝ)) 0 (1 / 7) (by norm_num) (by norm_num)
example := rotateHead_phase_error 4 (fun _ => 1 / 7) (fun _ => 0)
  (WithLp.toLp 2 ![1, 2, 3, 4]) (1 / 7) (by norm_num) (by intro c; norm_num)
example := frequency_phase_error (fun _ : Fin 1 => (1 : ℝ) / 7) (fun _ => 0)
  99 (1 / 7) (by intro c; norm_num) 0
example := Attention.headScore_phase_error 4 (by decide) (1 / 8388608) (by norm_num)
  (fun _ => 1 / 7) (fun _ => 2 / 7) (fun _ => 0) (fun _ => 0)
  (WithLp.toLp 2 ![1, 2, 3, 4]) (WithLp.toLp 2 ![2, 1, 4, 3])
  (1 / 7) (2 / 7) (by norm_num) (by norm_num) (by intro c; norm_num) (by intro c; norm_num)

end Transformer.SequenceModel
