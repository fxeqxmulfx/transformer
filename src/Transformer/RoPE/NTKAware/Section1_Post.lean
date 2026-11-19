module

public import Mathlib.Analysis.Real.Pi.Irrational
public import Transformer.RoPE.ReRoPE.Section3_SinglePass
public import Transformer.RoPE.YaRN.SectionA2_NTK

/-!
# NTK-aware scaled RoPE: the post

bloc97, *NTK-Aware Scaled RoPE allows LLaMA models to have extended (8k+) context size without any
fine-tuning and minimal perplexity degradation.*, r/LocalLLaMA post 14lz7j5, 2023-06-29. The post
replaces Position Interpolation's scale by a change of base, `base ↦ base·α^{d/(d−2)}` in its code,
which is YaRN's `adjustedBase` (arXiv:2309.00071v3, Eq. (19)); the frequency identities it implies
are in `YaRN.SectionA2_NTK`. The perplexity graphs and the question-answering transcript are
measurements and are not formalized. `rotary θ t` is RoFormer's rotary matrix and
`contrast ω t = 1 − cos ωt` the positional contrast of `RoPE.Basic`.

* `displacement_eq`: a rotated vector moves between positions `t` apart by
  `‖R^{m+t}x − R^m x‖² = 2 Σ_l (1 − cos tθ_l)|x_l|²`, the contrasts of its pairs;
* `tendsto_interpolated_displacement`: "simply interpolating the RoPE's fourier space linearly …
  prevents the network to distinguish the order and positions of tokens that are very close by":
  under Position Interpolation by `s` this displacement between adjacent positions tends to `0`;
* `adjusted_displacement_ge`: the NTK-aware base "does not scale the fourier features directly":
  the first pair keeps the frequency `1`, and the displacement between adjacent positions is at
  least `(8/9)|x_0|²` for every `α`;
* `eq_of_rotary_eq`, `adjusted_rotary_injective`: "all the positions are perfectly distinguishable
  from each other, even when taken to the extreme (eg. streched 1million times, which is
  effectively a context size of 2 Billion)": a pair of frequency `1` tells all integer positions
  apart, because `π` is irrational. Position Interpolation by a rational `s` tells them apart too;
  what it loses is the displacement.
-/

@[expose] public section

open Filter Matrix Topology
open scoped Real

namespace Transformer.RoPE.NTKAware

open RoFormer

variable {p : ℕ}

/-- `xᵀ R^s x = Σ_l cos(sθ_l)(x_{0,l}² + x_{1,l}²)`. -/
theorem dotProduct_rotary_mulVec (θ : Fin p → ℝ) (s : ℝ) (x : Fin 2 × Fin p → ℝ) :
    x ⬝ᵥ (rotary θ s *ᵥ x) = ∑ l, Real.cos (s * θ l) * (x (0, l) ^ 2 + x (1, l) ^ 2) := by
  rw [rotary_mulVec]
  simp only [dotProduct, Fintype.sum_prod_type, Fin.sum_univ_two, ← Finset.sum_add_distrib]
  refine Finset.sum_congr rfl fun l _ => ?_
  simp [rotateHalf]
  ring

/-- The squared displacement of a rotated vector between positions `t` apart,
`(R^{m+t}x − R^m x)ᵀ(R^{m+t}x − R^m x) = 2 Σ_l (1 − cos tθ_l)(x_{0,l}² + x_{1,l}²)`. -/
theorem displacement_eq (θ : Fin p → ℝ) (x : Fin 2 × Fin p → ℝ) (m t : ℝ) :
    (rotary θ (m + t) *ᵥ x - rotary θ m *ᵥ x) ⬝ᵥ (rotary θ (m + t) *ᵥ x - rotary θ m *ᵥ x) =
      2 * ∑ l, contrast (θ l) t * (x (0, l) ^ 2 + x (1, l) ^ 2) := by
  have hxx : x ⬝ᵥ x = ∑ l, (x (0, l) ^ 2 + x (1, l) ^ 2) := by
    simpa [rotary_zero] using dotProduct_rotary_mulVec θ 0 x
  have hneg : ∀ l, Real.cos (-t * θ l) = Real.cos (t * θ l) := fun l => by
    rw [neg_mul, Real.cos_neg]
  have hsum : ∑ l, contrast (θ l) t * (x (0, l) ^ 2 + x (1, l) ^ 2) =
      ∑ l, (x (0, l) ^ 2 + x (1, l) ^ 2) -
        ∑ l, Real.cos (t * θ l) * (x (0, l) ^ 2 + x (1, l) ^ 2) := by
    rw [← Finset.sum_sub_distrib]
    refine Finset.sum_congr rfl fun l _ => ?_
    rw [contrast, mul_comm (θ l) t]
    ring
  simp only [sub_dotProduct, dotProduct_sub, dotProduct_rotary_self, rotary_dotProduct]
  rw [show m - (m + t) = -t by ring, show m + t - m = t by ring, dotProduct_rotary_mulVec,
    dotProduct_rotary_mulVec, hxx, hsum]
  simp only [hneg]
  ring

/-- Position Interpolation by `s` puts the positions `m` and `m + 1` at `m/s` and `(m + 1)/s`; the
displacement of a rotated vector between them tends to `0` as `s → ∞`. -/
theorem tendsto_interpolated_displacement (θ : Fin p → ℝ) (x : Fin 2 × Fin p → ℝ) (m : ℝ) :
    Tendsto (fun s : ℝ => (rotary θ ((m + 1) / s) *ᵥ x - rotary θ (m / s) *ᵥ x) ⬝ᵥ
      (rotary θ ((m + 1) / s) *ᵥ x - rotary θ (m / s) *ᵥ x)) atTop (𝓝 0) := by
  have h1 : Tendsto (fun s : ℝ => 1 / s) atTop (𝓝 0) := tendsto_const_nhds.div_atTop tendsto_id
  have hg : Continuous fun u : ℝ => 2 * ∑ l, contrast (θ l) u * (x (0, l) ^ 2 + x (1, l) ^ 2) := by
    unfold contrast
    fun_prop
  have h0 : 2 * ∑ l, contrast (θ l) 0 * (x (0, l) ^ 2 + x (1, l) ^ 2) = 0 := by
    simp [contrast]
  have h := (hg.tendsto 0).comp h1
  rw [h0] at h
  exact h.congr fun s => by rw [Function.comp_apply, add_div, displacement_eq]

/-- The NTK-aware frequencies of a head of `p` pairs, `d = 2p`: RoPE's grid for the base
`base·α^{d/(d−2)}`. -/
noncomputable def adjustedFrequencies (base α : ℝ) (p : ℕ) : Fin p → ℝ :=
  fun l => frequency (YaRN.adjustedBase base α ((2 * p : ℕ) : ℝ)) (2 * p) l

/-- The NTK-aware base keeps the first frequency at `1`. -/
theorem adjustedFrequencies_zero (base α : ℝ) (p : ℕ) :
    adjustedFrequencies base α (p + 1) 0 = 1 := by
  simp [adjustedFrequencies, frequency]

/-- The post: the NTK-aware base does not slow the first pair, so a rotated vector moves between
adjacent positions by at least `2(1 − cos 1)|x_0|² ≥ (8/9)|x_0|²`, whatever `α`. -/
theorem adjusted_displacement_ge (base α : ℝ) (x : Fin 2 × Fin (p + 1) → ℝ) (m : ℝ) :
    8 / 9 * (x (0, 0) ^ 2 + x (1, 0) ^ 2) ≤
      (rotary (adjustedFrequencies base α (p + 1)) (m + 1) *ᵥ x -
          rotary (adjustedFrequencies base α (p + 1)) m *ᵥ x) ⬝ᵥ
        (rotary (adjustedFrequencies base α (p + 1)) (m + 1) *ᵥ x -
          rotary (adjustedFrequencies base α (p + 1)) m *ᵥ x) := by
  rw [displacement_eq]
  have hle := Finset.single_le_sum
    (f := fun l =>
      contrast (adjustedFrequencies base α (p + 1) l) 1 * (x (0, l) ^ 2 + x (1, l) ^ 2))
    (fun l _ => mul_nonneg (contrast_bounds _ _).1 (by positivity)) (Finset.mem_univ 0)
  simp only [adjustedFrequencies_zero] at hle
  rw [contrast, mul_one] at hle
  nlinarith [mul_nonneg (sub_nonneg.mpr Real.cos_one_le)
    (add_nonneg (sq_nonneg (x (0, 0))) (sq_nonneg (x (1, 0))))]

/-- The post: a pair of frequency `1` tells all integer positions apart. If `R^n = R^m`, the angles
`n` and `m` agree modulo `2π`, and `n − m = 2πk` forces `k = 0` since `π` is irrational. -/
theorem eq_of_rotary_eq {θ : Fin p → ℝ} {l : Fin p} (hl : θ l = 1) {n m : ℤ}
    (h : rotary θ n = rotary θ m) : n = m := by
  have hc := (ReRoPE.single_dotProduct_rotary θ n l).1
  have hs := (ReRoPE.single_dotProduct_rotary θ n l).2
  rw [h, (ReRoPE.single_dotProduct_rotary θ m l).1] at hc
  rw [h, (ReRoPE.single_dotProduct_rotary θ m l).2] at hs
  rw [hl, mul_one, mul_one] at hc hs
  obtain ⟨k, hk⟩ := Real.Angle.angle_eq_iff_two_pi_dvd_sub.mp (Real.Angle.cos_sin_inj hc hs)
  rcases eq_or_ne k 0 with rfl | hk0
  · simp only [Int.cast_zero, mul_zero, sub_eq_zero] at hk
    exact_mod_cast hk.symm
  · exfalso
    apply (irrational_pi.mul_intCast (m := 2 * k) (by omega)).ne_int (m - n)
    push_cast
    linear_combination -hk

example : (5 : ℤ) = 5 :=
  eq_of_rotary_eq (θ := fun _ : Fin 1 => 1) (l := 0) rfl rfl

/-- The post: the NTK-aware encodings of distinct integer positions differ, for every `α`. -/
theorem adjusted_rotary_injective (base α : ℝ) (p : ℕ) :
    Function.Injective fun n : ℤ => rotary (adjustedFrequencies base α (p + 1)) n :=
  fun _ _ h => eq_of_rotary_eq (adjustedFrequencies_zero base α p) h

example : Function.Injective fun n : ℤ => rotary (adjustedFrequencies 10000 1000000 64) n :=
  adjusted_rotary_injective 10000 1000000 63

end Transformer.RoPE.NTKAware
