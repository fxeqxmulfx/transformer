module

public import Mathlib.Analysis.Real.Pi.Bounds
public import Transformer.RoPE.BaseBounds.Section4_ExpectedScores

/-!
# §4.2 The cosine kernel changes sign

Men et al., *Base of RoPE Bounds Context Length*, arXiv:2405.14591v1, §4.2. After Theorem 1 the
source calls `B_{m,θ} = ∑_{i=0}^{d/2-1} cos(m θ_i)` the ability to attend more to similar tokens
than to random ones, "which decreases as the relative distance `m` increases, as shown in
Figure 4. For a very small base value, we can observe that the `B_{m,θ}` is even below zero at a
certain distance".

For distinct frequencies in `(0, π)` the kernel has bounded partial sums. A nonnegative kernel
would then average to zero, and testing it against `cos(m θ₀)` rules that out:

* `abs_sum_cos_le`: `|∑_{m<M} cos(θ m)| ≤ 1/|sin(θ/2)|`, the Dirichlet-kernel bound;
* `abs_sum_cosineKernel_le`: the partial sums of `B` are bounded;
* `exists_cosineKernel_neg`: `B` is negative at some integer distance for all distinct
  frequencies in `(0, π)`, not only for small bases;
* `cosineKernel_not_antitone`: `B` does not decrease with the distance. The source's
  "decreases" describes the trend of its Figure 4, not a monotone function.
-/

@[expose] public section

open scoped BigOperators

namespace Transformer.RoPE.BaseBounds

/-- The telescoping identity behind the Dirichlet-kernel bound:
`2 sin(θ/2) cos(θ m) = sin(θ m + θ/2) − sin(θ m − θ/2)`. -/
theorem two_sin_mul_sum_cos (θ : ℝ) (M : ℕ) :
    2 * Real.sin (θ / 2) * ∑ m ∈ Finset.range M, Real.cos (θ * m) =
      Real.sin (θ * M - θ / 2) + Real.sin (θ / 2) := by
  induction M with
  | zero => simp [Real.sin_neg]
  | succ M ih =>
    have h : θ * ((M + 1 : ℕ) : ℝ) - θ / 2 = θ * M + θ / 2 := by push_cast; ring
    rw [Finset.sum_range_succ, mul_add, ih, h, Real.sin_add, Real.sin_sub]
    ring

/-- The Dirichlet-kernel bound: away from the multiples of `2π` the cosine sums are bounded. -/
theorem abs_sum_cos_le {θ : ℝ} (hθ : Real.sin (θ / 2) ≠ 0) (M : ℕ) :
    |∑ m ∈ Finset.range M, Real.cos (θ * m)| ≤ 1 / |Real.sin (θ / 2)| := by
  rw [le_div_iff₀ (abs_pos.mpr hθ)]
  have h := congrArg abs (two_sin_mul_sum_cos θ M)
  rw [abs_mul, abs_mul, abs_two] at h
  have h2 := (abs_add_le (Real.sin (θ * M - θ / 2)) (Real.sin (θ / 2))).trans
    (add_le_add (Real.abs_sin_le_one _) (Real.abs_sin_le_one _))
  nlinarith

example (M : ℕ) : |∑ m ∈ Finset.range M, Real.cos (1 * m)| ≤ 1 / |Real.sin (1 / 2)| :=
  abs_sum_cos_le (Real.sin_pos_of_pos_of_lt_pi (by norm_num)
    (by linarith [Real.pi_gt_three])).ne' M

/-- The partial sums of the kernel are bounded when no frequency is a multiple of `2π`. -/
theorem abs_sum_cosineKernel_le {n : ℕ} {omega : Fin n → ℝ}
    (homega : ∀ i, Real.sin (omega i / 2) ≠ 0) (M : ℕ) :
    |∑ m ∈ Finset.range M, cosineKernel omega m| ≤ ∑ i, 1 / |Real.sin (omega i / 2)| := by
  unfold cosineKernel
  rw [Finset.sum_comm]
  exact (Finset.abs_sum_le_sum_abs _ _).trans
    (Finset.sum_le_sum fun i _ => abs_sum_cos_le (homega i) M)

example (M : ℕ) : |∑ m ∈ Finset.range M, cosineKernel (fun _ : Fin 1 => (1 : ℝ)) m| ≤
    ∑ _i : Fin 1, 1 / |Real.sin (1 / 2)| :=
  abs_sum_cosineKernel_le (fun _ => (Real.sin_pos_of_pos_of_lt_pi (by norm_num)
    (by linarith [Real.pi_gt_three])).ne') M

/-- Product to sum, summed over the distances. -/
theorem sum_cos_mul_cos (a b : ℝ) (M : ℕ) :
    ∑ m ∈ Finset.range M, Real.cos (a * m) * Real.cos (b * m) =
      (∑ m ∈ Finset.range M, Real.cos ((a - b) * m) +
        ∑ m ∈ Finset.range M, Real.cos ((a + b) * m)) / 2 := by
  rw [← Finset.sum_add_distrib, Finset.sum_div]
  refine Finset.sum_congr rfl fun m _ => ?_
  rw [sub_mul, add_mul, Real.cos_sub, Real.cos_add]
  ring

/-- §4.2: for distinct frequencies in `(0, π)` the kernel is negative at some integer distance.
Were `B ≥ 0`, testing it against `cos(θ₀ m)` would give, for every `M`,
`M/2 − K ≤ ∑_{m<M} B(m) cos(θ₀ m) ≤ ∑_{m<M} B(m) ≤ C`. -/
theorem exists_cosineKernel_neg {n : ℕ} [NeZero n] {omega : Fin n → ℝ}
    (hinj : Function.Injective omega) (hpos : ∀ i, 0 < omega i)
    (hlt : ∀ i, omega i < Real.pi) :
    ∃ m : ℕ, cosineKernel omega m < 0 := by
  by_contra! hB
  have hsin : ∀ i, Real.sin (omega i / 2) ≠ 0 := fun i =>
    (Real.sin_pos_of_pos_of_lt_pi (by linarith [hpos i]) (by linarith [hlt i, Real.pi_pos])).ne'
  have hplus : ∀ i, Real.sin ((omega i + omega 0) / 2) ≠ 0 := fun i =>
    (Real.sin_pos_of_pos_of_lt_pi (by linarith [hpos i, hpos 0])
      (by linarith [hlt i, hlt 0])).ne'
  have hminus : ∀ i, i ≠ 0 → Real.sin ((omega i - omega 0) / 2) ≠ 0 := by
    intro i hi h
    rw [Real.sin_eq_zero_iff_of_lt_of_lt (by linarith [hpos i, hlt 0, Real.pi_pos])
      (by linarith [hpos 0, hlt i, Real.pi_pos])] at h
    exact hi (hinj (by linarith))
  set c : Fin n → ℝ := fun i =>
    1 / |Real.sin ((omega i - omega 0) / 2)| + 1 / |Real.sin ((omega i + omega 0) / 2)|
  have key : ∀ M : ℕ, (M : ℝ) / 2 - ∑ i, c i / 2 ≤ ∑ i, 1 / |Real.sin (omega i / 2)| := by
    intro M
    have hterm : ∀ i, ((if i = 0 then (M : ℝ) else 0) - c i) / 2 ≤
        (∑ m ∈ Finset.range M, Real.cos ((omega i - omega 0) * m) +
          ∑ m ∈ Finset.range M, Real.cos ((omega i + omega 0) * m)) / 2 := by
      intro i
      have hp := (neg_abs_le _).trans' (neg_le_neg (abs_sum_cos_le (hplus i) M))
      have hm : (if i = 0 then (M : ℝ) else 0) - 1 / |Real.sin ((omega i - omega 0) / 2)| ≤
          ∑ m ∈ Finset.range M, Real.cos ((omega i - omega 0) * m) := by
        split_ifs with hi
        · subst hi
          simp
        · linarith [neg_abs_le (∑ m ∈ Finset.range M, Real.cos ((omega i - omega 0) * m)),
            abs_sum_cos_le (hminus i hi) M]
      simp only [c]
      linarith
    have hlow : ∑ i, ((if i = 0 then (M : ℝ) else 0) - c i) / 2 ≤
        ∑ m ∈ Finset.range M, cosineKernel omega m * Real.cos (omega 0 * m) := by
      unfold cosineKernel
      simp_rw [Finset.sum_mul]
      rw [Finset.sum_comm]
      exact Finset.sum_le_sum fun i _ => (hterm i).trans_eq (sum_cos_mul_cos _ _ M).symm
    have hup : ∑ m ∈ Finset.range M, cosineKernel omega m * Real.cos (omega 0 * m) ≤
        ∑ i, 1 / |Real.sin (omega i / 2)| :=
      (Finset.sum_le_sum fun m _ => mul_le_of_le_one_right (hB m) (Real.cos_le_one _)).trans
        ((le_abs_self _).trans (abs_sum_cosineKernel_le hsin M))
    have hsum : ∑ i, ((if i = 0 then (M : ℝ) else 0) - c i) / 2 =
        (M : ℝ) / 2 - ∑ i, c i / 2 := by
      simp only [sub_div, Finset.sum_sub_distrib, ← Finset.sum_div, Finset.sum_ite_eq',
        Finset.mem_univ, ite_true]
    linarith
  set C := ∑ i, 1 / |Real.sin (omega i / 2)|
  set K := ∑ i, c i / 2
  have hM := key (⌈2 * (C + K)⌉₊ + 1)
  push_cast at hM
  linarith [Nat.le_ceil (2 * (C + K))]

example : ∃ m : ℕ, cosineKernel ![(1 : ℝ), 1 / 2] m < 0 :=
  exists_cosineKernel_neg
    (by intro i j h; fin_cases i <;> fin_cases j <;> first | rfl | norm_num at h)
    (by intro i; fin_cases i <;> norm_num)
    (by intro i; fin_cases i <;> norm_num <;> linarith [Real.pi_gt_three])

/-- §4.2: the kernel does not decrease with the distance. An antitone kernel would stay below its
first negative value, and its partial sums would be unbounded. -/
theorem cosineKernel_not_antitone {n : ℕ} [NeZero n] {omega : Fin n → ℝ}
    (hinj : Function.Injective omega) (hpos : ∀ i, 0 < omega i)
    (hlt : ∀ i, omega i < Real.pi) :
    ¬ Antitone (fun m : ℕ => cosineKernel omega m) := by
  intro hanti
  obtain ⟨m₀, hm₀⟩ := exists_cosineKernel_neg hinj hpos hlt
  have hsin : ∀ i, Real.sin (omega i / 2) ≠ 0 := fun i =>
    (Real.sin_pos_of_pos_of_lt_pi (by linarith [hpos i]) (by linarith [hlt i, Real.pi_pos])).ne'
  set C := ∑ i, 1 / |Real.sin (omega i / 2)|
  have hbound : ∀ M : ℕ, (M : ℝ) * (-cosineKernel omega m₀) ≤ 2 * C := by
    intro M
    have h1 := abs_sum_cosineKernel_le hsin (m₀ + M)
    have h2 := abs_sum_cosineKernel_le hsin m₀
    rw [Finset.sum_range_add] at h1
    have h3 : ∑ j ∈ Finset.range M, cosineKernel omega ((m₀ + j : ℕ) : ℝ) ≤
        M * cosineKernel omega m₀ := by
      calc _ ≤ ∑ _j ∈ Finset.range M, cosineKernel omega m₀ :=
            Finset.sum_le_sum fun j _ => hanti (Nat.le_add_right m₀ j)
        _ = _ := by simp
    linarith [neg_abs_le (∑ m ∈ Finset.range m₀, cosineKernel omega m +
        ∑ j ∈ Finset.range M, cosineKernel omega ((m₀ + j : ℕ) : ℝ)),
      le_abs_self (∑ m ∈ Finset.range m₀, cosineKernel omega m)]
  have hM := hbound (⌈2 * C / (-cosineKernel omega m₀)⌉₊ + 1)
  have hceil := Nat.le_ceil (2 * C / (-cosineKernel omega m₀))
  rw [div_le_iff₀ (by linarith)] at hceil
  push_cast at hM
  nlinarith

example : ¬ Antitone (fun m : ℕ => cosineKernel ![(1 : ℝ), 1 / 2] m) :=
  cosineKernel_not_antitone
    (by intro i j h; fin_cases i <;> fin_cases j <;> first | rfl | norm_num at h)
    (by intro i; fin_cases i <;> norm_num)
    (by intro i; fin_cases i <;> norm_num <;> linarith [Real.pi_gt_three])

end Transformer.RoPE.BaseBounds
