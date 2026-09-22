/-
# The emergence of clusters in self-attention dynamics — `l:boundedxn`, the estimates

The pointwise and one-token estimates of the proof of `l:boundedxn` (§7 of
arXiv:2305.05465v6), on the scalar curves of `e:Idnonresca`:

* `mul_exp_neg_antitoneOn` — `ḟ ≤ f` makes `f e^{-t}` nonincreasing;
* `nonneg_of_bounded_max` — a largest token bounded from below is never
  negative: `ẋ_n ≤ x_n`, so once negative it would decay like `-e^t`.  This is
  point (1) of the source with `≥` in place of `>`, for every `t ≥ 0`;
* `mul_exp_le_of_min` — the smallest token is at least `x_1(0) e^t`;
* `deriv_mul_sub_le` — the estimate `e:epsxjxn` on
  `(d/dt) x_n (x_n - x_j)` at a time when `x_n (x_n - x_j) ≤ κ`.

Source: arXiv:2305.05465v6, proof of `l:boundedxn`.
-/

import Transformer.Clusters.Section7_OnlyOne

open scoped BigOperators
open Real Filter Topology

namespace Transformer
namespace Clusters

variable {m : ℕ}

/-- `ḟ ≤ f` on `[T, ∞)` makes `f(t) e^{-t}` nonincreasing there. -/
theorem mul_exp_neg_antitoneOn {f f' : ℝ → ℝ} {T : ℝ}
    (hf : ∀ t, T ≤ t → HasDerivAt f (f' t) t) (h : ∀ t, T ≤ t → f' t ≤ f t) :
    AntitoneOn (fun t => f t * Real.exp (-t)) (Set.Ici T) := by
  refine antitoneOn_Ici_of_hasDerivAt (fun t ht => (hf t ht).mul (hasDerivAt_neg t).exp) ?_
  intro t ht
  have := h t ht
  have he := Real.exp_pos (-t)
  nlinarith

/-- The drift is an average: `Σ_j P_ij y_j ≥ y_L` when `y_L` is the smallest
coordinate. -/
theorem le_drift (y : Idx (m + 1) → ℝ) (i L : Idx (m + 1)) (hmin : ∀ j, y L ≤ y j) :
    y L ≤ ∑ j, Perspective.softmaxWeight (fun l => y i * y l) j * y j := by
  calc y L = ∑ j, Perspective.softmaxWeight (fun l => y i * y l) j * y L := by
        rw [← Finset.sum_mul, Perspective.sum_softmaxWeight (Nat.succ_pos m), one_mul]
    _ ≤ _ := Finset.sum_le_sum fun j _ =>
        mul_le_mul_of_nonneg_left (hmin j) (Perspective.softmaxWeight_nonneg _ _)

/-- **Point (1) of `l:boundedxn`, in the form used.**  A largest token that
stays above `-R` is nonnegative at every time `t ≥ 0`.  The source states
`x_n(t) > 0` for large `t`; the proof of point (3) only uses `x_n ≥ 0`, which
is what is proved, and for every `t ≥ 0` rather than eventually.

Source: arXiv:2305.05465v6, proof of `l:boundedxn`, point (1). -/
theorem nonneg_of_bounded_max (x : ℝ → Idx (m + 1) → ℝ)
    (hder : ∀ t k, HasDerivAt (fun s => x s k)
      (∑ j, Perspective.softmaxWeight (fun l => x t k * x t l) j * x t j) t)
    (N : Idx (m + 1)) (hmax : ∀ t, 0 ≤ t → ∀ j, x t j ≤ x t N) {R : ℝ}
    (hR : ∀ t, 0 ≤ t → -R ≤ x t N) : ∀ t, 0 ≤ t → 0 ≤ x t N := by
  intro t₀ ht₀
  by_contra hneg
  push Not at hneg
  set a := x t₀ N
  have hanti := mul_exp_neg_antitoneOn (T := 0) (fun t _ => hder t N)
    fun t ht => drift_le (x t) N N (hmax t ht)
  set s := (|R| + 1) / -a
  have hs : 0 ≤ s := div_nonneg (by positivity) (by linarith)
  have h1 := hanti (Set.mem_Ici.2 ht₀) (Set.mem_Ici.2 (by linarith)) (le_add_of_nonneg_right hs)
  simp only at h1
  have he : Real.exp (-(t₀ + s)) * Real.exp s = Real.exp (-t₀) := by
    rw [← Real.exp_add]; ring_nf
  have h2 : x (t₀ + s) N ≤ a * Real.exp s := by
    have hp := Real.exp_pos (-(t₀ + s))
    have h3 : x (t₀ + s) N * Real.exp (-(t₀ + s)) ≤ a * Real.exp s * Real.exp (-(t₀ + s)) := by
      nlinarith
    exact le_of_mul_le_mul_right h3 hp
  have h4 : a * Real.exp s ≤ a * (s + 1) :=
    mul_le_mul_of_nonpos_left (Real.add_one_le_exp s) hneg.le
  have h5 : a * s = -(|R| + 1) := by simp only [s]; field_simp; rw [div_self hneg.ne]
  linarith [hR (t₀ + s) (by linarith), le_abs_self R]

/-- The smallest token is at least `x_L(0) e^t`: `ẋ_L ≥ x_L`.

Source: arXiv:2305.05465v6, proof of `l:boundedxn`, the bound `x_1 ≥ -c_1 e^t`. -/
theorem mul_exp_le_of_min (x : ℝ → Idx (m + 1) → ℝ)
    (hder : ∀ t k, HasDerivAt (fun s => x s k)
      (∑ j, Perspective.softmaxWeight (fun l => x t k * x t l) j * x t j) t)
    (L : Idx (m + 1)) (hmin : ∀ t, 0 ≤ t → ∀ j, x t L ≤ x t j) :
    ∀ t, 0 ≤ t → x 0 L * Real.exp t ≤ x t L := by
  intro t ht
  have hanti := mul_exp_neg_antitoneOn (T := 0) (fun t _ => (hder t L).neg)
    fun t ht => neg_le_neg (le_drift (x t) L L (hmin t ht))
  have h1 := hanti (Set.mem_Ici.2 le_rfl) (Set.mem_Ici.2 ht) ht
  simp only [Pi.neg_apply, neg_zero, Real.exp_zero, mul_one] at h1
  have he : Real.exp (-t) * Real.exp t = 1 := by rw [← Real.exp_add]; simp
  have := mul_le_mul_of_nonneg_right h1 (Real.exp_pos t).le
  rw [mul_assoc, he, mul_one] at this
  linarith

/-- **`e:epsxjxn`.**  With `y_N ≥ 0`, `y_J < 0` the largest coordinate
besides `y_N`, and `y_L` the smallest: if `y_N (y_N - y_J) ≤ κ` then `P_NJ ≥ ε = 1/(e^κ + n)`, and
`ẏ_N (2 y_N - y_J) - y_N ẏ_J ≤ 2 y_N² - 2 y_N y_L - ε y_J²`.

Source: arXiv:2305.05465v6, proof of `l:boundedxn`, `e:epsxjxn`. -/
theorem deriv_mul_sub_le (y : Idx (m + 1) → ℝ) (N J L : Idx (m + 1)) (hJN : J ≠ N)
    (hJ : ∀ k, k ≠ N → y k ≤ y J) (hmin : ∀ k, y L ≤ y k)
    (hN : 0 ≤ y N) (hJneg : y J < 0) {κ : ℝ} (hκ : y N * (y N - y J) ≤ κ) :
    (∑ j, Perspective.softmaxWeight (fun l => y N * y l) j * y j) * (2 * y N - y J) -
        y N * ∑ j, Perspective.softmaxWeight (fun l => y J * y l) j * y j ≤
      2 * y N ^ 2 - 2 * y N * y L - 1 / (Real.exp κ + ((m : ℝ) + 1)) * y J ^ 2 := by
  set p := Perspective.softmaxWeight (fun l => y N * y l)
  set ε := 1 / (Real.exp κ + ((m : ℝ) + 1))
  have hε : 0 < ε := by positivity
  set Z := ∑ k : Idx (m + 1), Real.exp (y N * y k)
  have hZ0 : 0 < Z := Perspective.softmaxPartition_pos (Nat.succ_pos m) _
  have hZ : Z ≤ Real.exp (y N * y N) + ((m : ℝ) + 1) * Real.exp (y N * y J) := by
    have hk : ∀ k, Real.exp (y N * y k) ≤
        (if k = N then Real.exp (y N * y N) else 0) + Real.exp (y N * y J) := by
      intro k
      by_cases hk : k = N
      · rw [hk, ite_eq_left rfl]; linarith [Real.exp_pos (y N * y J)]
      · rw [ite_eq_right hk, zero_add]
        exact Real.exp_le_exp.2 (mul_le_mul_of_nonneg_left (hJ k hk) hN)
    have := Finset.sum_le_sum fun k (_ : k ∈ Finset.univ) => hk k
    rw [Finset.sum_add_distrib, Finset.sum_ite_eq' Finset.univ N,
      ite_eq_left (Finset.mem_univ N)] at this
    simpa using this
  have hpJ : ε ≤ p J := by
    show ε ≤ Real.exp (y N * y J) / Z
    rw [le_div_iff₀ hZ0]
    have he : Real.exp (y N * y N) ≤ Real.exp (y N * y J) * Real.exp κ := by
      rw [← Real.exp_add]; exact Real.exp_le_exp.2 (by linarith)
    have h1 : ε * (Real.exp κ + ((m : ℝ) + 1)) = 1 := by
      simp only [ε]; field_simp
    nlinarith [Real.exp_pos (y N * y J)]
  have hp1 : p N ≤ 1 := by
    rw [← Perspective.sum_softmaxWeight (Nat.succ_pos m) (fun l => y N * y l)]
    exact Finset.single_le_sum (f := p) (fun k _ => Perspective.softmaxWeight_nonneg _ _)
      (Finset.mem_univ N)
  have hterm : ∀ k, p k * y k ≤ (if k = N then y N else 0) + (if k = J then ε * y J else 0) := by
    intro k
    by_cases hkN : k = N
    · rw [hkN, ite_eq_left rfl, ite_eq_right (Ne.symm hJN), add_zero]
      nlinarith
    · rw [ite_eq_right hkN, zero_add]
      by_cases hkJ : k = J
      · rw [hkJ, ite_eq_left rfl]
        exact mul_le_mul_of_nonpos_right hpJ hJneg.le
      · rw [ite_eq_right hkJ]
        exact mul_nonpos_of_nonneg_of_nonpos (Perspective.softmaxWeight_nonneg _ _)
          ((hJ k hkN).trans hJneg.le)
  have hD := Finset.sum_le_sum fun k (_ : k ∈ Finset.univ) => hterm k
  rw [Finset.sum_add_distrib, Finset.sum_ite_eq' Finset.univ N, Finset.sum_ite_eq' Finset.univ J,
    ite_eq_left (Finset.mem_univ N), ite_eq_left (Finset.mem_univ J)] at hD
  have hDJ := le_drift y J L hmin
  have hLJ := hmin J
  nlinarith [mul_le_mul_of_nonneg_right hD (by linarith : (0 : ℝ) ≤ 2 * y N - y J),
    mul_le_mul_of_nonneg_left hDJ hN, mul_le_mul_of_nonneg_left hLJ hN,
    mul_nonpos_of_nonneg_of_nonpos (mul_nonneg hε.le hN) hJneg.le]

/-- The hypotheses of `deriv_mul_sub_le` are satisfiable: `y = (-1, 0)`, with
`N = 1`, `J = L = 0` and `κ = 1`. -/
example : ∃ y : Idx 2 → ℝ, (∀ k, k ≠ 1 → y k ≤ y 0) ∧
    (∀ k, y 0 ≤ y k) ∧ 0 ≤ y 1 ∧ y 0 < 0 ∧ y 1 * (y 1 - y 0) ≤ 1 := by
  refine ⟨fun k => (k : ℝ) - 1, ?_, ?_, ?_, ?_, ?_⟩
  · intro k hk; fin_cases k <;> simp_all
  · intro k; fin_cases k <;> simp
  all_goals simp

end Clusters
end Transformer
