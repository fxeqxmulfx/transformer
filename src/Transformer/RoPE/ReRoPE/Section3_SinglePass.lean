module

public import Mathlib.Analysis.SpecialFunctions.Trigonometric.Angle
public import Transformer.RoPE.ReRoPE.Section2_Fusion
public import Transformer.RoPE.RoFormer.Section3_Properties

/-!
# ReRoPE, 计算 (Computation): one pass of RoPE does not suffice

Su, *Transformer升级之路：12、无限外推的ReRoPE？*, kexue.fm/archives/9708, section 计算
(Computation): "RoPE realizes relative positions through absolute ones, so it only gives linear
relative positions, while the matrices (leaky-rerope), (rerope) are nonlinear, piecewise linear;
realizing them takes computing the attention matrix twice and combining the two".

* `exists_int_of_single_pass`: one pass rotates the query at `i` to a position `f i` and the key at
  `j` to `g j`, so each frequency sees the relative position `g j − f i`. If its scores are
  `q_iᵀ R^{−h(i−j)} k_j` for all `j ≤ i`, all `q` and all `k`, the second difference
  `h(n+2) − 2h(n+1) + h(n)` is a whole number of turns at every frequency: `h` is linear up to
  whole turns;
* `not_single_pass_rectified`, `not_single_pass_leaky`: ReRoPE's positions have the second
  difference `−1` at the window, Leaky ReRoPE's `1/k − 1`. At a frequency `0 < θ_l < 2π`, which
  RoPE's `θ_0 = 1` is, neither is a whole number of turns, so no single pass gives either matrix.
  The window must be `w ≥ 1` and `k > 1`: `w = 0` gives NoPE and `k = 1` gives RoPE.
-/

@[expose] public section

open Matrix
open scoped Real

namespace Transformer.RoPE.ReRoPE

open RoFormer

variable {p : ℕ}

/-- `R^t e_{(0,l)}` has `cos(tθ_l)` at `(0, l)` and `sin(tθ_l)` at `(1, l)`. -/
theorem single_dotProduct_rotary (θ : Fin p → ℝ) (t : ℝ) (l : Fin p) :
    Pi.single (0, l) 1 ⬝ᵥ (rotary θ t *ᵥ Pi.single (0, l) 1) = Real.cos (t * θ l) ∧
      Pi.single (1, l) 1 ⬝ᵥ (rotary θ t *ᵥ Pi.single (0, l) 1) = Real.sin (t * θ l) := by
  simp [rotary, blockDiagonal_apply, rotation]

/-- Computation: if one RoPE pass, the query at `i` rotated to `f i` and the key at `j` to `g j`,
gives the scores `q_iᵀ R^{−h(i−j)} k_j` for all `j ≤ i` and all `q`, `k`, then at each frequency
`θ_l` the second difference `h(n+2) − 2h(n+1) + h(n)` is a whole number of turns. -/
theorem exists_int_of_single_pass (θ : Fin p → ℝ) (l : Fin p) {f g h : ℕ → ℝ}
    (hfg : ∀ i j, j ≤ i → ∀ q key, (rotary θ (f i) *ᵥ q) ⬝ᵥ (rotary θ (g j) *ᵥ key) =
      q ⬝ᵥ (rotary θ (-h (i - j)) *ᵥ key)) (n : ℕ) :
    ∃ m : ℤ, (h (n + 2) - 2 * h (n + 1) + h n) * θ l = 2 * π * m := by
  have hturn : ∀ i j, j ≤ i → ∃ m : ℤ, (g j - f i) * θ l - -h (i - j) * θ l = 2 * π * m := by
    intro i j hji
    have hc := hfg i j hji (Pi.single (0, l) 1) (Pi.single (0, l) 1)
    have hs := hfg i j hji (Pi.single (1, l) 1) (Pi.single (0, l) 1)
    rw [rotary_dotProduct, (single_dotProduct_rotary θ _ l).1,
      (single_dotProduct_rotary θ _ l).1] at hc
    rw [rotary_dotProduct, (single_dotProduct_rotary θ _ l).2,
      (single_dotProduct_rotary θ _ l).2] at hs
    exact Real.Angle.angle_eq_iff_two_pi_dvd_sub.mp (Real.Angle.cos_sin_inj hc hs)
  obtain ⟨m1, h1⟩ := hturn (n + 2) 0 (by omega)
  obtain ⟨m2, h2⟩ := hturn (n + 1) 0 (by omega)
  obtain ⟨m3, h3⟩ := hturn (n + 2) 1 (by omega)
  obtain ⟨m4, h4⟩ := hturn (n + 1) 1 (by omega)
  simp only [Nat.sub_zero, show n + 2 - 1 = n + 1 by omega, Nat.add_sub_cancel] at h1 h2 h3 h4
  refine ⟨m1 - m2 - m3 + m4, ?_⟩
  push_cast
  linear_combination h1 - h2 - h3 + h4

example (θ : Fin 64 → ℝ) (l : Fin 64) (n : ℕ) :
    ∃ m : ℤ, (((n + 2 : ℕ) : ℝ) - 2 * ((n + 1 : ℕ) : ℝ) + (n : ℝ)) * θ l = 2 * π * m :=
  exists_int_of_single_pass θ l (f := fun i => i) (g := fun j => j) (h := fun d => d)
    (fun i j hji q key => by rw [rotary_dotProduct, Nat.cast_sub hji, neg_sub]) n

/-- No angle strictly between `0` and `2π` is a whole number of turns. -/
theorem ne_two_pi_mul_int {x : ℝ} (h0 : 0 < x) (h1 : x < 2 * π) (m : ℤ) : x ≠ 2 * π * m := by
  rintro rfl
  have hπ := Real.pi_pos
  have hm0 : (0 : ℝ) < m := by nlinarith
  have hm1 : (m : ℝ) < 1 := by nlinarith
  have hm0' : (0 : ℤ) < m := by exact_mod_cast hm0
  have hm1' : m < 1 := by exact_mod_cast hm1
  omega

example : (1 : ℝ) ≠ 2 * π * (0 : ℤ) :=
  ne_two_pi_mul_int one_pos (by linarith [Real.pi_gt_three]) 0

/-- Computation: no single RoPE pass gives ReRoPE's scores. If some frequency has `0 < θ_l < 2π`
(RoPE's `θ_0 = 1` does) and the window is `w ≥ 1`, no positions `f i`, `g j` make
`(R^{f i} q)ᵀ(R^{g j} k) = q_iᵀ R^{−min(i−j, w)} k_j` for all `j ≤ i` and all `q`, `k`. -/
theorem not_single_pass_rectified (θ : Fin p → ℝ) (l : Fin p) (hθ0 : 0 < θ l)
    (hθ : θ l < 2 * π) {w : ℕ} (hw : 1 ≤ w) :
    ¬ ∃ f g : ℕ → ℝ, ∀ i j, j ≤ i → ∀ q key,
      (rotary θ (f i) *ᵥ q) ⬝ᵥ (rotary θ (g j) *ᵥ key) =
        q ⬝ᵥ (rotary θ (-rectifiedPosition w ((i - j : ℕ) : ℝ)) *ᵥ key) := by
  rintro ⟨f, g, hfg⟩
  obtain ⟨v, rfl⟩ : ∃ v, w = v + 1 := ⟨w - 1, by omega⟩
  obtain ⟨m, hm⟩ := exists_int_of_single_pass θ l
    (h := fun d => rectifiedPosition ((v + 1 : ℕ) : ℝ) d) hfg v
  have e2 : rectifiedPosition ((v + 1 : ℕ) : ℝ) ((v + 2 : ℕ) : ℝ) = v + 1 := by
    rw [rectifiedPosition, min_eq_right (by push_cast; linarith)]
    push_cast
    rfl
  have e1 : rectifiedPosition ((v + 1 : ℕ) : ℝ) ((v + 1 : ℕ) : ℝ) = v + 1 := by
    rw [rectifiedPosition, min_self]
    push_cast
    rfl
  have e0 : rectifiedPosition ((v + 1 : ℕ) : ℝ) (v : ℝ) = v := by
    rw [rectifiedPosition, min_eq_left (by push_cast; linarith)]
  rw [e2, e1, e0] at hm
  exact ne_two_pi_mul_int hθ0 hθ (-m) (by push_cast; linear_combination -hm)

example : ¬ ∃ f g : ℕ → ℝ, ∀ i j, j ≤ i → ∀ q key,
    (rotary (fun i : Fin 64 => frequency 10000 128 i) (f i) *ᵥ q) ⬝ᵥ
        (rotary (fun i : Fin 64 => frequency 10000 128 i) (g j) *ᵥ key) =
      q ⬝ᵥ (rotary (fun i : Fin 64 => frequency 10000 128 i)
        (-rectifiedPosition ((512 : ℕ) : ℝ) ((i - j : ℕ) : ℝ)) *ᵥ key) :=
  not_single_pass_rectified _ 0 (by simp [frequency])
    (by simp [frequency]; linarith [Real.pi_gt_three]) (by norm_num)

/-- Computation: no single RoPE pass gives Leaky ReRoPE's scores for `k > 1`: the second difference
`1/k − 1` of its positions at the window is not a whole number of turns at a frequency
`0 < θ_l < 2π`. -/
theorem not_single_pass_leaky (θ : Fin p → ℝ) (l : Fin p) (hθ0 : 0 < θ l) (hθ : θ l < 2 * π)
    {w : ℕ} (hw : 1 ≤ w) {k : ℝ} (hk : 1 < k) :
    ¬ ∃ f g : ℕ → ℝ, ∀ i j, j ≤ i → ∀ q key,
      (rotary θ (f i) *ᵥ q) ⬝ᵥ (rotary θ (g j) *ᵥ key) =
        q ⬝ᵥ (rotary θ (-leakyPosition w k ((i - j : ℕ) : ℝ)) *ᵥ key) := by
  rintro ⟨f, g, hfg⟩
  obtain ⟨v, rfl⟩ : ∃ v, w = v + 1 := ⟨w - 1, by omega⟩
  obtain ⟨m, hm⟩ := exists_int_of_single_pass θ l
    (h := fun d => leakyPosition ((v + 1 : ℕ) : ℝ) k d) hfg v
  have e2 : leakyPosition ((v + 1 : ℕ) : ℝ) k ((v + 2 : ℕ) : ℝ) = v + 1 + 1 / k := by
    rw [leakyPosition, ite_eq_right (by push_cast; linarith)]
    push_cast
    ring
  have e1 : leakyPosition ((v + 1 : ℕ) : ℝ) k ((v + 1 : ℕ) : ℝ) = v + 1 := by
    rw [leakyPosition, ite_eq_right (lt_irrefl _)]
    push_cast
    ring
  have e0 : leakyPosition ((v + 1 : ℕ) : ℝ) k (v : ℝ) = v := by
    rw [leakyPosition, ite_eq_left (by push_cast; linarith)]
  rw [e2, e1, e0] at hm
  have hk0 : 0 < k := by linarith
  have hk1 : 1 / k < 1 := (div_lt_one hk0).mpr hk
  have hk2 : 0 < 1 / k := by positivity
  exact ne_two_pi_mul_int (x := (1 - 1 / k) * θ l) (mul_pos (by linarith) hθ0) (by nlinarith)
    (-m) (by push_cast; linear_combination -hm)

example : ¬ ∃ f g : ℕ → ℝ, ∀ i j, j ≤ i → ∀ q key,
    (rotary (fun i : Fin 64 => frequency 10000 128 i) (f i) *ᵥ q) ⬝ᵥ
        (rotary (fun i : Fin 64 => frequency 10000 128 i) (g j) *ᵥ key) =
      q ⬝ᵥ (rotary (fun i : Fin 64 => frequency 10000 128 i)
        (-leakyPosition ((512 : ℕ) : ℝ) 16 ((i - j : ℕ) : ℝ)) *ᵥ key) :=
  not_single_pass_leaky _ 0 (by simp [frequency])
    (by simp [frequency]; linarith [Real.pi_gt_three]) (by norm_num) (by norm_num)

end Transformer.RoPE.ReRoPE
