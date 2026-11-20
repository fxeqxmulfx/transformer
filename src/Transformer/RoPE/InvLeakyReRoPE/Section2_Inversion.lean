module

public import Mathlib.Analysis.Real.Pi.Bounds
public import Transformer.RoPE.InvLeakyReRoPE.Section1_Review
public import Transformer.RoPE.ReRoPE.Section3_Computation
public import Transformer.RoPE.ReRoPE.Section3_SinglePass

/-!
# InvLeaky ReRoPE, 反转 (Inversion): train with Leaky ReRoPE, infer with RoPE

Su, *Transformer升级之路：13、逆用Leaky ReRoPE*, kexue.fm/archives/9728, section 反转
(Inversion): "if we turn it around, using Leaky ReRoPE in training with a step larger than `1`
outside the window, then by the principle 'at inference use a smaller step outside the window',
could inference use the step `1` outside the window and fall back to RoPE?" Training uses
`leakyPosition w k` with `k < 1`; inference uses RoPE's position `d`.

* `leakyPosition_step` (`Section1_Review`): outside the window the positions step by `1/k`,
  above RoPE's `1` when `k < 1`;
* `leakyPosition_leakyPosition`: for `k₂ > 0` the maps compose, `k₁` after `k₂` being `k₁k₂`;
  `leakyPosition_inv`: so RoPE's position `d` is the training position of the training distance
  `leakyPosition w k⁻¹ d`. Read in training distances, inference with RoPE is Leaky ReRoPE with
  `k⁻¹ > 1`, the "smaller step at inference", and `leakyPosition_inv_left` reverses it. The
  experiments' choice of `k` and `w` is `Section3_Experiments`;
* `not_single_pass_leaky_of_ne`: no single RoPE pass gives Leaky ReRoPE's scores on sequences of
  length `N ≥ w + 2` when `(1/k − 1)θ_l` is not a whole number of turns at some frequency; this
  holds for `k` on either side of `1`, where part 12's `ReRoPE.not_single_pass_leaky` takes
  `k > 1` and all positions (`exists_int_of_single_pass_lt` bounds part 12's lemma);
* `not_single_pass_invLeaky`: so training at the length `512` with `w = 128` and `k = 1/16` is not
  one RoPE pass at a frequency `θ_l = 1`, RoPE's `θ_0`, since `15` is not a multiple of `2π`
  (`fifteen_ne_two_pi_mul_int`). Part 12's bound for passes through features of any dimension
  `D`, `ReRoPE.not_single_pass_features_leaky`, needs more than `(D + 2)w` tokens, more than `512`
  once `D ≥ 2`; these theorems are about RoPE passes, which rotate `q` and `k`. The training takes
  part 12's two attention matrices, `invLeaky_merge`: this is its slowdown. Inference is one RoPE
  pass, `RoFormer.rotary_dotProduct`.
-/

@[expose] public section

open Matrix
open scoped Real

namespace Transformer.RoPE.InvLeakyReRoPE

open ReRoPE RoFormer

/-- Inversion: for `k₂ > 0` Leaky ReRoPE's maps compose: `k₁` after `k₂` is `k₁k₂`. The inner
map keeps the distances outside the window outside it, so `k₁` needs no sign. -/
theorem leakyPosition_leakyPosition (w k₁ : ℝ) {k₂ : ℝ} (h₂ : 0 < k₂) (d : ℝ) :
    leakyPosition w k₁ (leakyPosition w k₂ d) = leakyPosition w (k₁ * k₂) d := by
  by_cases hd : d < w
  · rw [leakyPosition_of_lt k₂ hd, leakyPosition_of_lt k₁ hd, leakyPosition_of_lt _ hd]
  · have h0 : 0 ≤ (d - w) / k₂ := div_nonneg (by linarith [not_lt.mp hd]) h₂.le
    have e : leakyPosition w k₂ d = w + (d - w) / k₂ := by
      rw [leakyPosition, ite_eq_right hd]
    rw [e, leakyPosition, ite_eq_right (not_lt.mpr (by linarith)), leakyPosition,
      ite_eq_right hd, add_sub_cancel_left, div_div, mul_comm k₂]

example (d : ℝ) : leakyPosition 128 16 (leakyPosition 128 2 d) = leakyPosition 128 (16 * 2) d :=
  leakyPosition_leakyPosition 128 16 (by norm_num) d

/-- Inversion: for `k > 0`, RoPE's position `d` is the training position of the training distance
`leakyPosition w k⁻¹ d`: inference with RoPE is Leaky ReRoPE with `k⁻¹` in training distances. -/
theorem leakyPosition_inv (w : ℝ) {k : ℝ} (hk : 0 < k) (d : ℝ) :
    leakyPosition w k (leakyPosition w k⁻¹ d) = d := by
  rw [leakyPosition_leakyPosition w k (inv_pos.mpr hk), mul_inv_cancel₀ hk.ne',
    leakyPosition_one]

example (d : ℝ) : leakyPosition 128 (1 / 16) (leakyPosition 128 (1 / 16)⁻¹ d) = d :=
  leakyPosition_inv 128 (by norm_num) d

/-- Inversion: for `k > 0` the training distance `d` reads, at RoPE's inference, as the distance
`leakyPosition w k d`, and Leaky ReRoPE with `k⁻¹` brings it back. -/
theorem leakyPosition_inv_left (w : ℝ) {k : ℝ} (hk : 0 < k) (d : ℝ) :
    leakyPosition w k⁻¹ (leakyPosition w k d) = d := by
  rw [leakyPosition_leakyPosition w k⁻¹ hk, inv_mul_cancel₀ hk.ne',
    leakyPosition_one]

example (d : ℝ) : leakyPosition 128 (1 / 16)⁻¹ (leakyPosition 128 (1 / 16) d) = d :=
  leakyPosition_inv_left 128 (by norm_num) d

variable {p : ℕ}

/-- Part 12's `ReRoPE.exists_int_of_single_pass` on sequences of length `N ≥ n + 3`: its proof
uses the queries `n + 1`, `n + 2` and the keys `0`, `1` only. If a RoPE pass gives the scores
`q_iᵀ R^{−h(i−j)} k_j` for `j ≤ i < N`, then `(h(n + 2) − 2h(n + 1) + h(n)) θ_l` is a whole number
of turns. -/
theorem exists_int_of_single_pass_lt (θ : Fin p → ℝ) (l : Fin p) {f g h : ℕ → ℝ} {N : ℕ}
    (hfg : ∀ i j, j ≤ i → i < N → ∀ q key, (rotary θ (f i) *ᵥ q) ⬝ᵥ (rotary θ (g j) *ᵥ key) =
      q ⬝ᵥ (rotary θ (-h (i - j)) *ᵥ key)) {n : ℕ} (hn : n + 3 ≤ N) :
    ∃ m : ℤ, (h (n + 2) - 2 * h (n + 1) + h n) * θ l = 2 * π * m := by
  have hturn : ∀ i j, j ≤ i → i < N →
      ∃ m : ℤ, (g j - f i) * θ l - -h (i - j) * θ l = 2 * π * m := by
    intro i j hji hiN
    have hc := hfg i j hji hiN (Pi.single (0, l) 1) (Pi.single (0, l) 1)
    have hs := hfg i j hji hiN (Pi.single (1, l) 1) (Pi.single (0, l) 1)
    rw [rotary_dotProduct, (single_dotProduct_rotary θ _ l).1,
      (single_dotProduct_rotary θ _ l).1] at hc
    rw [rotary_dotProduct, (single_dotProduct_rotary θ _ l).2,
      (single_dotProduct_rotary θ _ l).2] at hs
    exact Real.Angle.angle_eq_iff_two_pi_dvd_sub.mp (Real.Angle.cos_sin_inj hc hs)
  obtain ⟨m1, h1⟩ := hturn (n + 2) 0 (by omega) (by omega)
  obtain ⟨m2, h2⟩ := hturn (n + 1) 0 (by omega) (by omega)
  obtain ⟨m3, h3⟩ := hturn (n + 2) 1 (by omega) (by omega)
  obtain ⟨m4, h4⟩ := hturn (n + 1) 1 (by omega) (by omega)
  simp only [Nat.sub_zero, show n + 2 - 1 = n + 1 by omega, Nat.add_sub_cancel] at h1 h2 h3 h4
  refine ⟨m1 - m2 - m3 + m4, ?_⟩
  push_cast
  linear_combination h1 - h2 - h3 + h4

/-- RoPE itself, `h(d) = d`, on `512` tokens: the second difference `0` is no turn at all. -/
example (θ : Fin 64 → ℝ) (l : Fin 64) :
    ∃ m : ℤ, (((127 + 2 : ℕ) : ℝ) - 2 * ((127 + 1 : ℕ) : ℝ) + ((127 : ℕ) : ℝ)) * θ l = 2 * π * m :=
  exists_int_of_single_pass_lt θ l (f := fun i => i) (g := fun j => j) (h := fun d => d)
    (N := 512) (fun i j hji _ q key => by rw [rotary_dotProduct, Nat.cast_sub hji, neg_sub])
    (n := 127) (by norm_num)

/-- Inversion: for any `k`, if some frequency `θ_l` makes `(1/k − 1)θ_l` no whole number of
turns, no RoPE pass, the query at `i` rotated to `f i` and the key at `j` to `g j`, gives Leaky
ReRoPE's scores `q_iᵀ R^{−leakyPosition w k (i−j)} k_j` for all `j ≤ i < N` and all `q`, `k`, on
any sequence length `N ≥ w + 2`. -/
theorem not_single_pass_leaky_of_ne (θ : Fin p → ℝ) (l : Fin p) {k : ℝ}
    (hθ : ∀ m : ℤ, (1 / k - 1) * θ l ≠ 2 * π * m) {w N : ℕ} (hw : 1 ≤ w) (hN : w + 2 ≤ N) :
    ¬ ∃ f g : ℕ → ℝ, ∀ i j, j ≤ i → i < N → ∀ q key,
      (rotary θ (f i) *ᵥ q) ⬝ᵥ (rotary θ (g j) *ᵥ key) =
        q ⬝ᵥ (rotary θ (-leakyPosition w k ((i - j : ℕ) : ℝ)) *ᵥ key) := by
  rintro ⟨f, g, hfg⟩
  obtain ⟨v, rfl⟩ : ∃ v, w = v + 1 := ⟨w - 1, by omega⟩
  obtain ⟨m, hm⟩ := exists_int_of_single_pass_lt θ l
    (h := fun d => leakyPosition ((v + 1 : ℕ) : ℝ) k d) hfg (n := v) (by omega)
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
  exact hθ m (by linear_combination hm)

/-- `15` is not a whole number of turns: `4π < 15 < 6π`. -/
theorem fifteen_ne_two_pi_mul_int (m : ℤ) : (15 : ℝ) ≠ 2 * π * m := by
  intro h
  have h1 := Real.pi_gt_three
  have h2 := Real.pi_lt_d2
  have hm2 : (2 : ℝ) < m := by nlinarith
  have hm3 : (m : ℝ) < 3 := by nlinarith
  have hm2' : (2 : ℤ) < m := by exact_mod_cast hm2
  have hm3' : m < 3 := by exact_mod_cast hm3
  omega

/-- Training steps of `2` outside the window, `k = 1/2`, `w = 256` on `4096` tokens at RoPE's
frequencies with the base `10000`: `(2 − 1) · θ_0 = 1` is no whole number of turns. -/
example : ¬ ∃ f g : ℕ → ℝ, ∀ i j, j ≤ i → i < 4096 → ∀ q key,
    (rotary (fun i : Fin 64 => frequency 10000 128 i) (f i) *ᵥ q) ⬝ᵥ
        (rotary (fun i : Fin 64 => frequency 10000 128 i) (g j) *ᵥ key) =
      q ⬝ᵥ (rotary (fun i : Fin 64 => frequency 10000 128 i)
        (-leakyPosition ((256 : ℕ) : ℝ) (1 / 2) ((i - j : ℕ) : ℝ)) *ᵥ key) :=
  not_single_pass_leaky_of_ne _ 0 (fun m => by
    have h0 : frequency 10000 128 ((0 : Fin 64) : ℕ) = 1 := by simp [frequency]
    rw [h0, show (1 / (1 / 2 : ℝ) - 1) * 1 = 1 by norm_num]
    exact ne_two_pi_mul_int one_pos (by linarith [Real.pi_gt_three]) m) (by norm_num) (by norm_num)

/-- Inversion: InvLeaky ReRoPE's training at the length `512`, `w = 128` and `k = 1/16`, is no
single RoPE pass when some frequency is `θ_l = 1`, as RoPE's `θ_0 = base^0` is:
`(16 − 1) · 1 = 15` is not a whole number of turns. -/
theorem not_single_pass_invLeaky (θ : Fin p → ℝ) (l : Fin p) (hθ : θ l = 1) :
    ¬ ∃ f g : ℕ → ℝ, ∀ i j, j ≤ i → i < 512 → ∀ q key,
      (rotary θ (f i) *ᵥ q) ⬝ᵥ (rotary θ (g j) *ᵥ key) =
        q ⬝ᵥ (rotary θ (-leakyPosition ((128 : ℕ) : ℝ) (1 / 16) ((i - j : ℕ) : ℝ)) *ᵥ key) :=
  not_single_pass_leaky_of_ne θ l (fun m => by
    rw [hθ, show (1 / (1 / 16 : ℝ) - 1) * 1 = 15 by norm_num]
    exact fifteen_ne_two_pi_mul_int m) (by norm_num) (by norm_num)

/-- The `b8` run: RoPE's frequencies with the base `80000` start at `θ_0 = 1`. -/
example : ¬ ∃ f g : ℕ → ℝ, ∀ i j, j ≤ i → i < 512 → ∀ q key,
    (rotary (fun i : Fin 64 => frequency 80000 128 i) (f i) *ᵥ q) ⬝ᵥ
        (rotary (fun i : Fin 64 => frequency 80000 128 i) (g j) *ᵥ key) =
      q ⬝ᵥ (rotary (fun i : Fin 64 => frequency 80000 128 i)
        (-leakyPosition ((128 : ℕ) : ℝ) (1 / 16) ((i - j : ℕ) : ℝ)) *ᵥ key) :=
  not_single_pass_invLeaky _ 0 (by simp [frequency])

/-- Inversion: InvLeaky ReRoPE's training scores are part 12's two matrices merged at `w = 128`
and `k = 1/16`, the second with the queries at `16(i − 128) + 128` and the keys at `16j`. -/
theorem invLeaky_merge (θ : Fin p → ℝ) (q key : Fin 2 × Fin p → ℝ) (i j : ℝ) :
    (if i - j < 128 then (rotary θ i *ᵥ q) ⬝ᵥ (rotary θ j *ᵥ key)
      else (rotary θ (16 * (i - 128) + 128) *ᵥ q) ⬝ᵥ (rotary θ (16 * j) *ᵥ key)) =
      q ⬝ᵥ (rotary θ (-leakyPosition 128 (1 / 16) (i - j)) *ᵥ key) := by
  rw [← merge_leaky θ q key 128 (1 / 16) i j,
    show (i - 128) / (1 / 16 : ℝ) + 128 = 16 * (i - 128) + 128 by ring,
    show j / (1 / 16 : ℝ) = 16 * j by ring]

end Transformer.RoPE.InvLeakyReRoPE
