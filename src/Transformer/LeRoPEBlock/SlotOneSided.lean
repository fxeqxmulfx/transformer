module

public import Transformer.LeRoPEBlock.SlotWeight
public import Mathlib.Analysis.Convex.Jensen
public import Mathlib.Analysis.Convex.SpecificFunctions.Basic

/-!
# A LeRoPE head keeps a bounded share on a key with keys of its token on one side

Source: `review/lerope_block.md`, *Reading a slot* (added), RS3.

RS1 needs `R` keys of the slot's token on each side, so it says nothing about a slot near an end:
the key next to the query has one key on the near side, the first key of the context none on the
far side. Weights `w` on other keys, on either side or both, do the same job. Margins become
weight through Gibbs' inequality `∑ e^a ≥ e^{∑ w a + H}`, `H = −∑ w log w`
(`exp_add_entropy_le`): the slot keeps at most `1 / (1 + e^{H − M})` when the weighted margins
are at most `M` (`depthWeight_le_of_weighted`). A band's margins average under `w` to its value
times `1 − C` minus its quadrature times `S`, where `C + i S = ∑ w e^{i e θ}` over the offsets `e`
of the keys from the slot, so they are at most `V` times its amplitude when
`|1 − ∑ w e^{i e θ}| ≤ V` (`sum_weight_band_sub_le`); `SlotOneSidedHead` turns this into the cap
of a head of the main model.
-/

@[expose] public section

open Matrix Finset
open Transformer.AttentionResiduals Transformer.RoPE.RoFormer

namespace Transformer.LeRoPEBlock

variable {p : ℕ}

/-- **Gibbs' inequality for `exp`**: for weights `w ≥ 0` summing to one,
`exp(∑ w a + H) ≤ ∑ exp a`, with the entropy `H = −∑ w log w`. -/
theorem exp_add_entropy_le {ι : Type*} (S : Finset ι) (w a : ι → ℝ) (hw : ∀ x ∈ S, 0 ≤ w x)
    (hw1 : ∑ x ∈ S, w x = 1) :
    Real.exp (∑ x ∈ S, w x * a x - ∑ x ∈ S, w x * Real.log (w x)) ≤
      ∑ x ∈ S, Real.exp (a x) := by
  have hJ := convexOn_exp.map_sum_le hw hw1 fun x _ => Set.mem_univ (a x - Real.log (w x))
  simp only [smul_eq_mul] at hJ
  calc Real.exp (∑ x ∈ S, w x * a x - ∑ x ∈ S, w x * Real.log (w x))
      = Real.exp (∑ x ∈ S, w x * (a x - Real.log (w x))) := by
        simp only [mul_sub, sum_sub_distrib]
    _ ≤ ∑ x ∈ S, w x * Real.exp (a x - Real.log (w x)) := hJ
    _ ≤ ∑ x ∈ S, Real.exp (a x) := sum_le_sum fun x hx => by
        rcases (hw x hx).eq_or_lt with h | h
        · rw [← h, zero_mul]
          exact (Real.exp_pos _).le
        · rw [Real.exp_sub, Real.exp_log h, mul_div_cancel₀ _ h.ne']

/-- Two points of weight one half: `exp((a₀ + a₁)/2 + ln 2) ≤ e^{a₀} + e^{a₁}`. -/
example (a : ℕ → ℝ) := exp_add_entropy_le (range 2) (fun _ => 1 / 2) a (fun _ _ => by norm_num)
  (by norm_num)

/-- **A key keeps at most `1 / (1 + e^{H − M})` of the weight** if weights `w` of entropy
`H = −∑ w log w` on other keys, at the distances `k x ≠ t`, put the margins `f t − f (k x)` at
most `M` on average. -/
theorem depthWeight_le_of_weighted {ι : Type*} {i t : ℕ} (hti : t ≤ i) (S : Finset ι)
    (k : ι → ℕ) (hk : Set.InjOn k S) (hkt : ∀ x ∈ S, k x ≠ t) (hki : ∀ x ∈ S, k x ≤ i)
    (f : ℕ → ℝ) (w : ι → ℝ) (hw : ∀ x ∈ S, 0 ≤ w x) (hw1 : ∑ x ∈ S, w x = 1) {M : ℝ}
    (hM : ∑ x ∈ S, w x * (f t - f (k x)) ≤ M) :
    depthWeight (fun j : Fin (i + 1) => f (i - j)) ⟨i - t, by omega⟩ ≤
      1 / (1 + Real.exp (-∑ x ∈ S, w x * Real.log (w x) - M)) := by
  set Z := ∑ δ ∈ range (i + 1), Real.exp (f δ)
  set a := Real.exp (f t)
  have ha : 0 < a := Real.exp_pos _
  have hwt : depthWeight (fun j : Fin (i + 1) => f (i - j)) ⟨i - t, by omega⟩ = a / Z := by
    simp only [depthWeight, depthPartition, sum_fin_exp_sub, a, Z]
    rw [Nat.sub_sub_self hti]
  -- the slot and the weighted keys are distinct keys
  have hsub : insert t (S.image k) ⊆ range (i + 1) := by
    intro δ hδ
    rw [mem_insert, mem_image] at hδ
    rw [mem_range]
    rcases hδ with rfl | ⟨x, hx, rfl⟩
    · omega
    · have := hki x hx
      omega
  have htS : t ∉ S.image k := by
    rw [mem_image]
    rintro ⟨x, hx, h⟩
    exact hkt x hx h
  have hZ : a + ∑ x ∈ S, Real.exp (f (k x)) ≤ Z := by
    have h1 := sum_le_sum_of_subset_of_nonneg (f := fun δ => Real.exp (f δ)) hsub
      fun _ _ _ => (Real.exp_pos _).le
    rwa [sum_insert htS, sum_image hk] at h1
  -- Gibbs on the differences `f (k x) − f t`
  have hG := exp_add_entropy_le S w (fun x => f (k x) - f t) hw hw1
  have hsplit : ∑ x ∈ S, Real.exp (f (k x) - f t) = (∑ x ∈ S, Real.exp (f (k x))) / a := by
    rw [sum_div]
    exact sum_congr rfl fun x _ => Real.exp_sub _ _
  have hneg : ∑ x ∈ S, w x * (f (k x) - f t) = -∑ x ∈ S, w x * (f t - f (k x)) := by
    rw [← sum_neg_distrib]
    exact sum_congr rfl fun x _ => by ring
  have hle : -∑ x ∈ S, w x * Real.log (w x) - M ≤
      ∑ x ∈ S, w x * (f (k x) - f t) - ∑ x ∈ S, w x * Real.log (w x) := by
    rw [hneg]
    linarith
  have hE := (Real.exp_le_exp.2 hle).trans hG
  rw [hsplit, le_div_iff₀ ha] at hE
  have hZ0 : 0 < Z := by
    have : 0 ≤ ∑ x ∈ S, Real.exp (f (k x)) := sum_nonneg fun _ _ => (Real.exp_pos _).le
    linarith
  rw [hwt, div_le_div_iff₀ hZ0 (by positivity)]
  linarith

/-- Three keys, the slot at the distance `1`, all the weight on the key at the distance `2`. -/
example (f : ℕ → ℝ) (M : ℝ) (hM : ∑ x ∈ range 1, (fun _ => (1 : ℝ)) x *
    (f 1 - f ((fun _ => 2) x)) ≤ M) :
    depthWeight (fun j : Fin (2 + 1) => f (2 - j)) ⟨2 - 1, by omega⟩ ≤
      1 / (1 + Real.exp (-∑ x ∈ range 1, (fun _ => (1 : ℝ)) x *
        Real.log ((fun _ => (1 : ℝ)) x) - M)) :=
  depthWeight_le_of_weighted (by norm_num) (range 1) (fun _ => 2)
    (fun x hx y hy _ => by rw [coe_range, Set.mem_Iio] at hx hy; omega) (fun _ _ => by norm_num)
    (fun _ _ => le_rfl) f _ (fun _ _ => zero_le_one) (by simp) hM

/-- **The weighted margins of a band**: under weights summing to one at the offsets `e x`, they
are `b (1 − C) − b' S`, `b` the band at `c`, `b'` its quadrature and `C + i S = ∑ w e^{i e θ}`,
so at most `V A` whenever `(1 − C)² + S² ≤ V²` and `X² + Y² ≤ A²`. -/
theorem sum_weight_band_sub_le {ι : Type*} (S : Finset ι) (w e : ι → ℝ)
    (hw1 : ∑ x ∈ S, w x = 1) (X Y θ c : ℝ) {V A : ℝ} (hV0 : 0 ≤ V) (hA0 : 0 ≤ A)
    (hXY : X ^ 2 + Y ^ 2 ≤ A ^ 2)
    (hV : (1 - ∑ x ∈ S, w x * Real.cos (e x * θ)) ^ 2 +
      (∑ x ∈ S, w x * Real.sin (e x * θ)) ^ 2 ≤ V ^ 2) :
    ∑ x ∈ S, w x * (band X Y θ c - band X Y θ (c + e x)) ≤ V * A := by
  set C := ∑ x ∈ S, w x * Real.cos (e x * θ)
  set Sn := ∑ x ∈ S, w x * Real.sin (e x * θ)
  set b := band X Y θ c
  set b' := Y * Real.cos (c * θ) - X * Real.sin (c * θ)
  have hshift (x : ι) : band X Y θ (c + e x) =
      b * Real.cos (e x * θ) + b' * Real.sin (e x * θ) := by
    simp only [b, b', band]
    rw [add_mul, Real.cos_add, Real.sin_add]
    ring
  have hterm : ∀ x ∈ S, w x * (b - band X Y θ (c + e x)) =
      b * w x - b * (w x * Real.cos (e x * θ)) - b' * (w x * Real.sin (e x * θ)) :=
    fun x _ => by rw [hshift]; ring
  have hsum : ∑ x ∈ S, w x * (b - band X Y θ (c + e x)) = b * (1 - C) - b' * Sn := by
    rw [sum_congr rfl hterm, sum_sub_distrib, sum_sub_distrib, ← mul_sum, ← mul_sum, ← mul_sum,
      hw1]
    ring
  have hbb : b ^ 2 + b' ^ 2 = X ^ 2 + Y ^ 2 := by
    simp only [b, b', band]
    linear_combination (X ^ 2 + Y ^ 2) * Real.cos_sq_add_sin_sq (c * θ)
  have hm := mul_le_mul (hbb.trans_le hXY) hV (by positivity) (sq_nonneg A)
  have hl : (b * (1 - C) - b' * Sn) ^ 2 + (b * Sn + b' * (1 - C)) ^ 2 =
      (b ^ 2 + b' ^ 2) * ((1 - C) ^ 2 + Sn ^ 2) := by ring
  have hL : (b * (1 - C) - b' * Sn) ^ 2 ≤ (V * A) ^ 2 := by
    nlinarith [sq_nonneg (b * Sn + b' * (1 - C))]
  rw [hsum]
  nlinarith [mul_nonneg hV0 hA0]

/-- One offset of weight one: the margin of a band of amplitude one is at most `2`. -/
example (θ c : ℝ) : ∑ x ∈ range 1, (fun _ => (1 : ℝ)) x *
    (band 1 0 θ c - band 1 0 θ (c + (fun _ => (1 : ℝ)) x)) ≤ 2 * 1 :=
  sum_weight_band_sub_le (range 1) (fun _ => 1) (fun _ => 1) (by simp) 1 0 θ c (by norm_num)
    (by norm_num) (by norm_num) (by
      simp only [sum_range_one, one_mul]
      nlinarith [Real.cos_sq_add_sin_sq θ, Real.neg_one_le_cos θ, Real.cos_le_one θ])

/-- The pairs of a band of LeRoPE's score: `X² + Y² ≤ ((|q_l|² + |k_l|²)/2)²`. -/
lemma band_coeff_sq_le (q₀ q₁ k₀ k₁ : ℝ) :
    (q₀ * k₀ + q₁ * k₁) ^ 2 + (q₀ * k₁ - q₁ * k₀) ^ 2 ≤
      ((q₀ ^ 2 + q₁ ^ 2 + k₀ ^ 2 + k₁ ^ 2) / 2) ^ 2 := by
  have h : (q₀ * k₀ + q₁ * k₁) ^ 2 + (q₀ * k₁ - q₁ * k₀) ^ 2 =
      (q₀ ^ 2 + q₁ ^ 2) * (k₀ ^ 2 + k₁ ^ 2) := by ring
  rw [h]
  nlinarith [sq_nonneg (q₀ ^ 2 + q₁ ^ 2 - k₀ ^ 2 - k₁ ^ 2)]

end Transformer.LeRoPEBlock
