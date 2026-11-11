module

public import Transformer.Grokking.FlatSpectrum.Training.Scope.SignSystem

/-!
# The first mode of the sign sample overtakes the second

Source: `review/flat_spectrum.md`, *Training with cross-entropy* (added), item XE19,
statements 2 and 3.

For the system `IsSignSystem` of the squared norms `a = ‖U + V‖²` and `b = ‖U − V‖²` of the
sign sample without weight decay, at every step `η > 0`, from `a_0 > 0` and `b_0 ≥ 0`:

* while `a ≤ b` the real part `(a − b) / 4` of the product is not positive, the weight is at
  least `1`, and `(1 + s)² ≥ (1 + 4 η / (1 + 4 η²)) (1 − s)²` for `s = η w` in `[η, 2 η]`: the
  ratio `a / b` grows geometrically, so `a` exceeds `b` at some step
  (`IsSignSystem.exists_lt`);
* from that step `t₀` the ratio is at least `a_{t₀} / b_{t₀} > 1`, and `a_t → ∞`, so
  `a_t − b_t → ∞` (`IsSignSystem.tendsto_sub_atTop`);
* once `(a − b) / 4 ≥ η / 2` the step times the weight is at most `2` and `b` does not grow
  (`IsSignSystem.exists_forall_snd_le`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Training.Scope

open Filter Topology

namespace IsSignSystem

variable {η : ℝ} {a b : ℕ → ℝ}

/-- While `a ≤ b` the weight is at least `1`, and the ratio `a / b` grows by the factor
`1 + 4 η / (1 + 4 η²)` at every step. -/
private lemma pow_mul_le (h : IsSignSystem η a b) (hη : 0 < η) (ha0 : 0 < a 0) (hb0 : 0 ≤ b 0)
    (hcon : ∀ t, a t ≤ b t) (t : ℕ) :
    (1 + 4 * η / (1 + 4 * η ^ 2)) ^ t * a 0 * b t ≤ a t * b 0 := by
  have hD : (0 : ℝ) < 1 + 4 * η ^ 2 := by positivity
  have hq : 0 < 4 * η / (1 + 4 * η ^ 2) := by positivity
  induction t with
  | zero => simp
  | succ t ih =>
    rw [h.1 t, h.2 t]
    have hw1 : 1 ≤ signWeight ((a t - b t) / 4) := by
      rw [← signWeight_zero]
      exact signWeight_anti (by linarith [hcon t])
    have hw2 := signWeight_le_two ((a t - b t) / 4)
    obtain ⟨s, hsdef⟩ : ∃ s : ℝ, s = η * signWeight ((a t - b t) / 4) := ⟨_, rfl⟩
    rw [← hsdef]
    have hs1 : η ≤ s := by rw [hsdef]; nlinarith
    have hs2 : s ≤ 2 * η := by rw [hsdef]; nlinarith
    have h3 : (1 - s) ^ 2 ≤ 1 + 4 * η ^ 2 := by
      nlinarith [mul_nonneg (sub_nonneg.2 hs2) (by linarith : (0 : ℝ) ≤ 2 * η + s)]
    have h4 : 4 * η / (1 + 4 * η ^ 2) * (1 - s) ^ 2 ≤ 4 * η := by
      rw [div_mul_eq_mul_div, div_le_iff₀ hD]
      exact mul_le_mul_of_nonneg_left h3 (by positivity)
    have h1 : (1 + 4 * η / (1 + 4 * η ^ 2)) * (1 - s) ^ 2 ≤ (1 + s) ^ 2 := by nlinarith
    have h2 := mul_nonneg (h.fst_pos hη.le ha0 t).le hb0
    calc (1 + 4 * η / (1 + 4 * η ^ 2)) ^ (t + 1) * a 0 * ((1 - s) ^ 2 * b t)
        = (1 + 4 * η / (1 + 4 * η ^ 2)) * (1 - s) ^ 2 *
            ((1 + 4 * η / (1 + 4 * η ^ 2)) ^ t * a 0 * b t) := by ring
      _ ≤ (1 + 4 * η / (1 + 4 * η ^ 2)) * (1 - s) ^ 2 * (a t * b 0) :=
        mul_le_mul_of_nonneg_left ih (mul_nonneg (by linarith) (sq_nonneg _))
      _ ≤ (1 + s) ^ 2 * (a t * b 0) := mul_le_mul_of_nonneg_right h1 h2
      _ = (1 + s) ^ 2 * a t * b 0 := by ring

/-- **At some step the first mode exceeds the second** (`review/flat_spectrum.md`, item XE19,
statement 2). -/
lemma exists_lt (h : IsSignSystem η a b) (hη : 0 < η) (ha0 : 0 < a 0) (hb0 : 0 ≤ b 0) :
    ∃ t, b t < a t := by
  by_contra hcon
  replace hcon : ∀ t, a t ≤ b t := fun t => not_lt.1 fun ht => hcon ⟨t, ht⟩
  have hq : 0 < 4 * η / (1 + 4 * η ^ 2) := by positivity
  obtain ⟨t, ht⟩ := pow_unbounded_of_one_lt (b 0 / a 0)
    (by linarith : 1 < 1 + 4 * η / (1 + 4 * η ^ 2))
  have h1 := h.pow_mul_le hη ha0 hb0 hcon t
  have h2 := mul_le_mul_of_nonneg_left (hcon t)
    (mul_nonneg (pow_nonneg (by linarith : 0 ≤ 1 + 4 * η / (1 + 4 * η ^ 2)) t) ha0.le)
  have hat := h.fst_pos hη.le ha0 t
  rw [div_lt_iff₀ ha0] at ht
  nlinarith [mul_pos (sub_pos.2 ht) hat]

/-- **The difference of the modes tends to infinity** (`review/flat_spectrum.md`, item XE19,
statement 2), for every step `η > 0`. -/
theorem tendsto_sub_atTop (h : IsSignSystem η a b) (hη : 0 < η) (ha0 : 0 < a 0)
    (hb0 : 0 ≤ b 0) : Tendsto (fun t => a t - b t) atTop atTop := by
  obtain ⟨t₀, ht₀⟩ := h.exists_lt hη ha0 hb0
  have hat₀ := h.fst_pos hη.le ha0 t₀
  refine tendsto_atTop_mono' atTop ?_
    ((h.tendsto_fst_atTop hη ha0 hb0).const_mul_atTop (div_pos (sub_pos.2 ht₀) hat₀))
  filter_upwards [eventually_ge_atTop t₀] with t ht
  have h1 := h.mul_le_mul_of_le hη.le ha0 hb0 ht
  rw [div_mul_eq_mul_div, div_le_iff₀ hat₀]
  nlinarith

/-- **From some step on the second mode does not grow** (`review/flat_spectrum.md`, item XE19,
statement 3). -/
theorem exists_forall_snd_le (h : IsSignSystem η a b) (hη : 0 < η) (ha0 : 0 < a 0)
    (hb0 : 0 ≤ b 0) : ∃ T, ∀ t, T ≤ t → b t ≤ b T := by
  obtain ⟨T, hT⟩ := eventually_atTop.1
    ((h.tendsto_sub_atTop hη ha0 hb0).eventually_ge_atTop (2 * η))
  refine ⟨T, fun t ht => ?_⟩
  induction t, ht using Nat.le_induction with
  | base => exact le_rfl
  | succ t ht ih =>
    refine le_trans ?_ ih
    rw [h.2 t]
    have hs0 := mul_nonneg hη.le (signWeight_pos ((a t - b t) / 4)).le
    have hs2 := mul_signWeight_le_two (η := η) (r := (a t - b t) / 4) (by linarith [hT t ht])
    exact mul_le_of_le_one_left (h.snd_nonneg hb0 t)
      (by nlinarith [mul_nonneg hs0 (sub_nonneg.2 hs2)])

end IsSignSystem

/-! The witness: the system with the step `1` from `a_0 = 1` and `b = 0`. -/

example := isSignSystem_signWitness.exists_lt one_pos one_pos le_rfl

example := isSignSystem_signWitness.tendsto_sub_atTop one_pos one_pos le_rfl

example := isSignSystem_signWitness.exists_forall_snd_le one_pos one_pos le_rfl

end Transformer.Grokking.FlatSpectrum.Training.Scope
