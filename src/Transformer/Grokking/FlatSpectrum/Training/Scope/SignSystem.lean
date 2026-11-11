module

public import Transformer.Grokking.FlatSpectrum.Training.CrossEntropy

/-!
# The two modes of the sign sample without weight decay

Source: `review/flat_spectrum.md`, *Training with cross-entropy* (added), item XE19.

For the sample of XE3 and XE17, one sample of a `1 × 1` product read with the code `±1`, the
gradient is `−w(Re x)` with the weight `w(r) = 2 / (1 + exp (2 r))` (`signWeight`), twice the
probability of the wrong class. Without weight decay the squared norms `a = ‖U + V‖²` and
`b = ‖U − V‖²` follow the system `IsSignSystem`:

    a' = (1 + η w)² a,    b' = (1 − η w)² b,    w = w((a − b) / 4).

For every step `η > 0`, from `a_0 > 0` and `b_0 ≥ 0`:

* `a_t → ∞` (`IsSignSystem.tendsto_fst_atTop`): `a` does not decrease, and below a bound `l`
  the weight is at least `w(l / 4)`, so `a` would grow geometrically;
* the ratio `a / b` does not decrease (`IsSignSystem.mul_le_mul_of_le`).

`SignGap` has the difference `a − b` and the bound on `b`. The witness of the examples is
`signWitness`: the system with the step `1` from `a_0 = 1` and `b = 0`.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Training.Scope

open Filter Topology

/-- The weight of the gradient of the sign sample at a product of real part `r`
(`review/flat_spectrum.md`, item XE19): `2 / (1 + exp (2 r))`, twice the probability of the
wrong class. -/
noncomputable def signWeight (r : ℝ) : ℝ := 2 / (1 + Real.exp (2 * r))

lemma signWeight_pos (r : ℝ) : 0 < signWeight r := by
  unfold signWeight
  positivity

lemma signWeight_le_two (r : ℝ) : signWeight r ≤ 2 := by
  unfold signWeight
  rw [div_le_iff₀ (by positivity)]
  nlinarith [Real.exp_pos (2 * r)]

/-- The weight decreases with the real part of the product. -/
lemma signWeight_anti {r r' : ℝ} (h : r ≤ r') : signWeight r' ≤ signWeight r := by
  unfold signWeight
  exact div_le_div_of_nonneg_left zero_le_two (by positivity)
    (by linarith [Real.exp_le_exp.2 (by linarith : 2 * r ≤ 2 * r')])

lemma signWeight_zero : signWeight 0 = 1 := by
  simp [signWeight]
  norm_num

/-- Where the real part of the product is at least `η / 2`, the step times the weight is at
most `2`. -/
lemma mul_signWeight_le_two {η r : ℝ} (hr : η ≤ 2 * r) : η * signWeight r ≤ 2 := by
  unfold signWeight
  rw [mul_div_assoc', div_le_iff₀ (by positivity)]
  nlinarith [Real.add_one_le_exp (2 * r)]

/-- The system of the two modes (`review/flat_spectrum.md`, item XE19): `a` is multiplied by
`(1 + η w)²` and `b` by `(1 − η w)²`, with the weight `w` at `(a − b) / 4`. -/
def IsSignSystem (η : ℝ) (a b : ℕ → ℝ) : Prop :=
  (∀ t, a (t + 1) = (1 + η * signWeight ((a t - b t) / 4)) ^ 2 * a t) ∧
    ∀ t, b (t + 1) = (1 - η * signWeight ((a t - b t) / 4)) ^ 2 * b t

namespace IsSignSystem

variable {η : ℝ} {a b : ℕ → ℝ}

lemma snd_nonneg (h : IsSignSystem η a b) (hb0 : 0 ≤ b 0) (t : ℕ) : 0 ≤ b t := by
  induction t with
  | zero => exact hb0
  | succ t ih =>
    rw [h.2 t]
    exact mul_nonneg (sq_nonneg _) ih

lemma fst_pos (h : IsSignSystem η a b) (hη : 0 ≤ η) (ha0 : 0 < a 0) (t : ℕ) : 0 < a t := by
  induction t with
  | zero => exact ha0
  | succ t ih =>
    rw [h.1 t]
    have hs := mul_nonneg hη (signWeight_pos ((a t - b t) / 4)).le
    exact mul_pos (pow_pos (by linarith) 2) ih

lemma fst_mono (h : IsSignSystem η a b) (hη : 0 ≤ η) (ha0 : 0 < a 0) : Monotone a :=
  monotone_nat_of_le_succ fun t => by
    rw [h.1 t]
    have hs := mul_nonneg hη (signWeight_pos ((a t - b t) / 4)).le
    exact le_mul_of_one_le_left (h.fst_pos hη ha0 t).le (one_le_pow₀ (by linarith))

/-- **The first mode tends to infinity** (`review/flat_spectrum.md`, item XE19, statement 1),
for every step `η > 0`. -/
theorem tendsto_fst_atTop (h : IsSignSystem η a b) (hη : 0 < η) (ha0 : 0 < a 0)
    (hb0 : 0 ≤ b 0) : Tendsto a atTop atTop := by
  refine tendsto_atTop_atTop_of_monotone' (h.fst_mono hη.le ha0) fun hbdd => ?_
  have hl := tendsto_atTop_ciSup (h.fst_mono hη.le ha0) hbdd
  have hle : ∀ t, a t ≤ ⨆ i, a i := fun t => le_ciSup hbdd t
  obtain ⟨l, hldef⟩ : ∃ l : ℝ, l = ⨆ i, a i := ⟨_, rfl⟩
  rw [← hldef] at hl hle
  have hc0 : 0 < η * signWeight (l / 4) := mul_pos hη (signWeight_pos _)
  have hstep : ∀ t, (1 + η * signWeight (l / 4)) ^ 2 * a t ≤ a (t + 1) := fun t => by
    rw [h.1 t]
    have hw : signWeight (l / 4) ≤ signWeight ((a t - b t) / 4) :=
      signWeight_anti (by linarith [hle t, h.snd_nonneg hb0 t])
    have hs := mul_le_mul_of_nonneg_left hw hη.le
    exact mul_le_mul_of_nonneg_right (pow_le_pow_left₀ (by linarith) (by linarith) 2)
      (h.fst_pos hη.le ha0 t).le
  have h1 : Tendsto (fun t => a (t + 1)) atTop (𝓝 l) := hl.comp (tendsto_add_atTop_nat 1)
  have h2 := le_of_tendsto_of_tendsto' (hl.const_mul ((1 + η * signWeight (l / 4)) ^ 2)) h1
    hstep
  have hlpos : 0 < l := (h.fst_pos hη.le ha0 0).trans_le (hle 0)
  nlinarith [mul_pos hc0 hlpos, mul_pos (mul_pos hc0 hc0) hlpos]

/-- The ratio `a / b` does not decrease: `a t₀ * b t ≤ a t * b t₀` for `t₀ ≤ t`. -/
lemma mul_le_mul_of_le (h : IsSignSystem η a b) (hη : 0 ≤ η) (ha0 : 0 < a 0) (hb0 : 0 ≤ b 0)
    {t₀ t : ℕ} (ht : t₀ ≤ t) : a t₀ * b t ≤ a t * b t₀ := by
  induction t, ht using Nat.le_induction with
  | base => exact le_rfl
  | succ t ht ih =>
    rw [h.1 t, h.2 t]
    have hs := mul_nonneg hη (signWeight_pos ((a t - b t) / 4)).le
    obtain ⟨s, hsdef⟩ : ∃ s : ℝ, s = η * signWeight ((a t - b t) / 4) := ⟨_, rfl⟩
    rw [← hsdef] at hs ⊢
    have h1 : (1 - s) ^ 2 ≤ (1 + s) ^ 2 := by nlinarith
    have h2 := mul_nonneg (h.fst_pos hη ha0 t).le (h.snd_nonneg hb0 t₀)
    calc a t₀ * ((1 - s) ^ 2 * b t) = (1 - s) ^ 2 * (a t₀ * b t) := by ring
      _ ≤ (1 - s) ^ 2 * (a t * b t₀) := mul_le_mul_of_nonneg_left ih (sq_nonneg _)
      _ ≤ (1 + s) ^ 2 * (a t * b t₀) := mul_le_mul_of_nonneg_right h1 h2
      _ = (1 + s) ^ 2 * a t * b t₀ := by ring

end IsSignSystem

/-- The witness of the examples: the first mode of the system with the step `1` from
`a_0 = 1` and `b = 0`. -/
noncomputable def signWitness : ℕ → ℝ
  | 0 => 1
  | t + 1 => (1 + 1 * signWeight ((signWitness t - 0) / 4)) ^ 2 * signWitness t

lemma isSignSystem_signWitness : IsSignSystem 1 signWitness fun _ => 0 :=
  ⟨fun _ => rfl, fun _ => (mul_zero _).symm⟩

example := signWeight_anti (zero_le_one' ℝ)

example := mul_signWeight_le_two (η := 1) (r := 1) (by norm_num)

example := isSignSystem_signWitness.snd_nonneg le_rfl 3

example := isSignSystem_signWitness.fst_pos zero_le_one one_pos 3

example := isSignSystem_signWitness.fst_mono zero_le_one one_pos

example := isSignSystem_signWitness.tendsto_fst_atTop one_pos one_pos le_rfl

example := isSignSystem_signWitness.mul_le_mul_of_le zero_le_one one_pos le_rfl (Nat.le_succ 2)

end Transformer.Grokking.FlatSpectrum.Training.Scope
