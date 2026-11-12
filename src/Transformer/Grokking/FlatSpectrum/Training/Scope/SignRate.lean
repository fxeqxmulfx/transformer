module

public import Transformer.Grokking.FlatSpectrum.Training.Scope.SignRunLimit

/-!
# A rate for the run without weight decay on the sign sample

Source: `review/flat_spectrum.md`, *Training with cross-entropy* (added), item XE19,
statement 5; the item "Not covered" of XE17.

For the system of `SignSystem`, write `E = exp ((a − b) / 2)`, the exponential of the margin,
so that the weight is `2 / (1 + E)`.

* At a step with `a − b ≥ 2 η` the second mode does not grow and `E ≥ 1`, and
  `E' ≥ E + η a` (`IsSignSystem.exp_add_le_exp_succ`), by `exp x ≥ 1 + x`.
* So from some step `T` on, `E_t ≥ E_T + η a_T (t − T)` (`IsSignSystem.exists_exp_ge`).

For the run on the sign sample, with any step `η > 0` and a start with `U₀ + V₀ ≠ 0`, from
some step `T` on the cross-entropy is at most `1 / (1 + η ‖U_T + V_T‖² (t − T))`
(`exists_ceLoss_signProduct_le`), and `‖U_t‖² + ‖V_t‖² ≥ log (1 + η ‖U_T + V_T‖² (t − T))`
(`exists_log_le_frobeniusSq_decayRun_signProbe`): the loss falls at least like `1 / t`, and
the norm grows at least like `log t`.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Training.Scope

open Matrix Filter Topology

namespace IsSignSystem

variable {η : ℝ} {a b : ℕ → ℝ}

/-- **One step of the exponential of the gap** (`review/flat_spectrum.md`, item XE19, statement
5): once `a − b ≥ 2 η`, `exp ((a' − b') / 2) ≥ exp ((a − b) / 2) + η a`. -/
lemma exp_add_le_exp_succ (h : IsSignSystem η a b) (hη : 0 < η) {t : ℕ} (ha : 0 ≤ a t)
    (hb : 0 ≤ b t) (hgap : 2 * η ≤ a t - b t) :
    Real.exp ((a t - b t) / 2) + η * a t ≤ Real.exp ((a (t + 1) - b (t + 1)) / 2) := by
  have ha' := h.1 t
  have hb' := h.2 t
  have hw : signWeight ((a t - b t) / 4) = 2 / (1 + Real.exp ((a t - b t) / 2)) := by
    rw [signWeight]
    congr 3
    ring
  have hs0 : 0 < η * signWeight ((a t - b t) / 4) := mul_pos hη (signWeight_pos _)
  have hs2 : η * signWeight ((a t - b t) / 4) ≤ 2 := mul_signWeight_le_two (by linarith)
  have hE1 : 1 ≤ Real.exp ((a t - b t) / 2) := Real.one_le_exp (by linarith)
  have hEs : η ≤ Real.exp ((a t - b t) / 2) * (η * signWeight ((a t - b t) / 4)) := by
    rw [hw]
    have h1 : 1 ≤ Real.exp ((a t - b t) / 2) * (2 / (1 + Real.exp ((a t - b t) / 2))) := by
      rw [mul_div_assoc', le_div_iff₀ (by linarith)]
      linarith
    nlinarith
  generalize η * signWeight ((a t - b t) / 4) = s at ha' hb' hs0 hs2 hEs
  have hb1 : b (t + 1) ≤ b t := by
    rw [hb']
    nlinarith [mul_nonneg (mul_nonneg hs0.le (by linarith : (0 : ℝ) ≤ 2 - s)) hb]
  have hstep : Real.exp ((a t - b t) / 2) * (1 + (a (t + 1) - a t) / 2) ≤
      Real.exp ((a (t + 1) - b t) / 2) := by
    have h1 := Real.add_one_le_exp ((a (t + 1) - a t) / 2)
    have h2 : Real.exp ((a (t + 1) - b t) / 2) =
        Real.exp ((a t - b t) / 2) * Real.exp ((a (t + 1) - a t) / 2) := by
      rw [← Real.exp_add]
      congr 1
      ring
    rw [h2]
    exact mul_le_mul_of_nonneg_left (by linarith) (Real.exp_pos _).le
  have hmono : Real.exp ((a (t + 1) - b t) / 2) ≤ Real.exp ((a (t + 1) - b (t + 1)) / 2) :=
    Real.exp_le_exp.2 (by linarith)
  generalize Real.exp ((a t - b t) / 2) = E at hE1 hEs hstep ⊢
  have h3 : E + η * a t ≤ E * (1 + (a (t + 1) - a t) / 2) := by
    rw [ha']
    nlinarith [mul_nonneg (sub_nonneg.2 hEs) ha,
      mul_nonneg (mul_nonneg (by linarith : (0 : ℝ) ≤ E) (sq_nonneg s)) ha]
  exact h3.trans (hstep.trans hmono)

/-- **The exponential of the gap grows at least linearly** (`review/flat_spectrum.md`, item
XE19, statement 5): from some step `T` on,
`exp ((a_t − b_t) / 2) ≥ exp ((a_T − b_T) / 2) + η a_T (t − T)`, and `a_T ≥ b_T`. -/
theorem exists_exp_ge (h : IsSignSystem η a b) (hη : 0 < η) (ha0 : 0 < a 0) (hb0 : 0 ≤ b 0) :
    ∃ T, b T ≤ a T ∧ ∀ t, T ≤ t →
      Real.exp ((a T - b T) / 2) + η * a T * (t - T) ≤ Real.exp ((a t - b t) / 2) := by
  obtain ⟨T, hT⟩ := eventually_atTop.1
    (tendsto_atTop.1 (h.tendsto_sub_atTop hη ha0 hb0) (2 * η))
  refine ⟨T, by linarith [hT T le_rfl], fun t ht => ?_⟩
  induction t, ht using Nat.le_induction with
  | base => simp
  | succ t ht ih =>
    have hstep := h.exp_add_le_exp_succ hη (h.fst_pos hη.le ha0 t).le (h.snd_nonneg hb0 t)
      (hT t ht)
    have hmono : a T ≤ a t := h.fst_mono hη.le ha0 ht
    push_cast
    nlinarith

end IsSignSystem

variable {k : Type*} [Fintype k] {η : ℝ} {U₀ V₀ : Matrix (Fin 1) k ℂ}

local notation "R" => decayRun (ceGrad signProbe fun _ => true)

/-- **Without weight decay the loss falls at least like `1 / t`** (`review/flat_spectrum.md`,
item XE19, statement 5): on the sign sample, for every step `η > 0` and every start with
`U₀ + V₀ ≠ 0`, from some step `T` on the cross-entropy is at most
`1 / (1 + η ‖U_T + V_T‖² (t − T))`. -/
theorem exists_ceLoss_signProduct_le (hη : 0 < η) (h0 : U₀ + V₀ ≠ 0) :
    ∃ T, 0 < frobeniusSq ((R η 0 U₀ V₀ T).1 + (R η 0 U₀ V₀ T).2) ∧ ∀ t, T ≤ t →
      ceLoss signProbe (fun _ => true) (signProduct η U₀ V₀ t) ≤
        1 / (1 + η * frobeniusSq ((R η 0 U₀ V₀ T).1 + (R η 0 U₀ V₀ T).2) * (t - T)) := by
  have sys := isSignSystem_decayRun_signProbe η U₀ V₀
  have ha0 : 0 < frobeniusSq (U₀ + V₀) :=
    lt_of_le_of_ne (frobeniusSq_nonneg _) fun h => h0 (eq_zero_of_frobeniusSq_eq_zero h.symm)
  have hb0 := frobeniusSq_nonneg (U₀ - V₀)
  obtain ⟨T, hTab, hT⟩ := sys.exists_exp_ge hη ha0 hb0
  have haT := sys.fst_pos hη.le ha0 T
  refine ⟨T, haT, fun t ht => ?_⟩
  have h := ceLoss_le_mul_exp (A := signProbe) (y := fun _ => true)
    (X := signProduct η U₀ V₀ t) fun s c hc => margin_le_gap _ hc
  norm_num at h
  rw [margin_signProduct] at h
  have hE := hT t ht
  have hE1 : 1 ≤ Real.exp ((frobeniusSq ((R η 0 U₀ V₀ T).1 + (R η 0 U₀ V₀ T).2) -
      frobeniusSq ((R η 0 U₀ V₀ T).1 - (R η 0 U₀ V₀ T).2)) / 2) :=
    Real.one_le_exp (by linarith)
  have hpos : 0 < 1 + η * frobeniusSq ((R η 0 U₀ V₀ T).1 + (R η 0 U₀ V₀ T).2) * (t - T) := by
    have : (0 : ℝ) ≤ (t : ℝ) - T := sub_nonneg.2 (by exact_mod_cast ht)
    positivity
  refine h.trans ?_
  rw [Real.exp_neg, ← one_div]
  exact one_div_le_one_div_of_le hpos (by linarith)

/-- **Without weight decay the norm of the factors grows at least logarithmically**
(`review/flat_spectrum.md`, item XE19, statement 5): on the sign sample, for every step
`η > 0` and every start with `U₀ + V₀ ≠ 0`, from some step `T` on
`‖U_t‖² + ‖V_t‖² ≥ log (1 + η ‖U_T + V_T‖² (t − T))`. -/
theorem exists_log_le_frobeniusSq_decayRun_signProbe (hη : 0 < η) (h0 : U₀ + V₀ ≠ 0) :
    ∃ T, 0 < frobeniusSq ((R η 0 U₀ V₀ T).1 + (R η 0 U₀ V₀ T).2) ∧ ∀ t, T ≤ t →
      Real.log (1 + η * frobeniusSq ((R η 0 U₀ V₀ T).1 + (R η 0 U₀ V₀ T).2) * (t - T)) ≤
        frobeniusSq (R η 0 U₀ V₀ t).1 + frobeniusSq (R η 0 U₀ V₀ t).2 := by
  have sys := isSignSystem_decayRun_signProbe η U₀ V₀
  have ha0 : 0 < frobeniusSq (U₀ + V₀) :=
    lt_of_le_of_ne (frobeniusSq_nonneg _) fun h => h0 (eq_zero_of_frobeniusSq_eq_zero h.symm)
  have hb0 := frobeniusSq_nonneg (U₀ - V₀)
  obtain ⟨T, hTab, hT⟩ := sys.exists_exp_ge hη ha0 hb0
  have haT := sys.fst_pos hη.le ha0 T
  refine ⟨T, haT, fun t ht => ?_⟩
  have hE := hT t ht
  have hE1 : 1 ≤ Real.exp ((frobeniusSq ((R η 0 U₀ V₀ T).1 + (R η 0 U₀ V₀ T).2) -
      frobeniusSq ((R η 0 U₀ V₀ T).1 - (R η 0 U₀ V₀ T).2)) / 2) :=
    Real.one_le_exp (by linarith)
  have hpos : 0 < 1 + η * frobeniusSq ((R η 0 U₀ V₀ T).1 + (R η 0 U₀ V₀ T).2) * (t - T) := by
    have : (0 : ℝ) ≤ (t : ℝ) - T := sub_nonneg.2 (by exact_mod_cast ht)
    positivity
  have hlog := Real.log_le_log hpos (by linarith : 1 + η * frobeniusSq ((R η 0 U₀ V₀ T).1 +
    (R η 0 U₀ V₀ T).2) * (t - T) ≤ Real.exp ((frobeniusSq ((R η 0 U₀ V₀ t).1 +
      (R η 0 U₀ V₀ t).2) - frobeniusSq ((R η 0 U₀ V₀ t).1 - (R η 0 U₀ V₀ t).2)) / 2))
  rw [Real.log_exp] at hlog
  have h1 := frobeniusSq_add_frobeniusSq (R η 0 U₀ V₀ t).1 (R η 0 U₀ V₀ t).2
  have h2 := frobeniusSq_nonneg ((R η 0 U₀ V₀ t).1 - (R η 0 U₀ V₀ t).2)
  have h3 := frobeniusSq_nonneg ((R η 0 U₀ V₀ t).1 + (R η 0 U₀ V₀ t).2)
  linarith

example := exists_ceLoss_signProduct_le (η := 1 / 10) (by norm_num)
  fromCols_one_zero_add_ne_zero

example := exists_log_le_frobeniusSq_decayRun_signProbe (η := 1 / 10) (by norm_num)
  fromCols_one_zero_add_ne_zero

end Transformer.Grokking.FlatSpectrum.Training.Scope
