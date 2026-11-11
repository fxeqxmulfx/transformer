module

public import Transformer.Grokking.FlatSpectrum.Training.CrossEntropyRun
public import Transformer.Grokking.FlatSpectrum.Training.NoDecay

/-!
# The run without weight decay on separable samples

Source: `review/flat_spectrum.md`, *Training with cross-entropy* (added), item XE13, the
bullet "No weight decay", and item XE17. Let some product have a positive margin, let the
probes have norm at most `a`, and let the run of gradient descent on the factors have no
weight decay, `λ = 0`, and start at stacked factors of full row rank.

* The run has no limit when `2 a |S| η < 1` (`not_tendsto_decayRun_ceLoss_zero`). The step
  matrices `1 − η M_t` are then positive definite (WD6), so a limit would be a trained fit
  (WD5), and without weight decay there is none (`not_isDecayedMin_ceLoss_zero`).
* The run leaves every ball `‖U‖² + ‖V‖² ≤ ρ` for which its step is admissible,
  `η (2 a |S| + 2 |S| a² ρ) < 1` (`exists_lt_frobeniusSq_decayRun_ceLoss_zero`): a run that
  stays in such a ball ends at a trained fit
  (`exists_tendsto_decayRun_isDecayedMin_of_forall_le`), and there is none. So the squared
  norm of the factors exceeds every `ρ < (1/η − 2 a |S|) / (2 |S| a²)`.
* It does so at arbitrarily late steps (`frequently_lt_frobeniusSq_decayRun_ceLoss_zero`):
  the run keeps full row rank (`surjective_stack_decayRun`), and its tail is the run from
  its state (`decayRun_add`), so the argument restarts at every step. The run does not
  settle in a ball where its step is admissible.

Not proved: that the norm tends to infinity for one fixed step, that the loss tends to `0`,
and the direction of the factors. The witness is `signProbe`.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Training.Scope

open Matrix Filter Topology
open scoped ComplexOrder

variable {m n k S C : Type*} [Fintype m] [Fintype n] [Fintype k] [DecidableEq m]
  [DecidableEq n] [Fintype S] [Fintype C] {A : S → C → Matrix m n ℂ} {y : S → C}
  {X₀ : Matrix m n ℂ} {U₀ : Matrix m k ℂ} {V₀ : Matrix n k ℂ} {a η ρ : ℝ}

/-- **The run without weight decay has no limit** (`review/flat_spectrum.md`, item XE17,
statement 1): on samples that some product classifies with a positive margin, with probes of
norm at most `a` and a step `0 < η` with `2 a |S| η < 1`, the run of gradient descent on the
factors with `λ = 0` from stacked factors of full row rank converges to no pair of factors. -/
theorem not_tendsto_decayRun_ceLoss_zero [Nonempty S] [Nontrivial C] (ha0 : 0 ≤ a)
    (ha : ∀ s c, frobeniusSq (A s c) ≤ a ^ 2) (hsep : 0 < margin A y X₀) (hη : 0 < η)
    (hstep : η * (2 * a * Fintype.card S) < 1)
    (hfull : Function.Surjective (stack U₀ V₀).mulVec) (U : Matrix m k ℂ) (V : Matrix n k ℂ) :
    ¬ Tendsto (decayRun (ceGrad A y) η 0 U₀ V₀) atTop (𝓝 (U, V)) := fun hlim =>
  not_isDecayedMin_ceLoss_zero hsep U V
    (isDecayedMin_of_tendsto_decayRun (ceLoss_first_order A y) (continuous_ceGrad A y) hη hfull
      (fun _ => one_sub_smul_lift_posDef hη.le (by positivity) (frobeniusSq_ceGrad_le ha0 ha _)
        (by rwa [abs_zero, zero_add])) hlim)

/-- **The run without weight decay leaves every ball where its step is admissible**
(`review/flat_spectrum.md`, item XE17, statement 3): on samples that some product classifies
with a positive margin, with probes of norm at most `a`, if `η (2 a |S| + 2 |S| a² ρ) < 1`
then the run with `λ = 0` from stacked factors of full row rank has
`‖U_t‖² + ‖V_t‖² > ρ` at some step. -/
theorem exists_lt_frobeniusSq_decayRun_ceLoss_zero [Nonempty S] [Nontrivial C] (ha0 : 0 ≤ a)
    (ha : ∀ s c, frobeniusSq (A s c) ≤ a ^ 2) (hsep : 0 < margin A y X₀) (hη : 0 < η)
    (hdesc : η * (2 * a * Fintype.card S + 2 * (Fintype.card S * a ^ 2) * ρ) < 1)
    (hfull : Function.Surjective (stack U₀ V₀).mulVec) :
    ∃ t, ρ < frobeniusSq (decayRun (ceGrad A y) η 0 U₀ V₀ t).1 +
      frobeniusSq (decayRun (ceGrad A y) η 0 U₀ V₀ t).2 := by
  by_contra hno
  push Not at hno
  have hρ : 0 ≤ ρ :=
    (add_nonneg (frobeniusSq_nonneg _) (frobeniusSq_nonneg _)).trans (hno 0)
  have hle : η * (2 * a * Fintype.card S) ≤
      η * (2 * a * Fintype.card S + 2 * (Fintype.card S * a ^ 2) * ρ) :=
    mul_le_mul_of_nonneg_left (le_add_of_nonneg_right (by positivity)) hη.le
  obtain ⟨U, V, -, hmin⟩ := exists_tendsto_decayRun_isDecayedMin_of_forall_le
    (ceLoss_first_order A y) (continuous_ceGrad A y) (analyticAt_ceLoss A y) hη
    (γ := 2 * a * Fintype.card S) (s := Fintype.card S * a ^ 2) (by positivity) (by positivity)
    (fun X _ => frobeniusSq_ceGrad_le ha0 ha X) (fun X Y _ _ => ceLoss_le_add ha0 ha X Y)
    (by rw [zero_add]; exact hdesc.le) (by rw [abs_zero, zero_add]; exact hle.trans_lt hdesc)
    hno hfull
  exact not_isDecayedMin_ceLoss_zero hsep U V hmin

omit [Fintype S] [Fintype C] [DecidableEq m] [DecidableEq n] in
/-- The tail of a run is the run from its state. -/
lemma decayRun_add (G : Matrix m n ℂ → Matrix m n ℂ) (η lam : ℝ) (U₀ : Matrix m k ℂ)
    (V₀ : Matrix n k ℂ) (T t : ℕ) :
    decayRun G η lam (decayRun G η lam U₀ V₀ T).1 (decayRun G η lam U₀ V₀ T).2 t =
      decayRun G η lam U₀ V₀ (t + T) :=
  (Function.iterate_add_apply (decayStep G η lam) t T (U₀, V₀)).symm

omit [Fintype S] [Fintype C] in
/-- A run from stacked factors of full row rank keeps full row rank while its step matrices
are positive definite: each step multiplies the stack by an invertible matrix (WD1). -/
lemma surjective_stack_decayRun {G : Matrix m n ℂ → Matrix m n ℂ} {lam : ℝ}
    (hfull : Function.Surjective (stack U₀ V₀).mulVec)
    (hstep : ∀ t, (1 - η • lift lam (G ((decayRun G η lam U₀ V₀ t).1 *
      (decayRun G η lam U₀ V₀ t).2ᴴ))).PosDef) (t : ℕ) :
    Function.Surjective
      (stack (decayRun G η lam U₀ V₀ t).1 (decayRun G η lam U₀ V₀ t).2).mulVec := by
  induction t with
  | zero => exact hfull
  | succ t ih =>
    rw [stack_decayRun_succ]
    intro w
    obtain ⟨u, hu⟩ := (Matrix.mulVec_surjective_iff_isUnit.2 (hstep t).isUnit) w
    obtain ⟨v, hv⟩ := ih u
    exact ⟨v, by rw [← Matrix.mulVec_mulVec, hv, hu]⟩

/-- **The run without weight decay is outside the ball at arbitrarily late steps**
(`review/flat_spectrum.md`, item XE17, statement 3): under the hypotheses of
`exists_lt_frobeniusSq_decayRun_ceLoss_zero`, `‖U_t‖² + ‖V_t‖² > ρ` for infinitely many `t`.
So the run does not settle in a ball where its step is admissible. -/
theorem frequently_lt_frobeniusSq_decayRun_ceLoss_zero [Nonempty S] [Nontrivial C]
    (ha0 : 0 ≤ a) (ha : ∀ s c, frobeniusSq (A s c) ≤ a ^ 2) (hsep : 0 < margin A y X₀)
    (hη : 0 < η)
    (hdesc : η * (2 * a * Fintype.card S + 2 * (Fintype.card S * a ^ 2) * ρ) < 1)
    (hfull : Function.Surjective (stack U₀ V₀).mulVec) :
    ∃ᶠ t in atTop, ρ < frobeniusSq (decayRun (ceGrad A y) η 0 U₀ V₀ t).1 +
      frobeniusSq (decayRun (ceGrad A y) η 0 U₀ V₀ t).2 := by
  refine Filter.frequently_atTop.2 fun T => ?_
  rcases lt_or_ge ρ 0 with hρ | hρ
  · exact ⟨T, le_rfl, hρ.trans_le (add_nonneg (frobeniusSq_nonneg _) (frobeniusSq_nonneg _))⟩
  have hle : η * (2 * a * Fintype.card S) ≤
      η * (2 * a * Fintype.card S + 2 * (Fintype.card S * a ^ 2) * ρ) :=
    mul_le_mul_of_nonneg_left (le_add_of_nonneg_right (by positivity)) hη.le
  obtain ⟨t, ht⟩ := exists_lt_frobeniusSq_decayRun_ceLoss_zero ha0 ha hsep hη hdesc
    (surjective_stack_decayRun hfull (fun _ => one_sub_smul_lift_posDef hη.le (by positivity)
      (frobeniusSq_ceGrad_le ha0 ha _) (by rw [abs_zero, zero_add]; exact hle.trans_lt hdesc)) T)
  rw [decayRun_add] at ht
  exact ⟨t + T, Nat.le_add_left T t, ht⟩

/-! The witness: `signProbe`, one sample of two classes with probes of norm `1`, where the
product `1` has the margin `2`; the start `U₀ = [1 0]`, `V₀ = [0 1]` of full row rank and the
step `1/10`. The run has no limit, and its squared norm exceeds `2`, as
`(1/10) (2 + 2 · 2) < 1`. -/

example := decayRun_add (fun X : Matrix (Fin 1) (Fin 1) ℂ => X) 1 1
  (1 : Matrix (Fin 1) (Fin 1) ℂ) 1 2 3

example := surjective_stack_decayRun (G := ceGrad signProbe fun _ => true) (η := 1 / 10)
  (lam := 0) (U₀ := fromCols (1 : Matrix (Fin 1) (Fin 1) ℂ) (0 : Matrix (Fin 1) (Fin 1) ℂ))
  (V₀ := fromCols (0 : Matrix (Fin 1) (Fin 1) ℂ) (1 : Matrix (Fin 1) (Fin 1) ℂ))
  (fun v => ⟨v, by rw [stack_fromCols_one, one_mulVec]⟩)
  (fun _ => one_sub_smul_lift_posDef (η := 1 / 10) (lam := 0) (by norm_num) zero_le_two
    ((frobeniusSq_ceGrad_le zero_le_one frobeniusSq_signProbe_le _).trans_eq (by simp))
    (by norm_num)) 3

example := frequently_lt_frobeniusSq_decayRun_ceLoss_zero (A := signProbe)
  (y := fun _ => true) (X₀ := 1) (η := 1 / 10) (ρ := 2) zero_le_one frobeniusSq_signProbe_le
  (two_pos.trans_le (le_margin le_gap_signProbe_one)) (by norm_num) (by norm_num)
  (U₀ := fromCols (1 : Matrix (Fin 1) (Fin 1) ℂ) (0 : Matrix (Fin 1) (Fin 1) ℂ))
  (V₀ := fromCols (0 : Matrix (Fin 1) (Fin 1) ℂ) (1 : Matrix (Fin 1) (Fin 1) ℂ))
  fun v => ⟨v, by rw [stack_fromCols_one, one_mulVec]⟩

example (U V : Matrix (Fin 1) (Fin 1 ⊕ Fin 1) ℂ) :
    ¬ Tendsto (decayRun (ceGrad signProbe fun _ => true) (1 / 10) 0 (fromCols 1 0)
      (fromCols 0 1)) atTop (𝓝 (U, V)) :=
  not_tendsto_decayRun_ceLoss_zero (X₀ := 1) zero_le_one frobeniusSq_signProbe_le
    (two_pos.trans_le (le_margin le_gap_signProbe_one)) (by norm_num) (by norm_num)
    (fun v => ⟨v, by rw [stack_fromCols_one, one_mulVec]⟩) U V

example : ∃ t, 2 < frobeniusSq (decayRun (ceGrad signProbe fun _ => true) (1 / 10) 0
      (fromCols (1 : Matrix (Fin 1) (Fin 1) ℂ) (0 : Matrix (Fin 1) (Fin 1) ℂ))
      (fromCols (0 : Matrix (Fin 1) (Fin 1) ℂ) (1 : Matrix (Fin 1) (Fin 1) ℂ)) t).1 +
    frobeniusSq (decayRun (ceGrad signProbe fun _ => true) (1 / 10) 0
      (fromCols (1 : Matrix (Fin 1) (Fin 1) ℂ) (0 : Matrix (Fin 1) (Fin 1) ℂ))
      (fromCols (0 : Matrix (Fin 1) (Fin 1) ℂ) (1 : Matrix (Fin 1) (Fin 1) ℂ)) t).2 :=
  exists_lt_frobeniusSq_decayRun_ceLoss_zero (X₀ := 1) zero_le_one frobeniusSq_signProbe_le
    (two_pos.trans_le (le_margin le_gap_signProbe_one)) (by norm_num) (by norm_num)
    fun v => ⟨v, by rw [stack_fromCols_one, one_mulVec]⟩

end Transformer.Grokking.FlatSpectrum.Training.Scope
