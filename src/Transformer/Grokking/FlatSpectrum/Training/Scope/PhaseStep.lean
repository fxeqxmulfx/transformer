module

public import Transformer.Grokking.FlatSpectrum.Training.Scope.NoDecayRun

/-!
# The step of one phase

Source: `review/flat_spectrum.md`, *Training with cross-entropy* (added), item XE18, with
item XE3.

XE3 has two bounds on the step, for a weight decay `λ > 0` and a ball `‖U‖² + ‖V‖² ≤ ρ` with
`4 |S| log |C| ≤ λ ρ`. The ball of squared radius `ρ = ‖U‖² + ‖V‖² + 4 |S| log |C| / λ`
contains the factors `(U, V)` and satisfies that condition, and `phaseStep` is the largest
step that the two bounds admit in it. It is positive (`phaseStep_pos`). With it the run from
factors of full row rank converges to a trained fit for `λ` and has full row rank at every
step (`exists_tendsto_decayRun_phaseStep`): XE3 for a start that is given, not a ball that is
given.

`exists_seq_of_forall_exists` is the dependent choice that puts such phases one after another.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Training.Scope

open Matrix Filter Topology

variable {m n k S C : Type*} [Fintype m] [Fintype n] [Fintype k] [DecidableEq m]
  [DecidableEq n] [Fintype S] [Fintype C] {A : S → C → Matrix m n ℂ} {y : S → C} {a lam : ℝ}

/-- **The step of a phase** (`review/flat_spectrum.md`, item XE18): the largest step that the
two bounds of XE3 admit for probes of norm at most `a`, the weight decay `lam` and the ball of
squared radius `‖U‖² + ‖V‖² + 4 |S| log |C| / lam` around the factors `W = (U, V)` at which
the phase starts. -/
noncomputable def phaseStep (S C : Type*) [Fintype S] [Fintype C] (a lam : ℝ)
    (W : Matrix m k ℂ × Matrix n k ℂ) : ℝ :=
  min (lam / (2 * (lam ^ 2 + (2 * a * Fintype.card S) ^ 2)))
    (1 / (lam + 2 * a * Fintype.card S + 2 * (Fintype.card S * a ^ 2) *
      (frobeniusSq W.1 + frobeniusSq W.2 +
        4 * (Fintype.card S * Real.log (Fintype.card C)) / lam)))

omit [DecidableEq m] [DecidableEq n] in
/-- The step of a phase is positive. -/
lemma phaseStep_pos (ha0 : 0 ≤ a) (hlam : 0 < lam) (W : Matrix m k ℂ × Matrix n k ℂ) :
    0 < phaseStep S C a lam W := by
  have hb0 : (0 : ℝ) ≤ Fintype.card S * Real.log (Fintype.card C) :=
    mul_nonneg (Nat.cast_nonneg _) (Real.log_natCast_nonneg _)
  have h1 := frobeniusSq_nonneg W.1
  have h2 := frobeniusSq_nonneg W.2
  exact lt_min (by positivity) (by positivity)

/-- **The run with the step of a phase** (`review/flat_spectrum.md`, item XE18; XE3 in the
ball of the start): for probes of norm at most `a` and `λ > 0`, from stacked factors of full
row rank the run with the step `phaseStep` converges to a trained fit and has full row rank at
every step. -/
theorem exists_tendsto_decayRun_phaseStep (ha0 : 0 ≤ a)
    (ha : ∀ s c, frobeniusSq (A s c) ≤ a ^ 2) (hlam : 0 < lam) {U : Matrix m k ℂ}
    {V : Matrix n k ℂ} (hfull : Function.Surjective (stack U V).mulVec) :
    ∃ (Us : Matrix m k ℂ) (Vs : Matrix n k ℂ),
      Tendsto (decayRun (ceGrad A y) (phaseStep S C a lam (U, V)) lam U V) atTop
        (𝓝 (Us, Vs)) ∧
      IsDecayedMin (ceLoss A y) lam Us Vs ∧
      ∀ t, Function.Surjective
        (stack (decayRun (ceGrad A y) (phaseStep S C a lam (U, V)) lam U V t).1
          (decayRun (ceGrad A y) (phaseStep S C a lam (U, V)) lam U V t).2).mulVec := by
  have hb0 : (0 : ℝ) ≤ Fintype.card S * Real.log (Fintype.card C) :=
    mul_nonneg (Nat.cast_nonneg _) (Real.log_natCast_nonneg _)
  have hη := phaseStep_pos (S := S) (C := C) ha0 hlam (U, V)
  obtain ⟨η, hηdef⟩ : ∃ η : ℝ, η = phaseStep S C a lam (U, V) := ⟨_, rfl⟩
  obtain ⟨ρ, hρdef⟩ : ∃ ρ : ℝ, ρ = frobeniusSq U + frobeniusSq V +
      4 * (Fintype.card S * Real.log (Fintype.card C)) / lam := ⟨_, rfl⟩
  have hw0 : 0 ≤ frobeniusSq U + frobeniusSq V :=
    add_nonneg (frobeniusSq_nonneg _) (frobeniusSq_nonneg _)
  have hw : frobeniusSq U + frobeniusSq V ≤ ρ := by
    rw [hρdef]
    exact le_add_of_nonneg_right (by positivity)
  have hρ0 : 0 ≤ ρ := hw0.trans hw
  have hρ : 4 * (Fintype.card S * Real.log (Fintype.card C)) ≤ lam * ρ := by
    rw [hρdef, mul_add, mul_div_cancel₀ _ hlam.ne']
    exact le_add_of_nonneg_left (mul_nonneg hlam.le hw0)
  have hd1 : 0 < 2 * (lam ^ 2 + (2 * a * Fintype.card S) ^ 2) := by positivity
  have hd2 : 0 < lam + 2 * a * Fintype.card S + 2 * (Fintype.card S * a ^ 2) * ρ := by
    positivity
  have h1 : η ≤ lam / (2 * (lam ^ 2 + (2 * a * Fintype.card S) ^ 2)) :=
    hηdef.trans_le (min_le_left _ _)
  have h2 : η ≤ 1 / (lam + 2 * a * Fintype.card S + 2 * (Fintype.card S * a ^ 2) * ρ) := by
    rw [hηdef, hρdef]
    exact min_le_right _ _
  rw [← hηdef] at hη ⊢
  rw [le_div_iff₀ hd1] at h1
  rw [le_div_iff₀ hd2] at h2
  have hstep : 2 * η * (lam ^ 2 + (2 * a * Fintype.card S) ^ 2) ≤ lam := by linarith
  have hγ : ∀ X : Matrix m n ℂ, frobeniusSq X ≤ (ρ / 2) ^ 2 →
      frobeniusSq (ceGrad A y X) ≤ (2 * a * Fintype.card S) ^ 2 :=
    fun X _ => frobeniusSq_ceGrad_le ha0 ha X
  obtain ⟨Us, Vs, hlim, hmin⟩ := exists_tendsto_decayRun_ceLoss_isDecayedMin (A := A) (y := y)
    hη hlam ha0 ha hρ hstep h2 hw hfull
  exact ⟨Us, Vs, hlim, hmin, surjective_stack_decayRun hfull fun t =>
    one_sub_smul_lift_posDef_of_ball hη.le hlam (by positivity) hγ hstep
      (frobeniusSq_decayRun_le hη.le hlam hγ (fun X _ => neg_le_pairing_ceGrad A y X) hρ hstep
        hw t)⟩

omit [Fintype m] [Fintype n] [Fintype k] [DecidableEq m] [DecidableEq n] [Fintype S]
  [Fintype C] in
/-- Dependent choice along the naturals: from a start with the invariant `P`, steps that keep
`P` give a sequence whose consecutive terms are related by `R`. -/
lemma exists_seq_of_forall_exists {α : Type*} {P : α → Prop} {R : ℕ → α → α → Prop} {x₀ : α}
    (h0 : P x₀) (hstep : ∀ j x, P x → ∃ x', R j x x' ∧ P x') :
    ∃ x : ℕ → α, x 0 = x₀ ∧ ∀ j, R j (x j) (x (j + 1)) := by
  have : Nonempty α := ⟨x₀⟩
  choose! f hf using hstep
  have hP : ∀ j, P (Nat.rec x₀ (fun j x => f j x) j : α) := fun j => by
    induction j with
    | zero => exact h0
    | succ j ih => exact (hf j _ ih).2
  exact ⟨fun j => Nat.rec x₀ (fun j x => f j x) j, rfl, fun j => (hf j _ (hP j)).1⟩

/-! The witness: one sample of a `1 × 1` product read with the code `±1`, the weight decay `1`
and the start `U₀ = [1 0]`, `V₀ = [0 1]` of full row rank. -/

example := phaseStep_pos (S := Fin 1) (C := Bool) zero_le_one one_pos
  ((1, 1) : Matrix (Fin 1) (Fin 1) ℂ × Matrix (Fin 1) (Fin 1) ℂ)

example := exists_tendsto_decayRun_phaseStep (A := signProbe) (y := fun _ => true)
  zero_le_one frobeniusSq_signProbe_le one_pos
  (U := fromCols (1 : Matrix (Fin 1) (Fin 1) ℂ) (0 : Matrix (Fin 1) (Fin 1) ℂ))
  (V := fromCols (0 : Matrix (Fin 1) (Fin 1) ℂ) (1 : Matrix (Fin 1) (Fin 1) ℂ))
  fun v => ⟨v, by rw [stack_fromCols_one, one_mulVec]⟩

example := exists_seq_of_forall_exists (P := fun _ : ℕ => True)
  (R := fun _ x x' => x' = x + 1) (x₀ := 0) trivial fun _ x _ => ⟨x + 1, rfl, trivial⟩

end Transformer.Grokking.FlatSpectrum.Training.Scope
