module

public import Transformer.Grokking.FlatSpectrum.CoupledMoments.Schatten.DecodedPhase

/-!
# Reflection and equal probabilities on paired latent states

Source: `review/flat_spectrum.md`, CM16. Reflect the latent vocabulary by
negation, then average the two probability tables. Constant Fourier operator
and nuclear budgets survive this operation. An even decoder gives precisely
the same original-vocabulary predictions before and after averaging.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.CoupledMoments.Schatten

variable {q : ℕ} [NeZero q] {m n C : Type*}

/-- Exchange the latent coordinates indexed by opposite phases (CM16). -/
def reflectLatent : Table m n (ZMod q) →ₗ[ℝ] Table m n (ZMod q) where
  toFun P := fun i j z => P i j (-z)
  map_add' _ _ := rfl
  map_smul' _ _ := rfl

/-- A linear equality constraint, independent of input labels (CM16). -/
def EvenLatent (P : Table m n (ZMod q)) : Prop := ∀ i j z, P i j z = P i j (-z)

/-- Averaging two feasible tables will not introduce learned products (CM16). -/
noncomputable def pairLatent : Table m n (ZMod q) →ₗ[ℝ] Table m n (ZMod q) :=
  (1 / 2 : ℝ) • LinearMap.id + (1 / 2 : ℝ) • reflectLatent

/-- Negation only permutes the probability coordinates (CM16). -/
theorem rows_reflectLatent {P : Table m n (ZMod q)} (hP : Rows P) :
    Rows (reflectLatent P) := by
  refine ⟨fun i j z => hP.1 i j (-z), ?_⟩
  intro i j
  change (∑ z, P i j (-z)) = 1
  calc
    _ = ∑ z, P i j z := Fintype.sum_equiv (Equiv.neg (ZMod q)) _ _ (fun _ => rfl)
    _ = 1 := hP.2 i j

/-- Reflection exchanges Fourier frequencies, without changing input indices (CM16). -/
theorem moment_reflectLatent (P : Table m n (ZMod q)) (f : ZMod q) :
    moment (phaseCode (Equiv.refl (ZMod q)) f) (reflectLatent P) =
      moment (phaseCode (Equiv.refl (ZMod q)) (-f)) P := by
  ext i j
  change (∑ z, (P i j (-z) : ℂ) * phaseCode (Equiv.refl (ZMod q)) f z) = _
  apply Fintype.sum_equiv (Equiv.neg (ZMod q))
  intro z
  simp [phaseCode]

omit [NeZero q] in
/-- Equal paired probabilities define a convex linear subspace (CM16). -/
theorem evenLatent_convex : Convex ℝ {P : Table m n (ZMod q) | EvenLatent P} := by
  intro P hP Q hQ a b _ _ _ i j z
  change a * P i j z + b * Q i j z = a * P i j (-z) + b * Q i j (-z)
  rw [hP i j z, hQ i j z]

omit [NeZero q] in
/-- Averaging with reflection always satisfies the paired equalities (CM16). -/
theorem pairLatent_even (P : Table m n (ZMod q)) : EvenLatent (pairLatent P) := by
  intro i j z
  change (1 / 2 : ℝ) * P i j z + (1 / 2) * P i j (-z) =
    (1 / 2) * P i j (-z) + (1 / 2) * P i j (- -z)
  rw [neg_neg]
  ring

variable [DecidableEq C]

/-- Reflection is invisible through any fixed even decoder (CM16). -/
theorem decode_reflectLatent (d : ZMod q → C) (hd : ∀ z, d (-z) = d z)
    (P : Table m n (ZMod q)) : decode d (reflectLatent P) = decode d P := by
  ext i j c
  change (∑ z, if d z = c then P i j (-z) else 0) = _
  apply Fintype.sum_equiv (Equiv.neg (ZMod q))
  intro z
  simp only [Equiv.neg_apply, hd]

/-- Pairing preserves every output coordinate, hence every ordinary CE value (CM16). -/
theorem decode_pairLatent (d : ZMod q → C) (hd : ∀ z, d (-z) = d z)
    (P : Table m n (ZMod q)) : decode d (pairLatent P) = decode d P := by
  change decode d ((1 / 2 : ℝ) • P + (1 / 2 : ℝ) • reflectLatent P) = _
  rw [(decode d).map_add, (decode d).map_smul, (decode d).map_smul,
    decode_reflectLatent d hd]
  ext i j c
  change (1 / 2 : ℝ) * decode d P i j c + (1 / 2) * decode d P i j c = _
  ring

variable [Fintype m] [Fintype n] [DecidableEq n]

/-- Frequency-independent spectral upper bounds are closed under reflection (CM16). -/
theorem budget_reflectLatent {R B : ℝ} {P : Table m n (ZMod q)}
    (hP : BudgetFeasible (phaseCode (Equiv.refl (ZMod q))) (fun _ => R) (fun _ => B) P) :
    BudgetFeasible (phaseCode (Equiv.refl (ZMod q))) (fun _ => R) (fun _ => B)
      (reflectLatent P) := by
  refine ⟨⟨rows_reflectLatent hP.1.1, ?_⟩, ?_⟩
  · intro f
    rw [moment_reflectLatent]
    exact hP.1.2 (-f)
  · intro f
    rw [moment_reflectLatent]
    exact hP.2 (-f)

/-- Averaging enforces symmetry while retaining both spectral bounds (CM16). -/
theorem budget_pairLatent {R B : ℝ} {P : Table m n (ZMod q)}
    (hP : BudgetFeasible (phaseCode (Equiv.refl (ZMod q))) (fun _ => R) (fun _ => B) P) :
    BudgetFeasible (phaseCode (Equiv.refl (ZMod q))) (fun _ => R) (fun _ => B)
      (pairLatent P) := by
  exact budget_convex _ _ _ hP (budget_reflectLatent hP)
    (by norm_num : (0 : ℝ) ≤ 1 / 2) (by norm_num) (by norm_num)

/-- Predictions from symmetric latent probabilities under fixed Fourier bounds (CM16). -/
def PairedFeasible (d : ZMod q → C) (R B : ℝ) (Q : Table m n C) : Prop :=
  ∃ P : Table m n (ZMod q),
    BudgetFeasible (phaseCode (Equiv.refl (ZMod q))) (fun _ => R) (fun _ => B) P ∧
    EvenLatent P ∧ decode d P = Q

/-- Both spectral budgets and the additional linear equalities remain convex (CM16). -/
theorem pairedFeasible_convex (d : ZMod q → C) (R B : ℝ) :
    Convex ℝ {Q : Table m n C | PairedFeasible d R B Q} := by
  rintro Q ⟨P, hP, heP, rfl⟩ T ⟨U, hU, heU, rfl⟩ a b ha hb hab
  refine ⟨a • P + b • U, budget_convex _ _ _ hP hU ha hb hab,
    evenLatent_convex heP heU ha hb hab, ?_⟩
  rw [(decode d).map_add, (decode d).map_smul, (decode d).map_smul]

/-- For an even decoder, the symmetry restriction loses no feasible prediction (CM16).
The budgets are constant across the complete Fourier bank, as required above. -/
theorem pairedFeasible_iff (d : ZMod q → C) (hd : ∀ z, d (-z) = d z)
    (R B : ℝ) (Q : Table m n C) :
    PairedFeasible d R B Q ↔
      DecodedFeasible d (phaseCode (Equiv.refl (ZMod q))) (fun _ => R) (fun _ => B) Q := by
  constructor
  · rintro ⟨P, hP, _, he⟩
    exact ⟨P, hP, he⟩
  · rintro ⟨P, hP, he⟩
    exact ⟨pairLatent P, budget_pairLatent hP, pairLatent_even P,
      (decode_pairLatent d hd P).trans he⟩

/-! Nonconstant latent distributions inhabit the conditional claims. -/

private def witness : Table (Fin 2) (Fin 2) (ZMod 3) :=
  pointMass (phaseLabels (Equiv.refl (ZMod 3)) cornerPhase cornerPhase)
private def witnessDecoder (z : ZMod 3) : Bool := decide (z ≠ 0)

private theorem witnessDecoder_even (z : ZMod 3) : witnessDecoder (-z) = witnessDecoder z := by
  simp [witnessDecoder]

example := rows_reflectLatent (rows_pointMass
  (phaseLabels (Equiv.refl (ZMod 3)) cornerPhase cornerPhase))
example := moment_reflectLatent witness 1
example := pairLatent_even witness
example := decode_reflectLatent witnessDecoder witnessDecoder_even witness
example := decode_pairLatent witnessDecoder witnessDecoder_even witness
example := budget_reflectLatent
  (phaseCandidate_budget (Equiv.refl (ZMod 3)) cornerPhase cornerPhase)
example := budget_pairLatent (phaseCandidate_budget (Equiv.refl (ZMod 3)) cornerPhase cornerPhase)
example := pairedFeasible_iff witnessDecoder witnessDecoder_even 2 2 (decode witnessDecoder witness)
example := witnessDecoder_even 1

end Transformer.Grokking.FlatSpectrum.CoupledMoments.Schatten
