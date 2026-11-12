module

public import Transformer.Grokking.FlatSpectrum.CoupledMoments.Schatten.Basic

/-!
# A fixed decoder of spectrally bounded latent probabilities

Source: `review/flat_spectrum.md`, CM15. Sum latent probabilities over a fixed
decoder's fibers. The output domain and original-vocabulary ordinary CE remain
convex. The spectral constraints remain on the latent moments, not on newly
decoded moments. A fixed decoder need not be injective or surjective.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.CoupledMoments.Schatten

open Finset

variable {m n L C F S : Type*} [Fintype L] [DecidableEq C]

/-- Aggregate latent coordinates into their fixed output labels (CM15). -/
noncomputable def decode (d : L → C) : Table m n L →ₗ[ℝ] Table m n C where
  toFun P := fun i j c => ∑ l, if d l = c then P i j l else 0
  map_add' P Q := by
    ext i j c
    simp only [Pi.add_apply, ← Finset.sum_add_distrib]
    apply Finset.sum_congr rfl
    intro l _
    split_ifs <;> simp
  map_smul' a P := by
    ext i j c
    simp only [RingHom.id_apply, Pi.smul_apply, smul_eq_mul, Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro l _
    split_ifs <;> simp

/-- Decoding one-hot latent states produces the actual decoded labels (CM15). -/
theorem decode_pointMass [DecidableEq L] (d : L → C) (labels : m → n → L) :
    decode d (pointMass labels) = pointMass (fun i j => d (labels i j)) := by
  ext i j c
  change (∑ l, if d l = c then pointMass labels i j l else 0) = _
  rw [Finset.sum_eq_single (labels i j)]
  · simp [pointMass, eq_comm]
  · intro l _ hl
    simp [pointMass, hl]
  · simp

variable [Fintype C]

/-- Fibers partition the latent vocabulary, preserving each probability row (CM15). -/
theorem rows_decode (d : L → C) {P : Table m n L} (hP : Rows P) :
    Rows (decode d P) := by
  constructor
  · intro i j c
    apply Finset.sum_nonneg
    intro l _
    split_ifs
    · exact hP.1 i j l
    · exact le_rfl
  · intro i j
    change (∑ c, ∑ l, if d l = c then P i j l else 0) = 1
    rw [Finset.sum_comm]
    simpa using hP.2 i j

/-- Every original class retains the probability-logit range, including classes
with several latent preimages or none (CM15). -/
theorem decoded_coordinate_bounds (d : L → C) {P : Table m n L}
    (hP : Rows P) (i : m) (j : n) (c : C) :
    0 ≤ decode d P i j c ∧ decode d P i j c ≤ 1 := by
  have hQ := rows_decode d hP
  exact ⟨hQ.1 i j c, coordinate_le_one hQ i j c⟩

/-- The identity decoder preserves every original probability coordinate (CM15). -/
theorem decode_id (P : Table m n C) : decode id P = P := by
  ext i j c
  simp [decode]

/-- Output moments are latent moments of pulled-back codes (CM15). Bounds on
other latent codes therefore cannot silently be reused for the output bank. -/
theorem moment_decode (d : L → C) (φ : C → ℂ) (P : Table m n L) :
    moment φ (decode d P) = moment (fun l => φ (d l)) P := by
  ext i j
  simp [moment, decode, Finset.sum_mul, Finset.sum_comm, apply_ite]

variable [Fintype m] [Fintype n] [DecidableEq n]

/-- The output prediction set is the image of the latent spectral domain (CM15). -/
def DecodedFeasible (d : L → C) (φ : F → L → ℂ) (R B : F → ℝ)
    (Q : Table m n C) : Prop :=
  ∃ P, BudgetFeasible φ R B P ∧ decode d P = Q

omit [Fintype C] in
/-- Fixed many-to-one decoding preserves the convex spectral prediction family (CM15). -/
theorem decodedFeasible_convex (d : L → C) (φ : F → L → ℂ) (R B : F → ℝ) :
    Convex ℝ {Q : Table m n C | DecodedFeasible d φ R B Q} := by
  change Convex ℝ (decode d '' {P | BudgetFeasible φ R B P})
  exact (budget_convex φ R B).linear_image (decode d)

/-- The original fixed-code family is exactly the identity-decoder case (CM15). -/
theorem decoded_identity_iff (φ : F → C → ℂ) (R B : F → ℝ) (Q : Table m n C) :
    DecodedFeasible id φ R B Q ↔ BudgetFeasible φ R B Q := by
  constructor
  · rintro ⟨P, hP, he⟩
    rw [decode_id] at he
    exact he ▸ hP
  · intro hQ
    exact ⟨Q, hQ, decode_id Q⟩

omit [Fintype m] [Fintype n] [DecidableEq n] in
/-- CE is over all original output classes and is convex in all latent coordinates (CM15). -/
theorem decoded_loss_convex [Fintype S] (d : L → C) (cell : S → m × n) (labels : S → C) :
    ConvexOn ℝ Set.univ (fun P : Table m n L => loss cell labels (decode d P)) := by
  simpa only [Set.preimage_univ, Function.comp_def] using
    (loss_convex cell labels).comp_linearMap (decode d)

/-- Feasible decoded fitting is a global minimum in the output prediction family (CM15).
The hypotheses require only decoded labels, not a chosen latent label at each sample. -/
theorem decoded_fit_minimum [Fintype S] (d : L → C) (φ : F → L → ℂ)
    (R B : F → ℝ) (cell : S → m × n) (labels : S → C)
    {P : Table m n L} (hP : BudgetFeasible φ R B P)
    (hfit : ∀ s, decode d P (cell s).1 (cell s).2 = ConvexRecall.targetLogits (labels s)) :
    DecodedFeasible d φ R B (decode d P) ∧
      IsMinOn (loss cell labels) {Q | DecodedFeasible d φ R B Q} (decode d P) := by
  refine ⟨⟨P, hP, rfl⟩, ?_⟩
  rintro Q ⟨T, hT, rfl⟩
  rw [loss_eq_ideal_of_fit cell labels hfit]
  exact idealLoss_le cell labels (rows_decode d hT.1.1)

example := decode_pointMass (m := Fin 2) (n := Fin 2)
  (fun l : Fin 3 => decide (l ≠ 2))
  (fun (i j : Fin 2) => (⟨i.val + j.val, by omega⟩ : Fin 3))
example := rows_decode (m := Fin 2) (n := Fin 2)
  (fun l : Fin 3 => decide (l ≠ 2))
  (rows_pointMass (fun (i j : Fin 2) => (⟨i.val + j.val, by omega⟩ : Fin 3)))
example := decoded_coordinate_bounds
  (fun l : Fin 3 => decide (l ≠ 2))
  (rows_pointMass (fun (i j : Fin 2) => (⟨i.val + j.val, by omega⟩ : Fin 3))) 1 1 false
example := decode_id (pointMass (fun (i : Fin 2) (j : Fin 2) => decide (i = j)))
example := moment_decode (m := Fin 2) (n := Fin 2)
  (fun l : Fin 3 => decide (l ≠ 2)) (fun b : Bool => if b then (1 : ℂ) else -1)
  (pointMass (fun (i j : Fin 2) => (⟨i.val + j.val, by omega⟩ : Fin 3)))
example := decodedFeasible_convex (m := Fin 2) (n := Fin 2)
  (fun l : Fin 3 => decide (l ≠ 2)) (fun (_ : Fin 1) (l : Fin 3) => (l.val : ℂ))
  (fun _ => 2) (fun _ => 2)
example := decoded_loss_convex (m := Fin 2) (n := Fin 2)
  (fun l : Fin 3 => decide (l ≠ 2)) (fun s : Fin 2 => (s, s)) (fun _ => true)

end Transformer.Grokking.FlatSpectrum.CoupledMoments.Schatten
