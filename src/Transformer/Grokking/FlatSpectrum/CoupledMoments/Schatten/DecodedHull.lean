module

public import Transformer.Grokking.FlatSpectrum.CoupledMoments.Schatten.DecodedPhase

/-!
# A convex prediction family covering all fixed output decoders

Source: `review/flat_spectrum.md`, CM15. Decoder discovery is a proposal search,
but its candidate predictions belong to one convex hull in the original output
coordinates. A feasible exact decoded fit minimizes ordinary CE over that
entire hull. This certifies a successful candidate, not an efficient oracle or
convergence when no such candidate is found.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.CoupledMoments.Schatten

variable {m n L C F S : Type*} [Fintype m] [Fintype n] [DecidableEq n]
  [Fintype L] [Fintype C] [DecidableEq C]

/-- One convex set of original-vocabulary predictions covering every decoder (CM15).
The learned coordinates are the predictions, not products of decoder and latent weights. -/
def DecoderHull (φ : F → L → ℂ) (R B : F → ℝ) : Set (Table m n C) :=
  convexHull ℝ (⋃ d : L → C, {Q | DecodedFeasible d φ R B Q})

omit [Fintype C] in
/-- All decoder choices can be included without making the prediction domain nonconvex (CM15). -/
theorem decoderHull_convex (φ : F → L → ℂ) (R B : F → ℝ) :
    Convex ℝ (DecoderHull (m := m) (n := n) (C := C) φ R B) :=
  convex_convexHull _ _

omit [Fintype C] in
/-- Each fixed-decoder candidate belongs to the common convex family (CM15). -/
theorem decoderHull_member (d : L → C) (φ : F → L → ℂ) (R B : F → ℝ)
    {Q : Table m n C} (hQ : DecodedFeasible d φ R B Q) : Q ∈ DecoderHull φ R B :=
  subset_convexHull ℝ _ (Set.mem_iUnion.mpr ⟨d, hQ⟩)

/-- Expanding to all decoders retains every feasible point of the original
fixed-code model, including its successful endpoints (CM15). -/
theorem decoderHull_identity_member (φ : F → C → ℂ) (R B : F → ℝ)
    {P : Table m n C} (hP : BudgetFeasible φ R B P) : P ∈ DecoderHull φ R B := by
  apply decoderHull_member id φ R B
  exact (decoded_identity_iff φ R B P).mpr hP

/-- Convexifying decoder choices preserves all original probability rows (CM15). -/
theorem decoderHull_rows (φ : F → L → ℂ) (R B : F → ℝ)
    {Q : Table m n C} (hQ : Q ∈ DecoderHull φ R B) : Rows Q := by
  apply convexHull_min (t := {Q : Table m n C | Rows Q}) ?_ rows_convex hQ
  intro T hT
  obtain ⟨d, P, hP, rfl⟩ := Set.mem_iUnion.mp hT
  exact rows_decode d hP.1.1

variable [Fintype S]

/-- Ordinary CE is convex on the entire decoder hull (CM15). -/
theorem decoderHull_loss_convex (φ : F → L → ℂ) (R B : F → ℝ)
    (cell : S → m × n) (labels : S → C) :
    ConvexOn ℝ (DecoderHull φ R B) (loss cell labels) := by
  refine ⟨decoderHull_convex φ R B, ?_⟩
  intro P _ Q _ a b ha hb hab
  exact (loss_convex cell labels).2 (Set.mem_univ _) (Set.mem_univ _) ha hb hab

/-- Any feasible exact decoded fit minimizes CE across all decoder choices (CM15).
If no candidate attains this lower bound, the theorem gives no solver certificate. -/
theorem decoderHull_fit_minimum (d : L → C) (φ : F → L → ℂ) (R B : F → ℝ)
    (cell : S → m × n) (labels : S → C) {P : Table m n L}
    (hP : BudgetFeasible φ R B P)
    (hfit : ∀ t, decode d P (cell t).1 (cell t).2 = ConvexRecall.targetLogits (labels t)) :
    decode d P ∈ DecoderHull φ R B ∧
      IsMinOn (loss cell labels) (DecoderHull φ R B) (decode d P) := by
  refine ⟨decoderHull_member d φ R B ⟨P, hP, rfl⟩, ?_⟩
  intro Q hQ
  rw [loss_eq_ideal_of_fit cell labels hfit]
  exact idealLoss_le cell labels (decoderHull_rows φ R B hQ)

variable {q : ℕ} [NeZero q]

/-- The constructed latent phase candidate certifies global CE optimality over
the common decoder hull when it fits the observed original labels (CM15). -/
theorem decodedPhase_hull_minimum (d : ZMod q → C) (r s : n → ZMod q)
    (cell : S → n × n) (labels : S → C)
    (hfit : ∀ t, d (r (cell t).1 + s (cell t).2) = labels t) :
    pointMass (fun i j => d (r i + s j)) ∈
        DecoderHull (phaseCode (Equiv.refl (ZMod q)))
          (fun _ => Fintype.card n) (fun _ => Fintype.card n) ∧
      IsMinOn (loss cell labels)
        (DecoderHull (phaseCode (Equiv.refl (ZMod q)))
          (fun _ => Fintype.card n) (fun _ => Fintype.card n))
        (pointMass (fun i j => d (r i + s j))) := by
  have hp := phaseCandidate_budget (Equiv.refl (ZMod q)) r s
  have hf : ∀ t, decode d (pointMass (phaseLabels (Equiv.refl (ZMod q)) r s))
      (cell t).1 (cell t).2 = ConvexRecall.targetLogits (labels t) := by
    intro t
    rw [decode_pointMass]
    funext c
    simp [pointMass, phaseLabels, hfit t, ConvexRecall.targetLogits]
  simpa only [decode_pointMass, phaseLabels, Equiv.refl_symm, Equiv.refl_apply] using
    decoderHull_fit_minimum d _ _ _ cell labels hp hf

/-- Convexifying decoder choices does not remove the unseen-label ambiguity (CM15).
Both explicit endpoints still attain the global ordinary-CE bound. -/
theorem corner_decoderHull_ambiguity :
    let domain : Set (Table (Fin 2) (Fin 2) Bool) :=
      DecoderHull (phaseCode (Equiv.refl (ZMod 3))) (fun _ => 2) (fun _ => 2)
    let P := pointMass (fun i j => cornerDecoder (cornerPhase i + cornerPhase j))
    let Q : Table (Fin 2) (Fin 2) Bool := pointMass (fun _ _ => true)
    P ∈ domain ∧ Q ∈ domain ∧
      IsMinOn (loss threeCells (fun _ => true)) domain P ∧
      IsMinOn (loss threeCells (fun _ => true)) domain Q ∧ P ≠ Q := by
  have hp := decodedPhase_hull_minimum cornerDecoder cornerPhase cornerPhase
    threeCells (fun _ => true) (by
      intro t
      fin_cases t <;> norm_num [cornerDecoder, cornerPhase, threeCells] <;> decide)
  have hq := decodedPhase_hull_minimum cornerDecoder (fun _ : Fin 2 => 0) (fun _ => 0)
    threeCells (fun _ => true) (by intro t; norm_num [cornerDecoder]; decide)
  simp only [Fintype.card_fin, Nat.cast_ofNat] at hp
  norm_num [cornerDecoder, show (0 : ZMod 3) ≠ 2 by decide] at hq
  refine ⟨hp.1, hq.1, hp.2, hq.2, ?_⟩
  intro he
  have hx := congrArg (fun P => P 1 1 true) he
  norm_num [pointMass, cornerPhase, cornerDecoder] at hx

/-! Nonconstant decoded examples inhabit every conditional certificate. -/

private def fullCell : Fin 2 × Fin 2 → Fin 2 × Fin 2 := id
private def fullLabel (ij : Fin 2 × Fin 2) : Bool :=
  cornerDecoder (cornerPhase ij.1 + cornerPhase ij.2)

example := decoderHull_member _ _ _ _ corner_decoded_admitted
example := decoderHull_identity_member _ _ _
  (phaseCandidate_budget (Equiv.refl (ZMod 2))
    (fun i : Fin 2 => i.val) (fun j : Fin 2 => j.val))
example := decoderHull_rows _ _ _ (decoderHull_member _ _ _ _ corner_decoded_admitted)
example := decoderHull_fit_minimum cornerDecoder (phaseCode (Equiv.refl (ZMod 3)))
  (fun _ => 2) (fun _ => 2) fullCell fullLabel
  (by simpa using phaseCandidate_budget (Equiv.refl (ZMod 3)) cornerPhase cornerPhase)
  (by
    intro t
    rw [decode_pointMass]
    funext c
    simp [pointMass, phaseLabels, fullCell, fullLabel, ConvexRecall.targetLogits]
    rfl)
example := decodedPhase_hull_minimum cornerDecoder cornerPhase cornerPhase
  fullCell fullLabel (by intro t; rfl)

end Transformer.Grokking.FlatSpectrum.CoupledMoments.Schatten
