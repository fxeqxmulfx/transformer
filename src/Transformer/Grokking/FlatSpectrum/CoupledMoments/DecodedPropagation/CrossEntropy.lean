module

public import Transformer.Grokking.FlatSpectrum.CoupledMoments.DecodedPropagation.Identification

/-!
# Training-derived equalities inside the convex spectral decoder hull

Source: `review/flat_spectrum.md`, CM19. The domain of CM15 contains the true
phase table for any fixed categorical decoder. Intersect it with the linear
one-hot equalities certified by propagation over a declared finite decoder
bank. If the target belongs to that bank, the intersection retains its table
and the global ordinary-CE optimum. Every feasible prediction has its certified
answers. These are exact statements about a convex program, not about an
optimizer, convergence, a learned decoder, or completeness of propagation.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.CoupledMoments.DecodedPropagation

open Schatten DecodedRecovery

variable {q : ℕ} [NeZero q] {n S C K : Type*} [Fintype n] [DecidableEq n]
  [Fintype S] [Fintype C] [DecidableEq C] [Fintype K]
  {decoder : K → ZMod q → C} {cell : S → n × n} {labels : S → C}

/-- The spectral hull of CM15 at budgets `N`, restricted by answer equalities
computed only from training labels and the declared bank. -/
def CertifiedFeasible (decoder : K → ZMod q → C) (cell : S → n × n) (labels : S → C)
    (anchor : S) (steps : ℕ) (Q : Table n n C) : Prop :=
  Q ∈ DecoderHull (phaseCode (Equiv.refl (ZMod q)))
      (fun _ => Fintype.card n) (fun _ => Fintype.card n) ∧
    ∀ i j label, bankAnswers decoder cell labels anchor steps i j = {label} →
      Q i j = ConvexRecall.targetLogits label

omit [Fintype C] in
/-- The restriction is convex even when different cells have different
certified labels and some cells remain uncertified. -/
theorem certifiedFeasible_convex (decoder : K → ZMod q → C) (cell : S → n × n)
    (labels : S → C) (anchor : S) (steps : ℕ) :
    Convex ℝ {Q | CertifiedFeasible decoder cell labels anchor steps Q} := by
  intro P hP Q hQ a b ha hb hab
  refine ⟨decoderHull_convex _ _ _ hP.1 hQ.1 ha hb hab, fun i j label hanswer => ?_⟩
  change a • P i j + b • Q i j = _
  rw [hP.2 i j label hanswer, hQ.2 i j label hanswer, ← add_smul, hab, one_smul]

/-- The usual full-vocabulary cross-entropy remains convex on the restriction. -/
theorem certified_loss_convex (decoder : K → ZMod q → C) (cell : S → n × n)
    (labels : S → C) (anchor : S) (steps : ℕ) :
    ConvexOn ℝ {Q | CertifiedFeasible decoder cell labels anchor steps Q}
      (loss cell labels) :=
  (loss_convex cell labels).subset (fun _ _ => Set.mem_univ _)
    (certifiedFeasible_convex decoder cell labels anchor steps)

/-- Every fitting phase table from any bank member survives all the certified
equalities. The procedure never receives its unobserved phases or labels. -/
theorem certified_witness (k : K) {r s : n → ZMod q}
    (hfit : Fits (decoder k) cell labels r s) (anchor : S) (steps : ℕ) :
    CertifiedFeasible decoder cell labels anchor steps
      (pointMass (fun i j => decoder k (r i + s j))) := by
  refine ⟨(decodedPhase_hull_minimum _ _ _ _ _ hfit).1, fun i j label hanswer => ?_⟩
  funext c
  simp [pointMass, ConvexRecall.targetLogits, bank_identifies k hfit anchor steps hanswer]

/-- The retained witness attains the original global CE optimum on the
restricted domain as well. This proves existence, not a solver trajectory. -/
theorem certified_optimum (k : K) {r s : n → ZMod q}
    (hfit : Fits (decoder k) cell labels r s) (anchor : S) (steps : ℕ) :
    IsMinOn (loss cell labels)
      {Q | CertifiedFeasible decoder cell labels anchor steps Q}
      (pointMass (fun i j => decoder k (r i + s j))) := by
  intro Q hQ
  exact (decodedPhase_hull_minimum _ _ _ _ _ hfit).2 hQ.1

/-- **No loss of optimality and no wrong certified answer** (CM19). Realizability
is with an arbitrary bank member. Every restricted minimum is also a minimum
over the unrestricted spectral decoder hull, and its certified rows are the
target's rows. The equalities are imposed, not inferred from CE minimization. -/
theorem certified_restriction (k : K) {r s : n → ZMod q}
    (hfit : Fits (decoder k) cell labels r s) (anchor : S) (steps : ℕ)
    {Q : Table n n C} (hQ : CertifiedFeasible decoder cell labels anchor steps Q)
    (hmin : IsMinOn (loss cell labels)
      {P | CertifiedFeasible decoder cell labels anchor steps P} Q) :
    IsMinOn (loss cell labels)
        (DecoderHull (phaseCode (Equiv.refl (ZMod q)))
          (fun _ => Fintype.card n) (fun _ => Fintype.card n)) Q ∧
      ∀ i j label, bankAnswers decoder cell labels anchor steps i j = {label} →
        Q i j = ConvexRecall.targetLogits (decoder k (r i + s j)) := by
  constructor
  · intro T hT
    exact ((isMinOn_iff.1 hmin _ (certified_witness k hfit anchor steps)).trans
      ((decodedPhase_hull_minimum _ _ _ _ _ hfit).2 hT))
  · intro i j label hanswer
    rw [bank_identifies k hfit anchor steps hanswer]
    exact hQ.2 i j label hanswer

end Transformer.Grokking.FlatSpectrum.CoupledMoments.DecodedPropagation
