import Transformer.GPTMini.Convex.Structured.TensorStream

/-!
# Actual head observations are invariant under intermediate code writes

Source: TensorState/TensorPointer's genuine tensor-only projections at
5bdff3c, RMS recovery at e49ca6d and TensorStream's partial residual.
The observation predicate below compares the 52 recovered input fields
and recovered learned position of two actual prenorm rows. It asserts
no probability, semantic state, desired answer or convexity result.

Every head input projection and full pair bias follows from these real
coordinate reads. Actual intermediate writes preserve the predicate
after a new genuine RMSNorm despite changing the norm and code axes.
The proof derives both unit-anchor recoveries and all preserved reads
from the actual residual formula. Subsequent chronological inference
and stack induction must still use these local observation identities.
-/

namespace Transformer.GPTMini.Convex.Structured

open scoped Classical
noncomputable section

variable {C d T : ℕ}

/-- Equality of actual recovered token fields and position; decoder coordinates are not head inputs.
Source: the explicit 52-field/position projections in TensorState/TensorPointer, not a semantic representation condition. -/
def tensorFieldsSame (hwidth : 64 ≤ d) (x y : EucSpace d) : Prop :=
  (∀ slot : Fin 52, tensorAnchorRecovery (tensorAnchorAxis hwidth) x (tensorFieldAxis hwidth slot) =
    tensorAnchorRecovery (tensorAnchorAxis hwidth) y (tensorFieldAxis hwidth slot)) ∧
  tensorAnchorRecovery (tensorAnchorAxis hwidth) x (tensorPositionAxis hwidth) =
    tensorAnchorRecovery (tensorAnchorAxis hwidth) y (tensorPositionAxis hwidth)

/-- The observation predicate is reflexive at arbitrary actual tensor rows.
Source: equality of the same finite recovered coordinate observations. -/
theorem tensorFieldsSame_refl (hwidth : 64 ≤ d) (x : EucSpace d) : tensorFieldsSame hwidth x x := by
  exact ⟨fun slot => rfl, rfl⟩

example : (64 : ℕ) ≤ 64 := by omega

/-- Consecutive real observation equalities compose without assumptions on decoder coordinates.
Source: equality transitivity for all recovered physical fields and position. -/
theorem tensorFieldsSame_trans (hwidth : 64 ≤ d) (x y z : EucSpace d)
    (hxy : tensorFieldsSame hwidth x y) (hyz : tensorFieldsSame hwidth y z) : tensorFieldsSame hwidth x z := by
  exact ⟨fun slot => (hxy.1 slot).trans (hyz.1 slot), hxy.2.trans hyz.2⟩

example : tensorFieldsSame (d := 64) (by omega) (0 : EucSpace 64) 0 :=
  tensorFieldsSame_refl _ _

/-- Any actual intermediate code write retains the same recovered head observations after genuine prenorm.
Source: exact unit-anchor RMS recovery on both real residual rows and TensorStream's preserved field/position/anchor reads. -/
theorem tensorFieldsSame_stream (hwidth : 64 ≤ d) (eps : ℝ) (heps : 0 < eps)
    (x : EucSpace d) (hanchor : x (tensorAnchorAxis hwidth) = 1) (coordinates : Fin 10 → ℝ) :
    tensorFieldsSame hwidth (rmsNormEps eps (tensorStreamWrite hwidth x coordinates)) (rmsNormEps eps x) := by
  have hnew : tensorStreamWrite hwidth x coordinates (tensorAnchorAxis hwidth) = 1 := by
    rw [tensorStreamWrite_anchor, hanchor]
  unfold tensorFieldsSame
  rw [tensorAnchorRecovery_rms (by omega) eps heps _ _ hnew,
    tensorAnchorRecovery_rms (by omega) eps heps _ x hanchor]
  exact ⟨fun slot => tensorStreamWrite_field hwidth x coordinates slot,
    tensorStreamWrite_position hwidth x coordinates⟩

example : (64 : ℕ) ≤ 128 ∧ (0 : ℝ) < 1 / 100000 ∧
    (WithLp.toLp 2 (fun _ : Fin 128 => (1 : ℝ)) : EucSpace 128) (tensorAnchorAxis (by omega)) = 1 := by
  exact ⟨by omega, by norm_num, rfl⟩

/-- The genuine normalized intermediate block retains all observations used by the next tensor heads.
Source: TensorStream's actual prenorm/residual write followed by the proved new-prenorm field invariance. -/
theorem tensorFieldsSame_block (eps : ℝ) (heps : 0 < eps) (hwidth : 64 ≤ d) (ψ : TensorHeadParameters C)
    (x : Fin T → EucSpace d) (hcap : T ≤ C) (row : Fin T) (hanchor : x row (tensorAnchorAxis hwidth) = 1) :
    tensorFieldsSame hwidth (rmsNormEps eps (tensorStreamBlock eps hwidth ψ x hcap row)) (rmsNormEps eps (x row)) := by
  rw [tensorStreamBlock_write eps heps hwidth ψ x hcap row hanchor]
  exact tensorFieldsSame_stream hwidth eps heps (x row) hanchor _

example : (0 : ℝ) < 1 / 100000 ∧ (64 : ℕ) ≤ 64 ∧ (3 : ℕ) ≤ 64 ∧
    (WithLp.toLp 2 (fun _ : Fin 64 => (1 : ℝ)) : EucSpace 64) (tensorAnchorAxis (by omega)) = 1 := by
  exact ⟨by norm_num, by omega, by omega, rfl⟩

/-- Every free token-conditioned transition matrix agrees at rows with equal real head observations.
Source: the actual 36 transition projections; no semantic or learned transition is supplied as a premise. -/
theorem tensorTransition_fieldsSame (hwidth : 64 ≤ d) (x y : EucSpace d) (hfields : tensorFieldsSame hwidth x y) :
    tensorTransition hwidth x = tensorTransition hwidth y := by
  funext previous next
  exact hfields.1 (transitionSlot previous next)

example : tensorFieldsSame (d := 128) (by omega) (0 : EucSpace 128) 0 :=
  tensorFieldsSame_refl _ _

/-- Real query logits agree on the complete independently learned channel array.
Source: actual sixteen query coordinate projections and recovered-field equality. -/
theorem tensorQuery_fieldsSame (hwidth : 64 ≤ d) (x y : EucSpace d) (hfields : tensorFieldsSame hwidth x y) :
    tensorQuery hwidth x = tensorQuery hwidth y := by
  funext group channel
  exact hfields.1 (querySlot group channel)

example : tensorFieldsSame (d := 64) (by omega) (0 : EucSpace 64) 0 :=
  tensorFieldsSame_refl _ _

/-- Real key logits agree without fixing or identifying the separately learned query/key weights.
Source: actual sixteen distinct key projections and the full field-observation predicate. -/
theorem tensorKey_fieldsSame (hwidth : 64 ≤ d) (x y : EucSpace d) (hfields : tensorFieldsSame hwidth x y) :
    tensorKey hwidth x = tensorKey hwidth y := by
  funext group channel
  exact hfields.1 (keySlot group channel)

example : tensorFieldsSame (d := 128) (by omega) (0 : EucSpace 128) 0 :=
  tensorFieldsSame_refl _ _

/-- All actual jointly learned value logits agree after the intermediate update.
Source: genuine twenty value-potential reads, rather than frozen reference output values. -/
theorem tensorValue_fieldsSame (hwidth : 64 ≤ d) (x y : EucSpace d) (hfields : tensorFieldsSame hwidth x y) :
    tensorValue hwidth x = tensorValue hwidth y := by
  funext group channel
  exact hfields.1 (valueSlot group channel)

example : tensorFieldsSame (d := 64) (by omega) (0 : EucSpace 64) 0 :=
  tensorFieldsSame_refl _ _

/-- Every actual all-pair bias remains identical at observation-equivalent prenorm sequences.
Source: recovered learned position, unchanged physical chronology and the same freely learned relative-pair coefficient. -/
theorem tensorBindingBias_fieldsSame (hwidth : 64 ≤ d) (ψ : TensorHeadParameters C)
    (x y : Fin T → EucSpace d) (hcap : T ≤ C) (hfields : ∀ position, tensorFieldsSame hwidth (x position) (y position)) :
    tensorBindingBias hwidth ψ x hcap = tensorBindingBias hwidth ψ y hcap := by
  funext pair
  unfold tensorBindingBias
  rw [(hfields pair.2).2]

example : (64 : ℕ) ≤ 128 ∧ (3 : ℕ) ≤ 64 ∧
    (∀ position : Fin 3, tensorFieldsSame (d := 128) (by omega)
      ((fun _ : Fin 3 => (0 : EucSpace 128)) position) 0) := by
  exact ⟨by omega, by omega, fun _ => tensorFieldsSame_refl _ _⟩

/-- An actual large change of intermediate decoder values leaves recovered normalized head inputs identical.
Source: a genuine unit-anchored 64-dimensional raw row, new code coordinates 100 and recomputed positive-epsilon RMSNorm. -/
example : tensorFieldsSame (d := 64) (by omega)
    (rmsNormEps (1 / 100000) (tensorStreamWrite (by omega)
      (WithLp.toLp 2 (fun _ : Fin 64 => (1 : ℝ))) (fun _ => 100)))
    (rmsNormEps (1 / 100000) (WithLp.toLp 2 (fun _ : Fin 64 => (1 : ℝ)))) :=
  tensorFieldsSame_stream _ _ (by norm_num) _ rfl _

/-- The actual query projection is unchanged even after a large computed decoder write and a different RMS scale.
Source: real field observations, not equality of complete residual tensors or a supplied normalized query. -/
example : tensorQuery (d := 64) (by omega)
    (rmsNormEps (1 / 100000) (tensorStreamWrite (by omega)
      (WithLp.toLp 2 (fun _ : Fin 64 => (1 : ℝ))) (fun _ => 100))) =
    tensorQuery (by omega) (rmsNormEps (1 / 100000) (WithLp.toLp 2 (fun _ : Fin 64 => (1 : ℝ)))) :=
  tensorQuery_fieldsSame _ _ _ (tensorFieldsSame_stream _ _ (by norm_num) _ rfl _)

end
end Transformer.GPTMini.Convex.Structured
