import Transformer.DoubleDescent.Section5_Curves
import Mathlib.Data.List.GetD

/-!
# Predicates on recorded experimental curves

arXiv:1912.02292v1, Sections 5--7. An observation stores a parameter and
an exact rational encoding of a published numeric value. The predicates
compare the actual recorded parameters, not arbitrary out-of-range defaults.
They certify properties of a finite log, not statistical generalization.
-/

namespace Transformer.DoubleDescent

/-- Sections 5--7: one finite observation, with its model size, sample size,
or epoch coordinate and the recorded test metric. -/
structure Observation where
  parameter : ℕ
  value : ℚ
  deriving DecidableEq

/-- Sections 5 and 6: four recorded points with a first descent, ascent,
and second descent, in increasing order of the measured parameter. -/
def RecordedDoubleDescent (rows : List Observation) : Prop :=
  ∃ a b c d : Fin rows.length,
    rows[a].parameter < rows[b].parameter ∧
    rows[b].parameter < rows[c].parameter ∧
    rows[c].parameter < rows[d].parameter ∧
    rows[b].value < rows[a].value ∧
    rows[b].value < rows[c].value ∧
    rows[d].value < rows[c].value

/-- Section 6: the second descent improves on the earlier minimum. -/
def RecordedCorrection (rows : List Observation) : Prop :=
  ∃ a b c d : Fin rows.length,
    rows[a].parameter < rows[b].parameter ∧
    rows[b].parameter < rows[c].parameter ∧
    rows[c].parameter < rows[d].parameter ∧
    rows[b].value < rows[a].value ∧
    rows[b].value < rows[c].value ∧
    rows[d].value < rows[b].value

/-- Section 7: more data strictly worsens the recorded test metric. -/
def RecordedMoreDataHurts (rows : List Observation) : Prop :=
  ∃ a b : Fin rows.length,
    rows[a].parameter < rows[b].parameter ∧ rows[a].value < rows[b].value

/-- Section 6: a recorded correction of overfitting includes double descent. -/
theorem recordedCorrection_implies_doubleDescent {rows : List Observation}
    (h : RecordedCorrection rows) : RecordedDoubleDescent rows := by
  rcases h with ⟨a, b, c, d, hab, hbc, hcd, hba, hcb, hdb⟩
  exact ⟨a, b, c, d, hab, hbc, hcd, hba, hcb, hdb.trans hcb⟩

/-- Section 6: all the recorded-correction hypotheses are satisfiable. -/
example : RecordedCorrection [⟨0, 3⟩, ⟨1, 1⟩, ⟨2, 4⟩, ⟨3, 0⟩] := by
  refine ⟨⟨0, by decide⟩, ⟨1, by decide⟩, ⟨2, by decide⟩, ⟨3, by decide⟩, ?_⟩
  norm_num

end Transformer.DoubleDescent
