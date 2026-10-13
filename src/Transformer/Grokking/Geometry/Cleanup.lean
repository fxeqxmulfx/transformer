import Transformer.Grokking.Geometry.Decisions

/-!
# A correctness certificate using the actual masked cleanup energy

Source: Nanda et al., arXiv:2301.05217v1, section 5.1, the joint use of
restricted loss and excluded components. Deviation: the fixed-orbit
observer at 4436290 uses every class and actual held-out cells. The
certificate below uses absolute residual energy and the correct margin
of the cell mean after restoring any removed global class bias.

The observer records mean energies. Converting one to the summed energy
below requires multiplying by the number of retained scalar coordinates.
Using the mean directly would understate the bound. The pointwise result
in Decisions can give tighter certificates; the global bound here is a
conservative sufficient condition, not a necessary learning criterion.

Subtracting a row shift is harmless for classification. Subtracting a
different bias for each class is not; the latter appears explicitly in
the reference and raw predictions. A high energy fraction without a
correct margin or an absolute scale bound supplies no such certificate.

These are statements about the current forward logits. No future margin,
residual decrease or discrete AdamW update law is assumed to be proved.
-/

namespace Transformer.Grokking.Geometry

open scoped BigOperators
open Transformer.Grokking.NaiveLoss

variable {Q D C : Type*}

/-- Full-class residual energy of one actual input against its computed
cell mean. Source: the masked orbit observer at 4436290, adapting
arXiv:2301.05217v1, section 5.1. -/
noncomputable def pointResidualEnergy [Fintype C] (cells : Q → Finset D)
    (z : Q → D → C → ℝ) (q : Q) (d : D) : ℝ :=
  energyOver Finset.univ (fun c => z q d c - cellMean cells z q c)

/-- Each selected input's error is bounded by the actual total residual
energy. Source: the masked orbit observer at 4436290, adapting
arXiv:2301.05217v1, section 5.1; missing inputs are not covered. -/
theorem point_residual_le_total [Fintype C] (qs : Finset Q) (cells : Q → Finset D)
    (z : Q → D → C → ℝ) (q : Q) (d : D) (hq : q ∈ qs) (hd : d ∈ cells q) :
    pointResidualEnergy cells z q d ≤ cellResidualEnergy qs Finset.univ cells z := by
  have hn : ∀ q d, 0 ≤ pointResidualEnergy cells z q d := by
    intro q d
    unfold pointResidualEnergy energyOver
    exact Finset.sum_nonneg (fun c hc => sq_nonneg _)
  have he : cellResidualEnergy qs Finset.univ cells z =
      ∑ q ∈ qs, ∑ d ∈ cells q, pointResidualEnergy cells z q d := by
    unfold cellResidualEnergy residualOver pointResidualEnergy energyOver cellMean
    apply Finset.sum_congr rfl
    intro q hq
    exact Finset.sum_comm
  have hdle : pointResidualEnergy cells z q d ≤
      ∑ d ∈ cells q, pointResidualEnergy cells z q d :=
    Finset.single_le_sum (fun e he => hn q e) hd
  have hqle : (∑ d ∈ cells q, pointResidualEnergy cells z q d) ≤
      ∑ q ∈ qs, ∑ d ∈ cells q, pointResidualEnergy cells z q d :=
    Finset.single_le_sum (fun q hq => Finset.sum_nonneg (fun d hd => hn q d)) hq
  rw [he]
  exact le_trans hdle hqle

example : (0 : ℕ) ∈ ({0, 1} : Finset ℕ) ∧
    (1 : ℕ) ∈ ({0, 1} : Finset ℕ) := by norm_num

/-- A correct, bias-restored cell margin and sufficiently small absolute
cleanup energy certify every selected raw prediction. Source: the
restricted/excluded interpretation in arXiv:2301.05217v1, section 5.1,
with an explicit sufficient condition for the orbit observer at 4436290. -/
theorem cell_cleanup_certifies [Fintype C] (qs : Finset Q) (cells : Q → Finset D)
    (z : Q → D → C → ℝ) (q : Q) (d : D) (y : C) (bias : C → ℝ)
    (shift margin : ℝ) (hq : q ∈ qs) (hd : d ∈ cells q)
    (hm : MarginAtLeast (fun c => cellMean cells z q c + bias c) y margin)
    (hp : 0 < margin) (he : 2 * cellResidualEnergy qs Finset.univ cells z < margin ^ 2) :
    StrictCorrect (fun c => z q d c + bias c + shift) y := by
  apply (strictCorrect_shift_iff (fun c => z q d c + bias c) y shift).mpr
  apply strictCorrect_of_margin_and_energy (fun c => z q d c + bias c)
    (fun c => cellMean cells z q c + bias c) y margin hm hp
  have hpoint := point_residual_le_total qs cells z q d hq hd
  have herr : energyOver Finset.univ
      (fun c => (z q d c + bias c) - (cellMean cells z q c + bias c)) =
      pointResidualEnergy cells z q d := by
    unfold energyOver pointResidualEnergy
    apply Finset.sum_congr rfl
    intro c hc
    ring
  rw [herr]
  linarith

example : true ∈ ({false, true} : Finset Bool) ∧ (0 : ℕ) ∈ ({0, 1} : Finset ℕ) ∧
    MarginAtLeast (fun c => cellMean (fun _ : Bool => ({0, 1} : Finset ℕ))
      (fun q _ c => if c = q then (1 : ℝ) else -1) true c + 0) true 2 ∧
    0 < (2 : ℝ) ∧ 2 * cellResidualEnergy ({false, true} : Finset Bool) Finset.univ
      (fun _ => ({0, 1} : Finset ℕ)) (fun q _ c => if c = q then (1 : ℝ) else -1) < 2 ^ 2 := by
  refine ⟨by norm_num, by norm_num, ?_, by norm_num, ?_⟩
  · intro c hc
    cases c
    · norm_num [cellMean, meanOver]
    · exact False.elim (hc rfl)
  · norm_num [cellResidualEnergy, residualOver, energyOver, meanOver, Fintype.sum_bool]

/-- Residual energy equals one minus the structural fraction times total
energy when the latter is positive. Source: the orbit observer at 4436290,
adapting arXiv:2301.05217v1, section 5.1; an absolute scale remains. -/
theorem residual_from_fraction (qs : Finset Q) (cs : Finset C) (cells : Q → Finset D)
    (z : Q → D → C → ℝ) (ht : 0 < cellTotalEnergy qs cs cells z) :
    cellResidualEnergy qs cs cells z =
      (1 - cellEnergyFraction qs cs cells z) * cellTotalEnergy qs cs cells z := by
  have he := cell_energy_decomposition qs cs cells z
  have hn : cellTotalEnergy qs cs cells z ≠ 0 := by linarith
  unfold cellEnergyFraction
  field_simp
  nlinarith

example : 0 < cellTotalEnergy ({false} : Finset Bool) {true}
    (fun _ => ({0, 1} : Finset ℕ)) (fun _ _ _ => (1 : ℝ)) := by
  norm_num [cellTotalEnergy, energyOver]

/-- A fraction certifies raw decisions only together with total energy
and a correct bias-restored margin. Source: the orbit observer at 4436290,
adapting restricted/excluded logits in arXiv:2301.05217v1, section 5.1. -/
theorem cell_fraction_cleanup_certifies [Fintype C] (qs : Finset Q) (cells : Q → Finset D)
    (z : Q → D → C → ℝ) (q : Q) (d : D) (y : C) (bias : C → ℝ)
    (shift margin : ℝ) (hq : q ∈ qs) (hd : d ∈ cells q)
    (hm : MarginAtLeast (fun c => cellMean cells z q c + bias c) y margin)
    (hp : 0 < margin) (ht : 0 < cellTotalEnergy qs Finset.univ cells z)
    (he : 2 * (1 - cellEnergyFraction qs Finset.univ cells z) *
      cellTotalEnergy qs Finset.univ cells z < margin ^ 2) :
    StrictCorrect (fun c => z q d c + bias c + shift) y := by
  apply cell_cleanup_certifies qs cells z q d y bias shift margin hq hd hm hp
  rw [residual_from_fraction qs Finset.univ cells z ht]
  nlinarith

example : true ∈ ({false, true} : Finset Bool) ∧ (0 : ℕ) ∈ ({0, 1} : Finset ℕ) ∧
    MarginAtLeast (fun c => cellMean (fun _ : Bool => ({0, 1} : Finset ℕ))
      (fun q _ c => if c = q then (1 : ℝ) else -1) true c + 0) true 2 ∧
    0 < (2 : ℝ) ∧ 0 < cellTotalEnergy ({false, true} : Finset Bool) Finset.univ
      (fun _ => ({0, 1} : Finset ℕ)) (fun q _ c => if c = q then (1 : ℝ) else -1) ∧
    2 * (1 - cellEnergyFraction ({false, true} : Finset Bool) Finset.univ
      (fun _ => ({0, 1} : Finset ℕ)) (fun q _ c => if c = q then (1 : ℝ) else -1)) *
      cellTotalEnergy ({false, true} : Finset Bool) Finset.univ
        (fun _ => ({0, 1} : Finset ℕ)) (fun q _ c => if c = q then (1 : ℝ) else -1) < 2 ^ 2 := by
  refine ⟨by norm_num, by norm_num, ?_, by norm_num, ?_, ?_⟩
  · intro c hc
    cases c
    · norm_num [cellMean, meanOver]
    · exact False.elim (hc rfl)
  · norm_num [cellTotalEnergy, energyOver, Fintype.sum_bool]
  · norm_num [cellEnergyFraction, cellTotalEnergy, cellProjectedEnergy,
      cellMean, energyOver, meanOver, Fintype.sum_bool]

end Transformer.Grokking.Geometry
