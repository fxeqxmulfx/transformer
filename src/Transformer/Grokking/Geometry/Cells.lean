import Transformer.Grokking.Geometry.Basic

/-!
# Masked logit energies across several cells and classes

Source: Nanda et al., arXiv:2301.05217v1, section 5.1; the fixed-division
adaptation in lab.infrastructure.engine.grokking at 4436290. Cells index
denominators with the same quotient; a held-out mask can give each cell a
different size. Every retained class is included without selecting a
frequency from a future checkpoint.

Energies below are sums. Dividing all three by the same positive count
gives the observer's mean energies without changing the energy fraction.
The exact-real fraction is used only under positive total energy. Python
also applies a numerical floor and a minimum held-out coverage check,
returns None on failure and clips roundoff; that floating-point policy is
not identified with real division by zero here.

The input logits may already have their row shifts and global class bias
removed, as in the observer. No such centering is needed for this energy
identity. Correct predictions require additional information about targets.
-/

namespace Transformer.Grokking.Geometry

open scoped BigOperators

variable {Q D C : Type*}

/-- Actual class-coordinate mean on one selected cell. Source: the
fixed-orbit adaptation of arXiv:2301.05217v1, section 5.1, at 4436290. -/
noncomputable def cellMean (cells : Q → Finset D) (z : Q → D → C → ℝ)
    (q : Q) (c : C) : ℝ := meanOver (cells q) (fun d => z q d c)

/-- Squared logit energy on the selected cells and classes. Source:
the orbit observer at 4436290, adapting arXiv:2301.05217v1, section 5.1. -/
noncomputable def cellTotalEnergy (qs : Finset Q) (cs : Finset C)
    (cells : Q → Finset D) (z : Q → D → C → ℝ) : ℝ :=
  ∑ q ∈ qs, ∑ c ∈ cs, energyOver (cells q) (fun d => z q d c)

/-- Energy of each cell mean broadcast over its actual selected points.
Source: the orbit observer at 4436290; cardinality weights retain the
held-out mask sizes in the adaptation of arXiv:2301.05217v1, section 5.1. -/
noncomputable def cellProjectedEnergy (qs : Finset Q) (cs : Finset C)
    (cells : Q → Finset D) (z : Q → D → C → ℝ) : ℝ :=
  ∑ q ∈ qs, ∑ c ∈ cs, ((cells q).card : ℝ) * (cellMean cells z q c) ^ 2

/-- Energy after subtracting each class-coordinate's computed cell mean.
Source: the orbit adaptation of arXiv:2301.05217v1, section 5.1, at 4436290. -/
noncomputable def cellResidualEnergy (qs : Finset Q) (cs : Finset C)
    (cells : Q → Finset D) (z : Q → D → C → ℝ) : ℝ :=
  ∑ q ∈ qs, ∑ c ∈ cs, residualOver (cells q) (fun d => z q d c)

/-- Exact-real structural fraction; a positive denominator is a premise
of its interpretation. Source: the orbit-energy observer at 4436290,
adapting arXiv:2301.05217v1, section 5.1; numerical guards are not ported. -/
noncomputable def cellEnergyFraction (qs : Finset Q) (cs : Finset C)
    (cells : Q → Finset D) (z : Q → D → C → ℝ) : ℝ :=
  cellProjectedEnergy qs cs cells z / cellTotalEnergy qs cs cells z

/-- The summed energy uses exactly the selected quotient, denominator
and class coordinates. Source: the masked orbit observer at 4436290,
adapting arXiv:2301.05217v1, section 5.1. -/
theorem cell_total_matches_masked_sum (qs : Finset Q) (cs : Finset C)
    (cells : Q → Finset D) (z : Q → D → C → ℝ) :
    cellTotalEnergy qs cs cells z = ∑ q ∈ qs, ∑ d ∈ cells q, ∑ c ∈ cs, (z q d c) ^ 2 := by
  unfold cellTotalEnergy energyOver
  apply Finset.sum_congr rfl
  intro q hq
  exact Finset.sum_comm

/-- Exact energy decomposition survives nonuniform held-out cell sizes.
Source: the masked orbit observer at 4436290, adapting restricted and
excluded components of arXiv:2301.05217v1, section 5.1. -/
theorem cell_energy_decomposition (qs : Finset Q) (cs : Finset C)
    (cells : Q → Finset D) (z : Q → D → C → ℝ) :
    cellTotalEnergy qs cs cells z = cellProjectedEnergy qs cs cells z +
      cellResidualEnergy qs cs cells z := by
  unfold cellTotalEnergy cellProjectedEnergy cellResidualEnergy cellMean
  simp_rw [energy_decomposition, Finset.sum_add_distrib]

/-- Computed projection energy is nonnegative. Source: the masked orbit
observer at 4436290, adapting arXiv:2301.05217v1, section 5.1. -/
theorem cell_projected_nonneg (qs : Finset Q) (cs : Finset C)
    (cells : Q → Finset D) (z : Q → D → C → ℝ) :
    0 ≤ cellProjectedEnergy qs cs cells z := by
  unfold cellProjectedEnergy
  apply Finset.sum_nonneg
  intro q hq
  apply Finset.sum_nonneg
  intro c hc
  positivity

/-- Computed residual energy is nonnegative. Source: the masked orbit
observer at 4436290, adapting arXiv:2301.05217v1, section 5.1. -/
theorem cell_residual_nonneg (qs : Finset Q) (cs : Finset C)
    (cells : Q → Finset D) (z : Q → D → C → ℝ) :
    0 ≤ cellResidualEnergy qs cs cells z := by
  unfold cellResidualEnergy residualOver energyOver
  apply Finset.sum_nonneg
  intro q hq
  apply Finset.sum_nonneg
  intro c hc
  apply Finset.sum_nonneg
  intro d hd
  exact sq_nonneg _

/-- Cell projection cannot increase squared energy. Source: the masked
orbit observer at 4436290, adapting arXiv:2301.05217v1, section 5.1. -/
theorem cell_projection_contracts (qs : Finset Q) (cs : Finset C)
    (cells : Q → Finset D) (z : Q → D → C → ℝ) :
    cellProjectedEnergy qs cs cells z ≤ cellTotalEnergy qs cs cells z := by
  have he := cell_energy_decomposition qs cs cells z
  have hr := cell_residual_nonneg qs cs cells z
  linarith

/-- A positive-energy fraction lies in the unit interval before any
roundoff clipping. Source: the orbit observer at 4436290, adapting
arXiv:2301.05217v1, section 5.1; this is not a success certificate. -/
theorem cell_fraction_bounds (qs : Finset Q) (cs : Finset C)
    (cells : Q → Finset D) (z : Q → D → C → ℝ)
    (ht : 0 < cellTotalEnergy qs cs cells z) :
    0 ≤ cellEnergyFraction qs cs cells z ∧ cellEnergyFraction qs cs cells z ≤ 1 := by
  unfold cellEnergyFraction
  refine ⟨div_nonneg (cell_projected_nonneg qs cs cells z) ht.le, ?_⟩
  apply (div_le_iff₀ ht).mpr
  simpa only [one_mul] using cell_projection_contracts qs cs cells z

example : 0 < cellTotalEnergy ({false} : Finset Bool) {true}
    (fun _ => ({0, 1} : Finset ℕ)) (fun _ _ _ => (1 : ℝ)) := by
  norm_num [cellTotalEnergy, energyOver]

/-- A fraction of one says exactly that the selected residual energy is
zero. Source: the orbit observer at 4436290, adapting
arXiv:2301.05217v1, section 5.1. It says nothing about target labels. -/
theorem cell_fraction_one_iff (qs : Finset Q) (cs : Finset C)
    (cells : Q → Finset D) (z : Q → D → C → ℝ)
    (ht : 0 < cellTotalEnergy qs cs cells z) :
    cellEnergyFraction qs cs cells z = 1 ↔ cellResidualEnergy qs cs cells z = 0 := by
  have he := cell_energy_decomposition qs cs cells z
  unfold cellEnergyFraction
  rw [div_eq_one_iff_eq (by linarith : cellTotalEnergy qs cs cells z ≠ 0)]
  constructor <;> intro h <;> linarith

example : 0 < cellTotalEnergy ({false} : Finset Bool) {true}
    (fun _ => ({0, 1} : Finset ℕ)) (fun _ _ _ => (2 : ℝ)) := by
  norm_num [cellTotalEnergy, energyOver]

/-- Changes outside the selected points cannot affect their cell mean.
Source: the held-out-only projection at 4436290, adapting
arXiv:2301.05217v1, section 5.1; no training logits enter this average. -/
theorem cell_mean_local (cells : Q → Finset D) (z w : Q → D → C → ℝ)
    (q : Q) (c : C) (heq : ∀ d, d ∈ cells q → z q d c = w q d c) :
    cellMean cells z q c = cellMean cells w q c := by
  unfold cellMean meanOver
  congr 1
  apply Finset.sum_congr rfl
  exact heq

example : ∀ d : ℕ, d ∈ ({0} : Finset ℕ) →
    (if d = 0 then (1 : ℝ) else 0) = (if d = 0 then (1 : ℝ) else 100) := by
  intro d hd
  have heq : d = 0 := by simpa using hd
  subst d
  norm_num

end Transformer.Grokking.Geometry
