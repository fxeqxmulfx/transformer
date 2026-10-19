module

public import Transformer.Interpolation.Section1_PhaseDerivatives
public import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic

/-!
# Distinct arbitrarily close atomic inputs with a common hole

ArXiv:2411.04551v3, §1, `eq: first.assp` and `thm: targets.atoms`.
A concrete unit circle in the first two coordinates of the original
three-dimensional ambient space supplies distinct nearby atoms.
The third coordinate unit vector is a common hole, for every circle
point. Thus proximity of the inputs is compatible with all the
source's distinctness and hole hypotheses.

Continuity at angle zero makes the squared gap times any prescribed
finite real factor less than one. This construction will be paired
with the original complete-network separation estimate. Passing from
unit points to their actual probability Dirac laws preserves
injectivity, as follows by testing their original barycenters.
-/

@[expose] public section

open Real MeasureTheory Set

namespace Transformer.Interpolation

/-- A concrete circle in the original ambient space. Source auxiliary:
arXiv:2411.04551v3, §1, unit atoms for `thm: targets.atoms`. -/
noncomputable def atomicCircleVector (t : ℝ) : EucSpace 3 :=
  (EuclideanSpace.equiv (Fin 3) ℝ).symm ![cos t, sin t, 0]

/-- The concrete circle has the original unit Euclidean norm.
Source auxiliary: arXiv:2411.04551v3, §1, the sphere in `eq: vf`. -/
theorem norm_atomicCircleVector (t : ℝ) : ‖atomicCircleVector t‖ = 1 := by
  have hs : ‖atomicCircleVector t‖ ^ 2 = 1 := by
    rw [EuclideanSpace.real_norm_sq_eq]
    simp [atomicCircleVector, Fin.sum_univ_succ]
  have hn := norm_nonneg (atomicCircleVector t)
  nlinarith

/-- The circle gives actual unit input atoms. Source auxiliary:
arXiv:2411.04551v3, §1, `thm: targets.atoms`. -/
noncomputable def atomicCirclePoint (t : ℝ) : SSphere 3 :=
  ⟨atomicCircleVector t, mem_sphere_zero_iff_norm.mpr (norm_atomicCircleVector t)⟩

/-- The original ambient circle is continuous in its angle. Source
auxiliary: arXiv:2411.04551v3, §1, nearby unit atomic inputs. -/
theorem continuous_atomicCircleVector : Continuous atomicCircleVector := by
  change Continuous (fun t : ℝ => WithLp.toLp 2 (![cos t, sin t, 0] : Fin 3 → ℝ))
  apply (PiLp.continuous_toLp 2 (fun _ : Fin 3 => ℝ)).comp
  apply continuous_pi_iff.mpr
  intro i
  fin_cases i
  · exact continuous_cos
  · exact continuous_sin
  · exact continuous_const

/-- Every strict angle in the upper semicircle gives a genuinely new
original atom. Source auxiliary: arXiv:2411.04551v3, §1,
the distinctness assumptions for `thm: targets.atoms`. -/
theorem atomicCirclePoint_ne_zero (t : ℝ) (ht : 0 < t) (htπ : t < π) :
    atomicCirclePoint t ≠ atomicCirclePoint 0 := by
  intro he
  have hc := congrArg (fun p : SSphere 3 => (p : EucSpace 3) (1 : Fin 3)) he
  have hs : sin t = 0 := by simpa [atomicCirclePoint, atomicCircleVector] using hc
  exact (sin_pos_of_pos_of_lt_pi ht htπ).ne' hs

/-- A fixed perpendicular unit point will miss every input atom.
Source auxiliary: arXiv:2411.04551v3, §1, `eq: first.assp`. -/
noncomputable def atomicInputHole : SSphere 3 :=
  ⟨PiLp.single 2 (2 : Fin 3) (1 : ℝ), by
    rw [mem_sphere_zero_iff_norm, PiLp.norm_single]
    norm_num⟩

/-- The same original unit point misses the entire input circle.
Source auxiliary: arXiv:2411.04551v3, §1, `eq: first.assp`. -/
theorem atomicInputHole_ne_circle (t : ℝ) : atomicInputHole ≠ atomicCirclePoint t := by
  intro he
  have hc := congrArg (fun p : SSphere 3 => (p : EucSpace 3) (2 : Fin 3)) he
  norm_num [atomicInputHole, atomicCirclePoint, atomicCircleVector] at hc

/-- A common actual hole is compatible with distinct atomic inputs
whose squared gap is smaller than any prescribed finite amplification.
Source auxiliary: arXiv:2411.04551v3, §1, `thm: targets.atoms`. -/
theorem exists_close_distinct_atoms_with_hole (A : ℝ) :
    ∃ p q w : SSphere 3, p ≠ q ∧ w ≠ p ∧ w ≠ q ∧
      ‖(p : EucSpace 3) - (q : EucSpace 3)‖ ^ 2 * A < 1 := by
  let f (t : ℝ) := ‖(atomicCirclePoint 0 : EucSpace 3) - atomicCircleVector t‖ ^ 2 * A
  have hf : Continuous f :=
    (((continuous_const.sub continuous_atomicCircleVector).norm).pow 2).mul continuous_const
  have hf0 : f 0 = 0 := by simp [f, atomicCirclePoint]
  obtain ⟨δ, hδ, hnear⟩ := Metric.continuousAt_iff.mp hf.continuousAt 1 one_pos
  let t := min (δ / 2) (π / 2)
  have ht : 0 < t := lt_min_iff.mpr ⟨by positivity, by positivity⟩
  have htδ : t < δ := (min_le_left _ _).trans_lt (by linarith)
  have htπ : t < π := (min_le_right _ _).trans_lt (by linarith [pi_pos])
  have hnear' : dist (f t) (f 0) < 1 := hnear
    (by simpa only [Real.dist_eq, sub_zero, abs_of_pos ht] using htδ)
  have hsmall : f t < 1 := (le_abs_self _).trans_lt
    (by simpa only [Real.dist_eq, hf0, sub_zero] using hnear')
  exact ⟨atomicCirclePoint 0, atomicCirclePoint t, atomicInputHole,
    (atomicCirclePoint_ne_zero t ht htπ).symm,
    atomicInputHole_ne_circle 0, atomicInputHole_ne_circle t, hsmall⟩

/-- The actual probability Dirac embedding preserves injectivity of
any original input point family. Source auxiliary:
arXiv:2411.04551v3, §1, the distinct inputs in `thm: targets.atoms`. -/
theorem injective_diracProb_family {d : ℕ} {ι : Type*} (X : ι → SSphere d)
    (hX : Function.Injective X) : Function.Injective (fun i => Perspective.diracProb d (X i)) := by
  intro i j hij
  apply hX
  apply Subtype.ext
  have hb := congrArg (barycenter d) hij
  simpa [barycenter, Perspective.diracProb] using hb

/-- A full injective original two-input data set has a common support
hole and an arbitrarily small weighted squared gap. Source auxiliary:
arXiv:2411.04551v3, §1, all input assumptions of `thm: targets.atoms`. -/
theorem exists_close_injective_atomic_data (A : ℝ) :
    ∃ (X : Idx 2 → SSphere 3) (w : SSphere 3), Function.Injective X ∧
      Function.Injective (fun i => Perspective.diracProb 3 (X i)) ∧
      (∀ i : Idx 2, w ∉ (Perspective.diracProb 3 (X i) : Measure (SSphere 3)).support) ∧
      ‖(X 0 : EucSpace 3) - (X 1 : EucSpace 3)‖ ^ 2 * A < 1 := by
  obtain ⟨p, q, w, hpq, hwp, hwq, hgap⟩ := exists_close_distinct_atoms_with_hole A
  let X : Idx 2 → SSphere 3 := ![p, q]
  have hX : Function.Injective X := by
    intro i j he
    fin_cases i <;> fin_cases j
    · rfl
    · exact False.elim (hpq he)
    · exact False.elim (hpq he.symm)
    · rfl
  refine ⟨X, w, hX, injective_diracProb_family X hX, ?_, hgap⟩
  intro i hw
  have he : w = X i := eq_of_mem_support_dirac hw
  fin_cases i
  · exact hwp he
  · exact hwq he

/-- The geometric hole misses the actual support of every original
input Dirac law, not just its chosen atom. Source auxiliary:
arXiv:2411.04551v3, §1, `eq: first.assp`. -/
theorem atomicInputHole_notMem_support (t : ℝ) :
    atomicInputHole ∉ (Perspective.diracProb 3 (atomicCirclePoint t) : Measure (SSphere 3)).support := by
  intro hw
  exact atomicInputHole_ne_circle t (eq_of_mem_support_dirac hw)

/-- Positive strict circle angles and a genuinely injective two-atom
family realize all geometric hypotheses. -/
example : (0 : ℝ) < π / 2 ∧ π / 2 < π ∧
    Function.Injective (![atomicCirclePoint 0, atomicCirclePoint (π / 2)] : Fin 2 → SSphere 3) := by
  have h0 : (0 : ℝ) < π / 2 := by positivity
  have hπ : π / 2 < π := by linarith [pi_pos]
  refine ⟨h0, hπ, ?_⟩
  intro i j he
  fin_cases i <;> fin_cases j
  · rfl
  · exact False.elim ((atomicCirclePoint_ne_zero (π / 2) h0 hπ) he.symm)
  · exact False.elim ((atomicCirclePoint_ne_zero (π / 2) h0 hπ) he)
  · rfl

end Transformer.Interpolation
