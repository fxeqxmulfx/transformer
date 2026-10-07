import Transformer.GPTMini.Semantics.DepthPresence

/-!
# Six ordinary FFN units realize simultaneous depth detectors

Source: the original W_in/ReLU2/W_out at f11b6e2 and the preceding
three-hinge presence/type calculation for Basis E_2/E_4 at cbafbe9.
Two detectors use six disjoint actual hidden units. Each preactivation
is an ordinary shared linear form in the real signal, raw type and
protected constant coordinates; no bias or Boolean oracle is inserted.

The simultaneous output matrix writes the true hinge differences into
specified residual coordinates. Unwritten coordinates are exactly zero.
The actual RMS prenorm is retained as its positive quadratic multiplier,
not cancelled by a separately supplied position scale or encoder.

The dimensions are generic; six units fit both the existing 256-unit
small FFN and 512-unit large FFN without widening either model.
Actual raw embeddings, fused attention, amplitude propagation and full
depth readout remain separate obligations. No training claim follows.
-/

namespace Transformer.GPTMini.Semantics

open scoped BigOperators

/-- The genuine ordinary linear type-conditioned presence form for one original FFN row.
Source: the protected constant and actual signal/type coordinates of depthTypeStep. -/
noncomputable def depthDetectorForm {D : ℕ} (constant signal kind : Fin D) (cap : ℝ) : EucSpace D →L[ℝ] ℝ :=
  EuclideanSpace.proj signal + cap • (EuclideanSpace.proj kind - EuclideanSpace.proj constant)

/-- Reading this actual row yields the stated real linear combination of the same residual coordinates.
Source: ordinary coordinate projections and their linear-map operations. -/
theorem depthDetectorForm_apply {D : ℕ} (constant signal kind : Fin D) (cap : ℝ) (x : EucSpace D) :
    depthDetectorForm constant signal kind cap x = x signal + cap * (x kind - x constant) := by
  simp only [depthDetectorForm, add_apply, smul_apply, sub_apply, EuclideanSpace.proj, PiLp.proj_apply, smul_eq_mul]

/-- Two three-hinge branches occupy disjoint units in the unchanged actual FFN width.
Source: the explicit six-row assignment, with its only dimension requirement. -/
def depthDetectorIndex {H : ℕ} (hH : 6 ≤ H) (branch : Fin 2) (hinge : Fin 3) : Fin H :=
  ⟨3 * branch.val + hinge.val, by have hb := branch.isLt; have hr := hinge.isLt; omega⟩

/-- No two actual detector rows interfere in the shared hidden activation array.
Source: the six-row arithmetic assignment with exactly three units per branch. -/
theorem depthDetectorIndex_eq {H : ℕ} (hH : 6 ≤ H) (branch other : Fin 2) (hinge next : Fin 3) :
    depthDetectorIndex hH branch hinge = depthDetectorIndex hH other next ↔ branch = other ∧ hinge = next := by
  rw [Fin.ext_iff]
  change 3 * branch.val + hinge.val = 3 * other.val + next.val ↔ branch = other ∧ hinge = next
  constructor
  · intro h
    have hr := hinge.isLt
    have hn := next.isLt
    exact ⟨Fin.ext (by omega), Fin.ext (by omega)⟩
  · rintro ⟨rfl, rfl⟩
    rfl

example : (6 : ℕ) ≤ 256 ∧ (6 : ℕ) ≤ 512 := by decide

/-- The genuine quadratic finite-difference coefficients are 1,-2,1 in the three ordinary output rows.
Source: depthStep's actual real hinge formula. -/
def depthHingeCoefficient (hinge : Fin 3) : ℝ := if hinge = 1 then -2 else 1

/-- Actual bias-free input matrices realize all six type-conditioned shifted linear forms simultaneously.
Source: finite rank-one sums in the original W_in type, without extra hidden units. -/
noncomputable def depthDetectorIn {D H : ℕ} (hH : 6 ≤ H) (constant : Fin D)
    (signal kind : Fin 2 → Fin D) (a cap : ℝ) : EucSpace D →L[ℝ] EucSpace H :=
  ∑ branch : Fin 2, ∑ hinge : Fin 3,
    (depthDetectorForm constant (signal branch) (kind branch) cap -
      ((hinge.val : ℝ) * a) • EuclideanSpace.proj constant).smulRight
        (EuclideanSpace.single (depthDetectorIndex hH branch hinge) 1)

/-- The actual output matrix takes both three-hinge differences into the designated residual coordinates.
Source: ordinary shared W_out, including an explicit finite gain used by both branches. -/
noncomputable def depthDetectorOut {D H : ℕ} (hH : 6 ≤ H) (output : Fin 2 → Fin D) (gain : ℝ) :
    EucSpace H →L[ℝ] EucSpace D :=
  ∑ branch : Fin 2, ∑ hinge : Fin 3,
    (depthHingeCoefficient hinge • EuclideanSpace.proj (depthDetectorIndex hH branch hinge)).smulRight
      (gain • EuclideanSpace.single (output branch) 1)

/-- Every assigned coordinate of the actual fused input matrix equals its own genuine shifted preactivation.
Source: the complete finite rank-one input sum and proved hidden-row separation. -/
theorem depthDetectorIn_coordinate {D H : ℕ} (hH : 6 ≤ H) (constant : Fin D)
    (signal kind : Fin 2 → Fin D) (a cap : ℝ) (x : EucSpace D) (branch : Fin 2) (hinge : Fin 3) :
    depthDetectorIn hH constant signal kind a cap x (depthDetectorIndex hH branch hinge) =
      depthDetectorForm constant (signal branch) (kind branch) cap x - (hinge.val : ℝ) * a * x constant := by
  unfold depthDetectorIn
  simp only [sum_apply, ContinuousLinearMap.smulRight_apply, sub_apply, smul_apply,
    EuclideanSpace.proj, PiLp.proj_apply, smul_eq_mul, WithLp.ofLp_sum, Finset.sum_apply,
    WithLp.ofLp_smul, Pi.smul_apply, PiLp.single_apply]
  rw [Fintype.sum_eq_single branch]
  · rw [Fintype.sum_eq_single hinge]
    · simp
    · intro next hn
      have he : depthDetectorIndex hH branch hinge ≠ depthDetectorIndex hH branch next :=
        fun he => hn ((depthDetectorIndex_eq hH branch branch hinge next).mp he).2.symm
      simp only [ite_eq_right he, mul_zero]
  · intro other ho
    apply Finset.sum_eq_zero
    intro next hn
    have he : depthDetectorIndex hH branch hinge ≠ depthDetectorIndex hH other next :=
      fun he => ho ((depthDetectorIndex_eq hH branch other hinge next).mp he).1.symm
    simp only [ite_eq_right he, mul_zero]

example : (6 : ℕ) ≤ 256 := by decide

/-- The true ordinary W_in/ReLU2/W_out computes both stated continuous type-conditioned presence gates.
Source: all actual assigned input rows, the real componentwise activation and simultaneous output coefficients. -/
theorem depthDetectorFFN_apply {D H : ℕ} (hH : 6 ≤ H) (constant : Fin D)
    (signal kind output : Fin 2 → Fin D) (a cap gain : ℝ) (x : EucSpace D) :
    relu2FFN (depthDetectorIn hH constant signal kind a cap) (depthDetectorOut hH output gain) x =
      ∑ branch : Fin 2, (gain * depthTypeStep a cap (x (signal branch)) (x (kind branch)) (x constant)) •
        EuclideanSpace.single (output branch) 1 := by
  unfold relu2FFN depthDetectorOut
  simp only [sum_apply, ContinuousLinearMap.smulRight_apply, smul_apply, EuclideanSpace.proj,
    PiLp.proj_apply, smul_eq_mul, relu2Vec_apply, smul_smul]
  apply Finset.sum_congr rfl
  intro branch hb
  simp only [depthDetectorIn_coordinate, Fin.sum_univ_three, depthHingeCoefficient]
  norm_num
  rw [← neg_smul, ← add_smul, ← add_smul]
  unfold depthTypeStep depthStep
  rw [depthDetectorForm_apply]
  congr 1
  ring_nf

example : (6 : ℕ) ≤ 512 := by decide

/-- The actual original RMS prenorm preserves the shared quadratic gate computation and its genuine position-dependent scale.
Source: real rmsNormEps followed by the simultaneous six-unit ordinary FFN; no external scale or normalized-state premise. -/
theorem depthDetectorFFN_rms {D H : ℕ} (hH : 6 ≤ H) (constant : Fin D)
    (signal kind output : Fin 2 → Fin D) (a cap gain eps : ℝ) (x : EucSpace D) :
    relu2FFN (depthDetectorIn hH constant signal kind a cap) (depthDetectorOut hH output gain) (rmsNormEps eps x) =
      ∑ branch : Fin 2, (gain * (Real.sqrt (D : ℝ) / Real.sqrt (‖x‖ ^ 2 + (D : ℝ) * eps)) ^ 2 *
        depthTypeStep a cap (x (signal branch)) (x (kind branch)) (x constant)) • EuclideanSpace.single (output branch) 1 := by
  rw [depthDetectorFFN_apply]
  unfold rmsNormEps
  simp only [PiLp.smul_apply, smul_eq_mul]
  apply Finset.sum_congr rfl
  intro branch hb
  rw [depthTypeStep_scale a cap _ _ _ _ (by positivity)]
  congr 1
  ring

example : (6 : ℕ) ≤ 256 := by decide

/-- Every residual coordinate outside the two designated outputs is exactly unchanged by this genuine FFN contribution.
Source: the actual two-branch output vectors; raw constants/type channels can therefore remain protected. -/
theorem depthDetectorFFN_protected {D H : ℕ} (hH : 6 ≤ H) (constant : Fin D)
    (signal kind output : Fin 2 → Fin D) (a cap gain : ℝ) (x : EucSpace D) (c : Fin D)
    (hprotected : ∀ branch, c ≠ output branch) :
    relu2FFN (depthDetectorIn hH constant signal kind a cap) (depthDetectorOut hH output gain) x c = 0 := by
  rw [depthDetectorFFN_apply]
  simp only [WithLp.ofLp_sum, Finset.sum_apply, WithLp.ofLp_smul, Pi.smul_apply, smul_eq_mul, PiLp.single_apply]
  apply Finset.sum_eq_zero
  intro branch hb
  rw [ite_eq_right (hprotected branch), mul_zero]

example : (6 : ℕ) ≤ 256 ∧ ∀ branch : Fin 2, (0 : Fin 64) ≠ (if branch = 0 then 4 else 5) := by
  refine ⟨by decide, ?_⟩
  intro branch
  split_ifs <;> decide

end Transformer.GPTMini.Semantics
