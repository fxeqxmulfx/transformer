module

public import Transformer.Interpolation.Section1_AutonomousPrefix
public import Transformer.Interpolation.Section1_TimeTranslation
public import Transformer.Interpolation.Section3_TwoRidgeScaling

/-!
# Exact original weak transport after the final control switch

ArXiv:2411.04551v3, §1, `eq: cauchy.pb`, and Appendix,
the successive constant controls in `lem: tubular.mass.movement`.

Every original weak solution follows the actual autonomous two-neuron
law after its last switch. Reverse the clock from the desired endpoint:
the final phase becomes an autonomous prefix, whose weak characteristic
identity has already been proved. Applying the forward probability flow
inverts the reversed law and gives the exact original elapsed-time formula.

The earlier controls are arbitrary. The switch can occur at any real
time, and equality at the switch itself follows from the actual initial
law of the autonomous flow. No uniqueness hypothesis is assumed here.
-/

@[expose] public section

open Real MeasureTheory Set Filter
open scoped Topology NNReal

namespace Transformer.Interpolation

variable {d : ℕ}

/-- Negating both original outputs reverses the actual probability clock.
Source: arXiv:2411.04551v3, §1, `eq: vf` and `eq: cauchy.pb`. -/
theorem twoRidgeLaw_neg_output (p u : EucSpace d) (b : ℝ) (q v : EucSpace d) (c : ℝ)
    (ν : Perspective.ProbSphere d) (t : ℝ) :
    twoRidgeLaw (-p) u b (-q) v c ν t = twoRidgeLaw p u b q v c ν (-t) := by
  simpa only [neg_one_smul, neg_one_mul] using twoRidgeLaw_smul_output (-1) p u b q v c ν t

/-- Every original weak law follows the true autonomous probability flow
for every nonnegative elapsed time after the last switch.
Auxiliary for arXiv:2411.04551v3, Appendix, `lem: tubular.mass.movement`. -/
theorem cauchyPB_twoRidge_tail (i j : Fin d) (hij : i ≠ j)
    (p u : EucSpace d) (b : ℝ) (q v : EucSpace d) (c S : ℝ) (θ : TimeParams d)
    (hfield : ∀ (ρ : Perspective.ProbSphere d) (s : ℝ), S < s → ∀ x : SSphere d,
      fullVF d θ ρ s x = twoRidgeField p u b q v c x)
    (μ : ℝ → Perspective.ProbSphere d) (hμ : cauchyPB d θ μ) (t : ℝ) (ht : 0 ≤ t) :
    μ (S + t) = twoRidgeLaw p u b q v c (μ S) t := by
  by_cases hzero : t = 0
  · subst t
    rw [add_zero, twoRidgeLaw_zero]
  have hpos : 0 < t := lt_of_le_of_ne ht (Ne.symm hzero)
  have hrev (ρ : Perspective.ProbSphere d) (r : ℝ) (hr : r < t) (x : SSphere d) :
      fullVF d (reverseTimeParams (S + t) θ) ρ r x = twoRidgeField (-p) u b (-q) v c x := by
    rw [fullVF_reverseTimeParams, hfield ρ (S + t - r) (by linarith)]
    simpa only [neg_one_smul] using
      (twoRidgeField_smul_output (-1) p u b q v c (x : EucSpace d)).symm
  have hμrev := cauchyPB_reverseTimeParams (S + t) θ μ hμ
  have he := cauchyPB_twoRidge_prefix i j hij (-p) u b (-q) v c t hpos
    (reverseTimeParams (S + t) θ) hrev (fun r => μ (S + t - r)) hμrev t le_rfl
  simp only [add_sub_cancel_right, sub_zero, twoRidgeLaw_neg_output] at he
  have hh := congrArg (fun ρ => twoRidgeLaw p u b q v c ρ t) he
  rw [twoRidgeLaw_add, neg_add_cancel, twoRidgeLaw_zero] at hh
  exact hh.symm

/-- The final-phase formula expressed at the original absolute time.
Auxiliary for arXiv:2411.04551v3, Appendix, `lem: tubular.mass.movement`. -/
theorem cauchyPB_twoRidge_tail_after (i j : Fin d) (hij : i ≠ j)
    (p u : EucSpace d) (b : ℝ) (q v : EucSpace d) (c S : ℝ) (θ : TimeParams d)
    (hfield : ∀ (ρ : Perspective.ProbSphere d) (s : ℝ), S < s → ∀ x : SSphere d,
      fullVF d θ ρ s x = twoRidgeField p u b q v c x)
    (μ : ℝ → Perspective.ProbSphere d) (hμ : cauchyPB d θ μ) (t : ℝ) (ht : S ≤ t) :
    μ t = twoRidgeLaw p u b q v c (μ S) (t - S) := by
  have he := cauchyPB_twoRidge_tail i j hij p u b q v c S θ hfield μ hμ
    (t - S) (sub_nonneg.mpr ht)
  have hclock : S + (t - S) = t := by ring
  simpa only [hclock] using he

/-- The actual law at the switch is the prescribed initial law for the
last phase. Auxiliary for arXiv:2411.04551v3, Appendix,
the elapsed-time controls in `lem: tubular.mass.movement`. -/
theorem cauchyPB_twoRidge_tail_initial (i j : Fin d) (hij : i ≠ j)
    (p u : EucSpace d) (b : ℝ) (q v : EucSpace d) (c S : ℝ) (θ : TimeParams d)
    (hfield : ∀ (ρ : Perspective.ProbSphere d) (s : ℝ), S < s → ∀ x : SSphere d,
      fullVF d θ ρ s x = twoRidgeField p u b q v c x)
    (ν : Perspective.ProbSphere d) (μ : ℝ → Perspective.ProbSphere d)
    (hμ : cauchyPB d θ μ) (h0 : μ S = ν) (t : ℝ) (ht : S ≤ t) :
    μ t = twoRidgeLaw p u b q v c ν (t - S) := by
  rw [cauchyPB_twoRidge_tail_after i j hij p u b q v c S θ hfield μ hμ t ht, h0]

/-- Agreement at the last switch forces agreement throughout the tail.
Auxiliary for arXiv:2411.04551v3, Appendix, `lem: tubular.mass.movement`. -/
theorem cauchyPB_twoRidge_tail_unique (i j : Fin d) (hij : i ≠ j)
    (p u : EucSpace d) (b : ℝ) (q v : EucSpace d) (c S : ℝ) (θ : TimeParams d)
    (hfield : ∀ (ρ : Perspective.ProbSphere d) (s : ℝ), S < s → ∀ x : SSphere d,
      fullVF d θ ρ s x = twoRidgeField p u b q v c x)
    (μ ν : ℝ → Perspective.ProbSphere d) (hμ : cauchyPB d θ μ) (hν : cauchyPB d θ ν)
    (h0 : μ S = ν S) (t : ℝ) (ht : S ≤ t) : μ t = ν t := by
  rw [cauchyPB_twoRidge_tail_after i j hij p u b q v c S θ hfield μ hμ t ht,
    cauchyPB_twoRidge_tail_after i j hij p u b q v c S θ hfield ν hν t ht, h0]

/-- Appending the actual constant controls gives exactly the original
last-phase characteristic law, for every weak solution of the joined
equation. Source: arXiv:2411.04551v3, Appendix, `lem: tubular.mass.movement`. -/
theorem cauchyPB_twoRidge_concatenate_tail (i j : Fin d) (hij : i ≠ j)
    (p u : EucSpace d) (b : ℝ) (q v : EucSpace d) (c S : ℝ) (θ : TimeParams d)
    (μ : ℝ → Perspective.ProbSphere d)
    (hμ : cauchyPB d (concatenate S θ (fun _ => twoRidgeParams i j p u b q v c)) μ)
    (t : ℝ) (ht : S ≤ t) : μ t = twoRidgeLaw p u b q v c (μ S) (t - S) := by
  apply cauchyPB_twoRidge_tail_after i j hij p u b q v c S _ ?_ μ hμ t ht
  intro ρ s hs x
  have hh := fullVF_concatenate S θ (fun _ => twoRidgeParams i j p u b q v c)
    (fun _ => ρ) (fun _ => ρ) s x
  simp only [concatenate, ite_eq_right (not_lt_of_ge hs.le)] at hh
  exact hh.trans (fullVF_twoRidgeParams i j hij p u b q v c ρ (s - S) x)

/-- An actual two-phase neural weak flow jointly realizes all switch,
clock, prescribed-law and tail-uniqueness hypotheses for every initial
probability. Both nonzero neurons change speed at the switch. -/
example (ν : Perspective.ProbSphere 2) : ∃ (θ : TimeParams 2)
    (μ η : ℝ → Perspective.ProbSphere 2) (p q : EucSpace 2)
    (ρ : Perspective.ProbSphere 2),
    (0 : Fin 2) ≠ 1 ∧ (0 : ℝ) ≤ 1 ∧ (1 : ℝ) ≤ 2 ∧
    cauchyPB 2 (concatenate 1 θ (fun _ => twoRidgeParams 0 1 p 0 1 q 0 1)) μ ∧
    cauchyPB 2 (concatenate 1 θ (fun _ => twoRidgeParams 0 1 p 0 1 q 0 1)) η ∧
    μ 0 = ν ∧ μ 1 = ρ ∧ μ 1 = η 1 ∧
    (∀ (σ : Perspective.ProbSphere 2) (s : ℝ), 1 < s → ∀ x : SSphere 2,
      fullVF 2 (concatenate 1 θ (fun _ => twoRidgeParams 0 1 p 0 1 q 0 1)) σ s x =
        twoRidgeField p 0 1 q 0 1 x) := by
  let p : EucSpace 2 := basePoint 1
  let q : EucSpace 2 := EuclideanSpace.single 1 1
  let θ₁ : TimeParams 2 := fun _ => twoRidgeParams 0 1 p 0 1 q 0 1
  let θ₂ : TimeParams 2 :=
    fun _ => twoRidgeParams 0 1 ((2 : ℝ) • p) 0 1 ((2 : ℝ) • q) 0 1
  let μ₁ := twoRidgeLaw p 0 1 q 0 1 ν
  let μ₂ :=
    twoRidgeLaw ((2 : ℝ) • p) 0 1 ((2 : ℝ) • q) 0 1 (μ₁ 1)
  have hmatch : μ₂ 0 = μ₁ 1 := twoRidgeLaw_zero _ _ _ _ _ _ _
  have hweak := cauchyPB_concatenate 1 zero_lt_one θ₁ θ₂ μ₁ μ₂
    (cauchyPB_twoRidgeLaw 0 1 (by norm_num) _ _ _ _ _ _ _)
    (cauchyPB_twoRidgeLaw 0 1 (by norm_num) _ _ _ _ _ _ _) hmatch
  refine ⟨θ₁, concatenate 1 μ₁ μ₂, concatenate 1 μ₁ μ₂,
    (2 : ℝ) • p, (2 : ℝ) • q, μ₁ 1, by norm_num,
    zero_le_one, by norm_num, hweak, hweak, ?_, ?_, rfl, ?_⟩
  · rw [concatenate_zero 1 zero_lt_one]
    exact twoRidgeLaw_zero p 0 1 q 0 1 ν
  · rw [concatenate_switch, hmatch]
  · intro σ s hs x
    have hh := fullVF_concatenate 1 θ₁ θ₂ (fun _ => σ) (fun _ => σ) s x
    simp only [concatenate, ite_eq_right (not_lt_of_ge hs.le)] at hh
    exact hh.trans (fullVF_twoRidgeParams 0 1 (by norm_num)
      ((2 : ℝ) • p) 0 1 ((2 : ℝ) • q) 0 1 σ (s - 1) x)

end Transformer.Interpolation
