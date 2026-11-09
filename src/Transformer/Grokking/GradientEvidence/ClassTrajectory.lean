module

public import Transformer.Grokking.GradientEvidence.UnusedClassDamping
public import Transformer.Grokking.FlatSpectrum.FeatureCodes.UpdateAttribution

/-!
# Finite and cumulative class-energy attribution

Source: `review/flat_spectrum.md`, *Class-contrast discovery and persistence*,
CG4–CG5. Signed component energies telescope over an actual trajectory.
Relative active-class energy has a different attribution from absolute
energy: common decay cancels from its radial numerator. All finite quadratic
terms remain. These are identities and explicit sufficient conditions, not
an unconditional optimizer-discovery theorem.
-/

@[expose] public section
noncomputable section
namespace Transformer.Grokking.GradientEvidence

open Finset FlatSpectrum.FeatureCodes
variable {I K : Type*} [Fintype I] [Fintype K]

/-- The exact energy change retains the square of the total increment,
including every cross-component product, CG4. -/
theorem component_energy_identity (x : K → ℝ) (d : I → K → ℝ) :
    (∑ k, (x k + ∑ i, d i k) ^ 2) = (∑ k, x k ^ 2) +
      (∑ i, 2 * ∑ k, x k * d i k) + ∑ k, (∑ i, d i k) ^ 2 := by
  have hs := linearScore_sum_feature x d
  unfold linearScore at hs
  calc
    _ = (∑ k, x k ^ 2) + 2 * (∑ k, x k * ∑ i, d i k) +
        ∑ k, (∑ i, d i k) ^ 2 := by
      simp only [mul_sum, ← sum_add_distrib]
      apply sum_congr rfl
      intro k hk
      simp only [← mul_sum]
      ring
    _ = _ := by rw [hs, mul_sum]

/-- Every prefix of a finite realized trajectory is explained by the signed
component totals and total squared increments, CG4. -/
theorem trajectory_component_energy_identity (x : ℕ → K → ℝ) (d : ℕ → I → K → ℝ)
    (hstep : ∀ t k, x (t+1) k = x t k + ∑ i, d t i k) (n : ℕ) :
    (∑ k, x n k ^ 2) = (∑ k, x 0 k ^ 2) + ∑ t ∈ range n,
      ((∑ i, 2 * ∑ k, x t k * d t i k) + ∑ k, (∑ i, d t i k) ^ 2) := by
  have one (t : ℕ) : (∑ k, x (t+1) k ^ 2) - (∑ k, x t k ^ 2) =
      (∑ i, 2 * ∑ k, x t k * d t i k) + ∑ k, (∑ i, d t i k) ^ 2 := by
    simp_rw [hstep]
    rw [component_energy_identity]
    ring
  calc
    _ = (∑ k, x 0 k ^ 2) + ∑ t ∈ range n,
        ((∑ k, x (t+1) k ^ 2) - ∑ k, x t k ^ 2) :=
      eq_sum_range_sub (fun t => ∑ k, x t k ^ 2) n
    _ = _ := by simp_rw [one]

omit [Fintype I] [Fintype K] in
/-- Exact relative-signal change, CG5. Positive signal growth alone is
insufficient if the competing energy grows proportionally faster. -/
theorem class_share_change (S V ds dv : ℝ) (h0 : S+V ≠ 0)
    (h1 : S+V+ds+dv ≠ 0) :
    (S+ds)/(S+V+ds+dv) - S/(S+V) =
      (V*ds-S*dv)/((S+V)*(S+V+ds+dv)) := by
  field_simp
  ring

omit [Fintype K] in
/-- Attribute every finite change in active-class share to all radial terms
and the total quadratic term, CG5. No component is treated as an independent
counterfactual trajectory. -/
theorem class_share_component_identity (S V : ℝ) (ds dv : I → ℝ) (qs qv : ℝ)
    (h0 : S+V ≠ 0) (h1 : S+V+(∑ i, ds i)+qs+(∑ i, dv i)+qv ≠ 0) :
    (S+(∑ i, ds i)+qs)/(S+V+(∑ i, ds i)+qs+(∑ i, dv i)+qv) - S/(S+V) =
      ((∑ i, (V*ds i-S*dv i))+(V*qs-S*qv)) /
        ((S+V)*(S+V+(∑ i, ds i)+qs+(∑ i, dv i)+qv)) := by
  simp only [sum_sub_distrib, ← mul_sum]
  field_simp
  ring

omit [Fintype I] [Fintype K] in
/-- The exact numerator is a sufficient improvement condition, CG5.
It is computed from current energies and signed increment contributions. -/
theorem class_share_nondecrease (S V ds dv : ℝ) (h0 : 0 < S+V)
    (h1 : 0 < S+V+ds+dv) (hgain : S*dv ≤ V*ds) :
    S/(S+V) ≤ (S+ds)/(S+V+ds+dv) := by
  have h := class_share_change S V ds dv (ne_of_gt h0) (ne_of_gt h1)
  have hp := div_nonneg (sub_nonneg.mpr hgain) (mul_pos h0 h1).le
  linarith

omit [Fintype I] [Fintype K] in
/-- Common scaling preserves class-energy proportions, CG5. Thus pure
weight decay cannot by itself create active-class concentration. -/
theorem common_scaling_preserves_class_share (S V a : ℝ) (ha : a ≠ 0) :
    (a*S)/(a*S+a*V) = S/(S+V) := by
  rw [← mul_add, mul_div_mul_left _ _ ha]

example (n : ℕ) := trajectory_component_energy_identity (I := Bool) (K := Fin 1)
  (fun t _ => t+1) (fun _ i _ => if i then 3/4 else 1/4)
  (by intro t k; norm_num) n

example := class_share_component_identity (I := Bool) 2 3
  (fun i => if i then 1 else -1/4) (fun i => if i then -1/2 else 1/4) (1/8) (1/16)
  (by norm_num) (by norm_num)

example := class_share_nondecrease 2 3 (3/4) (-1/4)
  (by norm_num) (by norm_num) (by norm_num)

example := common_scaling_preserves_class_share 2 3 (81/100) (by norm_num)

example := class_share_change 2 3 (3/4) (-1/4) (by norm_num) (by norm_num)

end Transformer.Grokking.GradientEvidence
