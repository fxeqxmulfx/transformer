/-
# Excluding a zero scalar branch on the original circle

ArXiv:2601.21366v2, §5.2, the final non-analyticity contradiction
in `thm: circle`, and §6, `eq:ratio-const-correct`.
An infinite compact angular pattern cannot consist only of zeros
of a nonzero original neuron. Analytic uniqueness proves this without
discarding any activation boundary or replacing the original angles.

For two neurons, a zero signed Gram action either has no effective
neurons or makes the two nontrivial neuron directions collinear.
The preceding collinear calculation then gives an actual analytic
extension of the original primitive potential. Consequently the
non-analytic source case cannot have scalar zero for its selected
infinite pattern, regardless of the original attention matrix.
The proof separates a zero output weight from a zero input vector.
No nonzero-neuron assumption is added. Analytic uniqueness selects a
genuine nonzero input only when both effective terms are nontrivial.
The attention matrix does not enter this finite-neuron argument.
-/

import Transformer.Perceptron.Section5_CollinearReLU

open Real Set

namespace Transformer.Perceptron

/-- A nonzero original circle neuron is nonzero at some angle of an
infinite compact angular set. Source: arXiv:2601.21366v2, §5.2,
the finite activation-boundary observation in `thm: circle`. -/
theorem exists_inner_ne_zero_in_infinite_angles (u : EucSpace 2) (hu : u ≠ 0)
    (S : Set ℝ) (hS : S.Infinite) (hbound : S ⊆ Icc (-π) π) :
    ∃ θ ∈ S, inner (𝕜 := ℝ) u (circlePoint θ : EucSpace 2) ≠ 0 := by
  by_contra! hzero
  have hc : AnalyticOnNhd ℝ (fun r => (circlePoint r : EucSpace 2)) univ :=
    analyticOnNhd_greatCircle (basePoint 1 : EucSpace 2) secondAxis
  have hf : AnalyticOnNhd ℝ (fun r => inner (𝕜 := ℝ) u (circlePoint r : EucSpace 2)) univ :=
    fun r hr => ((innerSL ℝ u).analyticAt _).comp (hc r hr)
  have hg := eq_zero_of_analytic_infinite_zeros _ hf (Icc (-π) π) isCompact_Icc S hS hbound hzero
  let p : SSphere 2 := ⟨‖u‖⁻¹ • u, mem_sphere_zero_iff_norm.mpr (norm_smul_inv_norm hu)⟩
  obtain ⟨θ, -, hθ⟩ := exists_circlePoint_eq p
  have hz : inner (𝕜 := ℝ) u (p : EucSpace 2) = 0 := by
    rw [← hθ]
    exact hg θ
  change inner (𝕜 := ℝ) u (‖u‖⁻¹ • u) = 0 at hz
  rw [real_inner_smul_right] at hz
  exact (inner_self_ne_zero.mpr hu)
    ((mul_eq_zero.mp hz).resolve_left (inv_ne_zero (norm_ne_zero_iff.mpr hu)))

/-- Each nonzero original neuron has only finitely many activation
boundary angles in the original compact circle interval.
Source: arXiv:2601.21366v2, §5.2, the finiteness of `𝒵` in `thm: circle`. -/
theorem finite_inner_zero_angles (u : EucSpace 2) (hu : u ≠ 0) :
    {θ : ℝ | θ ∈ Icc (-π) π ∧ inner (𝕜 := ℝ) u (circlePoint θ : EucSpace 2) = 0}.Finite := by
  by_contra hInf
  obtain ⟨θ, hθ, hne⟩ := exists_inner_ne_zero_in_infinite_angles u hu _ hInf (fun _ h => h.1)
  exact hne hθ.2

/-- If one of two original effective neurons vanishes, a zero signed
Gram action makes the other effective neuron vanish too.
Source: arXiv:2601.21366v2, §6, `eq:ratio-const-correct`, scalar zero. -/
theorem zero_effective_neurons_of_two_zero_signedGram (ω : Idx 2 → ℝ)
    (a : Idx 2 → EucSpace 2) (J : Finset (Idx 2))
    (hz : ∀ x, signedGram ω a J x = 0) (htriv : ω 1 = 0 ∨ a 1 = 0) :
    ∀ j, ω j = 0 ∨ a j = 0 := by
  have hfirst (x : EucSpace 2) :
      (signedBranchWeight ω J 0 * inner (𝕜 := ℝ) (a 0) x) • a 0 = 0 := by
    have h := hz x
    rw [signedGram_two_eq] at h
    rcases htriv with hω | ha
    · have hw : signedBranchWeight ω J 1 = 0 := (signedBranchWeight_eq_zero_iff ω J 1).mpr hω
      simpa only [hw, zero_mul, zero_smul, add_zero] using h
    · simpa only [ha, smul_zero, add_zero] using h
  have h₀ : ω 0 = 0 ∨ a 0 = 0 := by
    by_cases ha : a 0 = 0
    · exact Or.inr ha
    · have h := (smul_eq_zero.mp (hfirst (a 0))).resolve_right ha
      have hw := (mul_eq_zero.mp h).resolve_right (inner_self_ne_zero.mpr ha)
      exact Or.inl ((signedBranchWeight_eq_zero_iff ω J 0).mp hw)
  intro j
  fin_cases j
  · exact h₀
  · exact htriv

/-- A zero signed Gram action on an actual infinite angular pattern
forces the original two-neuron potential to be analytic on the circle.
Source: arXiv:2601.21366v2, §5.2, the analytic-potential contradiction
in `thm: circle`, and §6, `eq:ratio-const-correct`, scalar zero.
The arbitrary original primitive, all neurons and every original
angle of the selected pattern are retained. -/
theorem isAnalyticOnSphere_of_zero_signedGram_infinite_pattern (φ : ℝ → ℝ)
    (hφ : ∀ s, HasDerivAt φ (2 * max s 0) s)
    (ω : Idx 2 → ℝ) (a : Idx 2 → EucSpace 2) (J : Finset (Idx 2))
    (S : Set ℝ) (hS : S.Infinite) (hbound : S ⊆ Icc (-π) π)
    (hpat : ∀ θ ∈ S, IsActivationPattern a J (circlePoint θ : EucSpace 2))
    (hz : ∀ x, signedGram ω a J x = 0) : IsAnalyticOnSphere φ ω a := by
  by_cases htriv : ω 1 = 0 ∨ a 1 = 0
  · exact isAnalyticOnSphere_of_zero_effective_neurons φ ω a
      (zero_effective_neurons_of_two_zero_signedGram ω a J hz htriv)
  have hw₁ : ω 1 ≠ 0 := fun h => htriv (Or.inl h)
  have ha₁ : a 1 ≠ 0 := fun h => htriv (Or.inr h)
  obtain ⟨c, hc⟩ := exists_smul_of_two_rankone_cancel (a 0) (a 1)
    (signedBranchWeight ω J 0) (signedBranchWeight ω J 1)
    (fun h => hw₁ ((signedBranchWeight_eq_zero_iff ω J 1).mp h)) ha₁
    (fun x => by rw [← signedGram_two_eq]; exact hz x)
  have ha₀ : a 0 ≠ 0 := by
    intro h
    rw [h, smul_zero] at hc
    exact ha₁ hc
  obtain ⟨θ, hθ, hne⟩ := exists_inner_ne_zero_in_infinite_angles (a 0) ha₀ S hS hbound
  have ha : a = ![a 0, c • a 0] := by
    funext j
    fin_cases j
    · rfl
    · exact hc
  have hpat' : IsActivationPattern ![a 0, c • a 0] J (circlePoint θ : EucSpace 2) := by
    rw [← ha]
    exact hpat θ hθ
  have hz' : ∀ x, signedGram ω ![a 0, c • a 0] J x = 0 := by
    rw [← ha]
    exact hz
  have hresult := isAnalyticOnSphere_collinear_of_zero_signedGram φ hφ (a 0) c ω J
    (circlePoint θ : EucSpace 2) hpat' hne hz'
  rwa [← ha] at hresult

/-- The nonzero-neuron and infinite compact-angle inputs are realized
by the genuine first circle axis and the full original angle interval. -/
example : (basePoint 1 : EucSpace 2) ≠ 0 ∧ (Icc (-π) π).Infinite ∧
    (Icc (-π) π ⊆ Icc (-π) π) := by
  refine ⟨?_, Icc_infinite (by linarith [Real.pi_pos]), subset_rfl⟩
  intro h
  have hn := norm_basePoint_one
  rw [h, norm_zero] at hn
  exact zero_ne_one hn

/-- The zero-effective-neuron and infinite-pattern hypotheses hold
jointly for original zero weights and zero neurons on the actual circle.
The squared-ReLU primitive remains an actual primitive of twice ReLU. -/
example : (∀ s, HasDerivAt (fun t : ℝ => max t 0 ^ 2) (2 * max s 0) s) ∧
    (Icc (-π) π).Infinite ∧ (Icc (-π) π ⊆ Icc (-π) π) ∧
    (∀ θ ∈ Icc (-π) π, IsActivationPattern (0 : Idx 2 → EucSpace 2) ∅
      (circlePoint θ : EucSpace 2)) ∧
    (∀ x : EucSpace 2, signedGram (0 : Idx 2 → ℝ) 0 ∅ x = 0) ∧
    ((0 : Idx 2 → ℝ) 1 = 0 ∨ (0 : Idx 2 → EucSpace 2) 1 = 0) := by
  refine ⟨hasDerivAt_reluSq, Icc_infinite (by linarith [Real.pi_pos]), subset_rfl,
    fun θ hθ => ⟨fun j hj => by simp at hj, fun j hj => by simp⟩, ?_, Or.inl rfl⟩
  intro x
  simp [signedGram]

end Transformer.Perceptron
