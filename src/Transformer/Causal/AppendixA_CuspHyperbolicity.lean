import Transformer.Causal.AppendixA_CuspCritical
import Transformer.Causal.AppendixA_CuspJacobian

/-!
# All printed sequential-flow hypotheses for the actual C¹ cusp energies

ArXiv:2411.04990v2, Appendix A, `lemma:convergence`, assumes
that every critical point of every truncation is strongly stable
or strongly unstable. This condition holds for the actual periodic
C¹ energies constructed here, with normalization identically one.

The actual first critical equation forces its sine to vanish.
The actual coupling then vanishes, so every remaining original
critical equation is exactly the corresponding sine equation.
The proved genuine Jacobian is diagonal with cosine entries;
at every actual critical point each entry is therefore one or
minus one. If all entries are negative, the original truncation
is strongly stable. Otherwise its actual complex characteristic
polynomial has the positive root one, making it strongly unstable.

All the source's original regularity, positive bounded normalization,
periodicity, prefix dependence, isolation and stability hypotheses
are proved jointly below for these actual two-particle energies.
There are actual stable critical points as well as the actual
unstable target `(π,0)`. This is the input verification preceding
the construction of trajectories for the C¹ regularity audit.
No conclusion of the printed convergence lemma is assumed.
-/

open Real
open scoped BigOperators
namespace Transformer.Causal

/-- Original truncated criticality forces every corresponding sine to vanish.
Source: arXiv:2411.04990v2, Appendix A, `lemma:convergence`, assumption 2. -/
theorem sine_zero_of_isCriticalPrefix_cuspEnergy (k : Idx 2) (ψ : Idx 2 → ℝ)
    (hcrit : IsCriticalPrefix 2 cuspEnergy (fun _ _ => 1) k ψ) :
    ∀ i : Prefix 2 k, Real.sin (ψ i.1) = 0 := by
  have h₀ : Real.sin (ψ 0) = 0 := by
    have h := hcrit ⟨0, by omega⟩
    rwa [seqVelocity_cuspEnergy_zero] at h
  intro i
  by_cases hi : i.1 = 0
  · simpa only [hi] using h₀
  · have hi₁ : i.1 = 1 := by
      apply Fin.ext
      have hn : (i.1 : ℕ) ≠ 0 := fun h => hi (Fin.ext h)
      omega
    have h := hcrit i
    rw [hi₁, seqVelocity_cuspEnergy_one, h₀] at h
    simpa only [hi₁, zero_pow (by decide : 2 ≠ 0), zero_mul, sub_zero] using h

/-- The original actual truncated critical points are exactly these sine zeros.
Source: arXiv:2411.04990v2, Appendix A, `lemma:convergence`, critical-point audit. -/
theorem isCriticalPrefix_cuspEnergy_iff_sine_zero (k : Idx 2) (ψ : Idx 2 → ℝ) :
    IsCriticalPrefix 2 cuspEnergy (fun _ _ => 1) k ψ ↔
      ∀ i : Prefix 2 k, Real.sin (ψ i.1) = 0 := by
  constructor
  · exact sine_zero_of_isCriticalPrefix_cuspEnergy k ψ
  · intro hs i
    have h₀ := hs ⟨0, by omega⟩
    by_cases hi : i.1 = 0
    · rw [hi, seqVelocity_cuspEnergy_zero, h₀]
    · have hi₁ : i.1 = 1 := by
        apply Fin.ext
        have hn : (i.1 : ℕ) ≠ 0 := fun h => hi (Fin.ext h)
        omega
      have h₁ := hs i
      rw [hi₁] at h₁
      rw [hi₁, seqVelocity_cuspEnergy_one, h₀, h₁]
      ring

/-- Every actual critical truncation satisfies the original spectral alternative.
Source: arXiv:2411.04990v2, Appendix A, `lemma:convergence`, assumption 2. -/
theorem stable_or_unstablePrefix_cuspEnergy (k : Idx 2) (ψ : Idx 2 → ℝ)
    (hcrit : IsCriticalPrefix 2 cuspEnergy (fun _ _ => 1) k ψ) :
    StronglyStablePrefix 2 cuspEnergy (fun _ _ => 1) k ψ ∨
      StronglyUnstablePrefix 2 cuspEnergy (fun _ _ => 1) k ψ := by
  classical
  have hs := sine_zero_of_isCriticalPrefix_cuspEnergy k ψ hcrit
  have h₀ := hs ⟨0, by omega⟩
  by_cases hall : ∀ i : Prefix 2 k, Real.cos (ψ i.1) = -1
  · left
    intro z hz
    rw [seqJacobian_cuspEnergy_charpoly_eval k ψ h₀] at hz
    obtain ⟨i, -, hi⟩ := Finset.prod_eq_zero_iff.mp hz
    have heq : z = (Real.cos (ψ i.1) : ℂ) := sub_eq_zero.mp hi
    rw [heq, hall i]
    norm_num
  · right
    push Not at hall
    obtain ⟨i, hi⟩ := hall
    have hsq : Real.cos (ψ i.1) ^ 2 = 1 := by
      have htrig := Real.sin_sq_add_cos_sq (ψ i.1)
      rw [hs i] at htrig
      nlinarith
    have hcos : Real.cos (ψ i.1) = 1 := (sq_eq_one_iff.mp hsq).resolve_right hi
    refine ⟨1, ?_, by norm_num⟩
    rw [seqJacobian_cuspEnergy_charpoly_eval k ψ h₀]
    apply Finset.prod_eq_zero_iff.mpr
    refine ⟨i, Finset.mem_univ i, ?_⟩
    rw [hcos]
    norm_num

/-- The genuine tuple `(π,π)` is strongly stable for every original truncation.
Source: arXiv:2411.04990v2, Appendix A, `lemma:convergence`, strong stability. -/
theorem stronglyStablePrefix_cuspEnergy_stable (k : Idx 2) :
    StronglyStablePrefix 2 cuspEnergy (fun _ _ => 1) k (![π, π] : Idx 2 → ℝ) := by
  intro z hz
  rw [seqJacobian_cuspEnergy_charpoly_eval k _ (by simp)] at hz
  obtain ⟨i, -, hi⟩ := Finset.prod_eq_zero_iff.mp hz
  have hall (j : Idx 2) : Real.cos ((![π, π] : Idx 2 → ℝ) j) = -1 := by
    fin_cases j <;> simp
  rw [hall i.1] at hi
  have heq : z = ((-1 : ℝ) : ℂ) := sub_eq_zero.mp hi
  rw [heq]
  norm_num

/-- Every actual critical Jacobian has no zero eigenvalue, in addition
to the printed spectral alternative. Source: arXiv:2411.04990v2,
Appendix A, `lemma:convergence`, hyperbolicity audit. -/
theorem zero_not_root_seqJacobian_cuspEnergy (k : Idx 2) (ψ : Idx 2 → ℝ)
    (hcrit : IsCriticalPrefix 2 cuspEnergy (fun _ _ => 1) k ψ) :
    (((seqJacobian 2 cuspEnergy (fun _ _ => 1) k ψ).map (fun r : ℝ => (r : ℂ))).charpoly).eval 0 ≠ 0 := by
  have hs := sine_zero_of_isCriticalPrefix_cuspEnergy k ψ hcrit
  rw [seqJacobian_cuspEnergy_charpoly_eval k ψ (hs ⟨0, by omega⟩)]
  intro hz
  obtain ⟨i, -, hi⟩ := Finset.prod_eq_zero_iff.mp hz
  have hcomplex : (Real.cos (ψ i.1) : ℂ) = 0 := (sub_eq_zero.mp hi).symm
  have hcos : Real.cos (ψ i.1) = 0 := by exact_mod_cast hcomplex
  have htrig := Real.sin_sq_add_cos_sq (ψ i.1)
  rw [hs i, hcos] at htrig
  norm_num at htrig

/-- All original hypotheses of the printed lemma hold jointly for the actual
C¹ energies and normalization one. Source: arXiv:2411.04990v2,
Appendix A, `lemma:convergence`; no derivative or stability input is postulated. -/
theorem cuspEnergy_sequential_hypotheses : (0 : ℝ) < 1 / 2 ∧
    (∀ k : Idx 2, ContDiff ℝ 1 (cuspEnergy k) ∧ ContDiff ℝ 1 (fun _ : Idx 2 → ℝ => (1 : ℝ))) ∧
    (∀ k : Idx 2, ∀ ψ : Idx 2 → ℝ, (1 / 2 : ℝ) < (fun _ _ => (1 : ℝ)) k ψ ∧
      (fun _ _ => (1 : ℝ)) k ψ < 2) ∧
    PeriodicAngles 2 cuspEnergy ∧ PeriodicAngles 2 (fun _ _ => (1 : ℝ)) ∧
    DependsOnPrefix 2 cuspEnergy ∧ DependsOnPrefix 2 (fun _ _ => (1 : ℝ)) ∧
    IsolatedCritical 2 cuspEnergy ∧
    (∀ k : Idx 2, ∀ ψ : Idx 2 → ℝ, IsCriticalPrefix 2 cuspEnergy (fun _ _ => 1) k ψ →
      StronglyStablePrefix 2 cuspEnergy (fun _ _ => 1) k ψ ∨
        StronglyUnstablePrefix 2 cuspEnergy (fun _ _ => 1) k ψ) :=
  ⟨by norm_num, fun k => ⟨contDiff_cuspEnergy k, contDiff_const⟩, fun _ _ => by norm_num,
    periodicAngles_cuspEnergy, fun _ _ _ => rfl, dependsOnPrefix_cuspEnergy, fun _ _ _ _ => rfl,
    isolatedCritical_cuspEnergy, stable_or_unstablePrefix_cuspEnergy⟩

/-- The actual tuple `(π,0)` realizes the critical-point hypothesis for
every original truncation, including the full two-particle system. -/
example : ∀ k : Idx 2, IsCriticalPrefix 2 cuspEnergy (fun _ _ => 1) k (![π, 0] : Idx 2 → ℝ) :=
  fun k => isCriticalPrefix_cuspEnergy_of_sine_zero k _ (by simp) (by simp)

end Transformer.Causal
