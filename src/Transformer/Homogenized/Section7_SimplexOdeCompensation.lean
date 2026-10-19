module

public import Transformer.Homogenized.Section7_AttentionMomentContinuity
public import Transformer.Homogenized.Section7_OverlapNoiseMaximal
public import Transformer.Homogenized.Section7_SimplexSourceEnvelope

/-!
# The actual ODE compensator in the original simplex comparison

ArXiv:2604.01978v1, §7, `eq:gamma_ODE_selfcontained` and
`eq:Gronwall_E_selfcontained` in the proof of `thm:clustering_random_init`.
The original scalar ODE is integrated using the continuity of its
actual attention-moment drift. The Gaussian overlap drift along
continuous unit token paths is integrable as well.

Consequently the actual residual R(t)-gamma0-∫D supplies the
source's exact difference decomposition with gamma(t), including
the original D-b(gamma) time integral. Its integrability and the
identity are proved from the original fields and ODE data.
The final pathwise estimate therefore needs only the bound on
this fixed actual residual. The finite-dimension bias remains in
the existing deterministic estimate and its source-radius budget.
-/

@[expose] public section

open MeasureTheory Set
open scoped BigOperators NNReal
namespace Transformer.Homogenized

/-- Integrating the original right-half-line ODE gives its exact increment.
Source: arXiv:2604.01978v1, §7, subtracting `eq:gamma_ODE_selfcontained`. -/
theorem integral_right_ode_drift_eq_sub (γ b : ℝ → ℝ)
    (hd : ∀ t ∈ Ici (0 : ℝ), HasDerivWithinAt γ (b t) (Ici 0) t)
    (hb : ContinuousOn b (Ici (0 : ℝ))) (t : ℝ) (ht : 0 ≤ t) :
    (∫ s in (0 : ℝ)..t, b s) = γ t - γ 0 := by
  have hγ : ContinuousOn γ (Icc (0 : ℝ) t) := fun s hs =>
    (hd s hs.1).continuousWithinAt.mono (fun u hu => hu.1)
  have hbt : ContinuousOn b (Icc (0 : ℝ) t) := hb.mono (fun s hs => hs.1)
  apply intervalIntegral.integral_eq_sub_of_hasDerivAt_of_le ht hγ
    (fun s hs => (hd s hs.1.le).hasDerivAt (Ici_mem_nhds hs.1))
  apply ContinuousOn.intervalIntegrable
  rw [uIcc_of_le ht]
  exact hbt

/-- The original ODE and overlap fields give the integrable exact residual decomposition.
Source: arXiv:2604.01978v1, §7, the difference identity before `eq:Gronwall_E_selfcontained`.
The time integrability is derived, rather than added to the original data. -/
theorem overlapNoiseResidual_simplex_ode_decomposition {d n : ℕ} (hd : 2 ≤ d) (hn : 2 ≤ n + 1)
    (β : ℝ) (σV σA : ℝ≥0) (ρ : Measure (HeadParam d)) (hρ : IsGaussianHeadLaw d σV σA ρ)
    (f g : ℝ → ℝ)
    (hf : ∀ (c : ℝ) (Y : Idx (n + 1) → EucSpace d), IsSimplexConfig c Y → ∀ i,
      (∫ θ, ∑ k, attnProb β θ.2 Y i k ^ 2 ∂ρ) = f c)
    (hg : ∀ (c : ℝ) (Y : Idx (n + 1) → EucSpace d), IsSimplexConfig c Y → ∀ i j, i ≠ j →
      (∫ θ, ∑ k, attnProb β θ.2 Y i k * attnProb β θ.2 Y j k ∂ρ) = g c)
    (γ₀ : ℝ) (hγ₀ : γ₀ ∈ Ioo (-(1 / (((n + 1 : ℕ) : ℝ) - 1))) 1)
    (γ : ℝ → ℝ) (hinit : γ 0 = γ₀)
    (hdγ : ∀ t ∈ Ici (0 : ℝ), HasDerivWithinAt γ (simplexDrift f g (γ t)) (Ici 0) t)
    (T : ℝ) (hT : 0 ≤ T) {Ω : Type*} (X : ℝ → Ω → (Idx (n + 1) → EucSpace d)) (ω : Ω)
    (hX₀ : IsSimplexConfig γ₀ (X 0 ω))
    (hX : ∀ t ∈ Icc (0 : ℝ) T, ∀ i, ‖X t ω i‖ = 1)
    (hcX : ContinuousOn (fun t => X t ω) (Icc (0 : ℝ) T)) (i j : Idx (n + 1)) :
    (∀ t ∈ Ici (0 : ℝ), (∫ s in (0 : ℝ)..t, simplexDrift f g (γ s)) = γ t - γ₀) ∧
    IntervalIntegrable (fun s => overlapDrift β ρ (X s ω) i j - simplexDrift f g (γ s)) volume 0 T ∧
    ∀ t ∈ Icc (0 : ℝ) T, inner (𝕜 := ℝ) (X t ω i) (X t ω j) - γ t =
      overlapNoiseResidual β ρ X γ₀ i j t ω +
        ∫ s in (0 : ℝ)..t, overlapDrift β ρ (X s ω) i j - simplexDrift f g (γ s) := by
  have : IsProbabilityMeasure ρ := isProbabilityMeasure_of_isGaussianHeadLaw hρ
  have hBc := continuousOn_simplex_ode_drift hn β ρ f g hf hg γ₀ hγ₀ (X 0 ω) hX₀ γ hinit hdγ
  have hFTC (t : ℝ) (ht : 0 ≤ t) : (∫ s in (0 : ℝ)..t, simplexDrift f g (γ s)) = γ t - γ₀ := by
    simpa only [hinit] using integral_right_ode_drift_eq_sub γ (fun s => simplexDrift f g (γ s)) hdγ hBc t ht
  have hDc : ContinuousOn (fun s => overlapDrift β ρ (X s ω) i j) (Icc (0 : ℝ) T) :=
    (continuousOn_overlapDrift_gaussian hd β σV σA ρ hρ i j).comp' hcX (fun s hs => hX s hs)
  have hBcT : ContinuousOn (fun s => simplexDrift f g (γ s)) (Icc (0 : ℝ) T) :=
    hBc.mono (fun s hs => hs.1)
  have hInt (t : ℝ) (ht : t ∈ Icc (0 : ℝ) T) :
      IntervalIntegrable (fun s => overlapDrift β ρ (X s ω) i j) volume 0 t ∧
      IntervalIntegrable (fun s => simplexDrift f g (γ s)) volume 0 t := by
    have hsub : uIcc (0 : ℝ) t ⊆ Icc (0 : ℝ) T := by
      rw [uIcc_of_le ht.1]
      exact fun s hs => ⟨hs.1, hs.2.trans ht.2⟩
    exact ⟨(hDc.mono hsub).intervalIntegrable, (hBcT.mono hsub).intervalIntegrable⟩
  refine ⟨hFTC, ((hInt T ⟨hT, le_rfl⟩).1.sub (hInt T ⟨hT, le_rfl⟩).2), ?_⟩
  intro t ht
  rw [intervalIntegral.integral_sub (hInt t ht).1 (hInt t ht).2, hFTC t ht.1]
  unfold overlapNoiseResidual
  ring
/-- The actual residual bound alone supplies the original source envelope.
Source: arXiv:2604.01978v1, §7, `thm:clustering_random_init`.
The decomposition and its drift integrability are proved from the original ODE and fields. -/
theorem simplexOverlapError_le_of_noiseResidual {d n : ℕ} (hd : 2 ≤ d) (hn : 2 ≤ n + 1)
    (β : ℝ) (σV σA : ℝ≥0) (ρ : Measure (HeadParam d)) (hρ : IsGaussianHeadLaw d σV σA ρ)
    (f g : ℝ → ℝ)
    (hf : ∀ (c : ℝ) (Y : Idx (n + 1) → EucSpace d), IsSimplexConfig c Y → ∀ i,
      (∫ θ, ∑ k, attnProb β θ.2 Y i k ^ 2 ∂ρ) = f c)
    (hg : ∀ (c : ℝ) (Y : Idx (n + 1) → EucSpace d), IsSimplexConfig c Y → ∀ i j, i ≠ j →
      (∫ θ, ∑ k, attnProb β θ.2 Y i k * attnProb β θ.2 Y j k ∂ρ) = g c)
    (γ₀ : ℝ) (hγ₀ : γ₀ ∈ Ioo (-(1 / (((n + 1 : ℕ) : ℝ) - 1))) 1)
    (γ : ℝ → ℝ) (hinit : γ 0 = γ₀)
    (hdγ : ∀ t ∈ Ici (0 : ℝ), HasDerivWithinAt γ (simplexDrift f g (γ t)) (Ici 0) t)
    (T : ℝ) (hT : 0 ≤ T) (δ : ℝ) (hδ : δ ∈ Ioo (0 : ℝ) 1)
    {Ω : Type*} (X : ℝ → Ω → (Idx (n + 1) → EucSpace d)) (ω : Ω)
    (hX₀ : IsSimplexConfig γ₀ (X 0 ω))
    (hX : ∀ t ∈ Icc (0 : ℝ) T, ∀ i, ‖X t ω i‖ = 1)
    (hcX : ContinuousOn (fun t => X t ω) (Icc (0 : ℝ) T))
    (hNoise : ∀ t ∈ Icc (0 : ℝ) T, ∀ p : DistinctPairIdx (n + 1),
      |overlapNoiseResidual β ρ X γ₀ p.1.1 p.1.2 t ω| ≤ (7 / 8 : ℝ) * simplexNoiseRadius d (n + 1) T δ) :
    ∀ t ∈ Icc (0 : ℝ) T, simplexOverlapError (X t ω) (γ t) ≤
      Real.exp (82 * (1 + (d : ℝ) * (σA : ℝ) ^ 4 * β ^ 2 * ((n + 1 : ℕ) : ℝ)) * T) *
        simplexNoiseRadius d (n + 1) T δ := by
  have hparts (p : DistinctPairIdx (n + 1)) := overlapNoiseResidual_simplex_ode_decomposition hd hn β σV σA ρ hρ
    f g hf hg γ₀ hγ₀ γ hinit hdγ T hT X ω hX₀ hX hcX p.1.1 p.1.2
  exact simplexOverlapError_le_source_envelope hd hn β σV σA ρ hρ f g hf hg γ₀ hγ₀ γ hinit hdγ
    T hT δ hδ (fun t => X t ω) hX₀ hX hcX
    (fun t p => overlapNoiseResidual β ρ X γ₀ p.1.1 p.1.2 t ω)
    (fun t ht p => (hparts p).2.2 t ht) hNoise (fun p => (hparts p).2.1)

/-- Uniform attention, orthonormal token paths and the nonconstant original
scalar solution satisfy every premise. Their actual residual is zero,
while their actual error relative to the scalar trajectory is positive at t=1. -/
example : ∃ (X : ℝ → Unit → (Idx 2 → EucSpace 2)) (f g γ : ℝ → ℝ),
    (2 : ℕ) ≤ 2 ∧ IsGaussianHeadLaw 2 0 0 (Measure.dirac (0 : HeadParam 2)) ∧
    (0 : ℝ) ∈ Ioo (-(1 / ((2 : ℝ) - 1))) 1 ∧
    (∀ (c : ℝ) (Y : Idx 2 → EucSpace 2), IsSimplexConfig c Y → ∀ i,
      (∫ θ, ∑ k, attnProb 1 θ.2 Y i k ^ 2 ∂(Measure.dirac (0 : HeadParam 2))) = f c) ∧
    (∀ (c : ℝ) (Y : Idx 2 → EucSpace 2), IsSimplexConfig c Y → ∀ i j, i ≠ j →
      (∫ θ, ∑ k, attnProb 1 θ.2 Y i k * attnProb 1 θ.2 Y j k ∂(Measure.dirac (0 : HeadParam 2))) = g c) ∧
    γ 0 = 0 ∧ (∀ t ∈ Ici (0 : ℝ), HasDerivWithinAt γ (simplexDrift f g (γ t)) (Ici 0) t) ∧
    ContinuousOn (fun t => simplexDrift f g (γ t)) (Ici (0 : ℝ)) ∧
    (0 : ℝ) ≤ 1 ∧ (1 / 2 : ℝ) ∈ Ioo (0 : ℝ) 1 ∧ IsSimplexConfig 0 (X 0 ()) ∧
    (∀ t ∈ Icc (0 : ℝ) 1, ∀ i, ‖X t () i‖ = 1) ∧
    ContinuousOn (fun t => X t ()) (Icc (0 : ℝ) 1) ∧
    (∀ t ∈ Icc (0 : ℝ) 1, ∀ p : DistinctPairIdx 2,
      |overlapNoiseResidual 1 (Measure.dirac 0) X 0 p.1.1 p.1.2 t ()| ≤
        (7 / 8 : ℝ) * simplexNoiseRadius 2 2 1 (1 / 2)) ∧ 0 < γ 1 := by
  let X : ℝ → Unit → (Idx 2 → EucSpace 2) := fun _ _ k => EuclideanSpace.single k 1
  let f : ℝ → ℝ := fun _ => 1 / 2
  let g : ℝ → ℝ := fun _ => 1 / 2
  let γ : ℝ → ℝ := fun t => (Real.exp t - 1) / (Real.exp t + 1)
  have hγ : Continuous γ := continuous_iff_continuousAt.mpr
    (fun t => (hasDerivAt_simplex_uniform_attention t).continuousAt)
  have hB : Continuous (fun t => simplexDrift f g (γ t)) := by
    dsimp [simplexDrift, f, g]
    fun_prop
  refine ⟨X, f, g, γ, le_rfl, isGaussianHeadLaw_dirac_zero 2, by norm_num,
    ?_, ?_, by simp [γ], fun t ht => (hasDerivAt_simplex_uniform_attention t).hasDerivWithinAt,
    hB.continuousOn, zero_le_one, by norm_num, ?_, fun t ht i => by simp [X], continuousOn_const, ?_, ?_⟩
  · intro c Y hY i
    norm_num [f, attnProb, attnWeight, qkMap, Fin.sum_univ_two]
  · intro c Y hY i j hij
    norm_num [g, attnProb, attnWeight, qkMap, Fin.sum_univ_two]
  · constructor
    · intro i; simp [X]
    · intro i j hij; simp [X, EuclideanSpace.inner_single_left, hij.symm]
  · intro t ht p
    have hD (s : ℝ) : overlapDrift 1 (Measure.dirac (0 : HeadParam 2)) (X s ()) p.1.1 p.1.2 = 0 := by
      norm_num [overlapDrift, X, EuclideanSpace.inner_single_left, p.2.symm, PiLp.single_apply]
    have hzero : overlapNoiseResidual 1 (Measure.dirac 0) X 0 p.1.1 p.1.2 t () = 0 := by
      unfold overlapNoiseResidual
      rw [intervalIntegral.integral_congr (fun s hs => hD s), intervalIntegral.integral_zero]
      norm_num [X, EuclideanSpace.inner_single_left, p.2.symm, PiLp.single_apply]
    rw [hzero, abs_zero]
    exact mul_nonneg (by norm_num) (Real.sqrt_nonneg _)
  · exact div_pos (sub_pos.mpr (Real.one_lt_exp_iff.mpr (by norm_num))) (by positivity)

end Transformer.Homogenized
