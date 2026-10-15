import Transformer.Homogenized.Section7_SimplexOdeCompensation

/-!
# The consensus endpoint of the original simplex comparison

ArXiv:2604.01978v1, §7, `eq:Phi_def_selfcontained`,
`eq:gamma_ODE_selfcontained` and `thm:clustering_random_init`.
The source starts strictly below consensus. The original scalar
drift also vanishes at consensus, and its bounded attention
moments force every solution starting there to stay there.

The proof bounds the actual drift by its distance from 1 and
applies Grönwall along each compact trajectory interval.
No regularity of arbitrary moment extensions outside the feasible
overlap domain is required. The [0,1] bounds are those supplied
by `simplex_overlap_wellDefined` for the source's moments.

At this endpoint the reference simplex is the actual initial
configuration itself. The original residual decomposition and
Gaussian drift continuity give the same source envelope, with
the previously retained finite-dimension bias and coefficient 82.
Together with the strict-interior result this includes consensus.
-/

open MeasureTheory Set
open scoped BigOperators NNReal
namespace Transformer.Homogenized

/-- Bounded actual moments give a linear drift bound at consensus.
Source: arXiv:2604.01978v1, §7, `eq:Phi_def_selfcontained`. -/
theorem abs_simplexDrift_le_distance_to_one (f g : ℝ → ℝ) (c : ℝ)
    (hf : f c ∈ Icc (0 : ℝ) 1) (hg : g c ∈ Icc (0 : ℝ) 1) :
    |simplexDrift f g c| ≤ (|c| + 1) * |c - 1| := by
  have hf' : |1 - f c| ≤ 1 := by
    rw [abs_of_nonneg (by linarith [hf.2] : 0 ≤ 1 - f c)]
    linarith [hf.1]
  have hg' : |g c| ≤ 1 := by rw [abs_of_nonneg hg.1]; exact hg.2
  rw [simplexDrift_eq_factor, abs_mul, abs_sub_comm 1 c]
  calc
    _ ≤ |c - 1| * (|c| * |1 - f c| + |g c|) := by
      gcongr
      exact (abs_add_le _ _).trans_eq (by rw [abs_mul])
    _ ≤ |c - 1| * (|c| * 1 + 1) := by gcongr
    _ = _ := by ring

/-- The original scalar ODE cannot leave consensus, even for merely bounded
moment extensions. Source: arXiv:2604.01978v1, §7, `eq:gamma_ODE_selfcontained`.
This adds the upper endpoint to the source's strict-interior setting. -/
theorem simplex_ode_eq_one (f g : ℝ → ℝ)
    (hf : ∀ c, f c ∈ Icc (0 : ℝ) 1) (hg : ∀ c, g c ∈ Icc (0 : ℝ) 1)
    (γ : ℝ → ℝ) (hinit : γ 0 = 1)
    (hd : ∀ t ∈ Ici (0 : ℝ), HasDerivWithinAt γ (simplexDrift f g (γ t)) (Ici 0) t)
    (T : ℝ) (hT : 0 ≤ T) : γ T = 1 := by
  have hc : ContinuousOn γ (Icc (0 : ℝ) T) := fun t ht =>
    (hd t ht.1).continuousWithinAt.mono (fun s hs => hs.1)
  obtain ⟨M, hM⟩ := isCompact_Icc.exists_bound_of_continuousOn hc
  have h := norm_le_gronwallBound_of_norm_deriv_right_le
    (f := fun t => γ t - 1) (f' := fun t => simplexDrift f g (γ t))
    (δ := 0) (K := M + 1) (ε := 0) (hc.sub continuousOn_const)
    (fun t ht => ((hd t ht.1).sub_const 1).mono (fun s hs => ht.1.trans hs))
    (by simp [hinit]) (fun t ht => ?_)
  · have ht := h T ⟨hT, le_rfl⟩
    rw [gronwallBound_ε0, zero_mul] at ht
    exact sub_eq_zero.mp (norm_eq_zero.mp (le_antisymm ht (norm_nonneg _)))
  · rw [Real.norm_eq_abs, Real.norm_eq_abs, add_zero]
    apply (abs_simplexDrift_le_distance_to_one f g (γ t) (hf _) (hg _)).trans
    apply mul_le_mul_of_nonneg_right _ (abs_nonneg _)
    have hMt : |γ t| ≤ M := by simpa only [Real.norm_eq_abs] using hM t ⟨ht.1, ht.2.le⟩
    linarith

/-- The actual residual supplies the same source envelope at consensus.
Source: arXiv:2604.01978v1, §7, `thm:clustering_random_init`, extended to γ₀=1.
No initial-interior realization or supplied drift integrability is needed. -/
theorem simplexOverlapError_le_of_noiseResidual_consensus {d n : ℕ} (hd : 2 ≤ d) (hn : 2 ≤ n + 1)
    (β : ℝ) (σV σA : ℝ≥0) (ρ : Measure (HeadParam d)) (hρ : IsGaussianHeadLaw d σV σA ρ)
    (f g : ℝ → ℝ)
    (hf : ∀ (c : ℝ) (Y : Idx (n + 1) → EucSpace d), IsSimplexConfig c Y → ∀ i,
      (∫ θ, ∑ k, attnProb β θ.2 Y i k ^ 2 ∂ρ) = f c)
    (hg : ∀ (c : ℝ) (Y : Idx (n + 1) → EucSpace d), IsSimplexConfig c Y → ∀ i j, i ≠ j →
      (∫ θ, ∑ k, attnProb β θ.2 Y i k * attnProb β θ.2 Y j k ∂ρ) = g c)
    (T : ℝ) (hT : 0 ≤ T) (δ : ℝ) (hδ : δ ∈ Ioo (0 : ℝ) 1)
    {Ω : Type*} (X : ℝ → Ω → (Idx (n + 1) → EucSpace d)) (ω : Ω)
    (hX₀ : IsSimplexConfig 1 (X 0 ω))
    (hX : ∀ t ∈ Icc (0 : ℝ) T, ∀ i, ‖X t ω i‖ = 1)
    (hcX : ContinuousOn (fun t => X t ω) (Icc (0 : ℝ) T))
    (hNoise : ∀ t ∈ Icc (0 : ℝ) T, ∀ p : DistinctPairIdx (n + 1),
      |overlapNoiseResidual β ρ X 1 p.1.1 p.1.2 t ω| ≤ (7 / 8 : ℝ) * simplexNoiseRadius d (n + 1) T δ) :
    ∀ t ∈ Icc (0 : ℝ) T, simplexOverlapError (X t ω) 1 ≤
      Real.exp (82 * (1 + (d : ℝ) * (σA : ℝ) ^ 4 * β ^ 2 * ((n + 1 : ℕ) : ℝ)) * T) *
        simplexNoiseRadius d (n + 1) T δ := by
  have hB : simplexDrift f g 1 = 0 := by simp [simplexDrift]
  have hDc (p : DistinctPairIdx (n + 1)) : ContinuousOn
      (fun s => overlapDrift β ρ (X s ω) p.1.1 p.1.2) (Icc (0 : ℝ) T) :=
    (continuousOn_overlapDrift_gaussian hd β σV σA ρ hρ p.1.1 p.1.2).comp' hcX (fun s hs => hX s hs)
  have h := simplexOverlapError_le_of_decomposition hd β σV σA ρ hρ f g hf hg T
    ((7 / 8 : ℝ) * simplexNoiseRadius d (n + 1) T δ) hT
    (mul_nonneg (by norm_num) (Real.sqrt_nonneg _))
    (fun t => X t ω) (fun _ => X 0 ω) (fun _ => 1) hX (fun t ht => hX₀) hcX continuousOn_const
    (fun t p => overlapNoiseResidual β ρ X 1 p.1.1 p.1.2 t ω)
    (fun t ht p => ?_) hNoise (fun p => ?_)
  · intro t ht
    exact (h t ht).trans (simplex_comparison_envelope d (n + 1) hd hn T hT δ hδ
      ((d : ℝ) * (σA : ℝ) ^ 4 * β ^ 2 * ((n + 1 : ℕ) : ℝ)) (by positivity))
  · rw [hB]
    simp only [overlapNoiseResidual, sub_zero]
    ring
  · rw [hB]
    simp only [sub_zero]
    apply ContinuousOn.intervalIntegrable
    simpa only [uIcc_of_le hT] using hDc p

/-- Unit consensus, actual uniform moments and zero residual simultaneously
realize every deterministic hypothesis, with positive horizon and radius. -/
example : ∃ (X : ℝ → Unit → (Idx 2 → EucSpace 2)) (f g γ : ℝ → ℝ),
    (2 : ℕ) ≤ 2 ∧ IsGaussianHeadLaw 2 0 0 (Measure.dirac (0 : HeadParam 2)) ∧
    (∀ c, f c ∈ Icc (0 : ℝ) 1) ∧ (∀ c, g c ∈ Icc (0 : ℝ) 1) ∧
    (∀ (c : ℝ) (Y : Idx 2 → EucSpace 2), IsSimplexConfig c Y → ∀ i,
      (∫ θ, ∑ k, attnProb 1 θ.2 Y i k ^ 2 ∂(Measure.dirac (0 : HeadParam 2))) = f c) ∧
    (∀ (c : ℝ) (Y : Idx 2 → EucSpace 2), IsSimplexConfig c Y → ∀ i j, i ≠ j →
      (∫ θ, ∑ k, attnProb 1 θ.2 Y i k * attnProb 1 θ.2 Y j k ∂(Measure.dirac (0 : HeadParam 2))) = g c) ∧
    γ 0 = 1 ∧ (∀ t ∈ Ici (0 : ℝ), HasDerivWithinAt γ (simplexDrift f g (γ t)) (Ici 0) t) ∧
    (0 : ℝ) ≤ 1 ∧ (1 / 2 : ℝ) ∈ Ioo (0 : ℝ) 1 ∧ IsSimplexConfig 1 (X 0 ()) ∧
    (∀ t ∈ Icc (0 : ℝ) 1, ∀ i, ‖X t () i‖ = 1) ∧
    ContinuousOn (fun t => X t ()) (Icc (0 : ℝ) 1) ∧
    ∀ t ∈ Icc (0 : ℝ) 1, ∀ p : DistinctPairIdx 2,
      |overlapNoiseResidual 1 (Measure.dirac 0) X 1 p.1.1 p.1.2 t ()| ≤
        (7 / 8 : ℝ) * simplexNoiseRadius 2 2 1 (1 / 2) := by
  let v : EucSpace 2 := EuclideanSpace.single (0 : Fin 2) 1
  let X : ℝ → Unit → (Idx 2 → EucSpace 2) := fun _ _ _ => v
  let f : ℝ → ℝ := fun _ => 1 / 2
  let g : ℝ → ℝ := fun _ => 1 / 2
  have hv : ‖v‖ = 1 := by simp [v]
  have hR : inner (𝕜 := ℝ) v v = 1 := by rw [real_inner_self_eq_norm_sq, hv, one_pow]
  refine ⟨X, f, g, fun _ => 1, le_rfl, isGaussianHeadLaw_dirac_zero 2,
    fun c => by norm_num [f], fun c => by norm_num [g], ?_, ?_, rfl, ?_, zero_le_one, by norm_num,
    ⟨fun i => hv, fun i j hij => hR⟩, fun t ht i => hv, continuousOn_const, ?_⟩
  · intro c Y hY i
    norm_num [f, attnProb, attnWeight, qkMap, Fin.sum_univ_two]
  · intro c Y hY i j hij
    norm_num [g, attnProb, attnWeight, qkMap, Fin.sum_univ_two]
  · intro t ht
    simpa [simplexDrift] using (hasDerivWithinAt_const t (Ici (0 : ℝ)) (1 : ℝ))
  · intro t ht p
    have hD (s : ℝ) : overlapDrift 1 (Measure.dirac (0 : HeadParam 2)) (X s ()) p.1.1 p.1.2 = 0 := by
      change overlapDrift 1 (Measure.dirac (0 : HeadParam 2)) (X s ()) p.1.1 p.1.1 = 0
      exact overlapDrift_self _ _ _ _ hv
    have hz : overlapNoiseResidual 1 (Measure.dirac 0) X 1 p.1.1 p.1.2 t () = 0 := by
      unfold overlapNoiseResidual
      rw [intervalIntegral.integral_congr (fun s hs => hD s), intervalIntegral.integral_zero]
      change inner (𝕜 := ℝ) v v - 1 - 0 = 0
      rw [hR]
      ring
    rw [hz, abs_zero]
    exact mul_nonneg (by norm_num) (Real.sqrt_nonneg _)

end Transformer.Homogenized
