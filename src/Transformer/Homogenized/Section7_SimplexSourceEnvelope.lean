module

public import Transformer.Homogenized.Section7_SimplexOdeDomain
public import Transformer.Homogenized.Section7_SimplexNoiseReserve
public import Transformer.Homogenized.Section7_SimplexPathComparison

/-!
# The original simplex comparison envelope with its finite dimension bias

ArXiv:2604.01978v1, §7, eq:Gronwall_E_selfcontained and
thm:clustering_random_init.
Combine the actual Gram drift comparison, the actual initial
simplex realization, the proved ODE domain and the noise reserve.
The resulting deterministic estimate has exactly the source's
displayed radius and universal exponential coefficient 82.

Only the original overlap decomposition, its time integrability,
continuous unit token paths and a bound on the actual noise path
remain explicit inputs. In particular no intermediate reference
simplex, ODE invariance or drift stability estimate is assumed.
The stochastic probability step is proved downstream in
Section7_ClusteringProbability using this pathwise comparison.
-/

@[expose] public section

open scoped BigOperators NNReal
open MeasureTheory Set

namespace Transformer.Homogenized

/-- The actual pathwise error fits the source's original envelope with
universal coefficient 82 on the original noise-bound event.
Source: arXiv:2604.01978v1, §7, thm:clustering_random_init.
The nonzero 3T/d simplex drift bias is retained and absorbed using
the improved noise constant. The decomposition and noise bound are
explicit deterministic inputs, not conclusions about an SDE law. -/
theorem simplexOverlapError_le_source_envelope {d n : ℕ} (hd : 2 ≤ d) (hn : 2 ≤ n + 1)
    (β : ℝ) (σV σA : ℝ≥0) (ρ : Measure (HeadParam d)) (hρ : IsGaussianHeadLaw d σV σA ρ)
    (f g : ℝ → ℝ)
    (hf : ∀ (c : ℝ) (Y : Idx (n + 1) → EucSpace d), IsSimplexConfig c Y → ∀ i,
      (∫ θ, ∑ k, attnProb β θ.2 Y i k ^ 2 ∂ρ) = f c)
    (hg : ∀ (c : ℝ) (Y : Idx (n + 1) → EucSpace d), IsSimplexConfig c Y →
      ∀ i j, i ≠ j → (∫ θ, ∑ k, attnProb β θ.2 Y i k * attnProb β θ.2 Y j k ∂ρ) = g c)
    (γ₀ : ℝ) (hγ₀ : γ₀ ∈ Ioo (-(1 / (((n + 1 : ℕ) : ℝ) - 1))) 1)
    (γ : ℝ → ℝ) (hinit : γ 0 = γ₀)
    (hdγ : ∀ t ∈ Ici (0 : ℝ), HasDerivWithinAt γ (simplexDrift f g (γ t)) (Ici 0) t)
    (T : ℝ) (hT : 0 ≤ T) (δ : ℝ) (hδ : δ ∈ Ioo (0 : ℝ) 1)
    (X : ℝ → Idx (n + 1) → EucSpace d) (hX₀ : IsSimplexConfig γ₀ (X 0))
    (hX : ∀ t ∈ Icc (0 : ℝ) T, ∀ i, ‖X t i‖ = 1)
    (hcontX : ContinuousOn X (Icc (0 : ℝ) T))
    (M : ℝ → DistinctPairIdx (n + 1) → ℝ)
    (hdecomp : ∀ t ∈ Icc (0 : ℝ) T, ∀ p : DistinctPairIdx (n + 1),
      inner (𝕜 := ℝ) (X t p.1.1) (X t p.1.2) - γ t =
        M t p + ∫ s in (0 : ℝ)..t, overlapDrift β ρ (X s) p.1.1 p.1.2 - simplexDrift f g (γ s))
    (hM : ∀ t ∈ Icc (0 : ℝ) T, ∀ p,
      |M t p| ≤ (7 / 8 : ℝ) * simplexNoiseRadius d (n + 1) T δ)
    (hV : ∀ p : DistinctPairIdx (n + 1), IntervalIntegrable
      (fun s => overlapDrift β ρ (X s) p.1.1 p.1.2 - simplexDrift f g (γ s)) volume 0 T) :
    ∀ t ∈ Icc (0 : ℝ) T, simplexOverlapError (X t) (γ t) ≤
      Real.exp (82 * (1 + (d : ℝ) * (σA : ℝ) ^ 4 * β ^ 2 * ((n + 1 : ℕ) : ℝ)) * T) *
        simplexNoiseRadius d (n + 1) T δ := by
  have : IsProbabilityMeasure ρ := isProbabilityMeasure_of_isGaussianHeadLaw hρ
  have hcontγ : ContinuousOn γ (Icc (0 : ℝ) T) := fun t ht =>
    (hdγ t ht.1).continuousWithinAt.mono (fun s hs => hs.1)
  have hY (t : ℝ) (ht : t ∈ Icc (0 : ℝ) T) :
      IsSimplexConfig (γ t) (simplexRecenter (X 0) γ₀ (γ t)) :=
    isSimplexConfig_ode_reference hn β ρ f g hf hg hγ₀ hX₀ γ hinit hdγ t ht.1
  have hR : 0 ≤ (7 / 8 : ℝ) * simplexNoiseRadius d (n + 1) T δ :=
    mul_nonneg (by norm_num) (Real.sqrt_nonneg _)
  have h := simplexOverlapError_le_of_decomposition hd β σV σA ρ hρ f g hf hg T _ hT hR
    X (fun t => simplexRecenter (X 0) γ₀ (γ t)) γ hX hY hcontX hcontγ M hdecomp hM hV
  intro t ht
  exact (h t ht).trans (simplex_comparison_envelope d (n + 1) hd hn T hT δ hδ
    ((d : ℝ) * (σA : ℝ) ^ 4 * β ^ 2 * ((n + 1 : ℕ) : ℝ)) (by positivity))

/-- A nonconstant original scalar solution and static orthonormal tokens
satisfy every deterministic premise. The error is nonzero for t>0;
the noise path is zero and the original overlap drift is zero in d=2. -/
example : ∀ t ∈ Icc (0 : ℝ) 1,
    simplexOverlapError (fun k : Idx 2 => (EuclideanSpace.single k 1 : EucSpace 2))
      ((Real.exp t - 1) / (Real.exp t + 1)) ≤
        Real.exp (82 * (1 + (2 : ℝ) * (0 : ℝ≥0) ^ 4 * 1 ^ 2 * (2 : ℝ)) * 1) *
          simplexNoiseRadius 2 2 1 (1 / 2) := by
  let X : ℝ → Idx 2 → EucSpace 2 := fun _ k => EuclideanSpace.single k 1
  let γ : ℝ → ℝ := fun t => (Real.exp t - 1) / (Real.exp t + 1)
  let f : ℝ → ℝ := fun _ => 1 / 2
  let g : ℝ → ℝ := fun _ => 1 / 2
  have hγ : Continuous γ := continuous_iff_continuousAt.mpr
    (fun t => (hasDerivAt_simplex_uniform_attention t).continuousAt)
  have hB : Continuous (fun t => simplexDrift f g (γ t)) := by
    dsimp [simplexDrift, f, g]
    fun_prop
  have hzero : γ 0 = 0 := by simp [γ]
  have hD (s : ℝ) (p : DistinctPairIdx 2) :
      overlapDrift 1 (Measure.dirac (0 : HeadParam 2)) (X s) p.1.1 p.1.2 = 0 := by
    norm_num [overlapDrift, X, EuclideanSpace.inner_single_left, p.2.symm, PiLp.single_apply]
  have hFTC (t : ℝ) : (∫ s in (0 : ℝ)..t, simplexDrift f g (γ s)) = γ t := by
    simpa [γ, f, g] using intervalIntegral.integral_eq_sub_of_hasDerivAt
      (fun s hs => hasDerivAt_simplex_uniform_attention s) (hB.intervalIntegrable 0 t)
  apply simplexOverlapError_le_source_envelope (d := 2) (n := 1) le_rfl le_rfl
    1 0 0 (Measure.dirac 0) (isGaussianHeadLaw_dirac_zero 2) f g
    (γ₀ := 0) (γ := γ) (T := 1) (δ := 1 / 2) (X := X) (M := fun _ _ => 0)
  · intro c Y hY i
    norm_num [f, attnProb, attnWeight, qkMap, Fin.sum_univ_two]
  · intro c Y hY i j hij
    norm_num [g, attnProb, attnWeight, qkMap, Fin.sum_univ_two]
  · norm_num
  · exact hzero
  · intro t ht
    exact (hasDerivAt_simplex_uniform_attention t).hasDerivWithinAt
  · norm_num
  · norm_num
  · constructor
    · intro i; simp [X]
    · intro i j hij; simp [X, EuclideanSpace.inner_single_left, hij.symm]
  · intro t ht i; simp [X]
  · exact continuousOn_const
  · intro t ht p
    have hint : (∫ s in (0 : ℝ)..t, overlapDrift 1 (Measure.dirac (0 : HeadParam 2)) (X s)
        p.1.1 p.1.2 - simplexDrift f g (γ s)) = -γ t := by
      rw [intervalIntegral.integral_congr (fun s hs => by rw [hD s p, zero_sub]),
        intervalIntegral.integral_neg, hFTC]
    rw [hint]
    norm_num [X, EuclideanSpace.inner_single_left, p.2.symm, PiLp.single_apply]
  · intro t ht p
    have hr : 0 ≤ (7 / 8 : ℝ) * simplexNoiseRadius 2 2 1 (1 / 2) :=
      mul_nonneg (by norm_num) (Real.sqrt_nonneg _)
    simpa only [abs_zero] using hr
  · intro p
    have hneg : Continuous (fun t => -simplexDrift f g (γ t)) := hB.neg
    simpa only [hD, zero_sub] using hneg.intervalIntegrable 0 1

/-- The preceding witness has genuinely positive error at time one.
Thus the deterministic example does not collapse to identical actual
and reference Gram matrices. Source: arXiv:2604.01978v1, §7,
the overlap error E(t) in eq:Gronwall_E_selfcontained. -/
example : let Z : Idx 2 → EucSpace 2 := fun k => EuclideanSpace.single k 1
    let c := (Real.exp 1 - 1) / (Real.exp 1 + 1)
    (∀ k, ‖Z k‖ = 1) ∧ 0 < c ∧ 0 < simplexOverlapError Z c := by
  dsimp only
  let Z : Idx 2 → EucSpace 2 := fun k => EuclideanSpace.single k 1
  let c := (Real.exp 1 - 1) / (Real.exp 1 + 1)
  change (∀ k, ‖Z k‖ = 1) ∧ 0 < c ∧ 0 < simplexOverlapError Z c
  have hc : 0 < c := div_pos
    (sub_pos.mpr (Real.one_lt_exp_iff.mpr (by norm_num))) (by positivity)
  have hpair := abs_overlap_sub_simplex_le_error Z c 0 1
    (by decide)
  have habs : |inner (𝕜 := ℝ) (Z 0) (Z 1) - c| = c := by
    simp [Z, EuclideanSpace.inner_single_left, abs_of_pos hc]
  rw [habs] at hpair
  exact ⟨fun k => by simp [Z], hc, hc.trans_le hpair⟩

end Transformer.Homogenized
