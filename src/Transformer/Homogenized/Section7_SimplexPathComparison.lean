module

public import Transformer.Homogenized.Section7_SimplexError
public import Transformer.Homogenized.Section7_SimplexGronwall
public import Mathlib.MeasureTheory.SpecificCodomains.Pi

/-!
# Pathwise simplex comparison with the original drift and noise

ArXiv:2604.01978v1, §7, eq:Gronwall_E_selfcontained and the final
step of the proof of thm:clustering_random_init.
The original overlap errors, actual drift differences and actual
noise values retain the original distinct token-pair indices.
Finite-coordinate integration is justified from the actual drift
integrability before applying the proved Grönwall estimate.

The comparison keeps the finite dimension term 3T/d in addition
to the bounded noise path. A realized simplex reference and
continuous token and scalar paths are explicit geometric inputs.
The event inclusion then transfers an established noise-event
measure bound to the corresponding uniform error event.
-/

@[expose] public section

open MeasureTheory Set
open scoped BigOperators NNReal

namespace Transformer.Homogenized

/-- The actual original overlap error is uniformly bounded by
(r + 3T/d) exp(80(1+dσA⁴β²n)T) when its actual noise path is bounded by r.
Source: arXiv:2604.01978v1, §7, eq:Gronwall_E_selfcontained.
The source's exact-drift substitution is corrected by retaining 3T/d.
The original pathwise decomposition and its integrability are explicit
inputs; all drift bounds are proved from the original Gaussian head law. -/
theorem simplexOverlapError_le_of_decomposition {d n : ℕ} (hd : 2 ≤ d) (β : ℝ) (σV σA : ℝ≥0)
    (ρ : Measure (HeadParam d)) (hρ : IsGaussianHeadLaw d σV σA ρ) (f g : ℝ → ℝ)
    (hf : ∀ (c : ℝ) (x : Idx (n + 1) → EucSpace d), IsSimplexConfig c x → ∀ k : Idx (n + 1),
      (∫ θ, ∑ l : Idx (n + 1), attnProb β θ.2 x k l ^ 2 ∂ρ) = f c)
    (hg : ∀ (c : ℝ) (x : Idx (n + 1) → EucSpace d), IsSimplexConfig c x →
      ∀ k l : Idx (n + 1), k ≠ l →
      (∫ θ, ∑ m : Idx (n + 1), attnProb β θ.2 x k m * attnProb β θ.2 x l m ∂ρ) = g c)
    (T r : ℝ) (hT : 0 ≤ T) (hr : 0 ≤ r)
    (X Y : ℝ → Idx (n + 1) → EucSpace d) (γ : ℝ → ℝ)
    (hX : ∀ t ∈ Icc (0 : ℝ) T, ∀ i, ‖X t i‖ = 1)
    (hY : ∀ t ∈ Icc (0 : ℝ) T, IsSimplexConfig (γ t) (Y t))
    (hcontX : ContinuousOn X (Icc (0 : ℝ) T)) (hcontγ : ContinuousOn γ (Icc (0 : ℝ) T))
    (M : ℝ → DistinctPairIdx (n + 1) → ℝ)
    (hdecomp : ∀ t ∈ Icc (0 : ℝ) T, ∀ p : DistinctPairIdx (n + 1),
      inner (𝕜 := ℝ) (X t p.1.1) (X t p.1.2) - γ t =
        M t p + ∫ s in (0 : ℝ)..t, overlapDrift β ρ (X s) p.1.1 p.1.2 - simplexDrift f g (γ s))
    (hM : ∀ t ∈ Icc (0 : ℝ) T, ∀ p, |M t p| ≤ r)
    (hV : ∀ p : DistinctPairIdx (n + 1), IntervalIntegrable
      (fun s => overlapDrift β ρ (X s) p.1.1 p.1.2 - simplexDrift f g (γ s)) volume 0 T) :
    ∀ t ∈ Icc (0 : ℝ) T, simplexOverlapError (X t) (γ t) ≤
      (r + 3 * T / (d : ℝ)) *
        Real.exp ((80 * (1 + (d : ℝ) * (σA : ℝ) ^ 4 * β ^ 2 * ((n + 1 : ℕ) : ℝ))) * T) := by
  let F (t : ℝ) (p : DistinctPairIdx (n + 1)) :=
    inner (𝕜 := ℝ) (X t p.1.1) (X t p.1.2) - γ t
  let V (t : ℝ) (p : DistinctPairIdx (n + 1)) :=
    overlapDrift β ρ (X t) p.1.1 p.1.2 - simplexDrift f g (γ t)
  let L := 80 * (1 + (d : ℝ) * (σA : ℝ) ^ 4 * β ^ 2 * ((n + 1 : ℕ) : ℝ))
  have hL : 0 ≤ L := by dsimp [L]; positivity
  have hb : 0 ≤ 3 / (d : ℝ) := by positivity
  have hF : ContinuousOn F (Icc (0 : ℝ) T) := continuousOn_pi.2 fun p =>
    (((continuous_apply p.1.1).comp_continuousOn hcontX).inner
      ((continuous_apply p.1.2).comp_continuousOn hcontX)).sub hcontγ
  have hiV : IntervalIntegrable V volume 0 T :=
    ⟨integrable_pi_iff.mpr fun p => (hV p).1, integrable_pi_iff.mpr fun p => (hV p).2⟩
  have hFdecomp (t : ℝ) (ht : t ∈ Icc (0 : ℝ) T) : F t = M t + ∫ s in (0 : ℝ)..t, V s := by
    have hsub : uIcc (0 : ℝ) t ⊆ uIcc (0 : ℝ) T := by
      rw [uIcc_of_le ht.1, uIcc_of_le hT]
      exact fun s hs => ⟨hs.1, hs.2.trans ht.2⟩
    funext p
    change inner (𝕜 := ℝ) (X t p.1.1) (X t p.1.2) - γ t = M t p + (∫ s in (0 : ℝ)..t, V s) p
    have heval := ((ContinuousLinearMap.proj (R := ℝ) p :
      (DistinctPairIdx (n + 1) → ℝ) →L[ℝ] ℝ).intervalIntegral_comp_comm (hiV.mono_set hsub)).symm
    change (∫ s in (0 : ℝ)..t, V s) p = ∫ s in (0 : ℝ)..t, V s p at heval
    rw [heval]
    exact hdecomp t ht p
  have hMb (t : ℝ) (ht : t ∈ Icc (0 : ℝ) T) : ‖M t‖ ≤ r := by
    apply (pi_norm_le_iff_of_nonneg hr).2
    intro p
    rw [Real.norm_eq_abs]
    exact hM t ht p
  have hVb (t : ℝ) (ht : t ∈ Icc (0 : ℝ) T) : ‖V t‖ ≤ L * ‖F t‖ + 3 / (d : ℝ) := by
    apply (pi_norm_le_iff_of_nonneg (add_nonneg (mul_nonneg hL (norm_nonneg _)) hb)).2
    intro p
    rw [Real.norm_eq_abs]
    exact abs_overlapDrift_sub_simplexDrift_le_error hd β σV σA ρ hρ f g hf hg
      (X t) (Y t) (hX t ht) (γ t) (hY t ht) p.1.1 p.1.2 p.2
  have h := norm_error_le_exp_uniform_of_decomposition F M V T r (3 / (d : ℝ)) L
    hT hr hb hL hF hiV hFdecomp hMb hVb
  intro t ht
  change ‖F t‖ ≤ (r + 3 * T / (d : ℝ)) * Real.exp (L * T)
  apply (h t ht).trans_eq
  congr 1
  ring

/-- The proved path comparison transfers the measure of the original
noise-bound event to the original error-bound event.
Source: arXiv:2604.01978v1, §7, the final probability step following
eq:Gronwall_E_selfcontained. The event inclusion is proved path by path. -/
theorem measure_noise_bound_le_error_bound {E Ω : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    [MeasurableSpace Ω] (P : Measure Ω) (F M V : Ω → ℝ → E) (T r b L : ℝ)
    (hT : 0 ≤ T) (hr : 0 ≤ r) (hb : 0 ≤ b) (hL : 0 ≤ L)
    (hF : ∀ ω, ContinuousOn (F ω) (Icc (0 : ℝ) T))
    (hV : ∀ ω, IntervalIntegrable (V ω) volume 0 T)
    (hdecomp : ∀ ω, ∀ t ∈ Icc (0 : ℝ) T, F ω t = M ω t + ∫ s in (0 : ℝ)..t, V ω s)
    (hbound : ∀ ω, ∀ t ∈ Icc (0 : ℝ) T, ‖V ω t‖ ≤ L * ‖F ω t‖ + b) :
    P {ω | ∀ t ∈ Icc (0 : ℝ) T, ‖M ω t‖ ≤ r} ≤
      P {ω | ∀ t ∈ Icc (0 : ℝ) T, ‖F ω t‖ ≤ (r + b * T) * Real.exp (L * T)} := by
  apply measure_mono
  intro ω hM
  exact norm_error_le_exp_uniform_of_decomposition (F ω) (M ω) (V ω) T r b L hT hr hb hL
    (hF ω) (hV ω) (hdecomp ω) hM (hbound ω)

/-- Positive-variance Gaussian heads and consensus unit tokens realize
the actual reference, moment, continuity and decomposition inputs. -/
example : ∀ t ∈ Icc (0 : ℝ) 1,
    simplexOverlapError (fun _ : Idx 2 => (EuclideanSpace.single (0 : Fin 2) 1 : EucSpace 2)) 1 ≤
      (0 + 3 * 1 / (2 : ℝ)) *
        Real.exp ((80 * (1 + (2 : ℝ) * (1 : ℝ≥0) ^ 4 * 1 ^ 2 * (2 : ℝ))) * 1) := by
  let X : ℝ → Idx 2 → EucSpace 2 := fun _ _ => EuclideanSpace.single (0 : Fin 2) 1
  obtain ⟨f, g, -, -, hf, hg⟩ :=
    simplex_overlap_wellDefined (d := 2) (n := 2) 1 1 1 _ (isGaussianHeadLaw_gaussHeadLaw 2 1 1)
  have hD (s : ℝ) (p : DistinctPairIdx 2) :
      overlapDrift 1 (gaussHeadLaw 2 1 1) (X s) p.1.1 p.1.2 - simplexDrift f g 1 = 0 := by
    dsimp [X, overlapDrift, simplexDrift]
    norm_num [EuclideanSpace.inner_single_left, PiLp.single_apply]
    ring
  apply simplexOverlapError_le_of_decomposition (d := 2) (n := 1) (by norm_num) 1 1 1 _
    (isGaussianHeadLaw_gaussHeadLaw 2 1 1) f g hf hg 1 0 (by norm_num) le_rfl X X (fun _ => 1) (M := fun _ _ => 0)
  · intro t ht i
    simp [X]
  · intro t ht
    refine ⟨fun i => by simp [X], ?_⟩
    intro i j hij
    simp [X]
  · exact continuousOn_const
  · exact continuousOn_const
  · intro t ht p
    have hint : (∫ s in (0 : ℝ)..t,
        overlapDrift 1 (gaussHeadLaw 2 1 1) (X s) p.1.1 p.1.2 - simplexDrift f g 1) = 0 := by
      rw [intervalIntegral.integral_congr (fun s hs => hD s p), intervalIntegral.integral_zero]
    rw [hint]
    simp [X]
  · intro t ht p
    norm_num
  · intro p
    simpa only [hD] using (intervalIntegrable_const : IntervalIntegrable (fun _ : ℝ => (0 : ℝ)) volume 0 1)

/-- The measure transfer also applies to a genuine positive-bias path:
F(t)=t, M(t)=0 and V(t)=1 on the full closed interval. -/
example :
    (Measure.dirac ()) (Set.ofPred (fun _ : Unit => ∀ t ∈ Icc (0 : ℝ) 1, ‖(0 : ℝ)‖ ≤ 0)) ≤
      (Measure.dirac ()) (Set.ofPred (fun _ : Unit => ∀ t ∈ Icc (0 : ℝ) 1, ‖t‖ ≤ (0 + 1 * 1) * Real.exp (0 * 1))) := by
  apply measure_noise_bound_le_error_bound (Measure.dirac ()) (fun _ t => t) (fun _ _ => (0 : ℝ))
    (fun _ _ => (1 : ℝ)) 1 0 1 0 (by norm_num) le_rfl (by norm_num) le_rfl
  · intro ω
    exact continuous_id.continuousOn
  · intro ω
    exact intervalIntegrable_const
  · intro ω t ht
    simp
  · intro ω t ht
    norm_num

end Transformer.Homogenized
