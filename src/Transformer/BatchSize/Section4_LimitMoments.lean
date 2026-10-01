/-
# Lower semicontinuity of fourth moments at actual Euler limits

arXiv:2506.12543v1, Section 4.3, equations (2)--(3), Theorem 1.
Mean-square convergence yields convergence in measure. An almost-everywhere
convergent subsequence and Fatou's lemma preserve proved moment bounds.
-/

import Transformer.BatchSize.Section4_L2Limits
import Mathlib.MeasureTheory.Function.ConvergenceInMeasure

open MeasureTheory Filter
open scoped NNReal ENNReal Topology

noncomputable section

namespace Transformer.BatchSize

/-- Actual mean-square convergence implies convergence in measure,
Section 4.3, Theorem 1. Both are measured under the same probability law. -/
theorem meanSquare_tendstoInMeasure {Ω E : Type*} [MeasurableSpace Ω]
    [NormedAddCommGroup E] [InnerProductSpace ℝ E]
    (P : Measure Ω) (X : ℕ → Ω → E) (hX : ∀ n, MemLp (X n) 2 P)
    (Y : Ω → E) (hY : MemLp Y 2 P)
    (hlim : Tendsto (fun n => ∫ ω, ‖X n ω - Y ω‖ ^ 2 ∂P) atTop (𝓝 0)) :
    TendstoInMeasure P X atTop Y := by
  have hLp := meanSquare_tendsto_toLp P X hX Y hY hlim
  have hn := (tendsto_iff_norm_sub_tendsto_zero.mp hLp)
  have heq (n : ℕ) : eLpNorm (X n - Y) 2 P =
      ENNReal.ofReal ‖(hX n).toLp (X n) - hY.toLp Y‖ := by
    rw [ofReal_norm, ← (hX n).toLp_sub hY, Lp.enorm_def]
    exact (eLpNorm_congr_ae ((hX n).sub hY).coeFn_toLp).symm
  have h := (ENNReal.continuous_ofReal.tendsto (0 : ℝ)).comp hn
  simp only [Function.comp_def, ENNReal.ofReal_zero] at h
  apply tendstoInMeasure_of_tendsto_eLpNorm (by norm_num : (2 : ℝ≥0∞) ≠ 0)
  simpa only [heq] using h

/-- Differences of two genuine mean-square convergent sequences
converge to the difference of their limits, Section 4.3, Theorem 1. -/
theorem meanSquare_sub_tendsto {Ω E : Type*} [MeasurableSpace Ω]
    [NormedAddCommGroup E] [InnerProductSpace ℝ E]
    (P : Measure Ω) (X Z : ℕ → Ω → E)
    (hX : ∀ n, MemLp (X n) 2 P) (hZ : ∀ n, MemLp (Z n) 2 P)
    (Y W : Ω → E) (hY : MemLp Y 2 P) (hW : MemLp W 2 P)
    (hXY : Tendsto (fun n => ∫ ω, ‖X n ω - Y ω‖ ^ 2 ∂P) atTop (𝓝 0))
    (hZW : Tendsto (fun n => ∫ ω, ‖Z n ω - W ω‖ ^ 2 ∂P) atTop (𝓝 0)) :
    Tendsto (fun n => ∫ ω, ‖(X n ω - Z n ω) - (Y ω - W ω)‖ ^ 2 ∂P) atTop (𝓝 0) := by
  have hLp := (meanSquare_tendsto_toLp P X hX Y hY hXY).sub
    (meanSquare_tendsto_toLp P Z hZ W hW hZW)
  have hn := (tendsto_iff_norm_sub_tendsto_zero.mp hLp).pow 2
  have heq (n : ℕ) : (∫ ω, ‖(X n ω - Z n ω) - (Y ω - W ω)‖ ^ 2 ∂P) =
      ‖((hX n).toLp (X n) - (hZ n).toLp (Z n)) - (hY.toLp Y - hW.toLp W)‖ ^ 2 := by
    have h := toLp_dist_meanSquare P (X n - Z n) (Y - W) ((hX n).sub (hZ n)) (hY.sub hW)
    rw [dist_eq_norm, (hX n).toLp_sub (hZ n), hY.toLp_sub hW] at h
    exact h.symm
  simpa only [heq, zero_pow (by norm_num : (2 : ℕ) ≠ 0)] using hn

/-- Joint nonvacuity of two-sequence mean-square hypotheses,
Section 4.3: any two nonzero constant random states. -/
example : ∀ c : ℝ, (∀ n : ℕ, MemLp ((fun _ : ℕ => fun _ : BrownianSample 1 => c) n) 2
    (brownianNoiseLaw 1)) ∧ MemLp (fun _ : BrownianSample 1 => c) 2 (brownianNoiseLaw 1) ∧
    Tendsto (fun _ : ℕ => ∫ ω : BrownianSample 1, ‖(fun _ : BrownianSample 1 => c) ω - c‖ ^ 2
      ∂brownianNoiseLaw 1) atTop (𝓝 0) := fun _ =>
  ⟨fun _ => memLp_const _, memLp_const _, by simp⟩

/-- A uniform fourth-moment bound passes to an actual mean-square
limit with genuine integrability, Section 4.3, Theorem 1.
The conclusion follows from Fatou, without assuming convergence of
fourth moments or uniform boundedness of the sample paths. -/
theorem fourthMoment_le_of_meanSquare_tendsto {Ω E : Type*} [MeasurableSpace Ω]
    [NormedAddCommGroup E] [InnerProductSpace ℝ E]
    (P : Measure Ω) (X : ℕ → Ω → E) (hX2 : ∀ n, MemLp (X n) 2 P)
    (hX4 : ∀ n, MemLp (X n) 4 P) (Y : Ω → E) (hY : MemLp Y 2 P)
    (hlim : Tendsto (fun n => ∫ ω, ‖X n ω - Y ω‖ ^ 2 ∂P) atTop (𝓝 0))
    (R : ℝ) (hR : 0 ≤ R) (hbound : ∀ n, (∫ ω, ‖X n ω‖ ^ 4 ∂P) ≤ R) :
    Integrable (fun ω => ‖Y ω‖ ^ 4) P ∧ (∫ ω, ‖Y ω‖ ^ 4 ∂P) ≤ R := by
  obtain ⟨ns, _, hae⟩ := (meanSquare_tendstoInMeasure P X hX2 Y hY hlim).exists_seq_tendsto_ae
  let G (n : ℕ) (ω : Ω) : ℝ := ‖X (ns n) ω‖ ^ 4
  let F (ω : Ω) : ℝ := ‖Y ω‖ ^ 4
  have hGi (n : ℕ) : Integrable (G n) P :=
    (hX4 (ns n)).integrable_norm_pow (by norm_num : (4 : ℕ) ≠ 0)
  have hFm : AEStronglyMeasurable F P :=
    (continuous_pow 4).comp_aestronglyMeasurable hY.aestronglyMeasurable.norm
  have hformula (f : Ω → ℝ) (hi : Integrable f P) (hf : ∀ ω, 0 ≤ f ω) :
      eLpNorm f 1 P = ENNReal.ofReal (∫ ω, f ω ∂P) := by
    rw [eLpNorm_one_eq_lintegral_enorm hi.aestronglyMeasurable,
      ← ofReal_integral_norm_eq_lintegral_enorm hi]
    congr 1
    apply integral_congr_ae
    exact Eventually.of_forall fun ω => Real.norm_of_nonneg (hf ω)
  have hGbound : ∀ᶠ n in atTop, eLpNorm (G n) 1 P ≤ ENNReal.ofReal R := Eventually.of_forall fun n => by
    rw [hformula (G n) (hGi n) (fun ω => by dsimp [G]; positivity)]
    exact ENNReal.ofReal_le_ofReal (hbound (ns n))
  have hGF : ∀ᵐ ω ∂P, Tendsto (fun n => G n ω) atTop (𝓝 (F ω)) := by
    filter_upwards [hae] with ω hω
    exact hω.norm.pow 4
  have hFbound : eLpNorm F 1 P ≤ ENNReal.ofReal R := Lp.eLpNorm_le_of_ae_tendsto hGbound
    (fun n => (hGi n).aestronglyMeasurable) hFm hGF
  have hFi : Integrable F P := memLp_one_iff_integrable.mp (show MemLp F 1 P from
    lt_of_le_of_lt hFbound ENNReal.ofReal_lt_top)
  refine ⟨hFi, ?_⟩
  rw [hformula F hFi (fun ω => by dsimp [F]; positivity)] at hFbound
  exact (ENNReal.ofReal_le_ofReal_iff hR).mp hFbound

/-- Joint nonvacuity of mean-square and fourth-moment limit assumptions,
Section 4.3: a nonzero constant sequence under the genuine Brownian law. -/
example : (∀ n : ℕ, MemLp ((fun _ : ℕ => fun _ : BrownianSample 1 => (1 : ℝ)) n) 2 (brownianNoiseLaw 1)) ∧
    (∀ n : ℕ, MemLp ((fun _ : ℕ => fun _ : BrownianSample 1 => (1 : ℝ)) n) 4 (brownianNoiseLaw 1)) ∧
    MemLp (fun _ : BrownianSample 1 => (1 : ℝ)) 2 (brownianNoiseLaw 1) ∧
    Tendsto (fun _ : ℕ => ∫ ω : BrownianSample 1,
      ‖(fun _ : BrownianSample 1 => (1 : ℝ)) ω - 1‖ ^ 2 ∂brownianNoiseLaw 1) atTop (𝓝 0) ∧
    (0 : ℝ) ≤ 1 ∧
    (∀ n : ℕ, (∫ ω, ‖(fun _ : ℕ => fun _ : BrownianSample 1 => (1 : ℝ)) n ω‖ ^ 4 ∂brownianNoiseLaw 1) ≤ 1) := by
  refine ⟨fun _ => memLp_const _, fun _ => memLp_const _, memLp_const _, by simp, by norm_num, ?_⟩
  intro n
  simp

end Transformer.BatchSize
