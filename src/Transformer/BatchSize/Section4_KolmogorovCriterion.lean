/-
# Converting actual fourth moments to Kolmogorov's condition

arXiv:2506.12543v1, Section 4.3, equations (2)--(3), Theorem 1.
This bridge keeps integrability explicit when passing from Bochner
moment estimates to the extended-distance condition for continuous paths.
-/

import Transformer.BatchSize.Section4_EulerTimeChange
import BrownianMotion.Continuity.KolmogorovChentsov

open MeasureTheory ProbabilityTheory Filter
open scoped NNReal ENNReal

noncomputable section

namespace Transformer.BatchSize

/-- A measurable process with quadratic fourth-moment increments
satisfies the genuine Kolmogorov condition with p=4 and q=2,
Section 4.3, Theorem 1. -/
theorem isKolmogorovProcess_of_fourthMoment {Ω E : Type*} [MeasurableSpace Ω]
    [NormedAddCommGroup E] [MeasurableSpace E] [BorelSpace E] [SecondCountableTopology E]
    (P : Measure Ω) (X : ℝ≥0 → Ω → E) (hX : ∀ t, Measurable (X t)) (C : ℝ≥0)
    (hbound : ∀ s v : ℝ≥0, s ≤ v →
      Integrable (fun ω => ‖X v ω - X s ω‖ ^ 4) P ∧
        (∫ ω, ‖X v ω - X s ω‖ ^ 4 ∂P) ≤ C * ((v : ℝ) - s) ^ 2) :
    IsKolmogorovProcess X P 4 2 C := by
  have hfour (x y : E) : edist x y ^ (4 : ℝ) = ENNReal.ofReal (‖x - y‖ ^ 4) := by
    rw [show (4 : ℝ) = (4 : ℕ) by rfl, ENNReal.rpow_natCast, edist_dist, dist_eq_norm,
      ← ENNReal.ofReal_pow (norm_nonneg _)]
  have hordered (s v : ℝ≥0) (hsv : s ≤ v) :
      (∫⁻ ω, edist (X v ω) (X s ω) ^ (4 : ℝ) ∂P) ≤ (C : ℝ≥0∞) * edist v s ^ (2 : ℝ) := by
    obtain ⟨hint, hb⟩ := hbound s v hsv
    have hδ : 0 ≤ (v : ℝ) - s := sub_nonneg.mpr (NNReal.coe_le_coe.mpr hsv)
    have htwo : edist v s ^ (2 : ℝ) = ENNReal.ofReal (((v : ℝ) - s) ^ 2) := by
      rw [show (2 : ℝ) = (2 : ℕ) by rfl, ENNReal.rpow_natCast, edist_dist, NNReal.dist_eq,
        abs_of_nonneg hδ, ← ENNReal.ofReal_pow hδ]
    calc
      _ = ENNReal.ofReal (∫ ω, ‖X v ω - X s ω‖ ^ 4 ∂P) := by
        simp_rw [hfour]
        exact (ofReal_integral_eq_lintegral_ofReal hint (Eventually.of_forall fun _ => by positivity)).symm
      _ ≤ ENNReal.ofReal (C * ((v : ℝ) - s) ^ 2) := ENNReal.ofReal_le_ofReal hb
      _ = _ := by rw [htwo, ENNReal.ofReal_mul C.coe_nonneg, ENNReal.ofReal_coe_nnreal]
  refine ⟨measurable_pair_of_measurable hX, ?_, by norm_num, by norm_num⟩
  intro s t
  rcases le_total t s with hts | hst
  · exact hordered t s hts
  · simpa only [edist_comm] using hordered s t hst

/-- Joint nonvacuity of the fourth-moment criterion, Section 4.3:
a nonzero constant process under the actual Brownian probability law. -/
example : (∀ t : ℝ≥0, Measurable ((fun _ : ℝ≥0 => fun _ : BrownianSample 1 => (1 : ℝ)) t)) ∧
    (∀ s v : ℝ≥0, s ≤ v →
      Integrable (fun ω : BrownianSample 1 => ‖(fun _ : ℝ≥0 => fun _ : BrownianSample 1 => (1 : ℝ)) v ω - 1‖ ^ 4)
        (brownianNoiseLaw 1) ∧
      (∫ ω : BrownianSample 1, ‖(fun _ : ℝ≥0 => fun _ : BrownianSample 1 => (1 : ℝ)) v ω - 1‖ ^ 4
        ∂brownianNoiseLaw 1) ≤ (0 : ℝ≥0) * ((v : ℝ) - s) ^ 2) := by
  refine ⟨fun _ => measurable_const, fun _ _ _ => ?_⟩
  simp only [sub_self, norm_zero, zero_pow (by norm_num : (4 : ℕ) ≠ 0),
    NNReal.coe_zero, zero_mul, integral_zero, le_refl, and_true]
  exact integrable_const 0

end Transformer.BatchSize
