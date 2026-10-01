/-
# Batch-size sensitivity and the corrected local approximation

arXiv:2506.12543v1, Section 4.3's takeaway following Theorem 1.
The sqrt(B) effect is the derivative at zero signal, not an exact law
for finite signal. Saturation is approached as B tends to infinity.
-/

import Transformer.BatchSize.Section4_Coefficients

open Filter
open scoped Topology

noncomputable section

namespace Transformer.BatchSize

/-- Signed drift vanishes at zero true gradient, Section 4.3, equation (3). -/
theorem signResponse_zero (B : ℕ) (σ : ℝ) : signResponse B σ 0 = 0 := by
  simp [signResponse, errorFunction_zero]

/-- Negative signals have opposite signed drift, Section 4.3's proof sketch. -/
theorem signResponse_neg (B : ℕ) (σ g : ℝ) :
    signResponse B σ (-g) = -signResponse B σ g := by
  simp only [signResponse, mul_neg, neg_div, errorFunction_neg]

/-- The exact zero-signal derivative gives the paper's sqrt(B) effect.
It corrects the claim of exact linearity; Section 4.3's takeaway. -/
theorem signResponse_hasDerivAt_zero (B : ℕ) (σ : ℝ) :
    HasDerivAt (signResponse B σ)
      ((2 / Real.sqrt Real.pi) * (Real.sqrt ((B : ℝ) / 2) / σ)) 0 := by
  change HasDerivAt (fun g => errorFunction (Real.sqrt ((B : ℝ) / 2) * g / σ)) _ 0
  have hlin := ((hasDerivAt_id (0 : ℝ)).const_mul
    (Real.sqrt ((B : ℝ) / 2))).div_const σ
  have h := (errorFunction_hasDerivAt (Real.sqrt ((B : ℝ) / 2) * 0 / σ)).comp 0 hlin
  simpa [Function.comp_def] using h

/-- Precise local linearization of the signed drift; Section 4.3's takeaway. -/
theorem signResponse_linearization (B : ℕ) (σ : ℝ) :
    Tendsto (fun g => signResponse B σ g / g) (𝓝[≠] 0)
      (𝓝 ((2 / Real.sqrt Real.pi) * (Real.sqrt ((B : ℝ) / 2) / σ))) := by
  simpa [signResponse_zero, div_eq_mul_inv, mul_comm] using
    (signResponse_hasDerivAt_zero B σ).tendsto_slope_zero

/-- At a fixed positive signal, increasing B strictly increases the
signed drift magnitude; Section 4.3's takeaway. -/
theorem signResponse_strictMono_batch (σ g : ℝ) (hσ : 0 < σ) (hg : 0 < g) :
    StrictMono (fun B : ℕ => signResponse B σ g) := by
  intro B C hBC
  apply errorFunction_strictMono
  apply (div_lt_div_iff_of_pos_right hσ).2
  apply mul_lt_mul_of_pos_right _ hg
  apply Real.sqrt_lt_sqrt (by positivity)
  have hcast : (B : ℝ) < C := by exact_mod_cast hBC
  linarith

/-- Nonvacuity of the drift-acceleration regime, Section 4.3. -/
example : (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 / 1000 := by norm_num

/-- Drift saturation at fixed positive signal, Section 4.3's takeaway. -/
theorem signResponse_tendsto_one (σ g : ℝ) (hσ : 0 < σ) (hg : 0 < g) :
    Tendsto (fun B : ℕ => signResponse B σ g) atTop (𝓝 1) := by
  have hcast : Tendsto (fun B : ℕ => (B : ℝ)) atTop atTop := tendsto_natCast_atTop_atTop
  have hB : Tendsto (fun B : ℕ => (B : ℝ) / 2) atTop atTop :=
    hcast.atTop_div_const (by norm_num)
  exact errorFunction_tendsto_atTop.comp
    (((Real.tendsto_sqrt_atTop.comp hB).atTop_mul_const hg).atTop_div_const hσ)

/-- Nonvacuity of the large-batch limit, Section 4.3. -/
example : (0 : ℝ) < 1 ∧ (0 : ℝ) < 2 := by norm_num

/-- For either sign of the true gradient, the large-batch response
tends to its noiseless sign; Section 4.3's takeaway. -/
theorem signResponse_tendsto_sign (σ g : ℝ) (hσ : 0 < σ) :
    Tendsto (fun B : ℕ => signResponse B σ g) atTop (𝓝 (Real.sign g)) := by
  rcases lt_trichotomy g 0 with hg | rfl | hg
  · have h := (signResponse_tendsto_one σ (-g) hσ (neg_pos.2 hg)).neg
    simpa only [signResponse_neg, neg_neg, Real.sign_of_neg hg] using h
  · simpa only [signResponse_zero, Real.sign_zero] using
      (tendsto_const_nhds : Tendsto (fun _ : ℕ => (0 : ℝ)) atTop (𝓝 0))
  · simpa only [Real.sign_of_pos hg] using signResponse_tendsto_one σ g hσ hg

/-- Nonvacuity of the full signed limit, Section 4.3. -/
example : (0 : ℝ) < 1 := by norm_num

end Transformer.BatchSize
