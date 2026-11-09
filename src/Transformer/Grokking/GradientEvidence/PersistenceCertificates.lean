module

public import Transformer.Grokking.GradientEvidence.DampedPersistence

/-!
# Readout energy floors and finite-window prediction transport

Source: `review/flat_spectrum.md`, *Observed finite-window persistence*, DP2–DP3.
Readout concentration and prediction correctness are separate conclusions.
The latter needs a future gap enclosure for the whole nonlinear model;
DP1 alone supplies no feature-alignment theorem.
-/

@[expose] public section
noncomputable section
namespace Transformer.Grokking.GradientEvidence

/-- DP2: lower signal energy and upper competing energy give a lower share.
Both denominators and the sign of the competing energy are explicit. -/
theorem energy_share_floor (S V L U : ℝ) (hL : 0 ≤ L) (hV : 0 ≤ V)
    (hS : L ≤ S) (hU : V ≤ U) (hd : 0 < L+U) (hactual : 0 < S+V) :
    L/(L+U) ≤ S/(S+V) := by
  rw [div_le_div_iff₀ hd hactual]
  have h1 := mul_le_mul_of_nonneg_left hU hL
  have h2 := mul_le_mul_of_nonneg_right hS (hV.trans hU)
  nlinarith

variable {E : Type*} [NormedAddCommGroup E]

/-- DP2: apply the scalar share bound to the three orthogonal class modes.
Orthogonality is needed to identify this ratio with centered matrix energy,
not for the norm-ratio inequality itself. L must be positive to certify a
nonzero share; the native implementation emits zero when it is exhausted. -/
theorem projected_norm_share_floor (active unused between : E) (L U B : ℝ)
    (hL : 0 < L) (ha : L ≤ ‖active‖) (hu : ‖unused‖ ≤ U) (hb : ‖between‖ ≤ B) :
    L^2/(L^2+U^2+B^2) ≤
      ‖active‖^2/(‖active‖^2+‖unused‖^2+‖between‖^2) := by
  have hU : 0 ≤ U := (norm_nonneg _).trans hu
  have hB : 0 ≤ B := (norm_nonneg _).trans hb
  have hA : L^2 ≤ ‖active‖^2 := by nlinarith [norm_nonneg active]
  have hN : ‖unused‖^2 + ‖between‖^2 ≤ U^2+B^2 := by
    nlinarith [norm_nonneg unused, norm_nonneg between]
  have h := energy_share_floor (‖active‖^2) (‖unused‖^2+‖between‖^2)
    (L^2) (U^2+B^2) (sq_nonneg _) (by positivity) hA hN
    (by positivity) (by nlinarith [sq_nonneg ‖unused‖, sq_nonneg ‖between‖])
  simpa only [add_assoc] using h

omit [NormedAddCommGroup E] in
/-- DP3: transport an old certified margin using a bound for the full
future rival gap. D includes both readout and feature movement. -/
theorem finite_margin_transport (M oldGap newGap upper D : ℝ)
    (hold : M ≤ -oldGap) (hfuture : newGap ≤ upper) (hchange : upper-oldGap ≤ D) :
    M-D ≤ -newGap := by
  linarith

omit [NormedAddCommGroup E] in
/-- DP3: one uniform bound on gap increase transports a certificate to
every rival. The native maximum ranges over all 96 rivals, including unused
classes. The old and new scores can be from any finite window. -/
theorem finite_window_prediction {C : Type*} (old next upper : C → ℝ)
    (target : C) (M D : ℝ)
    (hold : ∀ c, c ≠ target → M ≤ old target-old c)
    (hfuture : ∀ c, c ≠ target → next c-next target ≤ upper c)
    (hchange : ∀ c, c ≠ target → upper c-(old c-old target) ≤ D)
    (hpositive : D < M) :
    ∀ c, c ≠ target → next c < next target := by
  intro c hc
  have h := finite_margin_transport M (old c-old target) (next c-next target)
    (upper c) D (by linarith [hold c hc]) (hfuture c hc) (hchange c hc)
  linarith

example := energy_share_floor 9 5 4 8
  (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
example := projected_norm_share_floor (3 : ℝ) 2 1 2 3 2
  (by norm_num) (by norm_num) (by norm_num) (by norm_num)
example := finite_margin_transport 3 (-4) (-2) (-1) 3
  (by norm_num) (by norm_num) (by norm_num)
example := finite_window_prediction (C := Bool) (fun c => if c then 5 else 1)
  (fun c => if c then 4 else 2) (fun c => if c then 0 else -1) true (7/2) 3
  (by intro c hc; cases c <;> norm_num at *)
  (by intro c hc; cases c <;> norm_num at *)
  (by intro c hc; cases c <;> norm_num at *) (by norm_num)

end Transformer.Grokking.GradientEvidence
