module

public import Transformer.Grokking.FlatSpectrum.FeatureCodes.SignedTransport

/-!
# Probability and loss from signed directional transport

Source: `review/flat_spectrum.md`, *Signed directional transport*, ST3–ST5.
Keep training log-odds combinations signed and bound their remaining error.
These bounds use actual normalized states; they impose no query target-error
assumption and do not assert that learning supplies useful low-rank geometry.
-/

@[expose] public section
noncomputable section

namespace Transformer.Grokking.FlatSpectrum.FeatureCodes

open Finset Geometry Landscape.Neural

variable {I J C K : Type*} [Fintype I] [Fintype J] [Fintype K]

/-- Signed training log odds and directional residual allowance, ST3. -/
def signedLogOddsBound (E : C → K → ℝ) (T : I → K → ℝ) (center x : K → ℝ)
    (a : I → ℝ) (B : J → K → ℝ) (b : C → J → ℝ) (t c : C) : ℝ :=
  transportPrediction T center (fun k => E c k - E t k) a +
    directionalResidualBound B (transportResidual T center x a)
      (fun k => E c k - E t k) (b c)

/-- Every competitor's actual log odds satisfy the signed bound, ST3. -/
theorem relative_logit_le_signed (E : C → K → ℝ) (T : I → K → ℝ) (center x : K → ℝ)
    (a : I → ℝ) (B : J → K → ℝ) (b : C → J → ℝ) (t c : C) :
    linearScore (E c) x - linearScore (E t) x ≤ signedLogOddsBound E T center x a B b t c := by
  have h := signed_transport_upper T center x (fun k => E c k - E t k) a B (b c)
  simpa only [signedLogOddsBound, linearScore, sub_mul, sum_sub_distrib] using h

/-- Training-only signed score reconstruction gives a query probability floor,
ST3. The geometry still requires evaluating the query state. -/
theorem probability_ge_signed_bound [Fintype C] [DecidableEq C]
    (E : C → K → ℝ) (T : I → K → ℝ) (center x : K → ℝ)
    (a : I → ℝ) (B : J → K → ℝ) (b : C → J → ℝ) (t : C) :
    1 / (1 + oddsEnvelope (signedLogOddsBound E T center x a B b t) t) ≤
      probability (fun c => linearScore (E c) x) t := by
  apply probability_ge_of_contrast_bounds
  intro c hc
  exact relative_logit_le_signed E T center x a B b t c

/-- The matching query CE ceiling, including failed decision conditions, ST3. -/
theorem crossEntropy_le_signed_bound [Fintype C] [DecidableEq C]
    (E : C → K → ℝ) (T : I → K → ℝ) (center x : K → ℝ)
    (a : I → ℝ) (B : J → K → ℝ) (b : C → J → ℝ) (t : C) :
    crossEntropy (fun c => linearScore (E c) x) t ≤
      Real.log (1 + oddsEnvelope (signedLogOddsBound E T center x a B b t) t) := by
  apply crossEntropy_le_of_contrast_bounds
  intro c hc
  exact relative_logit_le_signed E T center x a B b t c

/-- A nonzero unseen residual is compatible with a useful signed bound, ST5.
The two training states have positive target margins; reconstruction extrapolates
beyond them. The directional allowance is exactly one quarter. -/
theorem signed_transport_nonzero_witness :
    let T : Bool → Bool → ℝ := fun i k => if k then 0 else if i then 3 / 2 else 1 / 2
    let center : Bool → ℝ := fun k => if k then 0 else 1
    let x : Bool → ℝ := fun k => if k then 1 / 4 else 3
    let w : Bool → ℝ := fun k => if k then 1 else -2
    let a : Bool → ℝ := fun i => if i then 2 else -2
    let B : Fin 1 → Bool → ℝ := fun _ k => if k then 0 else 1
    let b : Fin 1 → ℝ := fun _ => -2
    transportResidual T center x a true = 1 / 4 ∧
      transportPrediction T center w a +
        directionalResidualBound B (transportResidual T center x a) w b = -23 / 4 := by
  norm_num [transportResidual, transportPrediction, directionalResidualBound,
    readoutResidual, linearScore]

example := signed_transport_nonzero_witness
example := probability_ge_signed_bound (fun (c : Bool) (_ : Fin 1) => if c then (2 : ℝ) else 0)
  (fun (i : Bool) (_ : Fin 1) => if i then (3 : ℝ) else 1)
  (fun _ => 2) (fun _ => 5 / 2) (fun i => if i then 1 / 4 else -1 / 4)
  (fun (_ : Fin 1) (_ : Fin 1) => 1) (fun c _ => if c then 0 else -2) true
example := crossEntropy_le_signed_bound (fun (c : Bool) (_ : Fin 1) => if c then (2 : ℝ) else 0)
  (fun (i : Bool) (_ : Fin 1) => if i then (3 : ℝ) else 1)
  (fun _ => 2) (fun _ => 5 / 2) (fun i => if i then 1 / 4 else -1 / 4)
  (fun (_ : Fin 1) (_ : Fin 1) => 1) (fun c _ => if c then 0 else -2) true

end Transformer.Grokking.FlatSpectrum.FeatureCodes
