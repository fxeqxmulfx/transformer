module

public import Transformer.Grokking.GradientEvidence.ProbabilityMotion
public import Transformer.Grokking.GradientEvidence.FeatureMetricControl

/-!
# Feed checkpoint probability ceilings into the actual CE step condition

Source: `review/flat_spectrum.md`, *Prospective probability control*, PM4.
Old logits and bounded contrast motion replace the future-probability premise.
The conclusion controls the partial-CE reference from the future state.
It does not silently remove the full AdamW/encoder perturbations of FM3.
-/

@[expose] public section
noncomputable section
namespace Transformer.Grokking.GradientEvidence

open Finset Geometry FlatSpectrum.FeatureCodes
variable {I C K U : Type*} [Fintype I] [Fintype C] [Fintype K]
  [Fintype U] [Nonempty U] [DecidableEq C] [DecidableEq K]

omit [Fintype C] [Fintype U] [Nonempty U] [DecidableEq C] [DecidableEq K] in
/-- PM4: a probability cap q, metric cap R, feature-energy cap M and common
clipping at most one give the prospective coefficient q*R*M. -/
theorem metric_probability_step_condition (r : K → ℝ) (T : I → K → ℝ)
    (s q R M clip : ℝ) (hs : 0 ≤ s) (hq : 0 ≤ q) (hR : 0 ≤ R) (hM : 0 ≤ M)
    (hr : ∀ k, r k ≤ R) (hclip : 0 ≤ clip) (hclip1 : clip ≤ 1)
    (hfeature : matrixEnergy T/Fintype.card I ≤ M) (hstep : 19*s*(q*(R*M)) ≤ 2) :
    19*s*(clip*(q*(metricFeatureEnergy r T/Fintype.card I))) ≤ 2 := by
  have h := metric_uniform_step_condition r T (s*q) R M clip (mul_nonneg hs hq)
    hR hM hr hclip hclip1 hfeature (by nlinarith [hstep])
  nlinarith [h]

/-- PM4: small checkpoint probabilities remain usable for an actual future
partial-CE step when declared contrast motion and feature energy permit it.
The state E',T' may result from full tied training; the concluding step is
the explicitly written partial-CE reference with arbitrary saved momentum. -/
theorem metric_unused_ce_from_checkpoint (T T' : I → K → ℝ) (y' : I → C)
    (E E' : C → K → ℝ) (classes : U → C) (m : U × K → ℝ) (r : K → ℝ)
    (s q D R M clip : ℝ) (hr : ∀ k, 0 < r k) (hu : ∀ u i, classes u ≠ y' i)
    (hq : ∀ u i, probability (fun c => linearScore (E c) (T i)) (classes u) ≤ q)
    (hq0 : 0 ≤ q) (hq1 : q ≤ 1)
    (hmotion : ∀ u i c, c ≠ classes u →
      (linearScore (E' (classes u)) (T' i)-linearScore (E' c) (T' i))-
      (linearScore (E (classes u)) (T i)-linearScore (E c) (T i)) ≤ D)
    (hs : 0 ≤ s) (hR : 0 ≤ R) (hM : 0 ≤ M) (hmax : ∀ k, r k ≤ R)
    (hclip : 0 ≤ clip) (hclip1 : clip ≤ 1)
    (hfeature : matrixEnergy T'/Fintype.card I ≤ M)
    (hstep : 19*s*(probabilityCeiling q D*(R*M)) ≤ 2) :
    let w := unusedCoordinates E' classes
    let g := fun j => clip*unusedCoordinates (readoutGradient T' y' E') classes j
    metricMomentumPotential (fun j => r j.2) s
      (fun j => w j-s*r j.2*mainFirst (m j) (g j)) (fun j => mainFirst (m j) (g j)) ≤
        metricMomentumPotential (fun j => r j.2) s w m := by
  apply metric_unused_ce_nonincrease T' y' E' classes m r s (probabilityCeiling q D)
    clip hr hu
  · intro u i
    exact probability_le_checkpoint_ceiling _ _ (classes u) q D (hq u i) hq1 (hmotion u i)
  · exact hs
  · exact hclip
  · exact metric_probability_step_condition r T' s (probabilityCeiling q D) R M clip hs
      (probabilityCeiling_bounds q D hq0 hq1).1 hR hM hmax hclip hclip1 hfeature hstep

example := metric_probability_step_condition (I := Fin 1) (K := Bool)
  (fun k => if k then 2 else 1) (fun _ k => if k then 2 else 1)
  (1/100) (1/5) 2 5 (1/2)
  (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  (by intro k; cases k <;> norm_num) (by norm_num) (by norm_num)
  (by norm_num [matrixEnergy]) (by norm_num)
example := metric_unused_ce_from_checkpoint (I := Fin 1) (C := Fin 3) (K := Fin 1)
  (fun _ _ => 1) (fun _ _ => 1) (fun _ => 0)
  (fun c _ => c.val) (fun c _ => c.val) (fun u : Fin 2 => u.succ)
  (fun j => if j.1.val=0 then -1 else 2) (fun _ => 2) (1/100) 1 0 2 1 (1/2)
  (by intro k; norm_num) (by intro u i; exact Fin.succ_ne_zero u)
  (by intro u i; exact probability_le_one _ _) (by norm_num) (by norm_num)
  (by intro u i c hc; simp) (by norm_num) (by norm_num) (by norm_num)
  (by intro k; norm_num) (by norm_num) (by norm_num)
  (by norm_num [matrixEnergy]) (by norm_num [probabilityCeiling])

end Transformer.Grokking.GradientEvidence
