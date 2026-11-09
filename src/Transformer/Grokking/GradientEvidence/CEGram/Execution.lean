module

public import Transformer.Grokking.GradientEvidence.CEGram.Control
public import Transformer.Grokking.GradientEvidence.CEGram.Transport
public import Transformer.Grokking.Geometry.ProbabilityMotion

/-!
# Explicit execution allowances in query-weighted CE control

Source: `review/flat_spectrum.md`, *Query-weighted feature covariance*, QG3.
GPU logits and ideal real dot products need not coincide. Likewise, an ideal
feature-motion radius needs both endpoint execution errors before controlling
executed features. These premises are explicit; finite numerical checks do not
establish a universal rounding theorem.
-/

@[expose] public section
noncomputable section
namespace Transformer.Grokking.GradientEvidence.CEGram

open Finset Geometry FlatSpectrum.FeatureCodes

/-- QG3: a per-logit execution error contributes twice its allowance to a
class contrast. Transport the GPU ceiling to the real-CE score table. -/
theorem probability_of_execution {C : Type*} [Fintype C] [DecidableEq C]
    (z z' : C → ℝ) (u : C) (q e : ℝ)
    (hq : probability z u ≤ q) (hq1 : q ≤ 1) (he : ∀ c, |z' c-z c| ≤ e) :
    probability z' u ≤ probabilityCeiling q (2*e) := by
  apply probability_le_checkpoint_ceiling z z' u q (2*e) hq hq1
  intro c hc
  have hu := (abs_le.mp (he u)).2
  have hc' := (abs_le.mp (he c)).1
  linarith

/-- QG3: old and future feature execution errors both enter the transported
radius; none of the three terms is replaced by a future observed displacement. -/
theorem executed_feature_motion {H : Type*} [SeminormedAddCommGroup H]
    (x x' y y' : H) (S e e' : ℝ) (hx : ‖x'-x‖ ≤ S)
    (hy : ‖y-x‖ ≤ e) (hy' : ‖y'-x'‖ ≤ e') : ‖y'-y‖ ≤ S+e+e' := by
  have hxy : ‖x-y‖ ≤ e := by rwa [norm_sub_rev]
  calc
    _ = ‖(y'-x')+(x'-x)+(x-y)‖ := by congr 1; abel
    _ ≤ ‖(y'-x')+(x'-x)‖+‖x-y‖ := norm_add_le _ _
    _ ≤ (‖y'-x'‖+‖x'-x‖)+‖x-y‖ := add_le_add (norm_add_le _ _) le_rfl
    _ ≤ S+e+e' := by linarith

/-- QG3: the same execution allowance in the squared-energy form used by
the coefficient transport. The feature space has its Euclidean norm. -/
theorem executed_feature_motion_sq {K : Type*} [Fintype K]
    (x x' y y' : EuclideanSpace ℝ K) (S e e' : ℝ) (hx : ‖x'-x‖ ≤ S)
    (hy : ‖y-x‖ ≤ e) (hy' : ‖y'-x'‖ ≤ e') :
    (∑ k, (y' k-y k)^2) ≤ (S+e+e')^2 := by
  have h := executed_feature_motion x x' y y' S e e' hx hy hy'
  have hb : 0 ≤ S+e+e' := (norm_nonneg _).trans h
  have hs := (sq_le_sq₀ (norm_nonneg _) hb).2 h
  rw [EuclideanSpace.real_norm_sq_eq] at hs
  exact hs

/-- QG2/QG3: use probability ceilings on executed logits only after the
explicit readout-error conversion. Features and the partial CE gradient are
unchanged; all full-optimizer perturbations still require their own control. -/
theorem executed_ce_nonincrease {I C K U : Type*}
    [Fintype I] [Fintype C] [Fintype K] [Fintype U] [Nonempty U]
    [DecidableEq I] [DecidableEq C] [DecidableEq K]
    (T : I → K → ℝ) (y : I → C) (E : C → K → ℝ) (z : I → C → ℝ)
    (classes : U → C) (m : U × K → ℝ) (r : K → ℝ) (s L clip e : ℝ) (q : I → ℝ)
    (hr : ∀ k, 0 < r k) (hu : ∀ u i, classes u ≠ y i)
    (hq : ∀ u i, probability (z i) (classes u) ≤ q i) (hq1 : ∀ i, q i ≤ 1)
    (he : ∀ i c, |linearScore (E c) (T i)-z i c| ≤ e)
    (hL : coefficient (fun i => probabilityCeiling (q i) (2*e)) r T ≤ L)
    (hs : 0 ≤ s) (hclip : 0 ≤ clip) (hstep : 19*s*(clip*L) ≤ 2) :
    let w := unusedCoordinates E classes
    let g := fun j => clip*unusedCoordinates (readoutGradient T y E) classes j
    metricMomentumPotential (fun j => r j.2) s
      (fun j => w j-s*r j.2*mainFirst (m j) (g j)) (fun j => mainFirst (m j) (g j)) ≤
        metricMomentumPotential (fun j => r j.2) s w m := by
  apply unused_ce_nonincrease T y E classes m r s L clip
    (fun i => probabilityCeiling (q i) (2*e)) hr hu _ hL hs hclip hstep
  intro u i
  exact probability_of_execution (z i) _ (classes u) (q i) e (hq u i) (hq1 i) (he i)

example := probability_of_execution (fun c : Bool => if c then 2 else 1)
  (fun c => if c then 21/10 else 9/10) true
  (probability (fun c : Bool => if c then 2 else 1) true) (1/10)
  le_rfl (probability_le_one _ _) (by intro c; cases c <;> norm_num)
example := executed_feature_motion (H := ℝ) 2 3 (21/10) (32/10) 1 (1/10) (2/10)
  (by norm_num) (by norm_num) (by norm_num)
example := executed_feature_motion_sq
  ((EuclideanSpace.equiv (Fin 1) ℝ).symm (fun _ => 2))
  ((EuclideanSpace.equiv (Fin 1) ℝ).symm (fun _ => 3))
  ((EuclideanSpace.equiv (Fin 1) ℝ).symm (fun _ => 21/10))
  ((EuclideanSpace.equiv (Fin 1) ℝ).symm (fun _ => 32/10)) 1 (1/10) (2/10)
  (by norm_num [EuclideanSpace.norm_eq]) (by norm_num [EuclideanSpace.norm_eq])
  (by norm_num [EuclideanSpace.norm_eq])
example := executed_ce_nonincrease (I := Fin 1) (C := Fin 3) (K := Fin 1)
  (fun _ _ => 1) (fun _ => 0) (fun c _ => c.val)
  (fun _ c => c.val+(1/10 : ℝ)) (fun u : Fin 2 => u.succ)
  (fun j => if j.1.val=0 then -1 else 2) (fun _ => 1) (1/100) 1 (1/2) (1/10) (fun _ => 1)
  (by intro k; norm_num) (by intro u i; exact Fin.succ_ne_zero u)
  (by intro u i; exact probability_le_one _ _) (by intro i; norm_num)
  (by intro i c; norm_num [linearScore])
  (by
    have h := coefficient_le_trace (I := Fin 1) (K := Fin 1)
      (fun _ => probabilityCeiling 1 (2*(1/10))) (fun _ => 1) (fun _ _ => 1)
      (by intro i; exact (probabilityCeiling_bounds _ _ (by norm_num) (by norm_num)).1)
      (by intro k; norm_num)
    norm_num [traceCoefficient, probabilityCeiling] at h
    simpa [probabilityCeiling] using h)
  (by norm_num) (by norm_num) (by norm_num)

end Transformer.Grokking.GradientEvidence.CEGram
