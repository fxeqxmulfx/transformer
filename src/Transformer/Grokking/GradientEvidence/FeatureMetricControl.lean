module

public import Transformer.Grokking.GradientEvidence.FeatureMetricMomentum

/-!
# Apply a checkpoint-fixed feature metric to actual partial CE

Source: `review/flat_spectrum.md`, *Checkpoint-fixed feature metrics*, FM1–FM3.
The finite-prefix result permits changing features and batches but keeps the
metric and scalar step fixed. Full tied AdamW is retained as an explicit error.
-/

@[expose] public section
noncomputable section
namespace Transformer.Grokking.GradientEvidence

open Finset Geometry FlatSpectrum.FeatureCodes
variable {I C K U : Type*} [Fintype I] [Fintype C] [Fintype K]
  [Fintype U] [Nonempty U] [DecidableEq C] [DecidableEq K]

omit [Fintype C] [Fintype K] [Nonempty U] [DecidableEq C] [DecidableEq K] in
/-- FM1: feature-only scaling commutes with the computed unused-class mean. -/
theorem feature_scale_class_center (r : K → ℝ) (E : C → K → ℝ)
    (classes : U → C) (j : U × K) :
    unusedCoordinates (fun c k => r k*E c k) classes j = r j.2*unusedCoordinates E classes j := by
  simp only [unusedCoordinates, classMean, ← mul_sum, mul_div_assoc, mul_sub]

/-- FM2: actual partial CE and common clipping satisfy the weighted potential
bound, with arbitrary saved moment and an explicit finite-step condition. -/
theorem metric_unused_ce_nonincrease (T : I → K → ℝ) (y : I → C)
    (E : C → K → ℝ) (classes : U → C) (m : U × K → ℝ) (r : K → ℝ) (s q clip : ℝ)
    (hr : ∀ k, 0 < r k) (hu : ∀ u i, classes u ≠ y i)
    (hq : ∀ u i, probability (fun v => linearScore (E v) (T i)) (classes u) ≤ q)
    (hs : 0 ≤ s) (hclip : 0 ≤ clip)
    (hstep : 19*s*(clip*(q*(metricFeatureEnergy r T/Fintype.card I))) ≤ 2) :
    let w := unusedCoordinates E classes
    let g := fun j => clip*unusedCoordinates (readoutGradient T y E) classes j
    metricMomentumPotential (fun j => r j.2) s
      (fun j => w j-s*r j.2*mainFirst (m j) (g j)) (fun j => mainFirst (m j) (g j)) ≤
        metricMomentumPotential (fun j => r j.2) s w m := by
  dsimp only
  apply metric_momentum_step_nonincrease (fun j => r j.2)
    s (clip*(q*(metricFeatureEnergy r T/Fintype.card I))) _ m _ (fun j => hr j.2) hs _ _ hstep
  · have hd := mul_nonneg hclip (unused_gradient_radial_nonneg T y E classes hu)
    convert hd using 1
    rw [mul_sum]
    apply sum_congr rfl
    intro j hj
    ring
  · exact metric_scaled_cocoercivity _ _ _ _ clip
      (metric_unused_group_cocoercive T y E classes q r (fun k => (hr k).le) hu hq)

/-- FM2: one fixed metric and scalar step control every prefix of the
declared partial-CE recurrence, even when the feature table changes. -/
theorem metric_unused_ce_sequence (T : ℕ → I → K → ℝ) (y : ℕ → I → C)
    (E : ℕ → C → K → ℝ) (classes : U → C) (m : ℕ → U × K → ℝ)
    (r : K → ℝ) (s q : ℝ) (clip : ℕ → ℝ)
    (hr : ∀ k, 0 < r k) (hu : ∀ u t i, classes u ≠ y t i)
    (hq : ∀ u t i, probability (fun v => linearScore (E t v) (T t i)) (classes u) ≤ q)
    (hs : 0 ≤ s) (hclip : ∀ t, 0 ≤ clip t)
    (hstep : ∀ t, 19*s*(clip t*(q*(metricFeatureEnergy r (T t)/Fintype.card I))) ≤ 2)
    (hm : ∀ t j, m (t+1) j = mainFirst (m t j)
      (clip t*unusedCoordinates (readoutGradient (T t) (y t) (E t)) classes j))
    (hw : ∀ t j, unusedCoordinates (E (t+1)) classes j =
      unusedCoordinates (E t) classes j-s*r j.2*m (t+1) j) (n : ℕ) :
    metricMomentumPotential (fun j => r j.2) s (unusedCoordinates (E n) classes) (m n) ≤
      metricMomentumPotential (fun j => r j.2) s (unusedCoordinates (E 0) classes) (m 0) := by
  induction n with
  | zero => exact le_rfl
  | succ n ih =>
    have hn := metric_unused_ce_nonincrease (T n) (y n) (E n) classes (m n) r s q (clip n)
      hr (fun u => hu u n) (fun u => hq u n) hs (hclip n) (hstep n)
    have hm' : m (n+1) = fun j => mainFirst (m n j)
        (clip n*unusedCoordinates (readoutGradient (T n) (y n) (E n)) classes j) := funext (hm n)
    have hw' : unusedCoordinates (E (n+1)) classes = fun j =>
        unusedCoordinates (E n) classes j-s*r j.2*mainFirst (m n j)
          (clip n*unusedCoordinates (readoutGradient (T n) (y n) (E n)) classes j) := by
      funext j
      rw [hw, hm]
    rw [hw', hm']
    exact hn.trans ih

omit [Fintype C] [Fintype U] [Nonempty U] [DecidableEq C] [DecidableEq K] in
/-- FM2: a uniform q=1 coefficient follows from feature energy and metric
caps, without using the next batch's probabilities or clipping scalar. -/
theorem metric_uniform_step_condition (r : K → ℝ) (T : I → K → ℝ) (s R M clip : ℝ)
    (hs : 0 ≤ s) (hR : 0 ≤ R) (hM : 0 ≤ M) (hr : ∀ k, r k ≤ R)
    (hclip : 0 ≤ clip) (hclip1 : clip ≤ 1)
    (hfeature : matrixEnergy T/Fintype.card I ≤ M) (hstep : 19*s*(R*M) ≤ 2) :
    19*s*(clip*(1*(metricFeatureEnergy r T/Fintype.card I))) ≤ 2 := by
  have he := div_le_div_of_nonneg_right (metric_feature_energy_le r T R hr)
    (Nat.cast_nonneg (Fintype.card I) : (0 : ℝ) ≤ Fintype.card I)
  have hf : metricFeatureEnergy r T/Fintype.card I ≤ R*M := by
    calc
      _ ≤ (R*matrixEnergy T)/Fintype.card I := he
      _ = R*(matrixEnergy T/Fintype.card I) := by ring
      _ ≤ R*M := mul_le_mul_of_nonneg_left hfeature hR
  have hc := mul_le_mul_of_nonneg_left hf hclip
  have hb := mul_le_mul_of_nonneg_right hclip1 (mul_nonneg hR hM)
  have hm : clip*(metricFeatureEnergy r T/Fintype.card I) ≤ R*M := by linarith
  have h := mul_le_mul_of_nonneg_left hm (by positivity : (0 : ℝ) ≤ 19*s)
  simpa only [one_mul] using h.trans hstep

/-- FM2: this version uses the universal probability ceiling. Its coefficient
needs only the fixed metric cap and a declared feature-energy bound. -/
theorem metric_unused_ce_uniform (T : I → K → ℝ) (y : I → C)
    (E : C → K → ℝ) (classes : U → C) (m : U × K → ℝ) (r : K → ℝ)
    (s R M clip : ℝ) (hr : ∀ k, 0 < r k) (hu : ∀ u i, classes u ≠ y i)
    (hs : 0 ≤ s) (hR : 0 ≤ R) (hM : 0 ≤ M) (hmax : ∀ k, r k ≤ R)
    (hclip : 0 ≤ clip) (hclip1 : clip ≤ 1)
    (hfeature : matrixEnergy T/Fintype.card I ≤ M) (hstep : 19*s*(R*M) ≤ 2) :
    let w := unusedCoordinates E classes
    let g := fun j => clip*unusedCoordinates (readoutGradient T y E) classes j
    metricMomentumPotential (fun j => r j.2) s
      (fun j => w j-s*r j.2*mainFirst (m j) (g j)) (fun j => mainFirst (m j) (g j)) ≤
        metricMomentumPotential (fun j => r j.2) s w m := by
  exact metric_unused_ce_nonincrease T y E classes m r s 1 clip hr hu
    (fun u i => probability_le_one _ _) hs hclip
    (metric_uniform_step_condition r T s R M clip hs hR hM hmax hclip hclip1 hfeature hstep)

/-- FM3: the exact full AdamW remainder uses the feature-scaled reference.
The next second moment and encoder input are not assumed favorable. -/
theorem feature_metric_adam_error (w m v gR gE em ep eta s r rho eps : ℝ) (t : ℕ) :
    let mn := mainFirst m (gR+gE)+em
    bufferedAdamUpdate (9/10) (19/20) eps eta rho (t+1) w mn v+ep-
      (w-s*r*mainFirst m gR) =
      -eta*rho*w +
      (s*r-eta*adamDiagonal (19/20) eps (t+1) v/(1-(9/10 : ℝ)^(t+1)))*mn -
      s*r*(1/10)*gE + (ep-s*r*em) := by
  exact main_adam_reference_error w m v gR gE em ep eta (s*r) rho eps t

example := metric_unused_ce_nonincrease (I := Fin 1) (C := Fin 3) (K := Bool)
  (fun _ k => if k then 2 else 1) (fun _ => 0) (fun c _ => c.val)
  (fun u : Fin 2 => u.succ) (fun j => if j.1.val=0 then -1 else 2)
  (fun k => if k then 2 else 1) (1/100) 1 (1/2)
  (by intro k; cases k <;> norm_num) (by intro u i; exact Fin.succ_ne_zero u)
  (by intro u i; exact probability_le_one _ _) (by norm_num) (by norm_num)
  (by norm_num [metricFeatureEnergy])
example (n : ℕ) := metric_unused_ce_sequence (I := Fin 1) (C := Fin 3) (K := Fin 1)
  (fun _ _ _ => 0) (fun _ _ => 0) (fun _ c _ => c.val) (fun u : Fin 2 => u.succ)
  (fun _ _ => 0) (fun _ => 2) (1/100) 1 (fun _ => 1/2)
  (by intro k; norm_num) (by intro u t i; exact Fin.succ_ne_zero u)
  (by intro u t i; exact probability_le_one _ _) (by norm_num)
  (by intro t; norm_num) (by intro t; norm_num [metricFeatureEnergy])
  (by intro t j; simp [unusedCoordinates, readoutGradient_eq, classMean, mainFirst])
  (by intro t j; simp) n
example := metric_uniform_step_condition (I := Fin 1) (K := Bool)
  (fun k => if k then 2 else 1) (fun _ k => if k then 2 else 1) (1/1000) 2 5 (1/2)
  (by norm_num) (by norm_num) (by norm_num) (by intro k; cases k <;> norm_num)
  (by norm_num) (by norm_num) (by norm_num [matrixEnergy]) (by norm_num)
example := metric_unused_ce_uniform (I := Fin 1) (C := Fin 3) (K := Fin 1)
  (fun _ _ => 1) (fun _ => 0) (fun c _ => c.val) (fun u : Fin 2 => u.succ)
  (fun j => if j.1.val=0 then -1 else 2) (fun _ => 2) (1/100) 2 1 (1/2)
  (by intro k; norm_num) (by intro u i; exact Fin.succ_ne_zero u)
  (by norm_num) (by norm_num) (by norm_num) (by intro k; norm_num)
  (by norm_num) (by norm_num) (by norm_num [matrixEnergy]) (by norm_num)

end Transformer.Grokking.GradientEvidence
