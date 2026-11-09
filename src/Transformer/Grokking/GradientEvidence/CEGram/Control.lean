module

public import Transformer.Grokking.GradientEvidence.CEGram.CE
public import Transformer.Grokking.GradientEvidence.FeatureMetricMomentum

/-!
# Query-weighted CE control with arbitrary saved momentum

Source: `review/flat_spectrum.md`, *Query-weighted feature covariance*, QG2/QG4.
The reference is actual partial CE with common clipping, a fixed feature metric
and a fixed scalar step. Its batch type may change at every step; this includes
the original source's shorter final batch. Full AdamW errors remain separate.
-/

@[expose] public section
noncomputable section
namespace Transformer.Grokking.GradientEvidence.CEGram

open Finset Geometry FlatSpectrum.FeatureCodes
variable {C K U : Type*} [Fintype C] [Fintype K] [Fintype U] [Nonempty U]
  [DecidableEq C] [DecidableEq K]

/-- QG2: the probability-weighted operator upper bound supplies the finite
momentum condition for the actual partial CE gradient and any saved memory. -/
theorem unused_ce_nonincrease {I : Type*} [Fintype I] [DecidableEq I]
    (T : I → K → ℝ) (y : I → C) (E : C → K → ℝ) (classes : U → C)
    (m : U × K → ℝ) (r : K → ℝ) (s L clip : ℝ) (q : I → ℝ)
    (hr : ∀ k, 0 < r k) (hu : ∀ u i, classes u ≠ y i)
    (hq : ∀ u i, probability (fun v => linearScore (E v) (T i)) (classes u) ≤ q i)
    (hL : coefficient q r T ≤ L) (hs : 0 ≤ s) (hclip : 0 ≤ clip)
    (hstep : 19*s*(clip*L) ≤ 2) :
    let w := unusedCoordinates E classes
    let g := fun j => clip*unusedCoordinates (readoutGradient T y E) classes j
    metricMomentumPotential (fun j => r j.2) s
      (fun j => w j-s*r j.2*mainFirst (m j) (g j)) (fun j => mainFirst (m j) (g j)) ≤
        metricMomentumPotential (fun j => r j.2) s w m := by
  dsimp only
  apply metric_momentum_step_nonincrease (fun j => r j.2)
    s (clip*L) _ m _ (fun j => hr j.2) hs _ _ hstep
  · have hd := mul_nonneg hclip (unused_gradient_radial_nonneg T y E classes hu)
    convert hd using 1
    rw [mul_sum]
    apply sum_congr rfl
    intro j hj
    ring
  · exact metric_scaled_cocoercivity _ _ _ _ clip
      (group_cocoercive_of_upper T y E classes q r L (fun k => (hr k).le) hu hq hL)

/-- QG4: all prefixes of the actual partial-CE recurrence obey the fixed
potential bound. Batch cardinalities, features, ceilings and clipping may vary;
each premise is required only inside the declared finite window. -/
theorem unused_ce_prefix (I : ℕ → Type*) [∀ t, Fintype (I t)] [∀ t, DecidableEq (I t)]
    (T : ∀ t, I t → K → ℝ) (y : ∀ t, I t → C) (E : ℕ → C → K → ℝ)
    (classes : U → C) (m : ℕ → U × K → ℝ) (r : K → ℝ) (s : ℝ)
    (L clip : ℕ → ℝ) (q : ∀ t, I t → ℝ) (n : ℕ)
    (hr : ∀ k, 0 < r k) (hu : ∀ t < n, ∀ u i, classes u ≠ y t i)
    (hq : ∀ t < n, ∀ u i, probability (fun v => linearScore (E t v) (T t i)) (classes u) ≤ q t i)
    (hL : ∀ t < n, coefficient (q t) r (T t) ≤ L t)
    (hs : 0 ≤ s) (hclip : ∀ t < n, 0 ≤ clip t)
    (hstep : ∀ t < n, 19*s*(clip t*L t) ≤ 2)
    (hm : ∀ t < n, ∀ j, m (t+1) j = mainFirst (m t j)
      (clip t*unusedCoordinates (readoutGradient (T t) (y t) (E t)) classes j))
    (hw : ∀ t < n, ∀ j, unusedCoordinates (E (t+1)) classes j =
      unusedCoordinates (E t) classes j-s*r j.2*m (t+1) j) :
    ∀ j ≤ n, metricMomentumPotential (fun j => r j.2) s (unusedCoordinates (E j) classes) (m j) ≤
      metricMomentumPotential (fun j => r j.2) s (unusedCoordinates (E 0) classes) (m 0) := by
  intro j
  induction j with
  | zero => intro hj; exact le_rfl
  | succ j ih =>
    intro hj
    have hjn : j < n := by omega
    have hn := unused_ce_nonincrease (T j) (y j) (E j) classes (m j) r s (L j) (clip j)
      (q j) hr (hu j hjn) (hq j hjn) (hL j hjn) hs (hclip j hjn) (hstep j hjn)
    have hm' : m (j+1) = fun k => mainFirst (m j k)
        (clip j*unusedCoordinates (readoutGradient (T j) (y j) (E j)) classes k) :=
      funext (hm j hjn)
    have hw' : unusedCoordinates (E (j+1)) classes = fun k =>
        unusedCoordinates (E j) classes k-s*r k.2*mainFirst (m j k)
          (clip j*unusedCoordinates (readoutGradient (T j) (y j) (E j)) classes k) := by
      funext k
      rw [hw j hjn k, hm j hjn k]
    rw [hw', hm']
    exact hn.trans (ih (by omega))

example := unused_ce_nonincrease (I := Fin 1) (C := Fin 3) (K := Fin 1)
  (fun _ _ => 3) (fun _ => 0) (fun c _ => c.val) (fun u : Fin 2 => u.succ)
  (fun j => if j.1.val=0 then -1 else 2) (fun _ => 2) (1/100) 18 (1/2) (fun _ => 1)
  (by intro k; norm_num) (by intro u i; exact Fin.succ_ne_zero u)
  (by intro u i; exact probability_le_one _ _)
  (by
    have h := coefficient_le_trace (I := Fin 1) (K := Fin 1)
      (fun _ => 1) (fun _ => 2) (fun _ _ => 3) (by intro i; norm_num) (by intro k; norm_num)
    norm_num [traceCoefficient] at h
    exact h)
  (by norm_num) (by norm_num) (by norm_num)

/-- A nonzero CE step, nonzero memory, positive step and two unused classes.
The batch family changes cardinality; the declared window has one update. -/
example :
    let T := fun (t : ℕ) (_ : Fin (t+1)) (_ : Fin 1) => (1 : ℝ)
    let y := fun (t : ℕ) (_ : Fin (t+1)) => (0 : Fin 3)
    let E₀ := fun (c : Fin 3) (_ : Fin 1) => (c.val : ℝ)
    let M₀ := fun (c : Fin 3) (_ : Fin 1) => (c.val : ℝ)-1
    let M₁ := fun c k => mainFirst (M₀ c k) ((1/2)*readoutGradient (T 0) (y 0) E₀ c k)
    let E := fun (t : ℕ) => if t=0 then E₀ else fun c k => E₀ c k-(1/100)*M₁ c k
    let classes := fun u : Fin 2 => u.succ
    let m := fun (t : ℕ) => unusedCoordinates (if t=0 then M₀ else M₁) classes
    ∀ j ≤ 1, metricMomentumPotential (fun _ => 1) (1/100) (unusedCoordinates (E j) classes) (m j) ≤
      metricMomentumPotential (fun _ => 1) (1/100) (unusedCoordinates (E 0) classes) (m 0) := by
  dsimp only
  apply unused_ce_prefix (fun t => Fin (t+1))
    (fun _ _ _ => 1) (fun _ _ => 0) _ _ _ (fun _ => 1) (1/100)
    (fun _ => 1) (fun _ => 1/2) (fun _ _ => 1) 1
  · intro k; norm_num
  · intro t ht u i; exact Fin.succ_ne_zero u
  · intro t ht u i; exact probability_le_one _ _
  · intro t ht
    have ht0 : t=0 := by omega
    subst t
    have h := coefficient_le_trace (I := Fin 1) (K := Fin 1)
      (fun _ => 1) (fun _ => 1) (fun _ _ => 1) (by intro i; norm_num) (by intro k; norm_num)
    norm_num [traceCoefficient] at h
    exact h
  · norm_num
  · intro t ht; norm_num
  · intro t ht; norm_num
  · intro t ht j
    have ht0 : t=0 := by omega
    subst t
    simp only [ite_true, zero_add, show ¬(1 : ℕ)=0 by decide, ite_false]
    simp [unusedCoordinates, classMean, Fin.sum_univ_succ, mainFirst]
    ring
  · intro t ht j
    have ht0 : t=0 := by omega
    subst t
    simp only [ite_true, zero_add, show ¬(1 : ℕ)=0 by decide, ite_false]
    simp [unusedCoordinates, classMean, Fin.sum_univ_succ]
    ring

end Transformer.Grokking.GradientEvidence.CEGram
