module

public import Transformer.Grokking.FiniteRecovery.MeanFieldWidth.Initialization
public import Transformer.Grokking.FiniteRecovery.ArgandObservedJacobian

/-! The softmax certificate. The certificate `marginalBound` of the Argand
recovery theorem depends on the weights only through the logits: it is
`softmaxBound` of the logit table, the misses 1 - softmax at the observed
labels plus count times the misses at a feasible completion. As a function
of the score table, `softmaxBound` dominates the structural risk of the
table law and is smooth, and `certificateMap` of every C² map is Lipschitz
in ℓ¹ on every box, which is the hypothesis `Regular.map`. A training whose
scores are width averages mean ψ therefore recovers with the certificate
softmaxBound ∘ mean ψ (`permanent_of_softmax_certificate`). Argand's logits
are not width averages, so Argand's certificate stays outside the class. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.MeanFieldWidth

open MeasureTheory Finset Filter Transformer.Grokking.FiniteRecovery.ObservedConstraints

variable {P K X Act : Type*}

/-- The softmax bound of a score table s: the misses 1 - softmax at the
observed labels y plus count times the misses at a completion g. -/
noncomputable def softmaxBound [Fintype X] [Fintype Act] (count : ℕ) (S : Finset X) (y : S → Act)
    (g : X → Act) (s : X × Act → ℝ) : ℝ :=
  let miss x a := 1 - Real.exp (s (x, a)) / ∑ b, Real.exp (s (x, b))
  (∑ x : S, miss x.val (y x)) + count * ∑ x, miss x (g x)

/-- For a completion g of the observations that every rule admits, the
softmax bound of a score table dominates the structural risk of its table
law. -/
theorem risk_le_softmaxBound [Fintype X] [Fintype Act] [Nonempty Act] [MeasurableSpace Act]
    [MeasurableSingletonClass Act] {R : Type*} [Fintype R] (S : Finset X) (y : S → Act)
    (g : X → Act) (rule : R → Set (X → Act)) (feasible : ∀ j, g ∈ rule j)
    (fit : ∀ x : S, g x = y x) (s : X × Act → ℝ) :
    risk (tableLaw fun x a => s (x, a)) S y rule ≤ softmaxBound (Fintype.card R) S y g s := by
  let score : X → Act → ℝ := fun x a => s (x, a)
  have miss (x : X) : (tableLaw score).real {k | k x ≠ g x} =
      1 - Real.exp (s (x, g x)) / ∑ b, Real.exp (s (x, b)) := by
    let E := {k : X → Act | k x = g x}
    change (tableLaw score).real Eᶜ = _
    rw [measureReal_compl (measurableSet_eq_fun (measurable_pi_apply x) measurable_const),
      probReal_univ]
    exact congrArg (fun v : ℝ => 1-v) (table_marginal score x (g x))
  rw [← funext fit]
  simpa only [softmaxBound, miss] using risk_le_of_feasible (tableLaw score) S rule g feasible

/-- The softmax bound is smooth in the score table. -/
theorem contDiff_softmaxBound [Fintype X] [Fintype Act] [Nonempty Act] (count : ℕ)
    (S : Finset X) (y : S → Act) (g : X → Act) {n : WithTop ℕ∞} :
    ContDiff ℝ n (softmaxBound count S y g) := by
  have miss (x : X) (a : Act) : ContDiff ℝ n fun s : X × Act → ℝ =>
      1 - Real.exp (s (x, a)) / ∑ b, Real.exp (s (x, b)) :=
    contDiff_const.sub (((contDiff_apply ℝ (fun _ => ℝ) (x, a)).exp).div
      (ContDiff.sum fun b _ => (contDiff_apply ℝ (fun _ => ℝ) (x, b)).exp)
      fun s => (Finset.sum_pos (fun b _ => Real.exp_pos _) Finset.univ_nonempty).ne')
  unfold softmaxBound
  exact (ContDiff.sum fun (x : S) _ => miss x.val (y x)).add
    (contDiff_const.mul (ContDiff.sum fun x _ => miss x (g x)))

/-- For a C² map F, `certificateMap F ρ c` is Lipschitz in ℓ¹ on every box:
the hypothesis `Regular.map` holds with some constant. -/
theorem certificateMap_lipschitz {J : Type*} [Fintype J] (F : (J → ℝ) → ℝ) (hF : ContDiff ℝ 2 F)
    (ρ c Cχ : ℝ) : ∃ LΓ, 0 ≤ LΓ ∧ ∀ a a' : Fin 3 × J → ℝ, (∀ q, |a q| ≤ Cχ) →
      (∀ q, |a' q| ≤ Cχ) → certificateMap F ρ c a - certificateMap F ρ c a' ≤ LΓ * l1 (a - a') := by
  have p (k : Fin 3) : ContDiff ℝ 1 fun a : Fin 3 × J → ℝ => fun j => a (k, j) :=
    contDiff_pi.2 fun j => contDiff_apply ℝ (fun _ => ℝ) (k, j)
  have hC : ContDiff ℝ 1 (certificateMap F ρ c) :=
    (((hF.fderiv_right one_add_one_eq_two.le).comp (p 0)).clm_apply (p 1)).add
      (contDiff_const.mul (((hF.of_le one_le_two).comp (p 2)).sub contDiff_const))
  obtain ⟨K, hK⟩ := hC.contDiffOn.exists_lipschitzOnWith one_ne_zero
    (convex_closedBall (0 : Fin 3 × J → ℝ) (max Cχ 0)) (isCompact_closedBall _ _)
  refine ⟨K, K.2, fun a a' ha ha' => ?_⟩
  have mem (b : Fin 3 × J → ℝ) (hb : ∀ q, |b q| ≤ Cχ) : b ∈ Metric.closedBall 0 (max Cχ 0) := by
    rw [mem_closedBall_zero_iff, pi_norm_le_iff_of_nonneg (le_max_right _ _)]
    exact fun q => (Real.norm_eq_abs _ ▸ hb q).trans (le_max_left _ _)
  have hn : ‖a - a'‖ ≤ l1 (a - a') := by
    refine (pi_norm_le_iff_of_nonneg (l1_nonneg _)).2 fun q => ?_
    rw [Real.norm_eq_abs]
    exact Finset.single_le_sum (f := fun q => |(a - a') q|) (fun _ _ => abs_nonneg _)
      (Finset.mem_univ q)
  have hd := hK.dist_le_mul a (mem a ha) a' (mem a' ha')
  rw [dist_eq_norm, dist_eq_norm, Real.norm_eq_abs] at hd
  calc _ ≤ |certificateMap F ρ c a - certificateMap F ρ c a'| := le_abs_self _
    _ ≤ K * ‖a - a'‖ := hd
    _ ≤ K * l1 (a - a') := mul_le_mul_of_nonneg_left hn K.2

/-- `marginalBound`, the certificate of the Argand recovery theorem, is the
softmax bound of Argand's logit table. -/
theorem marginalBound_eq_softmaxBound [Fintype X] [Fintype Act] {H d L T : ℕ} (r count : ℕ)
    (S : Finset X) (y : S → Act) (g : X → Act) (prompt : X → Word Act (T+1))
    (pos : X → Fin (T+1)) (w : Argand.Weights Act H d L) :
    Argand.ObservedDrift.marginalBound r count S y g prompt pos w = softmaxBound count S y g
      fun q => (Argand.Model.fromWeights w).logits r (prompt q.1) (pos q.1) q.2 :=
  rfl

/-- **Recovery with the softmax certificate.** For a training whose scores are
the width average mean ψ of a score table, the softmax bound of the target f
is a certificate F ∘ mean ψ dominating the risk, and recovery from an
initialization law whose coordinates leave [-C₀, C₀] with probability at most
δ holds with probability above q - N · |P| · δ. -/
theorem permanent_of_softmax_certificate [Fintype X] [Fintype Act] [Nonempty Act]
    [MeasurableSpace Act] [MeasurableSingletonClass Act] {R : Type*} [Fintype R] [Fintype P]
    [Fintype K] (A : Training P K) (μ : Measure (P → ℝ)) [IsProbabilityMeasure μ]
    (ψ : (P → ℝ) → X × Act → ℝ) (hψ : Differentiable ℝ ψ) {S : Finset X} {f : X → Act}
    {η LG Lφ Cφ LΓ Lχ Mχ Cχ C₀ δ ρ r q rbar : ℝ} (hδ : ∀ p, μ.real {θ | C₀ < |θ p|} ≤ δ)
    (h : Regular A (certificateMap (softmaxBound (Fintype.card R) S (fun x => f x) f) ρ r)
      (certificateFeatures ψ (η * (max C₀ 120 / 10 + 12))) η LG Lφ Cφ LΓ Lχ Mχ Cχ) {N : ℕ}
    (hN : 0 < N) {B H n : ℕ} (hn : 0 < n) (rule : R → Set (X → Act)) (hf : ∀ j, f ∈ rule j)
    (hid : Set.InjOn (fun g : X → Act => fun x : S => g x) (⋂ j, rule j))
    (hρ : 0 ≤ ρ) (hlt : ρ < 1) (hr : r < 1/2)
    (hlim : ∀ t ∈ Icc B H, ∀ u ∈ Set.Ioo (0:ℝ) 1, limitResidual A μ
      (certificateMap (softmaxBound (Fintype.card R) S (fun x => f x) f) ρ r)
      (certificateFeatures ψ (η * (max C₀ 120 / 10 + 12))) t u ≤ rbar)
    (tail : (Fin N → P → ℝ) → ℝ) (htm : Measurable tail)
    (hti : Integrable tail (Measure.pi fun _ : Fin N => μ)) (htn : ∀ z, 0 ≤ tail z)
    (htail : ∀ z t, H < t → ∀ u ∈ Set.Ioo (0:ℝ) 1, residual A
      (certificateMap (softmaxBound (Fintype.card R) S (fun x => f x) f) ρ r)
      (certificateFeatures ψ (η * (max C₀ 120 / 10 + 12))) z t u ≤ tail z)
    (small : max rbar 0 + widthError A.eps η LG Lφ Cφ LΓ Lχ Mχ Cχ (Fintype.card K)
        (Fintype.card (Fin 3 × (X × Act))) N B H n +
        ∫ z, tail z ∂(Measure.pi fun _ : Fin N => μ) < (1-q)*((1-ρ)*(1/2-r))) :
    ∃ G ⊆ box N C₀, MeasurableSet G ∧
      q - N * Fintype.card P * δ < (Measure.pi fun _ : Fin N => μ).real G ∧
      ∀ z ∈ G, ∀ᶠ t in atTop, StrictOn (fun x a => meanField ψ (weightsAt A z t) (x, a)) f :=
  permanent_of_certificate A μ _ ψ ((contDiff_softmaxBound _ S _ f (n := 1)).differentiable
    one_ne_zero) hψ hδ h hN hn (fun v x a => meanField ψ v (x, a)) S rule f hf hid
    (fun v => risk_le_softmaxBound S (fun x => f x) f rule hf (fun _ => rfl) (meanField ψ v))
    hρ hlt hr hlim tail htm hti htn htail small

section Examples

theorem softmaxBound_unit (count : ℕ) (S : Finset (Fin 1)) (y : S → Fin 1) (g : Fin 1 → Fin 1) :
    softmaxBound count S y g = fun _ => 0 := by
  funext s
  have h (x a : Fin 1) : 1 - Real.exp (s (x, a)) / ∑ b, Real.exp (s (x, b)) = 0 := by
    rw [Fin.sum_univ_one, Subsingleton.elim a 0, div_self (Real.exp_pos _).ne', sub_self]
  simp only [softmaxBound, h, Finset.sum_const_zero, mul_zero, add_zero]

/-- The identity training with the softmax certificate of one input and one
label is regular. -/
theorem softmaxRegular (count : ℕ) (S : Finset (Fin 1)) (y : S → Fin 1) (g : Fin 1 → Fin 1)
    (ρ r D : ℝ) : Regular identityTraining (certificateMap (softmaxBound count S y g) ρ r)
      (certificateFeatures (fun _ (_ : Fin 1 × Fin 1) => (0:ℝ)) D) 1 1 0 0 0 0 0 0 := by
  rw [softmaxBound_unit]
  exact identityCertificateRegular ρ r D

example := risk_le_softmaxBound (X := Fin 1) (Act := Fin 1) (R := Fin 1) Finset.univ
  (fun _ => 0) (fun _ => 0) (fun _ => Set.univ) (fun _ => Set.mem_univ _) (fun _ => rfl) (fun _ => 0)
example := contDiff_softmaxBound (X := Fin 1) (Act := Fin 1) 1 Finset.univ (fun _ => 0)
  (fun _ => 0) (n := 2)
example := certificateMap_lipschitz (J := Fin 1) (fun a => a 0)
  (contDiff_apply ℝ (fun _ => ℝ) 0) 0 0 1
example := permanent_of_softmax_certificate identityTraining (Measure.dirac 0) (R := Fin 1)
  (fun _ (_ : Fin 1 × Fin 1) => (0:ℝ)) (differentiable_const _) (S := ∅) (f := fun _ => 0)
  (C₀ := 1) (δ := 1) (fun _ => measureReal_le_one) (softmaxRegular _ _ _ _ 0 0 _) (N := 2) two_pos
  (B := 1) (H := 3) (n := 4) (by norm_num) (fun _ => Set.univ) (fun _ => Set.mem_univ _)
  (fun _ _ _ _ _ => Subsingleton.elim _ _) (q := 0) (rbar := 0) le_rfl one_pos (by norm_num)
  (fun _ _ _ _ => by simp [limitResidual, certificateMap, softmaxBound_unit]) (fun _ => 0)
  measurable_const (integrable_const _) (fun _ => le_rfl)
  (fun _ _ _ _ _ => by simp [residual, certificateMap, softmaxBound_unit]) (by norm_num [widthError])

end Examples

end Transformer.Grokking.FiniteRecovery.MeanFieldWidth
