module

public import Transformer.Grokking.FiniteRecovery.MeanFieldWidth.Transfer
public import Transformer.Grokking.FiniteRecovery.MeanFieldWidth.Bounds
public import Transformer.Grokking.FiniteRecovery.ObservedDrift

/-! Certificates read through width averages. For U(w) = F(mean_i ψ(w_i))
the chain rule writes the residual DU(w + uΔ)[Δ] + (1-ρ)(U(w) - c) of the
recovery theorem as `certificateMap` of the empirical mean of three blocks
of segment features: ψ at the point u of the segment, its derivative along
the displacement there, and ψ at the start. The features clamp the fraction
to [0,1] and each displacement coordinate to [-D, D]; neither clamp acts on
an update whose displacements lie within D, which `Bounds` guarantees for
D = η · (max(C₀, 120)/10 + 12) when every initial weight lies within C₀. On
such starts `certificate_slope` gives the slope hypothesis of the width
transfer at every update. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.MeanFieldWidth

open Finset

variable {P K J : Type*}

/-- Coordinatewise clamp of a displacement to [-D, D]. -/
def clampDisp (D : ℝ) (δ : P → ℝ) : P → ℝ := fun p => max (-D) (min D (δ p))

/-- Segment features of the certificate: ψ at the fraction u of the clamped
segment, its derivative along the clamped displacement, and ψ at the start. -/
noncomputable def certificateFeatures (ψ : (P → ℝ) → J → ℝ) (D u : ℝ) (θ θ' : P → ℝ) :
    Fin 3 × J → ℝ :=
  fun q => ![ψ (θ + max 0 (min 1 u) • clampDisp D (θ' - θ)) q.2,
    fderiv ℝ ψ (θ + max 0 (min 1 u) • clampDisp D (θ' - θ)) (clampDisp D (θ' - θ)) q.2,
    ψ θ q.2] q.1

/-- The residual map of the certificate F ∘ mean ψ for contraction ρ and floor c. -/
noncomputable def certificateMap (F : (J → ℝ) → ℝ) (ρ c : ℝ) (a : Fin 3 × J → ℝ) : ℝ :=
  fderiv ℝ F (fun j => a (0, j)) (fun j => a (1, j)) + (1 - ρ) * (F (fun j => a (2, j)) - c)

theorem meanField_eq_smul_sum [Fintype J] {N : ℕ} (ψ : (P → ℝ) → J → ℝ) (v : Fin N → P → ℝ) :
    meanField ψ v = (N:ℝ)⁻¹ • ∑ i, ψ (v i) := by
  funext j
  simp [meanField, Finset.sum_apply, div_eq_inv_mul]

theorem hasFDerivAt_meanField [Fintype P] [Fintype J] {N : ℕ} {ψ : (P → ℝ) → J → ℝ}
    (hψ : Differentiable ℝ ψ) (x : Fin N → P → ℝ) :
    HasFDerivAt (meanField ψ)
      ((N:ℝ)⁻¹ • ∑ i, (fderiv ℝ ψ (x i)).comp (ContinuousLinearMap.proj i)) x := by
  have e : meanField ψ = fun v : Fin N → P → ℝ => (N:ℝ)⁻¹ • ∑ i, ψ (v i) :=
    funext (meanField_eq_smul_sum ψ)
  rw [e]
  exact (HasFDerivAt.fun_sum fun i _ =>
    (hψ (x i)).hasFDerivAt.comp x (hasFDerivAt_apply i x)).fun_const_smul _

/-- The chain rule for a certificate read through width averages. -/
theorem fderiv_certificate [Fintype P] [Fintype J] (F : (J → ℝ) → ℝ) (ψ : (P → ℝ) → J → ℝ)
    (hF : Differentiable ℝ F) (hψ : Differentiable ℝ ψ) {N : ℕ} (x Δ : Fin N → P → ℝ) :
    fderiv ℝ (fun v : Fin N → P → ℝ => F (meanField ψ v)) x Δ =
      fderiv ℝ F (meanField ψ x) (fun j => (∑ i, fderiv ℝ ψ (x i) (Δ i) j) / N) := by
  change fderiv ℝ (F ∘ meanField ψ) x Δ = _
  rw [((hF _).hasFDerivAt.comp x (hasFDerivAt_meanField hψ x)).fderiv, ContinuousLinearMap.comp_apply]
  congr 1
  funext j
  simp [Finset.sum_apply, div_eq_inv_mul]

theorem clampDisp_of_abs_le {D : ℝ} {δ : P → ℝ} (h : ∀ p, |δ p| ≤ D) : clampDisp D δ = δ := by
  funext p
  have := abs_le.1 (h p)
  simp [clampDisp, min_eq_right this.2, max_eq_right this.1]

/-- The residual of the recovery theorem for the certificate F ∘ mean ψ is
`certificateMap` of the empirical mean of its segment features. -/
theorem certificate_residual [Fintype P] [Fintype J] (F : (J → ℝ) → ℝ) (ψ : (P → ℝ) → J → ℝ)
    (hF : Differentiable ℝ F) (hψ : Differentiable ℝ ψ) {N : ℕ} {D : ℝ} (w Δ : Fin N → P → ℝ)
    (hΔ : ∀ i p, |Δ i p| ≤ D) {u : ℝ} (hu0 : 0 ≤ u) (hu1 : u ≤ 1) (ρ c : ℝ) :
    fderiv ℝ (fun v : Fin N → P → ℝ => F (meanField ψ v)) (w + u • Δ) Δ +
        (1 - ρ) * (F (meanField ψ w) - c) =
      certificateMap F ρ c (meanField (fun q : (P → ℝ) × (P → ℝ) => certificateFeatures ψ D u q.1 q.2)
        fun i => (w i, w i + Δ i)) := by
  have hc i : clampDisp D (w i + Δ i - w i) = Δ i := by
    rw [add_sub_cancel_left]
    exact clampDisp_of_abs_le (hΔ i)
  have hu : max 0 (min 1 u) = u := by rw [min_eq_right hu1, max_eq_right hu0]
  rw [fderiv_certificate F ψ hF hψ, certificateMap]
  simp only [meanField, certificateFeatures, hc, hu, Matrix.cons_val_zero, Matrix.cons_val_one,
    Matrix.cons_val_two, Matrix.head_cons, Matrix.tail_cons, Pi.add_apply, Pi.smul_apply]
  rfl

/-- Along the N-particle system, the recovery residual of the certificate is
`residual` of `certificateMap` and the certificate features. -/
theorem certificate_residual_system [Fintype P] [Fintype J] (A : Training P K) (F : (J → ℝ) → ℝ)
    (ψ : (P → ℝ) → J → ℝ) (hF : Differentiable ℝ F) (hψ : Differentiable ℝ ψ) {N : ℕ}
    (init : Fin N → P → ℝ) {D : ℝ} (t : ℕ)
    (hΔ : ∀ i p, |(system A init (t+1) i).weights p - (system A init t i).weights p| ≤ D) {u : ℝ}
    (hu0 : 0 ≤ u) (hu1 : u ≤ 1) (ρ c : ℝ) :
    let w := fun i => (system A init t i).weights
    let Δ := fun i => (system A init (t+1) i).weights - (system A init t i).weights
    fderiv ℝ (fun v : Fin N → P → ℝ => F (meanField ψ v)) (w + u • Δ) Δ +
        (1 - ρ) * (F (meanField ψ w) - c) =
      residual A (certificateMap F ρ c) (certificateFeatures ψ D) init t u := by
  intro w Δ
  rw [certificate_residual F ψ hF hψ w Δ hΔ hu0 hu1 ρ c, residual]
  simp only [w, Δ, add_sub_cancel]

/-- With initial weights within C₀ and rates in [0, η], η ≤ 10, the recovery
residual of the certificate F ∘ mean ψ along the system is `residual` of
`certificateMap` and the certificate features clamped at
D = η · (max(C₀, 120)/10 + 12), at every update and fraction. -/
theorem certificate_residual_of_init [Fintype P] [Fintype J] (A : Training P K)
    (F : (J → ℝ) → ℝ) (ψ : (P → ℝ) → J → ℝ) (hF : Differentiable ℝ F) (hψ : Differentiable ℝ ψ)
    (he : 0 ≤ A.eps) {η : ℝ} (hr : ∀ t, 0 ≤ A.rate t ∧ A.rate t ≤ η) (hη : η ≤ 10) {N : ℕ}
    (init : Fin N → P → ℝ) {C₀ : ℝ} (hinit : ∀ i p, |init i p| ≤ C₀) (t : ℕ) {u : ℝ}
    (hu0 : 0 ≤ u) (hu1 : u ≤ 1) (ρ c : ℝ) :
    let w := fun i => (system A init t i).weights
    let Δ := fun i => (system A init (t+1) i).weights - (system A init t i).weights
    fderiv ℝ (fun v : Fin N → P → ℝ => F (meanField ψ v)) (w + u • Δ) Δ +
        (1 - ρ) * (F (meanField ψ w) - c) =
      residual A (certificateMap F ρ c) (certificateFeatures ψ (η * (max C₀ 120 / 10 + 12))) init t u := by
  refine certificate_residual_system A F ψ hF hψ init t (fun i p => ?_) hu0 hu1 ρ c
  have h10 : ∀ t, 0 ≤ A.rate t ∧ A.rate t ≤ 10 := fun t => ⟨(hr t).1, (hr t).2.trans hη⟩
  refine (abs_displacement_le A he h10 init t i p).trans ?_
  have hm : max |init i p| 120 ≤ max C₀ 120 := max_le_max_right _ (hinit i p)
  have hpos : 0 ≤ max |init i p| 120 / 10 + 12 := by positivity
  calc A.rate t * (max |init i p| 120 / 10 + 12) ≤ η * (max |init i p| 120 / 10 + 12) :=
        mul_le_mul_of_nonneg_right (hr t).2 hpos
    _ ≤ _ := by gcongr; exact (hr t).1.trans (hr t).2

/-- Along a segment of a differentiable risk the derivative is its Fréchet
derivative along the displacement. -/
theorem hasDerivAt_segmentRisk {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E] {V : E → ℝ}
    (hV : Differentiable ℝ V) (w Δ : E) (u : ℝ) :
    HasDerivAt (ObservedConstraints.segmentRisk V w Δ) (fderiv ℝ V (w + u • Δ) Δ) u := by
  have hγ : HasDerivAt (fun s : ℝ => w + s • Δ) Δ u := by
    simpa using ((hasDerivAt_id u).smul_const Δ).const_add w
  exact (hV _).hasFDerivAt.comp_hasDerivAt u hγ

/-- When every initial weight lies within C₀, the certificate F ∘ mean ψ meets
the slope hypothesis of the width transfer at every update: its segment is
differentiable, and the derivative plus the contraction term equals the
`residual` of `certificateMap` with the features clamped at
D = η · (max(C₀, 120)/10 + 12). -/
theorem certificate_slope [Fintype P] [Fintype J] (A : Training P K) (F : (J → ℝ) → ℝ)
    (ψ : (P → ℝ) → J → ℝ) (hF : Differentiable ℝ F) (hψ : Differentiable ℝ ψ) (he : 0 ≤ A.eps)
    {η : ℝ} (hr : ∀ t, 0 ≤ A.rate t ∧ A.rate t ≤ η) (hη : η ≤ 10) {N : ℕ}
    (init : Fin N → P → ℝ) {C₀ : ℝ} (hinit : ∀ i p, |init i p| ≤ C₀) (ρ c : ℝ) (t : ℕ) :
    let V := fun v : Fin N → P → ℝ => F (meanField ψ v)
    let w := fun i => (system A init t i).weights
    let line := ObservedConstraints.segmentRisk V w
      ((fun i => (system A init (t+1) i).weights) - w)
    ContinuousOn line (Set.Icc 0 1) ∧ DifferentiableOn ℝ line (Set.Ioo 0 1) ∧
      ∀ u ∈ Set.Ioo (0:ℝ) 1, deriv line u + (1-ρ)*(V w - c) ≤ residual A (certificateMap F ρ c)
        (certificateFeatures ψ (η * (max C₀ 120 / 10 + 12))) init t u := by
  intro V w line
  have hV : Differentiable ℝ V := hF.comp fun x => (hasFDerivAt_meanField hψ x).differentiableAt
  have hd (u : ℝ) := hasDerivAt_segmentRisk hV w ((fun i => (system A init (t+1) i).weights) - w) u
  refine ⟨fun u _ => (hd u).continuousAt.continuousWithinAt,
    fun u _ => (hd u).differentiableAt.differentiableWithinAt, fun u hu => ?_⟩
  rw [(hd u).deriv]
  exact (certificate_residual_of_init A F ψ hF hψ he hr hη init hinit t hu.1.le hu.2.le ρ c).le

section Examples

example := meanField_eq_smul_sum (P := Fin 1) (J := Fin 1) (N := 2) (fun θ _ => θ 0) (fun _ => 0)
example := hasFDerivAt_meanField (P := Fin 1) (J := Fin 1) (N := 2) (ψ := fun θ _ => θ 0)
  (by fun_prop) (fun _ => 0)
example := fderiv_certificate (P := Fin 1) (J := Fin 1) (fun a => a 0) (fun θ _ => θ 0) (by fun_prop)
  (by fun_prop) (N := 2) (fun _ => 0) (fun _ => 0)
example := clampDisp_of_abs_le (P := Fin 1) (D := 1) (δ := 0) fun _ => by simp
example := certificate_residual (P := Fin 1) (J := Fin 1) (fun a => a 0) (fun θ _ => θ 0)
  (by fun_prop) (by fun_prop) (N := 2) (D := 1) (fun _ => 0) (fun _ => 0) (fun _ _ => by simp)
  (u := 1/2) (by norm_num) (by norm_num) (1/2) 0
example := certificate_residual_system identityTraining (fun a : Fin 1 → ℝ => a 0)
  (fun θ _ => θ 0) (by fun_prop) (by fun_prop) (N := 2) (fun _ => 0) (D := 24) 3
  (fun i p => (abs_displacement_le identityTraining zero_le_one
    (fun _ => ⟨zero_le_one, by norm_num [identityTraining]⟩) _ 3 i p).trans
      (by simp [identityTraining]; norm_num)) (u := 1/2) (by norm_num) (by norm_num) (1/2) 0
example := certificate_residual_of_init identityTraining (fun a : Fin 1 → ℝ => a 0)
  (fun θ _ => θ 0) (by fun_prop) (by fun_prop) zero_le_one (η := 1)
  (fun _ => ⟨zero_le_one, le_rfl⟩) (by norm_num) (N := 2) (fun _ => 0) (C₀ := 0)
  (fun _ _ => by simp) 3 (u := 1/2) (by norm_num) (by norm_num) (1/2) 0
example := hasDerivAt_segmentRisk (E := ℝ) differentiable_id 0 1 (1/2)
example := certificate_slope identityTraining (fun a : Fin 1 → ℝ => a 0) (fun θ _ => θ 0)
  (by fun_prop) (by fun_prop) zero_le_one (η := 1) (fun _ => ⟨zero_le_one, le_rfl⟩) (by norm_num)
  (N := 2) (fun _ => 0) (C₀ := 0) (fun _ _ => by simp) (1/2) 0 3

end Examples

end Transformer.Grokking.FiniteRecovery.MeanFieldWidth
