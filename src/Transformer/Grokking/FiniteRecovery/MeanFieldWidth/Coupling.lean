module

public import Transformer.Grokking.FiniteRecovery.MeanFieldWidth.Basic

/-! Synchronous coupling of the N-particle system with N limiting particles
started from the same weights and fed the same minibatches. Each step
multiplies the mean distance by a fixed factor and adds the sampling
error of the limiting particles' field. A discrete Gronwall bound then
controls every time up to the horizon by one sum of sampling errors. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.MeanFieldWidth

open Finset

variable {P K : Type*}

theorem cone_system (A : Training P K) {N : ℕ} (init : Fin N → P → ℝ) (t : ℕ) (i : Fin N) :
    Cone (system A init t i) := by
  induction t with
  | zero => intro p; simp [system]
  | succ t ih => rw [system_succ]; exact cone_step _ _ _ _ ih

theorem cone_flow (A : Training P K) (μ : MeasureTheory.Measure (P → ℝ)) (t : ℕ) (θ : P → ℝ) :
    Cone (flow A μ t θ) := by
  induction t with
  | zero => intro p; simp [flow]
  | succ t ih => rw [flow_succ]; exact cone_step _ _ _ _ ih

theorem l1_nonneg {ι : Type*} [Fintype ι] (v : ι → ℝ) : 0 ≤ l1 v :=
  sum_nonneg fun _ _ => abs_nonneg _

theorem l1_sub_le {ι : Type*} [Fintype ι] (a b c : ι → ℝ) : l1 (a - c) ≤ l1 (a - b) + l1 (b - c) := by
  unfold l1
  rw [← sum_add_distrib]
  exact sum_le_sum fun i _ => abs_sub_le (a i) (b i) (c i)

theorem gap_nonneg [Fintype P] (s s' : AdamW.State P) : 0 ≤ gap s s' :=
  sum_nonneg fun _ _ => by positivity

theorem l1_weights_le_gap [Fintype P] (s s' : AdamW.State P) : l1 (s.weights - s'.weights) ≤ gap s s' :=
  sum_le_sum fun p _ => by
    simp only [Pi.sub_apply]
    linarith [abs_nonneg (s.first p - s'.first p),
      abs_nonneg (Real.sqrt (s.second p) - Real.sqrt (s'.second p))]

/-- An ℓ¹-Lipschitz feature moves the empirical field by at most its
constant times the mean ℓ¹ distance of the particles. -/
theorem meanField_sub_le [Fintype P] [Fintype K] {N : ℕ} (φ : (P → ℝ) → K → ℝ) {L : ℝ}
    (hφ : ∀ θ θ', l1 (φ θ - φ θ') ≤ L * l1 (θ - θ')) (θ θ' : Fin N → P → ℝ) :
    l1 (meanField φ θ - meanField φ θ') ≤ L * (∑ i, l1 (θ i - θ' i)) / N := by
  have hN : (0:ℝ) ≤ N := Nat.cast_nonneg N
  calc l1 (meanField φ θ - meanField φ θ') = (∑ k, |∑ i, (φ (θ i) k - φ (θ' i) k)|) / N := by
        unfold l1 meanField
        rw [sum_div]
        refine sum_congr rfl fun k _ => ?_
        rw [Pi.sub_apply, ← sub_div, abs_div, abs_of_nonneg hN, sum_sub_distrib]
    _ ≤ (∑ i, l1 (φ (θ i) - φ (θ' i))) / N := by
        refine div_le_div_of_nonneg_right ?_ hN
        unfold l1
        rw [sum_comm]
        exact sum_le_sum fun k _ => abs_sum_le_sum_abs _ _
    _ ≤ L * (∑ i, l1 (θ i - θ' i)) / N := by
        refine div_le_div_of_nonneg_right ?_ hN
        rw [mul_sum]
        exact sum_le_sum fun i _ => hφ _ _

theorem coupling_zero [Fintype P] (A : Training P K) (μ : MeasureTheory.Measure (P → ℝ)) {N : ℕ}
    (init : Fin N → P → ℝ) : coupling A μ init 0 = 0 := by
  simp [coupling, system, flow, gap]

/-- One coupled step. The factor 1 + 12η/eps comes from the Adam
direction, the factor 1 + L_G(1 + L_φ) from the gradient and the field. -/
theorem coupling_succ_le [Fintype P] [Fintype K] (A : Training P K) (μ : MeasureTheory.Measure (P → ℝ))
    {N : ℕ} (hN : 0 < N) (init : Fin N → P → ℝ) {LG Lφ η : ℝ} (he : 0 < A.eps)
    (hr : ∀ t, 0 ≤ A.rate t ∧ A.rate t ≤ η) (hη : η ≤ 10) (hLG : 0 ≤ LG) (hLφ : 0 ≤ Lφ)
    (hG : ∀ t θ θ' Φ Φ', l1 (A.grad t θ Φ - A.grad t θ' Φ') ≤ LG * (l1 (θ - θ') + l1 (Φ - Φ')))
    (hφ : ∀ θ θ', l1 (A.field θ - A.field θ') ≤ Lφ * l1 (θ - θ')) (t : ℕ) :
    coupling A μ init (t+1) ≤ (1 + 12*η/A.eps) *
      ((1 + LG*(1+Lφ)) * coupling A μ init t + LG * fluctuation A μ init t) := by
  set s := system A init t
  set X := fun i => flow A μ t (init i)
  set D := coupling A μ init t
  set ζ := fluctuation A μ init t
  have hNr : (0:ℝ) < N := Nat.cast_pos.2 hN
  have hlam : 1 + 12 * A.rate t / A.eps ≤ 1 + 12 * η / A.eps := by
    have := div_le_div_of_nonneg_right (mul_le_mul_of_nonneg_left (hr t).2 (by norm_num : (0:ℝ) ≤ 12)) he.le
    linarith
  have hlam0 : 0 ≤ 1 + 12 * A.rate t / A.eps := by
    have := div_nonneg (mul_nonneg (by norm_num : (0:ℝ) ≤ 12) (hr t).1) he.le; linarith
  have hfield : l1 (meanField A.field (fun j => (s j).weights) - limitField A μ t) ≤ Lφ * D + ζ := by
    refine (l1_sub_le _ (meanField A.field fun j => (X j).weights) _).trans (add_le_add ?_ le_rfl)
    refine (meanField_sub_le A.field hφ _ _).trans ?_
    have hsum : ∑ i, l1 ((s i).weights - (X i).weights) ≤ ∑ i, gap (s i) (X i) :=
      sum_le_sum fun i _ => l1_weights_le_gap _ _
    rw [mul_div_assoc]
    exact mul_le_mul_of_nonneg_left (div_le_div_of_nonneg_right hsum (Nat.cast_nonneg N)) hLφ
  have hi : ∀ i, gap (system A init (t+1) i) (flow A μ (t+1) (init i)) ≤
      (1 + 12*η/A.eps) * ((1 + LG) * gap (s i) (X i) + LG * (Lφ * D + ζ)) := by
    intro i
    rw [system_succ, flow_succ]
    refine (gap_step he (hr t).1 ((hr t).2.trans hη) t (cone_system A init t i)
      (second_nonneg_of_cone (cone_flow A μ t (init i))) _ _).trans ?_
    have hg := hG t (s i).weights (X i).weights (meanField A.field fun j => (s j).weights)
      (limitField A μ t)
    have hw := l1_weights_le_gap (s i) (X i)
    have h0 : 0 ≤ gap (s i) (X i) + l1 (A.grad t (s i).weights (meanField A.field fun j => (s j).weights) -
        A.grad t (X i).weights (limitField A μ t)) := add_nonneg (gap_nonneg _ _) (l1_nonneg _)
    refine (mul_le_mul_of_nonneg_right hlam h0).trans (mul_le_mul_of_nonneg_left ?_ (by linarith))
    nlinarith [mul_le_mul_of_nonneg_left hw hLG, mul_le_mul_of_nonneg_left hfield hLG]
  have e : (∑ i, (1 + 12*η/A.eps) * ((1 + LG) * gap (s i) (X i) + LG * (Lφ * D + ζ))) / N =
      (1 + 12*η/A.eps) * ((1 + LG*(1+Lφ)) * D + LG * ζ) := by
    rw [← mul_sum, sum_add_distrib, ← mul_sum, sum_const, card_univ, Fintype.card_fin, nsmul_eq_mul]
    simp only [D, coupling]
    field_simp
    ring
  rw [← e]
  exact div_le_div_of_nonneg_right (sum_le_sum fun i _ => hi i) hNr.le

/-- Discrete Gronwall: from zero, a growth factor a ≥ 1 and nonnegative
forcing b ζ, every time t is bounded by a^t b times the forcing so far. -/
theorem gronwall {D ζ : ℕ → ℝ} {a b : ℝ} (ha : 1 ≤ a) (hb : 0 ≤ b) (hζ : ∀ t, 0 ≤ ζ t)
    (h0 : D 0 = 0) (hstep : ∀ t, D (t+1) ≤ a * D t + b * ζ t) (t : ℕ) :
    D t ≤ a^t * b * ∑ s ∈ range t, ζ s := by
  induction t with
  | zero => simp [h0]
  | succ t ih =>
    have hS : 0 ≤ ∑ s ∈ range t, ζ s := sum_nonneg fun s _ => hζ s
    have hat : 1 ≤ a^t * a := by rw [← pow_succ]; exact one_le_pow₀ ha
    rw [sum_range_succ, pow_succ]
    nlinarith [hstep t, mul_le_mul_of_nonneg_left ih (by linarith : (0:ℝ) ≤ a),
      mul_le_mul_of_nonneg_right hat (mul_nonneg hb (hζ t))]

/-- Growth factor of the coupled distance per step. -/
noncomputable def growth (eps η LG Lφ : ℝ) : ℝ := (1 + 12*η/eps) * (1 + LG*(1+Lφ))

/-- Weight of the sampling error in the coupled step. -/
noncomputable def forcing (eps η LG : ℝ) : ℝ := (1 + 12*η/eps) * LG

/-- Pathwise propagation of chaos: for every initial draw, at every time
up to the horizon H, the mean coupled distance is bounded by growth^H
times the summed sampling errors of the limiting particles. -/
theorem coupling_le [Fintype P] [Fintype K] (A : Training P K) (μ : MeasureTheory.Measure (P → ℝ))
    {N : ℕ} (hN : 0 < N) (init : Fin N → P → ℝ) {LG Lφ η : ℝ} (he : 0 < A.eps)
    (hr : ∀ t, 0 ≤ A.rate t ∧ A.rate t ≤ η) (hη : η ≤ 10) (hLG : 0 ≤ LG) (hLφ : 0 ≤ Lφ)
    (hG : ∀ t θ θ' Φ Φ', l1 (A.grad t θ Φ - A.grad t θ' Φ') ≤ LG * (l1 (θ - θ') + l1 (Φ - Φ')))
    (hφ : ∀ θ θ', l1 (A.field θ - A.field θ') ≤ Lφ * l1 (θ - θ')) {H t : ℕ} (ht : t ≤ H) :
    coupling A μ init t ≤
      growth A.eps η LG Lφ ^ H * forcing A.eps η LG * ∑ s ∈ range H, fluctuation A μ init s := by
  have hη0 : 0 ≤ η := (hr 0).1.trans (hr 0).2
  have hk : 0 ≤ 12 * η / A.eps := div_nonneg (by linarith) he.le
  have ha : 1 ≤ growth A.eps η LG Lφ := one_le_mul_of_one_le_of_one_le (by linarith)
    (by nlinarith [mul_nonneg hLG hLφ])
  have hb : 0 ≤ forcing A.eps η LG := mul_nonneg (by linarith) hLG
  have hζ s : 0 ≤ fluctuation A μ init s := l1_nonneg _
  have hD := gronwall ha hb hζ (coupling_zero A μ init) (fun s => by
    have h := coupling_succ_le A μ hN init he hr hη hLG hLφ hG hφ s
    simp only [growth, forcing]
    linarith) t
  refine hD.trans (mul_le_mul (mul_le_mul_of_nonneg_right (pow_le_pow_right₀ ha ht) hb) ?_
    (sum_nonneg fun s _ => hζ s) (mul_nonneg (pow_nonneg (by linarith) _) hb))
  exact sum_le_sum_of_subset_of_nonneg (range_subset_range.2 ht) fun s _ _ => hζ s

section Examples

open MeasureTheory

/-- One particle coordinate whose gradient is its own weight; no field. -/
noncomputable def identityTraining : Training (Fin 1) (Fin 1) := ⟨1, fun _ => 1, fun _ θ _ => θ, fun _ _ => 0⟩

theorem identityTraining_grad : ∀ t θ θ' (Φ Φ' : Fin 1 → ℝ),
    l1 (identityTraining.grad t θ Φ - identityTraining.grad t θ' Φ') ≤ 1 * (l1 (θ - θ') + l1 (Φ - Φ')) :=
  fun _ θ θ' Φ Φ' => by
    show l1 (θ - θ') ≤ _
    linarith [l1_nonneg (Φ - Φ')]

theorem identityTraining_field : ∀ θ θ' : Fin 1 → ℝ,
    l1 (identityTraining.field θ - identityTraining.field θ') ≤ 0 * l1 (θ - θ') :=
  fun _ _ => by simp [identityTraining, l1]

example := meanField_sub_le (N := 2) identityTraining.field identityTraining_field 0 0
example := gronwall (D := fun _ => 0) (ζ := fun _ => 0) le_rfl le_rfl (fun _ => le_rfl) rfl
  (fun _ => by simp) 3
example := coupling_succ_le identityTraining (Measure.dirac 0) one_pos (fun _ => 0) (η := 1) one_pos
  (fun _ => ⟨zero_le_one, le_rfl⟩) (by norm_num) zero_le_one le_rfl identityTraining_grad
  identityTraining_field 0
example := coupling_le identityTraining (Measure.dirac 0) one_pos (fun _ => 0) (η := 1) one_pos
  (fun _ => ⟨zero_le_one, le_rfl⟩) (by norm_num) zero_le_one le_rfl identityTraining_grad
  identityTraining_field (le_refl 5)

end Examples

end Transformer.Grokking.FiniteRecovery.MeanFieldWidth
