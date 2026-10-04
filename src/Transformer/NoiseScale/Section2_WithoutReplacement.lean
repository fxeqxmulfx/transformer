/-
# Batches sampled without replacement

arXiv:1812.06162, §2.2, the footnote to eq. (2.2): "when batches are sampled without
replacement from a dataset of size `D`, the variance instead scales like `(1/B - 1/D)`".

A batch is an injection `σ : Fin B ↪ Fin D`, uniform, and its gradient is
`G_est = (1/B) Σ_i g_{σ(i)}` for the per-example gradients `g_1, …, g_D`.  It is unbiased
(`integral_batchGrad_embedding`), and its covariance is `(1/B - 1/D) D/(D-1) Σ` for `Σ` the
covariance of the gradient of a uniform example (`covMatrix_batchGrad_embedding`), where
independent examples give `Σ/B` (`covMatrix_batchGrad`).

The restriction of a uniform injection to `Fin n` is uniform (`sum_trans_mul_card`), as the
permutations of the dataset act transitively on the injections `Fin n ↪ Fin D`
(`Equiv.Perm.isMultiplyPretransitive`).
-/

import Mathlib.GroupTheory.GroupAction.MultipleTransitivity
import Mathlib.Probability.UniformOn
import Transformer.NoiseScale.Section2_Batches

open MeasureTheory ProbabilityTheory Finset
open scoped Matrix

namespace Transformer.NoiseScale

variable {B D : ℕ}

/-- Every set of batches is measurable. -/
scoped instance : MeasurableSpace (Fin B ↪ Fin D) := ⊤

scoped instance : DiscreteMeasurableSpace (Fin B ↪ Fin D) :=
  ⟨fun _ => MeasurableSpace.measurableSet_top⟩

/-- Summed over the injections `σ : Fin B ↪ Fin D`, a function of the restriction `r.trans σ`
is `|Fin B ↪ Fin D| / |Fin n ↪ Fin D|` times its sum over the injections `Fin n ↪ Fin D`. -/
theorem sum_trans_mul_card {n : ℕ} (r : Fin n ↪ Fin B) (φ : (Fin n ↪ Fin D) → ℝ) :
    (∑ σ : Fin B ↪ Fin D, φ (r.trans σ)) * Fintype.card (Fin n ↪ Fin D) =
      Fintype.card (Fin B ↪ Fin D) * ∑ x, φ x := by
  classical
  have := Equiv.Perm.isMultiplyPretransitive (Fin D) n
  set N : (Fin n ↪ Fin D) → ℕ := fun x => #{σ : Fin B ↪ Fin D | r.trans σ = x}
  have hfib : ∑ σ : Fin B ↪ Fin D, φ (r.trans σ) = ∑ x, (N x : ℝ) * φ x := by
    rw [← sum_fiberwise univ (fun σ => r.trans σ)]
    refine sum_congr rfl fun x _ => ?_
    rw [sum_congr rfl fun σ hσ => by rw [(mem_filter.1 hσ).2], sum_const, nsmul_eq_mul]
  have hsmul : ∀ (π : Equiv.Perm (Fin D)) (σ : Fin B ↪ Fin D),
      r.trans (π • σ) = π • r.trans σ := fun π σ => by ext; simp
  have hinv : ∀ (π : Equiv.Perm (Fin D)) x, N (π • x) = N x := fun π x =>
    card_nbij' (π⁻¹ • ·) (π • ·) (fun σ hσ => by simp_all) (fun σ hσ => by simp_all)
      (fun σ _ => smul_inv_smul π σ) (fun σ _ => inv_smul_smul π σ)
  have htot : ∑ x, N x = Fintype.card (Fin B ↪ Fin D) :=
    (card_eq_sum_card_fiberwise fun σ _ => mem_univ _).symm
  rcases isEmpty_or_nonempty (Fin n ↪ Fin D) with h | h
  · simp
  obtain ⟨x₀⟩ := h
  have hc : ∀ x, N x = N x₀ := fun x => by
    obtain ⟨π, rfl⟩ := MulAction.exists_smul_eq (Equiv.Perm (Fin D)) x₀ x
    exact hinv π x₀
  have h1 : ∑ x, (N x : ℝ) * φ x = N x₀ * ∑ x, φ x := by
    rw [mul_sum]
    exact sum_congr rfl fun x _ => by rw [hc x]
  have h2 : (Fintype.card (Fin B ↪ Fin D) : ℝ) = Fintype.card (Fin n ↪ Fin D) * N x₀ := by
    rw [← htot, sum_congr rfl fun x _ => hc x, sum_const, card_univ, smul_eq_mul, Nat.cast_mul]
  rw [hfib, h1, h2]
  ring

/-- `Fin 2 ↪ Fin D` are the pairs of distinct examples. -/
theorem sum_embedding_two (h : Fin D → Fin D → ℝ) :
    ∑ x : Fin 2 ↪ Fin D, h (x 0) (x 1) = ∑ d, ∑ e, h d e - ∑ d, h d d := by
  classical
  have : ∑ x : Fin 2 ↪ Fin D, h (x 0) (x 1) =
      ∑ p ∈ univ.filter (fun p : Fin D × Fin D => p.1 ≠ p.2), h p.1 p.2 := by
    refine sum_bij' (fun x _ => (x 0, x 1))
      (fun p hp => ⟨![p.1, p.2], fun k l hkl => by
        have := (mem_filter.1 hp).2
        fin_cases k <;> fin_cases l <;> simp_all [eq_comm]⟩)
      (fun x _ => mem_filter.2 ⟨mem_univ _, x.injective.ne (by decide)⟩)
      (fun _ _ => mem_univ _) (fun x _ => by ext k; fin_cases k <;> rfl) (fun _ _ => rfl)
      (fun _ _ => rfl)
  rw [this, sum_filter, Fintype.sum_prod_type, ← sum_sub_distrib]
  refine sum_congr rfl fun d _ => ?_
  rw [← sum_filter, filter_ne, sum_erase_eq_sub (mem_univ d)]

/-- The example `σ i` of a uniform batch is uniform. -/
theorem sum_apply_mul (i : Fin B) (h : Fin D → ℝ) :
    (∑ σ : Fin B ↪ Fin D, h (σ i)) * D = Fintype.card (Fin B ↪ Fin D) * ∑ d, h d := by
  have := sum_trans_mul_card (⟨fun _ => i, fun a b _ => Subsingleton.elim a b⟩ : Fin 1 ↪ Fin B)
    (fun x => h (x 0))
  rw [Fintype.card_congr Equiv.uniqueEmbeddingEquivResult, Fintype.card_fin,
    Fintype.sum_equiv Equiv.uniqueEmbeddingEquivResult (fun x => h (x 0)) h fun _ => rfl] at this
  exact this

/-- Two examples `σ i`, `σ j`, `i ≠ j`, of a uniform batch are a uniform pair of distinct
examples. -/
theorem sum_apply_apply_mul {i j : Fin B} (hij : i ≠ j) (h : Fin D → Fin D → ℝ) :
    (∑ σ : Fin B ↪ Fin D, h (σ i) (σ j)) * (D * D - D) =
      Fintype.card (Fin B ↪ Fin D) * (∑ d, ∑ e, h d e - ∑ d, h d d) := by
  have hr : Function.Injective ![i, j] := fun k l hkl => by
    fin_cases k <;> fin_cases l <;> simp_all [eq_comm]
  have := sum_trans_mul_card ⟨![i, j], hr⟩ (fun x => h (x 0) (x 1))
  have hc : (Fintype.card (Fin 2 ↪ Fin D) : ℝ) = D * D - D := by
    simpa using sum_embedding_two (D := D) fun _ _ => 1
  rwa [hc, sum_embedding_two] at this

/-- The hypothesis of `sum_apply_apply_mul` is satisfiable: two distinct positions. -/
example (h : Fin 3 → Fin 3 → ℝ) := sum_apply_apply_mul (B := 2) (i := 0) (j := 1) (by decide) h

/-- The mean under the uniform law on a finite type. -/
theorem integral_uniformOn_univ {α : Type*} [Fintype α] [MeasurableSpace α]
    [MeasurableSingletonClass α] (f : α → ℝ) :
    ∫ x, f x ∂uniformOn (Set.univ : Set α) = (Fintype.card α : ℝ)⁻¹ * ∑ x, f x := by
  rw [integral_fintype .of_finite, mul_sum]
  refine sum_congr rfl fun x _ => ?_
  rw [measureReal_def, uniformOn_univ, Measure.count_singleton, smul_eq_mul]
  simp

variable {ι : Type*}

/-- The footnote to eq. (2.2): the gradient of a batch sampled without replacement is
unbiased, of mean that of the gradient of a uniform example. -/
theorem integral_batchGrad_embedding (hB : B ≠ 0) (hBD : B ≤ D) (g : Fin D → ι → ℝ) (a : ι) :
    ∫ σ, batchGrad (fun i (σ : Fin B ↪ Fin D) => g (σ i)) σ a ∂uniformOn Set.univ =
      ∫ d, g d a ∂uniformOn Set.univ := by
  have hN : (Fintype.card (Fin B ↪ Fin D) : ℝ) ≠ 0 :=
    Nat.cast_ne_zero.2 (Fintype.card_pos_iff.2 ⟨Fin.castLEEmb hBD⟩).ne'
  have hD : (D : ℝ) ≠ 0 := Nat.cast_ne_zero.2 (by omega)
  have hB0 : (B : ℝ) ≠ 0 := Nat.cast_ne_zero.2 hB
  simp only [integral_uniformOn_univ, batchGrad_apply, Fintype.card_fin, ← mul_sum]
  rw [sum_comm, sum_congr rfl fun i _ => eq_div_of_mul_eq hD (sum_apply_mul i fun d => g d a),
    sum_const, card_univ, Fintype.card_fin, nsmul_eq_mul]
  field_simp

/-- The hypotheses of `integral_batchGrad_embedding` are satisfiable: one of two examples. -/
example (g : Fin 2 → ι → ℝ) (a : ι) :=
  integral_batchGrad_embedding (B := 1) one_ne_zero one_le_two g a

/-- **The footnote to eq. (2.2)**: for a batch sampled without replacement the covariance of
the batch gradient is `(1/B - 1/D) D/(D-1)` times the covariance `Σ` of the gradient of a
uniform example, where independent examples give `Σ/B`: it "scales like `1/B - 1/D`". -/
theorem covMatrix_batchGrad_embedding (hB : B ≠ 0) (hBD : B ≤ D) (hD : 1 < D)
    (g : Fin D → ι → ℝ) :
    covMatrix (batchGrad fun i (σ : Fin B ↪ Fin D) => g (σ i)) (uniformOn Set.univ) =
      ((1 / B - 1 / D) * (D / (D - 1)) : ℝ) • covMatrix g (uniformOn Set.univ) := by
  ext a b
  have hN : (Fintype.card (Fin B ↪ Fin D) : ℝ) ≠ 0 :=
    Nat.cast_ne_zero.2 (Fintype.card_pos_iff.2 ⟨Fin.castLEEmb hBD⟩).ne'
  have hB0 : (B : ℝ) ≠ 0 := Nat.cast_ne_zero.2 hB
  have hD0 : (D : ℝ) ≠ 0 := Nat.cast_ne_zero.2 (by omega)
  have hD1 : (D : ℝ) - 1 ≠ 0 := sub_ne_zero.2 (by exact_mod_cast hD.ne')
  have hDD : (D : ℝ) * D - D ≠ 0 := by
    rw [show (D : ℝ) * D - D = D * (D - 1) by ring]
    exact mul_ne_zero hD0 hD1
  simp only [covMatrix, Matrix.of_apply, Matrix.smul_apply, smul_eq_mul, covariance,
    integral_batchGrad_embedding hB hBD g]
  simp only [integral_uniformOn_univ, Fintype.card_fin]
  obtain ⟨m, hm⟩ : ∃ m : ι → ℝ, ∀ a, (D : ℝ)⁻¹ * ∑ d, g d a = m a := ⟨_, fun _ => rfl⟩
  simp only [hm]
  have hc : ∀ a, ∑ d, (g d a - m a) = 0 := fun a => by
    rw [sum_sub_distrib, sum_const, card_univ, Fintype.card_fin, nsmul_eq_mul, ← hm]
    field_simp
    ring
  have hG : ∀ (σ : Fin B ↪ Fin D) a, batchGrad (fun i (σ : Fin B ↪ Fin D) => g (σ i)) σ a - m a =
      (B : ℝ)⁻¹ * ∑ i, (g (σ i) a - m a) := fun σ a => by
    rw [batchGrad_apply, sum_sub_distrib, sum_const, card_univ, Fintype.card_fin, nsmul_eq_mul]
    field_simp
  set V := ∑ d, (g d a - m a) * (g d b - m b)
  set N := (Fintype.card (Fin B ↪ Fin D) : ℝ)
  have hS : ∀ i j : Fin B, ∑ σ : Fin B ↪ Fin D, (g (σ i) a - m a) * (g (σ j) b - m b) =
      -(N * V / (D * D - D)) + if i = j then N * V / D + N * V / (D * D - D) else 0 := by
    intro i j
    split_ifs with hij
    · subst hij
      rw [eq_div_of_mul_eq hD0 (sum_apply_mul i fun d => (g d a - m a) * (g d b - m b))]
      ring
    · have := sum_apply_apply_mul hij fun d e => (g d a - m a) * (g e b - m b)
      rw [← sum_mul_sum, hc a, zero_mul, zero_sub] at this
      rw [eq_div_of_mul_eq hDD this]
      ring
  have hL : ∀ σ : Fin B ↪ Fin D,
      (batchGrad (fun i (σ : Fin B ↪ Fin D) => g (σ i)) σ a - m a) *
        (batchGrad (fun i (σ : Fin B ↪ Fin D) => g (σ i)) σ b - m b) =
      (B : ℝ)⁻¹ * (B : ℝ)⁻¹ * ∑ i, ∑ j, (g (σ i) a - m a) * (g (σ j) b - m b) := fun σ => by
    rw [hG, hG, mul_mul_mul_comm, sum_mul_sum]
  have hswap : ∑ σ : Fin B ↪ Fin D, ∑ i, ∑ j, (g (σ i) a - m a) * (g (σ j) b - m b) =
      ∑ i, ∑ j, ∑ σ : Fin B ↪ Fin D, (g (σ i) a - m a) * (g (σ j) b - m b) := by
    rw [sum_comm]
    exact sum_congr rfl fun i _ => sum_comm
  simp only [hL]
  rw [← mul_sum, hswap]
  simp only [hS, sum_add_distrib, sum_const, card_univ, Fintype.card_fin, nsmul_eq_mul,
    sum_ite_eq, mem_univ, ite_true]
  field_simp
  ring

/-- The hypotheses of `covMatrix_batchGrad_embedding` are satisfiable: one of two examples. -/
example (g : Fin 2 → ι → ℝ) :=
  covMatrix_batchGrad_embedding (B := 1) one_ne_zero one_le_two one_lt_two g

end Transformer.NoiseScale
