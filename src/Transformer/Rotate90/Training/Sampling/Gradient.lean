module

public import Transformer.Rotate90.Training.Sampling.Basic

/-!
# The sampled gradient measures label imbalance at a collapsed embedding

Source: `review/flat_spectrum.md`, *Sampled updates can break aliases*, equations
(S3)--(S5), (S7). Batches are finite indexed families, so repeated examples are allowed.
Their loss is the ordinary mean of actual tied-model cross-entropies. A collapsed
embedding has a common hidden state on all inputs, for arbitrary branch weights.
The gradient formula depends on empirical label frequencies, not just the task's
full-table spectrum. It is an exact local formula, not a convergence assertion.
-/

@[expose] public section

namespace Transformer.Rotate90.Training.Sampling

open Landscape.Neural
open scoped BigOperators Classical

variable {V K H J : Type*} [Fintype V] [Fintype K] [Fintype H] [Fintype J]

/-- An indexed batch allows sampling with replacement as well as fixed subsets. -/
noncomputable def batchLoss (eps : ℝ) (blocks : List (Power.BlockWeights K H))
    (a b target : J → V) (E : Table V (Channels K)) : ℝ :=
  (∑ j, rowLoss eps blocks (a j) (b j) (target j) E) / Fintype.card J

/-- Empirical target frequency; the later nonempty hypothesis excludes 0/0. -/
noncomputable def frequency [DecidableEq V] (target : J → V) (q : V) : ℝ :=
  (∑ j, if target j = q then (1 : ℝ) else 0) / Fintype.card J

/-- Analyticity justifies differentiation of the actual sampled objective. -/
theorem analyticAt_batchLoss {eps : ℝ} (heps : 0 < eps)
    (blocks : List (Power.BlockWeights K H)) (a b target : J → V)
    (E : Table V (Channels K)) : AnalyticAt ℝ (batchLoss eps blocks a b target) E := by
  exact (Finset.analyticAt_fun_sum Finset.univ fun j _ =>
    analyticAt_rowLoss heps blocks (a j) (b j) (target j) E).div_const

/-- The exact singleton embedding gradient. Input and output occurrences are
all tied; the cancellation theorem removes their common hidden derivative. -/
theorem gradient_rowLoss_constant [DecidableEq V] {eps : ℝ} (heps : 0 < eps)
    (blocks : List (Power.BlockWeights K H)) (v : Channels K → ℝ)
    (a b target q : V) (i : Channels K) :
    entry (gradient (rowLoss eps blocks a b target) (constantTable v)) q i =
      ((Fintype.card V : ℝ)⁻¹ - if target = q then 1 else 0) *
        hidden eps blocks v v i := by
  classical
  let h : Table V (Channels K) → Channels K → ℝ := fun E =>
    hidden eps blocks (entry E a) (entry E b)
  have hh : ∀ c, DifferentiableAt ℝ (fun E => h E c) (constantTable v) := by
    intro c
    exact (analyticAt_hidden heps blocks
      (fun j => (PiLp.proj (𝕜 := ℝ) 2 (fun _ : V × Channels K => ℝ) (a, j)).analyticAt _)
      (fun j => (PiLp.proj (𝕜 := ℝ) 2 (fun _ : V × Channels K => ℝ) (b, j)).analyticAt _)
      c).differentiableAt
  rw [gradient_entry]
  change fderiv ℝ (fun E => crossEntropy (tiedReadout (entry E) (h E)) target)
    (constantTable v) (basis q i) = _
  rw [fderiv_tiedLoss_constant v hh]
  have hb (r : V) :
      (∑ c, entry (basis q i) r c * h (constantTable v) c) =
        if r = q then hidden eps blocks v v i else 0 := by
    rw [show h (constantTable v) = hidden eps blocks v v from rfl]
    by_cases hr : r = q
    · subst r
      simp [entry, basis, PiLp.single_apply]
    · simp [entry, basis, hr]
  simp_rw [hb]
  simp only [Finset.sum_ite_eq', Finset.mem_univ, ite_true]
  split_ifs <;> simp only [div_eq_mul_inv] <;> ring

/-- Averaging singleton derivatives gives the exact empirical-frequency formula.
No assumption about the batch's relation to the complete table is used. -/
theorem gradient_batchLoss_constant [DecidableEq V] [Nonempty J] {eps : ℝ} (heps : 0 < eps)
    (blocks : List (Power.BlockWeights K H)) (v : Channels K → ℝ)
    (a b target : J → V) (q : V) (i : Channels K) :
    entry (gradient (batchLoss eps blocks a b target) (constantTable v)) q i =
      ((Fintype.card V : ℝ)⁻¹ - frequency target q) * hidden eps blocks v v i := by
  classical
  have hd := (HasFDerivAt.fun_sum (u := Finset.univ) fun j _ =>
    (analyticAt_rowLoss heps blocks (a j) (b j) (target j)
      (constantTable v)).differentiableAt.hasFDerivAt).mul_const (Fintype.card J : ℝ)⁻¹
  rw [gradient_entry]
  unfold batchLoss
  simp only [div_eq_mul_inv]
  rw [hd.fderiv]
  simp only [smul_apply, sum_apply, smul_eq_mul]
  simp_rw [← gradient_entry, gradient_rowLoss_constant heps]
  simp only [sub_mul, Finset.sum_sub_distrib, Finset.sum_const, Finset.card_univ,
    nsmul_eq_mul, ← Finset.sum_mul]
  have hn : (Fintype.card J : ℝ) ≠ 0 := by exact_mod_cast ne_of_gt (Fintype.card_pos)
  unfold frequency
  field_simp

/-- Unequal empirical counts split any two collapsed rows when their common
hidden coordinate is nonzero. Uniform softmax terms cancel from the difference. -/
theorem gradient_row_difference [DecidableEq V] [Nonempty J] {eps : ℝ} (heps : 0 < eps)
    (blocks : List (Power.BlockWeights K H)) (v : Channels K → ℝ)
    (a b target : J → V) (q r : V) (i : Channels K) :
    entry (gradient (batchLoss eps blocks a b target) (constantTable v)) q i -
      entry (gradient (batchLoss eps blocks a b target) (constantTable v)) r i =
        (frequency target r - frequency target q) * hidden eps blocks v v i := by
  rw [gradient_batchLoss_constant heps, gradient_batchLoss_constant heps]
  ring

/-- A balanced batch has zero embedding gradient at this collapsed point.
This concerns the embedding partial gradient; no convergence follows. -/
theorem gradient_batch_zero_of_uniform [DecidableEq V] [Nonempty J] {eps : ℝ} (heps : 0 < eps)
    (blocks : List (Power.BlockWeights K H)) (v : Channels K → ℝ)
    (a b target : J → V) (hfreq : ∀ q, frequency target q = (Fintype.card V : ℝ)⁻¹) :
    gradient (batchLoss eps blocks a b target) (constantTable v) = 0 := by
  ext ⟨q, i⟩
  change entry (gradient (batchLoss eps blocks a b target) (constantTable v)) q i = 0
  rw [gradient_batchLoss_constant heps, hfreq, sub_self, zero_mul]

/-- Uniformly sampled targets have a nonzero mean squared row-separating
gradient whenever the common hidden coordinate is nonzero: its exact value is
2/card(V) times that coordinate squared. The mean gradient may still vanish.
Source: the note's equation (S7); arbitrary selected operand pairs are allowed. -/
theorem singleton_transverse_energy {eps : ℝ} (heps : 0 < eps)
    (blocks : List (Power.BlockWeights K H)) (v : Channels K → ℝ)
    (a b : V → V) (q r : V) (hne : q ≠ r) (i : Channels K) :
    (∑ t : V, (entry (gradient (rowLoss eps blocks (a t) (b t) t) (constantTable v)) q i -
      entry (gradient (rowLoss eps blocks (a t) (b t) t) (constantTable v)) r i) ^ 2) /
        Fintype.card V = 2 / (Fintype.card V : ℝ) * hidden eps blocks v v i ^ 2 := by
  classical
  let z := hidden eps blocks v v i
  have hc (t : V) :
      (((Fintype.card V : ℝ)⁻¹ - if t = q then 1 else 0) * z -
        ((Fintype.card V : ℝ)⁻¹ - if t = r then 1 else 0) * z) ^ 2 =
          (if t = q then z ^ 2 else 0) + (if t = r then z ^ 2 else 0) := by
    by_cases hq : t = q
    · subst t
      simp only [ite_true, hne, ite_false]
      ring
    · by_cases hr : t = r
      · subst t
        simp only [hne.symm, ite_false, ite_true]
        ring
      · simp only [hq, hr, ite_false]
        ring
  simp_rw [gradient_rowLoss_constant heps]
  change (∑ t : V, (((Fintype.card V : ℝ)⁻¹ - if t = q then 1 else 0) * z -
    ((Fintype.card V : ℝ)⁻¹ - if t = r then 1 else 0) * z) ^ 2) /
      Fintype.card V = 2 / (Fintype.card V : ℝ) * z ^ 2
  simp_rw [hc]
  rw [Finset.sum_add_distrib]
  simp only [Finset.sum_ite_eq', Finset.mem_univ, ite_true]
  ring

/-- A singleton has a negative target gradient and positive competitor gradient
along every positive common hidden coordinate of the collapsed model. -/
theorem row_gradient_signs [Nontrivial V] {eps : ℝ} (heps : 0 < eps)
    (blocks : List (Power.BlockWeights K H)) (v : Channels K → ℝ)
    (a b target other : V) (hne : target ≠ other) (i : Channels K)
    (hh : 0 < hidden eps blocks v v i) :
    entry (gradient (rowLoss eps blocks a b target) (constantTable v)) target i < 0 ∧
      0 < entry (gradient (rowLoss eps blocks a b target) (constantTable v)) other i := by
  classical
  have hn : (1 : ℝ) < Fintype.card V := by
    exact_mod_cast (Fintype.one_lt_card_iff_nontrivial.mpr inferInstance : 1 < Fintype.card V)
  have hp : (0 : ℝ) < Fintype.card V := lt_trans zero_lt_one hn
  rw [gradient_rowLoss_constant heps, gradient_rowLoss_constant heps]
  simp only [ite_true, hne, ite_false, sub_zero]
  exact ⟨mul_neg_of_neg_of_pos (sub_neg.mpr ((inv_lt_one₀ hp).mpr hn)) hh,
    mul_pos (inv_pos.mpr hp) hh⟩

example := row_gradient_signs (V := Fin 3) (K := Fin 8) (H := Fin 43)
  (eps := 1 / 8388608) (by norm_num) [Power.zeroBlock] (fun _ => 1)
  1 1 1 2 (by decide) (0, 1) (hidden_ones_pos (by norm_num) 1 0)
example := analyticAt_batchLoss (V := Fin 3) (J := Fin 2) (K := Fin 1) (H := Fin 2)
  (eps := 1 / 8388608) (by norm_num) [Power.zeroBlock]
  (fun _ => 1) (fun _ => 1) (fun _ => 1) (constantTable fun _ => 1)
example := gradient_rowLoss_constant (V := Fin 3) (K := Fin 1) (H := Fin 2)
  (eps := 1 / 8388608) (by norm_num) [Power.zeroBlock] (fun _ => 1) 1 1 1 2 (0, 1)
example := gradient_batchLoss_constant (V := Fin 3) (J := Fin 2) (K := Fin 1) (H := Fin 2)
  (eps := 1 / 8388608) (by norm_num) [Power.zeroBlock] (fun _ => 1)
  (fun _ => 1) (fun _ => 1) (fun _ => 1) 2 (0, 1)
example := gradient_row_difference (V := Fin 3) (J := Fin 2) (K := Fin 1) (H := Fin 2)
  (eps := 1 / 8388608) (by norm_num) [Power.zeroBlock] (fun _ => 1)
  (fun _ => 1) (fun _ => 1) (fun _ => 1) 1 2 (0, 1)
example := gradient_batch_zero_of_uniform (V := Fin 3) (J := Fin 3) (K := Fin 1) (H := Fin 2)
  (eps := 1 / 8388608) (by norm_num) [Power.zeroBlock] (fun _ => 1)
  (fun _ => 1) (fun _ => 1) id
  (by intro q; fin_cases q <;> norm_num [frequency, Fin.sum_univ_succ])
example := singleton_transverse_energy (V := ZMod 3) (K := Fin 8) (H := Fin 43)
  (eps := 1 / 8388608) (by norm_num) [Power.zeroBlock] (fun _ => 1)
  id (fun _ => 1) 1 2 (by decide) (0, 1)

end Transformer.Rotate90.Training.Sampling
