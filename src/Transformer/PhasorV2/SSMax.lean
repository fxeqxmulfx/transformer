module

public import Transformer.AttentionResiduals.Section3_Attention
public import Mathlib.Analysis.SpecialFunctions.Pow.Real

/-!
# Phasor v2: Scalable-Softmax

Phasor v2 keeps only those elements of Phasor (`Transformer.Phasor`, the main model of DiffGPT)
that are chosen for it one at a time. The first is Scalable-Softmax (SSMax; Nakanishi,
arXiv:2501.19399v1, §2). Over the `n` keys a query sees, SSMax multiplies the scores `z` by
`s log n` before the softmax, Eq. (2):

    z_j ↦ n^{s z_j} / ∑_k n^{s z_k} = e^{(s log n) z_j} / ∑_k e^{(s log n) z_k}

(`ssmaxWeight`, `ssmaxWeight_eq_rpow`). Eq. (11) applies the same map to the scores
`q_n K_{1:n}ᵀ / √d` of the `n`-th query. Phasor fixes the scale at `s = 1` on scores divided by
the head width; here `s` is a parameter of the head, as in the source.

§2.2 bounds the weight of one key against the `n − 1` others. Eqs. (6), (8) and (9) are stated for
the maximum, the minimum and the second-largest score; they hold for every key `j` and for every
bound on the other scores:

* Eq. (6), softmax: if `m` bounds every score from below, the weight of `j` is at most
  `1 / ((n − 1)/e^{z_j − m} + 1)` (`depthWeight_le_of_forall_le`);
* the same argument from above: if `M` bounds every other score, the weight is at least
  `1 / ((n − 1)/e^{z_j − M} + 1)` (`le_depthWeight_of_forall_le`);
* Eqs. (8) and (9), SSMax with `s ≥ 0`: the same bounds with `e^{z_j − m}` replaced by
  `n^{s(z_j − m)}` (`ssmaxWeight_le`, `le_ssmaxWeight`);
* footnote 1: the case `s < 0` is the case `−s > 0` for the scores `−z` (`ssmaxWeight_neg`).

`Transformer.PhasorV2.SSMaxLength` lets `n` grow in these bounds.
-/

@[expose] public section

open Transformer.AttentionResiduals

namespace Transformer.PhasorV2

variable {ι : Type*} [Fintype ι]

/-- **Scalable-Softmax** (arXiv:2501.19399v1, Eq. (2)): the weight of the key `j` among the
`n = |ι|` keys with the scores `z`, the softmax of `(s log n) z`. -/
noncomputable def ssmaxWeight (s : ℝ) (z : ι → ℝ) (j : ι) : ℝ :=
  depthWeight (fun k => s * Real.log (Fintype.card ι) * z k) j

/-- Eq. (2): the SSMax weights are `n^{s z_j} / ∑_k n^{s z_k}`. -/
theorem ssmaxWeight_eq_rpow [Nonempty ι] (s : ℝ) (z : ι → ℝ) (j : ι) :
    ssmaxWeight s z j =
      (Fintype.card ι : ℝ) ^ (s * z j) / ∑ k, (Fintype.card ι : ℝ) ^ (s * z k) := by
  have hn : (0 : ℝ) < Fintype.card ι := by exact_mod_cast Fintype.card_pos
  have h (x : ℝ) : (Fintype.card ι : ℝ) ^ (s * x) =
      Real.exp (s * Real.log (Fintype.card ι) * x) := by
    rw [Real.rpow_def_of_pos hn]
    ring_nf
  simp only [ssmaxWeight, depthWeight, depthPartition, h]

/-- Three keys at `s = 1`. -/
example (z : Fin 3 → ℝ) (j : Fin 3) := ssmaxWeight_eq_rpow 1 z j

/-- **Eq. (6)**: if `m` bounds every score from below, the softmax weight of the key `j` is at
most `1 / ((n − 1)/e^{w_j − m} + 1)`. The source takes the maximum for `j` and the minimum
for `m`. -/
theorem depthWeight_le_of_forall_le (w : ι → ℝ) (j : ι) {m : ℝ} (hm : ∀ k, m ≤ w k) :
    depthWeight w j ≤ 1 / (((Fintype.card ι : ℝ) - 1) / Real.exp (w j - m) + 1) := by
  classical
  have h1 : 1 ≤ Fintype.card ι := Fintype.card_pos_iff.2 ⟨j⟩
  have hn : (1 : ℝ) ≤ Fintype.card ι := by exact_mod_cast h1
  have hrest : ((Fintype.card ι : ℝ) - 1) * Real.exp m ≤
      ∑ k ∈ Finset.univ.erase j, Real.exp (w k) := by
    have h := Finset.card_nsmul_le_sum (Finset.univ.erase j) (fun k => Real.exp (w k))
      (Real.exp m) fun k _ => Real.exp_le_exp.2 (hm k)
    rwa [Finset.card_erase_of_mem (Finset.mem_univ j), Finset.card_univ, nsmul_eq_mul,
      Nat.cast_sub h1, Nat.cast_one] at h
  have hsplit : ∑ k, Real.exp (w k) =
      Real.exp (w j) + ∑ k ∈ Finset.univ.erase j, Real.exp (w k) :=
    (Finset.add_sum_erase _ _ (Finset.mem_univ j)).symm
  have he := Real.exp_pos (w j)
  have hm0 := Real.exp_pos m
  rw [Real.exp_sub, div_div_eq_mul_div, div_add_one he.ne', one_div_div, depthWeight,
    depthPartition, hsplit]
  exact div_le_div_of_nonneg_left he.le
    (add_pos_of_nonneg_of_pos (mul_nonneg (by linarith) hm0.le) he) (by linarith)

/-- Constant scores, bounded below by `0`. -/
example := depthWeight_le_of_forall_le (ι := Fin 3) (fun _ => (0 : ℝ)) 0 (m := 0)
  fun _ => le_rfl

/-- **The lower bound of §2.2** for softmax: if `M` bounds every score but that of `j`, the
weight of `j` is at least `1 / ((n − 1)/e^{w_j − M} + 1)`. -/
theorem le_depthWeight_of_forall_le (w : ι → ℝ) (j : ι) {M : ℝ}
    (hM : ∀ k ≠ j, w k ≤ M) :
    1 / (((Fintype.card ι : ℝ) - 1) / Real.exp (w j - M) + 1) ≤ depthWeight w j := by
  classical
  have h1 : 1 ≤ Fintype.card ι := Fintype.card_pos_iff.2 ⟨j⟩
  have hrest : ∑ k ∈ Finset.univ.erase j, Real.exp (w k) ≤
      ((Fintype.card ι : ℝ) - 1) * Real.exp M := by
    have h := Finset.sum_le_card_nsmul (Finset.univ.erase j) (fun k => Real.exp (w k))
      (Real.exp M) fun k hk => Real.exp_le_exp.2 (hM k (Finset.ne_of_mem_erase hk))
    rwa [Finset.card_erase_of_mem (Finset.mem_univ j), Finset.card_univ, nsmul_eq_mul,
      Nat.cast_sub h1, Nat.cast_one] at h
  have hsplit : ∑ k, Real.exp (w k) =
      Real.exp (w j) + ∑ k ∈ Finset.univ.erase j, Real.exp (w k) :=
    (Finset.add_sum_erase _ _ (Finset.mem_univ j)).symm
  have he := Real.exp_pos (w j)
  have hS : 0 ≤ ∑ k ∈ Finset.univ.erase j, Real.exp (w k) :=
    Finset.sum_nonneg fun k _ => (Real.exp_pos (w k)).le
  rw [Real.exp_sub, div_div_eq_mul_div, div_add_one he.ne', one_div_div, depthWeight,
    depthPartition, hsplit]
  exact div_le_div_of_nonneg_left he.le (by linarith) (by linarith)

/-- Constant scores, bounded above by `0`. -/
example := le_depthWeight_of_forall_le (ι := Fin 3) (fun _ => (0 : ℝ)) 0 (M := 0)
  fun _ _ => le_rfl

/-- `e^{c(z_j − m)}` at `c = s log n` is `n^{s(z_j − m)}`. -/
theorem exp_mul_log_sub (s : ℝ) {n : ℝ} (hn : 0 < n) (a b : ℝ) :
    Real.exp (s * Real.log n * a - s * Real.log n * b) = n ^ (s * (a - b)) := by
  rw [Real.rpow_def_of_pos hn]
  congr 1
  ring

/-- Two keys at `s = 1`. -/
example := exp_mul_log_sub 1 (n := 2) (by norm_num) 1 0

/-- **Eq. (8)**: for `s ≥ 0`, if `m` bounds every score from below, the SSMax weight of the key
`j` is at most `1 / ((n − 1)/n^{s(z_j − m)} + 1)`. -/
theorem ssmaxWeight_le {s : ℝ} (hs : 0 ≤ s) (z : ι → ℝ) (j : ι) {m : ℝ} (hm : ∀ k, m ≤ z k) :
    ssmaxWeight s z j ≤
      1 / (((Fintype.card ι : ℝ) - 1) / (Fintype.card ι : ℝ) ^ (s * (z j - m)) + 1) := by
  have hn : (1 : ℝ) ≤ Fintype.card ι := by exact_mod_cast Fintype.card_pos_iff.2 ⟨j⟩
  have hc : 0 ≤ s * Real.log (Fintype.card ι) := mul_nonneg hs (Real.log_nonneg hn)
  have h := depthWeight_le_of_forall_le (fun k => s * Real.log (Fintype.card ι) * z k) j
    (m := s * Real.log (Fintype.card ι) * m) fun k => mul_le_mul_of_nonneg_left (hm k) hc
  rwa [exp_mul_log_sub s (by linarith)] at h

/-- The scores `−2, −2, −2, −2, 3` of the source's Fig. 1 at its `s = 0.43`. -/
example := ssmaxWeight_le (s := 0.43) (by norm_num)
  (fun k : Fin 5 => if k = 4 then (3 : ℝ) else -2) 4 (m := -2)
  fun k => by split_ifs <;> norm_num

/-- **Eq. (9)**: for `s ≥ 0`, if `M` bounds every score but that of `j`, the SSMax weight of `j`
is at least `1 / ((n − 1)/n^{s(z_j − M)} + 1)`. The source takes `M = z_2nd`. -/
theorem le_ssmaxWeight {s : ℝ} (hs : 0 ≤ s) (z : ι → ℝ) (j : ι) {M : ℝ}
    (hM : ∀ k ≠ j, z k ≤ M) :
    1 / (((Fintype.card ι : ℝ) - 1) / (Fintype.card ι : ℝ) ^ (s * (z j - M)) + 1) ≤
      ssmaxWeight s z j := by
  have hn : (1 : ℝ) ≤ Fintype.card ι := by exact_mod_cast Fintype.card_pos_iff.2 ⟨j⟩
  have hc : 0 ≤ s * Real.log (Fintype.card ι) := mul_nonneg hs (Real.log_nonneg hn)
  have h := le_depthWeight_of_forall_le (fun k => s * Real.log (Fintype.card ι) * z k) j
    (M := s * Real.log (Fintype.card ι) * M)
    fun k hk => mul_le_mul_of_nonneg_left (hM k hk) hc
  rwa [exp_mul_log_sub s (by linarith)] at h

/-- Fig. 1 again: the key `4` leads the others by `5`. -/
example := le_ssmaxWeight (s := 0.43) (by norm_num)
  (fun k : Fin 5 => if k = 4 then (3 : ℝ) else -2) 4 (M := -2)
  fun k hk => by simp [hk]

/-- **Footnote 1 of §2.2**: SSMax at `−s` on the scores `−z` is SSMax at `s` on `z`, so the
case `s < 0` reduces to `s > 0`. -/
theorem ssmaxWeight_neg (s : ℝ) (z : ι → ℝ) (j : ι) :
    ssmaxWeight (-s) (fun k => -z k) j = ssmaxWeight s z j := by
  simp only [ssmaxWeight, neg_mul, mul_neg, neg_neg]

end Transformer.PhasorV2
