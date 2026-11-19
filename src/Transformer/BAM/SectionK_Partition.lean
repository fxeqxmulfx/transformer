module

public import Transformer.BAM.Section2_BayesianAttention

/-!
# Appendix K.1–K.2: the normalizer `Z` in statistical physics and geometry

Bianchessi, Aguirre, Barros, Kupssinskü, arXiv:2505.22842v4, Appendix K.1–K.2.

With `Z_cont = Σ_j e^{f_j}`, `Z_pos = Σ_j e^{g_j}` and `Z_joint = Σ_j e^{f_j + g_j}`
(`partition`), Appendix K calls `Z = Z_cont Z_pos / Z_joint` the normalizer that restores
`softmax(f + g)_j = Z · softmax(f)_j · softmax(g)_j` (`softmax_add_eq_mul`), and equates it with
`Σ_j softmax(f)_j softmax(g)_j = ⟨p_cont, p_pos⟩` (`overlap`), in K.1 and again in K.2. These are
reciprocals (`partition_mul_div_partition_add`, §2.1), as K.2 concludes: "`Z = 1/⟨p_cont,
p_pos⟩`". The equation is false (`partition_ratio_ne_overlap`), and so is "`Z = Σ_j p p` restores
`Σ_j [p p] Z = 1`", which holds with `Z = 1/⟨p_cont, p_pos⟩` (`sum_mul_div_overlap`).

* K.1: `ln Z = ln Z_cont + ln Z_pos − ln Z_joint` (`log_partition_ratio`). An inverse temperature
  `γ` controls how sharply attention peaks: the top key's weight in `softmax(γ s)` grows with `γ`
  (`softmax_smul_le`).
* K.2: `⟨p_cont, p_pos⟩ ∈ [0, 1]` (`overlap_nonneg`, `overlap_le_one`; with `g = f`,
  `‖p‖₂² ≤ 1`), and it is `0` exactly for disjoint supports (`overlap_eq_zero_iff`). It is not
  "maximal when they coincide" (`overlap_self_lt`). The product `p_cont ⊙ p_pos` sums to
  `⟨p_cont, p_pos⟩`, which is `< 1` for real content scores on two keys or more
  (`overlap_lt_one`), but `1` on one key (`overlap_fin_one`).
-/

@[expose] public section

namespace Transformer.BAM

open Finset

variable {ι : Type*} [Fintype ι]

/-- **Appendix K.1: `Z = Z_cont Z_pos / Z_joint` restores the factorization**
`softmax(f + g)_j = Z · softmax(f)_j · softmax(g)_j`. -/
theorem softmax_add_eq_mul (f g : ι → EReal) (j : ι) :
    softmax (f + g) j =
      partition f * partition g / partition (f + g) * (softmax f j * softmax g j) := by
  rw [softmax_add, partition_mul_div_partition_add]
  ring

/-- **Appendix K.1–K.2 equate reciprocals.** "`Z = Z_cont Z_pos / Z_joint =
Σ_j softmax(f_j) × softmax(g_j)`" (K.1; `Z = ⟨p_cont, p_pos⟩` in K.2) fails for zero scores on
two keys, where `Z = 2` and the sum is `1/2`; so does "`Z = Σ_j p p` restores `Σ_j [p p] Z = 1`"
(K.2), as `(1/2) · (1/2) ≠ 1`. -/
theorem partition_ratio_ne_overlap :
    partition (0 : Fin 2 → EReal) * partition (0 : Fin 2 → EReal) /
          partition ((0 : Fin 2 → EReal) + 0) ≠ overlap (0 : Fin 2 → EReal) 0 ∧
      overlap (0 : Fin 2 → EReal) 0 * overlap (0 : Fin 2 → EReal) 0 ≠ 1 := by
  norm_num [overlap, softmax, partition]

/-- **Appendix K.2: `1/⟨p_cont, p_pos⟩` re-normalizes the product** `p_cont ⊙ p_pos`, the
paper's "`Σ_j [p p] Z = 1`" with `Z = 1/⟨p_cont, p_pos⟩`. -/
theorem sum_mul_div_overlap {f g : ι → EReal} (h : overlap f g ≠ 0) :
    ∑ j, softmax f j * softmax g j / overlap f g = 1 := by
  rw [← sum_div]
  exact div_self h

/-- The hypothesis of `sum_mul_div_overlap` holds for zero scores on two keys. -/
example : ∑ j : Fin 2, softmax (0 : Fin 2 → EReal) j * softmax (0 : Fin 2 → EReal) j /
    overlap (0 : Fin 2 → EReal) 0 = 1 :=
  sum_mul_div_overlap (by norm_num [overlap, softmax, partition])

/-- **Appendix K.1: the interaction free energy**,
`ln Z = ln Z_cont + ln Z_pos − ln Z_joint`. -/
theorem log_partition_ratio {f g : ι → EReal} (hf : partition f ≠ 0) (hg : partition g ≠ 0)
    (hfg : partition (f + g) ≠ 0) :
    Real.log (partition f * partition g / partition (f + g)) =
      Real.log (partition f) + Real.log (partition g) - Real.log (partition (f + g)) := by
  rw [Real.log_div (mul_ne_zero hf hg) hfg, Real.log_mul hf hg]

/-- The hypotheses of `log_partition_ratio` hold for zero scores on two keys. -/
example : Real.log (partition (0 : Fin 2 → EReal) * partition (0 : Fin 2 → EReal) /
      partition ((0 : Fin 2 → EReal) + 0)) =
    Real.log (partition (0 : Fin 2 → EReal)) + Real.log (partition (0 : Fin 2 → EReal)) -
      Real.log (partition ((0 : Fin 2 → EReal) + 0)) :=
  log_partition_ratio (by norm_num [partition]) (by norm_num [partition])
    (by norm_num [partition])

/-- **Appendix K.1: an inverse temperature `γ` sharpens the peak.** With scores `γ s`, the weight
of a top-scoring key `j` grows with `γ`. -/
theorem softmax_smul_le {s : ι → ℝ} {j : ι} (hj : ∀ z, s z ≤ s j) {γ₁ γ₂ : ℝ} (h : γ₁ ≤ γ₂) :
    softmax (fun z => ((γ₁ * s z : ℝ) : EReal)) j ≤
      softmax (fun z => ((γ₂ * s z : ℝ) : EReal)) j := by
  have key : ∀ γ : ℝ, softmax (fun z => ((γ * s z : ℝ) : EReal)) j =
      1 / ∑ z, Real.exp (γ * s z - γ * s j) := fun γ => by
    simp only [softmax, partition, weight_coe, Real.exp_sub, ← sum_div, one_div_div]
  rw [key, key]
  refine one_div_le_one_div_of_le (sum_pos (fun z _ => Real.exp_pos _) ⟨j, mem_univ j⟩)
    (sum_le_sum fun z _ => Real.exp_le_exp.2 ?_)
  nlinarith [mul_nonneg (sub_nonneg.2 h) (sub_nonneg.2 (hj z))]

/-- The hypotheses of `softmax_smul_le` hold for the scores `(0, 1)`, the key `1`, `γ = 1, 2`. -/
example : softmax (fun z : Fin 2 => ((1 * ![0, 1] z : ℝ) : EReal)) 1 ≤
    softmax (fun z : Fin 2 => ((2 * ![0, 1] z : ℝ) : EReal)) 1 :=
  softmax_smul_le (fun z => by fin_cases z <;> simp) one_le_two

/-- **Appendix K.2: `⟨p_cont, p_pos⟩ ≥ 0`.** -/
theorem overlap_nonneg (f g : ι → EReal) : 0 ≤ overlap f g :=
  sum_nonneg fun z _ => mul_nonneg (softmax_nonneg f z) (softmax_nonneg g z)

/-- **Appendix K.2: `⟨p_cont, p_pos⟩ ≤ 1`**; with `g = f`, `‖p‖₂² ≤ 1`. -/
theorem overlap_le_one (f g : ι → EReal) : overlap f g ≤ 1 :=
  (sum_le_sum fun z _ => mul_le_of_le_one_left (softmax_nonneg g z) (softmax_le_one f z)).trans
    (sum_softmax_le_one g)

/-- **Appendix K.2: the overlap is minimal, `0`, exactly for disjoint supports.** -/
theorem overlap_eq_zero_iff (f g : ι → EReal) :
    overlap f g = 0 ↔ ∀ j, softmax f j = 0 ∨ softmax g j = 0 := by
  rw [overlap, sum_eq_zero_iff_of_nonneg fun z _ =>
    mul_nonneg (softmax_nonneg f z) (softmax_nonneg g z)]
  simp [mul_eq_zero]

/-- **Appendix K.2: the overlap is not "maximal when they coincide".** For `p = (3/5, 2/5)`,
`⟨p, p⟩ = 13/25` is below `⟨p, δ₀⟩ = 3/5`, where masking key `1` gives `δ₀`. -/
theorem overlap_self_lt :
    overlap ![((Real.log 3 : ℝ) : EReal), ((Real.log 2 : ℝ) : EReal)]
        ![((Real.log 3 : ℝ) : EReal), ((Real.log 2 : ℝ) : EReal)] <
      overlap ![((Real.log 3 : ℝ) : EReal), ((Real.log 2 : ℝ) : EReal)] ![0, ⊥] := by
  norm_num [overlap, softmax, partition, Fin.sum_univ_two, Real.exp_log]

/-- Scores with a positive partition function give some key a positive weight. -/
theorem exists_softmax_pos {g : ι → EReal} (hg : partition g ≠ 0) : ∃ k, 0 < softmax g k := by
  by_contra hcon
  push Not at hcon
  have h := sum_nonpos fun j (_ : j ∈ univ) => hcon j
  rw [sum_softmax hg] at h
  norm_num at h

/-- The hypothesis of `exists_softmax_pos` holds for zero scores on two keys. -/
example : ∃ k, 0 < softmax (0 : Fin 2 → EReal) k :=
  exists_softmax_pos (by norm_num [partition])

/-- **Appendix K.2: `⟨p_cont, p_pos⟩ < 1` for real content on two keys or more**: every content
weight is below `1`, so `p_cont ⊙ p_pos` sums to less than `Σ_j p_pos(j) = 1`. -/
theorem overlap_lt_one {f : ι → ℝ} {g : ι → EReal} (hι : 1 < Fintype.card ι)
    (hg : partition g ≠ 0) : overlap (fun z => (f z : EReal)) g < 1 := by
  have hp : ∀ z, softmax (fun z => (f z : EReal)) z < 1 := fun z => by
    obtain ⟨y, hy⟩ := Fintype.exists_ne_of_one_lt_card hι z
    have hlt : weight (f z : EReal) < partition (fun z => (f z : EReal)) := by
      unfold partition
      exact single_lt_sum (f := fun k => weight ((f k : ℝ) : EReal)) hy (mem_univ z) (mem_univ y)
        (by simp [Real.exp_pos]) (fun k _ _ => weight_nonneg _)
    rw [softmax, div_lt_one ((weight_nonneg _).trans_lt hlt)]
    exact hlt
  obtain ⟨k, hk⟩ := exists_softmax_pos hg
  calc overlap (fun z => (f z : EReal)) g < ∑ j, softmax g j :=
        sum_lt_sum (fun j _ => mul_le_of_le_one_left (softmax_nonneg g j) (hp j).le)
          ⟨k, mem_univ k, by nlinarith [mul_pos (sub_pos.2 (hp k)) hk]⟩
    _ = 1 := sum_softmax hg

/-- The hypotheses of `overlap_lt_one` hold for two keys and zero position scores. -/
example (f : Fin 2 → ℝ) : overlap (fun z => (f z : EReal)) 0 < 1 :=
  overlap_lt_one (by simp) (by norm_num [partition])

/-- For real content scores, the overlap with any position marginal is positive. -/
theorem overlap_pos {f : ι → ℝ} {g : ι → EReal} (hg : partition g ≠ 0) :
    0 < overlap (fun z => (f z : EReal)) g := by
  obtain ⟨k, hk⟩ := exists_softmax_pos hg
  have hp : 0 < softmax (fun z => (f z : EReal)) k :=
    div_pos (by simp [Real.exp_pos]) (partition_pos (z := k) (by simp) (by simp))
  exact (mul_pos hp hk).trans_le (single_le_sum (fun j _ =>
    mul_nonneg (softmax_nonneg _ j) (softmax_nonneg g j)) (mem_univ k))

/-- The hypothesis of `overlap_pos` holds for zero position scores on two keys. -/
example (f : Fin 2 → ℝ) : 0 < overlap (fun z => (f z : EReal)) 0 :=
  overlap_pos (by norm_num [partition])

/-- **Appendix K.2: on one key, `⟨p_cont, p_pos⟩ = 1`**, so "sums to `⟨p_cont, p_pos⟩ ≠ 1`" needs
two keys or more. -/
theorem overlap_fin_one (f g : Fin 1 → ℝ) :
    overlap (fun z => (f z : EReal)) (fun z => (g z : EReal)) = 1 := by
  simp [overlap, softmax, partition, Real.exp_ne_zero]

end Transformer.BAM
