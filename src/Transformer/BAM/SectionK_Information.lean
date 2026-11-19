module

public import Transformer.BAM.SectionK_Partition

/-!
# Appendix K.3: `Z` as the exponential of an information

Bianchessi, Aguirre, Barros, Kupssinskü, arXiv:2505.22842v4, Appendix K.3.

Appendix K.3 calls `I = Σ_j p_joint(j) ln(p_joint(j) / (p_cont(j) p_pos(j)))`, with
`p_joint = softmax(f + g)`, the mutual information between content and position (`mutualInfo`),
and shows `I = ln(Z_cont Z_pos / Z_joint) = ln Z` (`mutualInfo_eq`). The three distributions are
over the same keys `j`, and `p_cont ⊙ p_pos` sums to `⟨p_cont, p_pos⟩`, not `1`: `I` is the
divergence of `p_joint` from that unnormalized product, `I = −ln⟨p_cont, p_pos⟩ ≥ 0`
(`mutualInfo_eq_neg_log_overlap`, `mutualInfo_nonneg`).

* "`Z = exp(I(cont; pos))`" (`exp_mutualInfo`): `Z ≥ 1` "grows exponentially with the amount of
  shared information".
* "`Z = 1` if and only if content and position are statistically independent (`I = 0`)":
  `I = 0 ↔ Z = 1` (`mutualInfo_eq_zero_iff`), and `Z = 1 ↔ p_joint = p_cont · p_pos`
  (`partition_ratio_eq_one_iff`). For real content scores on two keys or more this never
  happens: `I > 0` (`mutualInfo_pos`).
-/

@[expose] public section

namespace Transformer.BAM

open Finset

variable {ι : Type*} [Fintype ι]

/-- The mutual information between content and position of Appendix K.3,
`I = Σ_j p_joint(j) ln(p_joint(j) / (p_cont(j) p_pos(j)))` with `p_joint = softmax(f + g)`. -/
noncomputable def mutualInfo (f g : ι → EReal) : ℝ :=
  ∑ j, softmax (f + g) j * Real.log (softmax (f + g) j / (softmax f j * softmax g j))

/-- **Appendix K.3: `I(cont; pos) = ln(Z_cont Z_pos / Z_joint) = ln Z`.** -/
theorem mutualInfo_eq (f g : ι → EReal) :
    mutualInfo f g = Real.log (partition f * partition g / partition (f + g)) := by
  have hterm : ∀ j, softmax (f + g) j * Real.log (softmax (f + g) j / (softmax f j * softmax g j)) =
      softmax (f + g) j * Real.log (partition f * partition g / partition (f + g)) := fun j => by
    rcases eq_or_ne (softmax (f + g) j) 0 with h | h
    · rw [h, zero_mul, zero_mul]
    have hfac := softmax_add_eq_mul f g j
    have hpq : softmax f j * softmax g j ≠ 0 := fun h0 => h (by rw [hfac, h0, mul_zero])
    have hratio : softmax (f + g) j / (softmax f j * softmax g j) =
        partition f * partition g / partition (f + g) := by
      rw [hfac, mul_div_cancel_right₀ _ hpq]
    rw [hratio]
  rw [mutualInfo, sum_congr rfl fun j _ => hterm j, ← sum_mul]
  by_cases h : partition (f + g) = 0
  · simp [softmax, h]
  · rw [sum_softmax h, one_mul]

/-- **Appendix K.3: `I = −ln⟨p_cont, p_pos⟩`**, as `Z = 1/⟨p_cont, p_pos⟩`. -/
theorem mutualInfo_eq_neg_log_overlap (f g : ι → EReal) :
    mutualInfo f g = -Real.log (overlap f g) := by
  rw [mutualInfo_eq, partition_mul_div_partition_add, one_div, Real.log_inv]

/-- **`I ≥ 0`**, as the overlap is at most `1`. -/
theorem mutualInfo_nonneg (f g : ι → EReal) : 0 ≤ mutualInfo f g := by
  rw [mutualInfo_eq_neg_log_overlap, neg_nonneg]
  exact Real.log_nonpos (overlap_nonneg f g) (overlap_le_one f g)

/-- A nonzero joint partition function makes the content and position ones positive. -/
theorem partition_pos_of_add {f g : ι → EReal} (h : partition (f + g) ≠ 0) :
    0 < partition f ∧ 0 < partition g := by
  refine ⟨(partition_nonneg f).lt_of_ne fun h0 => h ?_,
    (partition_nonneg g).lt_of_ne fun h0 => h ?_⟩
  · simp [partition, weight_add, weight_eq_zero_of_partition_eq_zero h0.symm]
  · simp [partition, weight_add, weight_eq_zero_of_partition_eq_zero h0.symm]

/-- The hypothesis of `partition_pos_of_add` holds for zero scores on two keys. -/
example : 0 < partition (0 : Fin 2 → EReal) ∧ 0 < partition (0 : Fin 2 → EReal) :=
  partition_pos_of_add (by norm_num [partition])

/-- **Appendix K.3: `Z = exp(I(cont; pos))`.** -/
theorem exp_mutualInfo {f g : ι → EReal} (h : partition (f + g) ≠ 0) :
    Real.exp (mutualInfo f g) = partition f * partition g / partition (f + g) := by
  obtain ⟨hf, hg⟩ := partition_pos_of_add h
  rw [mutualInfo_eq, Real.exp_log (div_pos (mul_pos hf hg) ((partition_nonneg _).lt_of_ne' h))]

/-- The hypothesis of `exp_mutualInfo` holds for zero scores on two keys. -/
example : Real.exp (mutualInfo (0 : Fin 2 → EReal) 0) =
    partition (0 : Fin 2 → EReal) * partition (0 : Fin 2 → EReal) /
      partition ((0 : Fin 2 → EReal) + 0) :=
  exp_mutualInfo (by norm_num [partition])

/-- **Appendix K.3: `I = 0` exactly when `Z = 1`.** -/
theorem mutualInfo_eq_zero_iff {f g : ι → EReal} (h : partition (f + g) ≠ 0) :
    mutualInfo f g = 0 ↔ partition f * partition g / partition (f + g) = 1 := by
  rw [← exp_mutualInfo h, Real.exp_eq_one_iff]

/-- The hypothesis of `mutualInfo_eq_zero_iff` holds for zero scores on two keys. -/
example : mutualInfo (0 : Fin 2 → EReal) 0 = 0 ↔
    partition (0 : Fin 2 → EReal) * partition (0 : Fin 2 → EReal) /
      partition ((0 : Fin 2 → EReal) + 0) = 1 :=
  mutualInfo_eq_zero_iff (by norm_num [partition])

/-- **Appendix K.3: `Z = 1` exactly when content and position are independent**,
`p_joint(j) = p_cont(j) · p_pos(j)` at every key. -/
theorem partition_ratio_eq_one_iff {f g : ι → EReal} (h : partition (f + g) ≠ 0) :
    partition f * partition g / partition (f + g) = 1 ↔
      ∀ j, softmax (f + g) j = softmax f j * softmax g j := by
  constructor
  · intro hZ j
    rw [softmax_add_eq_mul, hZ, one_mul]
  · intro hj
    have h1 : overlap f g = 1 :=
      calc overlap f g = ∑ j, softmax (f + g) j := sum_congr rfl fun j _ => (hj j).symm
        _ = 1 := sum_softmax h
    rw [partition_mul_div_partition_add, h1, div_one]

/-- The hypothesis of `partition_ratio_eq_one_iff` holds for zero scores on two keys. -/
example : partition (0 : Fin 2 → EReal) * partition (0 : Fin 2 → EReal) /
      partition ((0 : Fin 2 → EReal) + 0) = 1 ↔
    ∀ j, softmax ((0 : Fin 2 → EReal) + 0) j =
      softmax (0 : Fin 2 → EReal) j * softmax (0 : Fin 2 → EReal) j :=
  partition_ratio_eq_one_iff (by norm_num [partition])

/-- **For real content scores on two keys or more, `I > 0`**: content and position are never
independent in the sense of Appendix K.3. -/
theorem mutualInfo_pos {f : ι → ℝ} {g : ι → EReal} (hι : 1 < Fintype.card ι)
    (hg : partition g ≠ 0) : 0 < mutualInfo (fun z => (f z : EReal)) g := by
  rw [mutualInfo_eq_neg_log_overlap, neg_pos]
  exact Real.log_neg (overlap_pos hg) (overlap_lt_one hι hg)

/-- The hypotheses of `mutualInfo_pos` hold for two keys and zero position scores. -/
example (f : Fin 2 → ℝ) : 0 < mutualInfo (fun z => (f z : EReal)) 0 :=
  mutualInfo_pos (by simp) (by norm_num [partition])

end Transformer.BAM
