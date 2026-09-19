/-
# What Four Over Six does to the witness, coin by coin

arXiv:2601.22813v2, §4.2.

With the scales of `Transformer.Quartet.Section4_Witness` in hand the two
branches are explicit functions of the one coin that matters, the coin of the
single non-zero entry.  On the `4.0` branch that entry is presented to E2M1 as
`64/17` and comes back as `17/16` with probability `13/17` and as `51/64`
otherwise; on the `6.0` branch it is `96/17` and comes back as `17/16` with
probability `14/17` and as `17/24` otherwise.

The selection then reads those realized values.  Below `13/17` both branches
round up and their errors agree, so the tie goes to `4.0`; between `13/17` and
`14/17` the `4.0` branch has already fallen to `51/64` while `6.0` still
rounds up, so `6.0` wins; above `14/17` both have fallen, and the coarser
`6.0` step makes it the worse of the two, so `4.0` wins again.  The upshot is
`q46At_biasWitness`: the entry comes back as `17/16` on the first `14/17` of
the coin and as `51/64` on the rest — the distribution of the `6.0` branch
with the values of the `4.0` branch, which is nobody's unbiased estimator.
-/

import Transformer.Quartet.Section4_Witness

namespace Transformer
namespace Quartet

/-- The zero entries of the witness are on the E2M1 grid at either scale, so
they are returned exactly and the coin decides nothing (§3.1). -/
theorem qSRAt_biasWitness_ne (c : ℝ) (hc : c ≠ 0) (j : Fin 16) (hj : j ≠ 0) (t : ℝ) :
    qSRAt c biasWitness 0 j t = 0 := by
  unfold qSRAt
  rw [groupScaleSR_biasWitness c hc, show biasWitness 0 j = 0 from by simp [biasWitness, hj],
    zero_div, sr, floorOn_fp4_zero, ceilOn_fp4_zero]
  norm_num

/-- The hypotheses of `qSRAt_biasWitness_ne` are satisfiable: the grid maximum
`4.0` and the second entry of the group. -/
example : (4 : ℝ) ≠ 0 ∧ (1 : Fin 16) ≠ 0 := ⟨by norm_num, by decide⟩

/-- The `4.0` branch on the non-zero entry: `64/17` rounds up to `4` — that is
`17/16` after dequantization — when the coin falls below `13/17`, and down to
`3`, that is `51/64`, otherwise (§4.2). -/
theorem qSRAt_biasWitness_four (t : ℝ) :
    qSRAt 4 biasWitness 0 0 t = if t < 13 / 17 then 17 / 16 else 51 / 64 := by
  unfold qSRAt
  rw [groupScaleSR_biasWitness 4 (by norm_num), tensorScaleSR_biasWitness,
    show biasWitness 0 0 / (448 * (17 / (4 * 7168))) = 64 / 17 from by
      rw [show biasWitness 0 0 = 1 from by simp [biasWitness]]; norm_num,
    sr, floorOn_fp4_64_17, ceilOn_fp4_64_17]
  simp only [show ((4 : ℝ) - 3) = 1 from by norm_num, mul_one]
  split_ifs <;> first | (exfalso; linarith) | norm_num

/-- The `6.0` branch on the same entry: `96/17` rounds up to `6`, that is
`17/16`, when the coin falls below `14/17`, and down to `4`, that is `17/24`,
otherwise (§4.2). -/
theorem qSRAt_biasWitness_six (t : ℝ) :
    qSRAt 6 biasWitness 0 0 t = if t < 14 / 17 then 17 / 16 else 17 / 24 := by
  unfold qSRAt
  rw [groupScaleSR_biasWitness 6 (by norm_num), tensorScaleSR_biasWitness,
    show biasWitness 0 0 / (448 * (17 / (6 * 7168))) = 96 / 17 from by
      rw [show biasWitness 0 0 = 1 from by simp [biasWitness]]; norm_num,
    sr, floorOn_fp4_96_17, ceilOn_fp4_96_17]
  simp only [show ((6 : ℝ) - 4) = 2 from by norm_num]
  split_ifs <;> first | (exfalso; linarith) | norm_num

/-- The realized squared error of the witness's only group is carried by its
only non-zero entry (§4.2, "picks the one that yields lower MSE"). -/
theorem groupErr_biasWitness (c : ℝ) (hc : c ≠ 0) (t : Fin 16 → ℝ) :
    groupErr c biasWitness 0 t = (qSRAt c biasWitness 0 0 (t 0) - 1) ^ 2 := by
  unfold groupErr
  rw [Finset.sum_eq_single (0 : Fin 16)]
  · rw [show biasWitness 0 0 = 1 from by simp [biasWitness]]
  · intro j _ hj
    rw [qSRAt_biasWitness_ne c hc j hj, show biasWitness 0 j = 0 from by simp [biasWitness, hj]]
    norm_num
  · intro h
    exact absurd (Finset.mem_univ _) h

/-- The hypothesis of `groupErr_biasWitness` is satisfiable at both grid
maxima. -/
example : (4 : ℝ) ≠ 0 ∧ (6 : ℝ) ≠ 0 := ⟨by norm_num, by norm_num⟩

/-- **Four Over Six on the witness** (§4.2): the entry comes back as `17/16`
while the coin is below `14/17` and as `51/64` above it.  The branch that
wins is `4.0` at both ends and `6.0` in the middle, and what survives is the
*upper* threshold of the `6.0` branch carrying the *values* of the `4.0`
branch — the coupling between the choice and the coins the paper points at. -/
theorem q46At_biasWitness (t : Fin 16 → ℝ) :
    q46At biasWitness 0 0 t = if t 0 < 14 / 17 then 17 / 16 else 51 / 64 := by
  unfold q46At fourOverSix
  rw [groupErr_biasWitness 4 (by norm_num), groupErr_biasWitness 6 (by norm_num)]
  rcases lt_or_ge (t 0) (13 / 17) with h | h
  · have h4 : qSRAt 4 biasWitness 0 0 (t 0) = 17 / 16 := by
      rw [qSRAt_biasWitness_four]; simp [h]
    have h6 : qSRAt 6 biasWitness 0 0 (t 0) = 17 / 16 := by
      rw [qSRAt_biasWitness_six]; simp [show t 0 < 14 / 17 by linarith]
    rw [h4, h6]
    norm_num
    rw [qSRAt_biasWitness_four]
    simp [h, show t 0 < 14 / 17 by linarith]
  · rcases lt_or_ge (t 0) (14 / 17) with h' | h'
    · have h4 : qSRAt 4 biasWitness 0 0 (t 0) = 51 / 64 := by
        rw [qSRAt_biasWitness_four]; simp [not_lt.mpr h]
      have h6 : qSRAt 6 biasWitness 0 0 (t 0) = 17 / 16 := by
        rw [qSRAt_biasWitness_six]; simp [h']
      rw [h4, h6]
      norm_num
      rw [qSRAt_biasWitness_six]
      simp [h']
    · have h4 : qSRAt 4 biasWitness 0 0 (t 0) = 51 / 64 := by
        rw [qSRAt_biasWitness_four]; simp [not_lt.mpr h]
      have h6 : qSRAt 6 biasWitness 0 0 (t 0) = 17 / 24 := by
        rw [qSRAt_biasWitness_six]; simp [not_lt.mpr h']
      rw [h4, h6]
      norm_num
      rw [qSRAt_biasWitness_four]
      simp [not_lt.mpr h, not_lt.mpr h']

end Quartet
end Transformer
