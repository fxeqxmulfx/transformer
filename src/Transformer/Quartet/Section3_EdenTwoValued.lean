/-
# `MS-EDEN` on a two-valued rotation

Support for the counterexample of `Transformer.Quartet.Section3_EdenBias` to the Corollary of
§3.3.  When the rotation `y = RHT(x)` of a tensor takes two values only — `P` at even positions
and `R` at odd ones (`twoValued`), with `M = max(|P|, |R|)` — every number Algorithm 1 computes
is explicit:

* the per-tensor scale is `M / (256 s)` and every group scale is `RTN_FP8(256) = 256`;
* the E2M1 entries are `q_P = RTN_FP4(P s / M)` and `q_R = RTN_FP4(R s / M)`, times `M/s`;
* the EDEN correction of every group is `S = s (P² + R²) / (M (P q_P + R q_R))`;
* the output entry at position `j` is `q_{j mod 2} · SR_FP8(256 S) · M / (256 s)`.

The stochastic roundings are what is left random, one coin per group; the inverse rotation at
the first entry then reads `± c · 8 (q_P + q_R) · M/(256 s) · Σ_g SR_FP8(256 S)(u_g)` with
`c = 1/√d`, the first row of the Hadamard matrix (`hadamard_zero_row`).

Source: arXiv:2601.22813v2, §3.1 (`Q_RTN`), §3.3 and Algorithm 1 (`MS-EDEN`).
-/

import Transformer.Quartet.Section3_Eden
import Transformer.Quartet.Section3_EdenPair
import Transformer.Quartet.Fp4Grid

namespace Transformer
namespace Quartet

variable {k : ℕ}

/-- A tensor of `2^k` groups whose entries are `P` at even positions and `R` at odd ones. -/
def twoValued (k : ℕ) (P R : ℝ) : Fin (2 ^ k) → Fin 16 → ℝ :=
  fun _ j => if (j : ℕ) % 2 = 0 then P else R

section TwoValued

variable {P R M s : ℝ}

/-- The largest entry of every group of a two-valued tensor is `M`, as soon as `M` bounds `|P|`,
`|R|` and equals one of them. -/
theorem groupAbsMax_twoValued (hP : |P| ≤ M) (hR : |R| ≤ M) (h : |P| = M ∨ |R| = M)
    (i : Fin (2 ^ k)) : groupAbsMax (twoValued k P R) i = M := by
  refine le_antisymm ?_ ?_
  · unfold groupAbsMax
    refine Finset.sup'_le _ _ fun j _ => ?_
    unfold twoValued
    split
    · exact hP
    · exact hR
  · rcases h with h | h
    · exact h.symm.le.trans (by simpa [twoValued] using abs_le_groupAbsMax (twoValued k P R) i 0)
    · exact h.symm.le.trans (by simpa [twoValued] using abs_le_groupAbsMax (twoValued k P R) i 1)

/-- And so is the largest entry of the tensor. -/
theorem absMax_twoValued (hP : |P| ≤ M) (hR : |R| ≤ M) (h : |P| = M ∨ |R| = M) :
    absMax (twoValued k P R) = M :=
  le_antisymm (Finset.sup'_le _ _ fun p _ => (abs_le_groupAbsMax _ p.1 p.2).trans
    (groupAbsMax_twoValued hP hR h p.1).le)
    ((groupAbsMax_twoValued hP hR h 0).symm.le.trans (groupAbsMax_le_absMax _ 0))

/-- The per-tensor scale of `Q_RTN` on a two-valued tensor is `M / (256 s)`. -/
theorem tensorScaleRTN_twoValued (hP : |P| ≤ M) (hR : |R| ≤ M) (h : |P| = M ∨ |R| = M) :
    tensorScaleRTN s (twoValued k P R) = M / (s * 256) := by
  rw [tensorScaleRTN, absMax_twoValued hP hR h]

/-- **Every group scale of `Q_RTN` on a two-valued tensor is `256`**: the group maximum over the
tensor scale is exactly `256`, a point of the E4M3 grid. -/
theorem groupScaleRTN_twoValued (hs : 0 < s) (hM : 0 < M) (hP : |P| ≤ M) (hR : |R| ≤ M)
    (h : |P| = M ∨ |R| = M) (i : Fin (2 ^ k)) : groupScaleRTN s (twoValued k P R) i = 256 := by
  have h256 : (256 : ℝ) ∈ fp8 := ⟨by norm_num, 8, 5, by norm_num⟩
  rw [groupScaleRTN, groupAbsMax_twoValued hP hR h i, tensorScaleRTN_twoValued hP hR h,
    show M / (M / (s * 256) * s) = 256 by field_simp]
  exact rtn_self h256

/-- The entries of `Q_RTN` on a two-valued tensor. -/
theorem qRTN_twoValued (hs : 0 < s) (hM : 0 < M) (hP : |P| ≤ M) (hR : |R| ≤ M)
    (h : |P| = M ∨ |R| = M) (i : Fin (2 ^ k)) (j : Fin 16) :
    qRTN s (twoValued k P R) i j = rtn fp4 (twoValued k P R i j * s / M) * (M / s) := by
  unfold qRTN
  rw [groupScaleRTN_twoValued hs hM hP hR h, tensorScaleRTN_twoValued hP hR h,
    show twoValued k P R i j / (256 * (M / (s * 256))) = twoValued k P R i j * s / M by field_simp,
    mul_assoc, show (256 : ℝ) * (M / (s * 256)) = M / s by field_simp]

/-- A sum over the `16` positions of a group splits by parity. -/
theorem sum_parity (a b : ℝ) :
    ∑ j : Fin 16, (if (j : ℕ) % 2 = 0 then a else b) = 8 * a + 8 * b := by
  simp [Fin.sum_univ_succ]
  ring

/-- **The EDEN correction of a two-valued tensor**, `S = s (P² + R²) / (M (P q_P + R q_R))`. -/
theorem edenScale_twoValued {qP qR : ℝ} (hs : 0 < s) (hM : 0 < M) (hP : |P| ≤ M) (hR : |R| ≤ M)
    (h : |P| = M ∨ |R| = M) (hqP : rtn fp4 (P * s / M) = qP) (hqR : rtn fp4 (R * s / M) = qR)
    (hD : P * qP + R * qR ≠ 0) (i : Fin (2 ^ k)) :
    edenScale s (twoValued k P R) i = s * (P ^ 2 + R ^ 2) / (M * (P * qP + R * qR)) := by
  have hq : ∀ j : Fin 16, rtn fp4 (twoValued k P R i j * s / M) =
      if (j : ℕ) % 2 = 0 then qP else qR := by
    intro j
    unfold twoValued
    split
    · exact hqP
    · exact hqR
  have hnum : ∑ j : Fin 16, twoValued k P R i j * twoValued k P R i j =
      8 * (P * P) + 8 * (R * R) := by
    rw [← sum_parity (P * P) (R * R)]
    refine Finset.sum_congr rfl fun j _ => ?_
    unfold twoValued
    split_ifs <;> rfl
  have hden : ∑ j : Fin 16, twoValued k P R i j * qRTN s (twoValued k P R) i j =
      (8 * (P * qP) + 8 * (R * qR)) * (M / s) := by
    rw [← sum_parity (P * qP) (R * qR), Finset.sum_mul]
    refine Finset.sum_congr rfl fun j _ => ?_
    rw [qRTN_twoValued hs hM hP hR h, hq]
    unfold twoValued
    split_ifs <;> ring
  unfold edenScale
  rw [hnum, hden]
  field_simp

/-- **`MS-EDEN` on a two-valued rotation**, entry by entry: `q_{j mod 2}` times the stochastic
rounding of the corrected scale `256 S`, times the tensor scale `M / (256 s)`. -/
theorem msEden_twoValued {x : Fin (2 ^ k) → Fin 16 → ℝ} {ε : Fin (2 ^ k) → Fin 16 → Bool}
    {qP qR : ℝ} (hy : rht k ε x = twoValued k P R) (hs : 0 < s) (hM : 0 < M) (hP : |P| ≤ M)
    (hR : |R| ≤ M) (h : |P| = M ∨ |R| = M) (hqP : rtn fp4 (P * s / M) = qP)
    (hqR : rtn fp4 (R * s / M) = qR) (hD : P * qP + R * qR ≠ 0) (u : Fin (2 ^ k) → ℝ)
    (i : Fin (2 ^ k)) (j : Fin 16) :
    msEden k s x ε u i j = (if (j : ℕ) % 2 = 0 then qP else qR) *
      sr fp8 (s * (P ^ 2 + R ^ 2) / (M * (P * qP + R * qR)) * 256) (u i) * (M / (s * 256)) := by
  have hq : rtn fp4 (twoValued k P R i j * s / M) = if (j : ℕ) % 2 = 0 then qP else qR := by
    unfold twoValued
    split
    · exact hqP
    · exact hqR
  simp only [msEden, hy]
  rw [groupScaleRTN_twoValued hs hM hP hR h, tensorScaleRTN_twoValued hP hR h,
    edenScale_twoValued hs hM hP hR h hqP hqR hD,
    show twoValued k P R i j / (256 * (M / (s * 256))) = twoValued k P R i j * s / M by
      field_simp,
    hq]

/-- **The first entry of the inverse rotation of `MS-EDEN` on a two-valued rotation**:
`± c · 8 (q_P + q_R) · M/(256 s) · Σ_g SR_FP8(256 S)(u_g)`, with `c = 1/√(2^{k+4})` the first row
of the Hadamard matrix and `±` the sign the seed gives that entry. -/
theorem rhtInv_msEden_twoValued {x : Fin (2 ^ k) → Fin 16 → ℝ}
    {ε : Fin (2 ^ k) → Fin 16 → Bool} {qP qR : ℝ} (hy : rht k ε x = twoValued k P R)
    (hs : 0 < s) (hM : 0 < M) (hP : |P| ≤ M) (hR : |R| ≤ M) (h : |P| = M ∨ |R| = M)
    (hqP : rtn fp4 (P * s / M) = qP) (hqR : rtn fp4 (R * s / M) = qR)
    (hD : P * qP + R * qR ≠ 0) (u : Fin (2 ^ k) → ℝ) :
    rhtInv k ε (msEden k s x ε u) 0 0 = (if ε 0 0 then -1 else 1) *
      (1 / Real.sqrt (2 ^ (k + 4)) * (8 * (qP + qR)) * (M / (s * 256))) *
        ∑ i' : Fin (2 ^ k),
          sr fp8 (s * (P ^ 2 + R ^ 2) / (M * (P * qP + R * qR)) * 256) (u i') := by
  unfold rhtInv
  have hsum : ∀ i' : Fin (2 ^ k), ∑ j' : Fin 16, hadamard k 0 i' 0 j' * msEden k s x ε u i' j' =
      1 / Real.sqrt (2 ^ (k + 4)) * (8 * (qP + qR)) * (M / (s * 256)) *
        sr fp8 (s * (P ^ 2 + R ^ 2) / (M * (P * qP + R * qR)) * 256) (u i') := by
    intro i'
    have : ∀ j' : Fin 16, hadamard k 0 i' 0 j' * msEden k s x ε u i' j' =
        1 / Real.sqrt (2 ^ (k + 4)) * (M / (s * 256)) *
          sr fp8 (s * (P ^ 2 + R ^ 2) / (M * (P * qP + R * qR)) * 256) (u i') *
            (if (j' : ℕ) % 2 = 0 then qP else qR) := by
      intro j'
      rw [hadamard_zero_row, msEden_twoValued hy hs hM hP hR h hqP hqR hD]
      ring
    rw [Finset.sum_congr rfl fun j' _ => this j', ← Finset.mul_sum, sum_parity qP qR]
    ring
  rw [Finset.sum_congr rfl fun i' _ => hsum i', ← Finset.mul_sum]
  ring

end TwoValued

/-- The hypotheses of the two-valued formulas are satisfiable, by the rotation of `edenPair`
at the smallest dimension, its all-`false` seed, and `s = 6`: `P = 11/40`, `R = 9/40`,
`M = 11/40`, `q_P = 6`, `q_R = 4`. -/
example : ∃ (x : Fin (2 ^ 0) → Fin 16 → ℝ) (ε : Fin (2 ^ 0) → Fin 16 → Bool)
    (P R M s qP qR : ℝ), rht 0 ε x = twoValued 0 P R ∧ 0 < s ∧ 0 < M ∧ |P| ≤ M ∧ |R| ≤ M ∧
      (|P| = M ∨ |R| = M) ∧ rtn fp4 (P * s / M) = qP ∧ rtn fp4 (R * s / M) = qR ∧
      P * qP + R * qR ≠ 0 := by
  have h4 : √((2 : ℝ) ^ (0 + 4)) = 4 := by
    rw [show ((2 : ℝ) ^ (0 + 4)) = 4 ^ 2 by norm_num, Real.sqrt_sq (by norm_num)]
  refine ⟨edenPair 0, fun _ _ => false, 11 / 40, 9 / 40, 11 / 40, 6, 6, 4, ?_, by norm_num,
    by norm_num, by norm_num [abs_of_pos], by norm_num [abs_of_pos],
    Or.inl (by norm_num [abs_of_pos]), ?_, ?_, by norm_num⟩
  · rw [rht_edenPair, h4]
    funext i j
    unfold twoValued
    norm_num
  · rw [show (11 / 40 : ℝ) * 6 / (11 / 40) = 6 by norm_num]
    exact rtn_fp4_six (by norm_num)
  · rw [show (9 / 40 : ℝ) * 6 / (11 / 40) = 54 / 11 by norm_num]
    exact rtn_fp4_four (by norm_num) (by norm_num)

end Quartet
end Transformer
