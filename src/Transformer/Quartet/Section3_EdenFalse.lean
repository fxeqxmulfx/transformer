/-
# The Corollary of §3.3 is false for large `s`

arXiv:2601.22813v2, §3.3: "For all `x ∈ ℝ^d` and scale `s ≠ 0`",
`E_{ω_RHT, ω_SR} RHT⁻¹(x̂, ω_RHT) = x` for the output `x̂` of `MS-EDEN`.

The quantifier over `s` is too wide.  The group scales of `Q_RTN` are capped at
`256` so that the EDEN correction `S` can raise them to at most `448`, the top
of E4M3; that headroom is a factor `1.75`.  With a large clipping factor `s`
the E2M1 entries clip at `±6`, `S` grows like `s`, and the stochastic rounding
of `S · 256` saturates at `448` whatever the coin says: the correction is lost,
and with it unbiasedness.

The witness is a single group, `d = 16`, and the first basis vector.  Its
rotation is `±(1/4, …, 1/4)` for every sign seed, so at `s = 21` the entries
round to `±6`, `S = 7/2`, the scale saturates at `448 < 7/2 · 256`, and the
estimate is exactly half the input: `MS-EDEN` returns `x/2` in expectation.

The statement the paper means, at its own clipping factor, is
`Transformer.Quartet.mean_rhtInv_msEden`.
-/

import Transformer.Quartet.Section3_Eden

namespace Transformer
namespace Quartet

/-- The first basis vector of `ℝ^16`, one NVFP4 group. -/
def edenWitness : Fin (2 ^ 0) → Fin 16 → ℝ := fun _ j => if j = 0 then 1 else 0

/-- Its rotation is flat: every entry is `±1/4`, with the sign of the seed at
the one nonzero entry. -/
theorem rht_edenWitness (ε : Fin (2 ^ 0) → Fin 16 → Bool) :
    rht 0 ε edenWitness = fun _ _ => (if ε 0 0 then -1 else 1) / 4 := by
  have h4 : √(16 : ℝ) = 4 := by
    rw [show (16 : ℝ) = 4 ^ 2 by norm_num, Real.sqrt_sq (by norm_num)]
  have hi : ∀ a : Fin (2 ^ 0), a = 0 := fun a => Fin.ext (by have := a.isLt; simp at this; omega)
  funext i j
  unfold rht edenWitness hadamard
  rw [Finset.sum_eq_single (0 : Fin (2 ^ 0)) (fun b _ hb => absurd (hi b) hb) (by simp),
    Finset.sum_eq_single (0 : Fin 16) (fun b _ hb => by simp [hb]) (by simp)]
  cases ε 0 0 <;> norm_num [h4]

/-- **`MS-EDEN` at `s = 21` halves a flat group.**  Group scale `256`, entries
clipped to `±6`, correction `S = 7/2`, and the corrected scale `896` saturated
to `448`, whatever the coin `u`. -/
theorem msEden_const {c : ℝ} (hc : |c| = 1 / 4) (u : Fin (2 ^ 0) → ℝ)
    (ε : Fin (2 ^ 0) → Fin 16 → Bool) (hy : rht 0 ε edenWitness = fun _ _ => c) :
    msEden 0 21 edenWitness ε u = fun _ _ => c / 2 := by
  have habs : absMax (fun _ _ => c : Fin (2 ^ 0) → Fin 16 → ℝ) = 1 / 4 := by
    unfold absMax
    exact le_antisymm (Finset.sup'_le _ _ fun _ _ => hc.le)
      (hc.symm.le.trans (Finset.le_sup' (fun p : Fin (2 ^ 0) × Fin 16 => |c|)
        (Finset.mem_univ (0, 0))))
  have hgabs : ∀ i, groupAbsMax (fun _ _ => c : Fin (2 ^ 0) → Fin 16 → ℝ) i = 1 / 4 := by
    intro i
    unfold groupAbsMax
    exact le_antisymm (Finset.sup'_le _ _ fun _ _ => hc.le)
      (hc.symm.le.trans (Finset.le_sup' (fun _ : Fin 16 => |c|) (Finset.mem_univ 0)))
  have hT : tensorScaleRTN 21 (fun _ _ => c : Fin (2 ^ 0) → Fin 16 → ℝ) = 1 / 21504 := by
    rw [tensorScaleRTN, habs]; norm_num
  have h256 : (256 : ℝ) ∈ fp8 := ⟨by norm_num, 8, 5, by norm_num⟩
  have h448 : (448 : ℝ) ∈ fp8 := ⟨by norm_num, 14, 5, by norm_num⟩
  have hG : ∀ i, groupScaleRTN 21 (fun _ _ => c : Fin (2 ^ 0) → Fin 16 → ℝ) i = 256 := by
    intro i
    rw [groupScaleRTN, hgabs, hT, show (1 / 4 : ℝ) / (1 / 21504 * 21) = 256 by norm_num, rtn,
      floorOn_eq h256 le_rfl fun _ _ hle => hle, ceilOn_eq h256 le_rfl fun _ _ hle => hle]
    norm_num
  have hq : rtn fp4 (c / (256 * (1 / 21504))) = 24 * c := by
    rcases abs_eq (by norm_num : (0 : ℝ) ≤ 1 / 4) |>.mp hc with h | h <;> subst h
    · rw [rtn_of_forall_le (g := 6) (by norm_num [fp4]) (fun _ hy => (mem_Icc_of_mem_fp4 hy).2)
        (by norm_num)]; norm_num
    · rw [rtn_of_le_forall (g := -6) (by norm_num [fp4]) (fun _ hy => (mem_Icc_of_mem_fp4 hy).1)
        (by norm_num)]; norm_num
  have hc2 : c * c = 1 / 16 := by
    rcases abs_eq (by norm_num : (0 : ℝ) ≤ 1 / 4) |>.mp hc with h | h <;> subst h <;> norm_num
  have hS : ∀ i, edenScale 21 (fun _ _ => c : Fin (2 ^ 0) → Fin 16 → ℝ) i = 7 / 2 := by
    intro i
    unfold edenScale qRTN
    simp only [hG, hT, hq]
    simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
    rw [show c * (24 * c * 256 * (1 / 21504)) = c * c * (2 / 7) by ring, hc2]
    norm_num
  funext i j
  simp only [msEden, hy, hS, hG, hT, hq]
  rw [sr_of_forall_le h448 (fun y hy => (abs_le.mp hy.1).2) (by norm_num)]
  ring

/-- The hypotheses of `msEden_const` are satisfiable: they are what
`rht_edenWitness` gives at the all-`false` seed. -/
example : |((if (fun _ _ => false : Fin (2 ^ 0) → Fin 16 → Bool) 0 0 then -1 else 1 : ℝ) / 4)|
      = 1 / 4 ∧
    rht 0 (fun _ _ => false) edenWitness = fun _ _ => (1 : ℝ) / 4 :=
  ⟨by norm_num, by rw [rht_edenWitness]; norm_num⟩

/-- The expectation of Algorithm 1 fixes constants: the seeds are uniform and
the coins range over a cube of volume `1`. -/
theorem mean_const (k : ℕ) (a : ℝ) : mean k (fun _ _ => a) = a := by
  unfold mean
  rw [Finset.sum_const, Finset.card_univ, nsmul_eq_mul, MeasureTheory.setIntegral_const,
    MeasureTheory.measureReal_def, Set.pi_univ_Icc, Real.volume_Icc_pi]
  simp

/-- `RHT⁻¹` is linear; halving is all that is needed here. -/
theorem rhtInv_div_two (k : ℕ) (ε : Fin (2 ^ k) → Fin 16 → Bool) (y : Fin (2 ^ k) → Fin 16 → ℝ)
    (i : Fin (2 ^ k)) (j : Fin 16) :
    rhtInv k ε (fun a b => y a b / 2) i j = rhtInv k ε y i j / 2 := by
  unfold rhtInv
  simp only [mul_div_assoc', ← Finset.sum_div]

/-- **The Corollary of §3.3 is false as stated**: "for all `x` and scale
`s ≠ 0`" fails at `s = 21`, where `MS-EDEN` returns `x/2` in expectation for
the first basis vector of one group.

Source: arXiv:2601.22813v2, §3.3, the Corollary after Algorithm 1. -/
theorem not_mean_rhtInv_msEden :
    ¬ ∀ {k : ℕ} {s : ℝ}, s ≠ 0 → ∀ (x : Fin (2 ^ k) → Fin 16 → ℝ) (i : Fin (2 ^ k))
      (j : Fin 16), mean k (fun ε u => rhtInv k ε (msEden k s x ε u) i j) = x i j := by
  intro h
  have hpt : ∀ ε u, rhtInv 0 ε (msEden 0 21 edenWitness ε u) 0 0 = 1 / 2 := by
    intro ε u
    have hy := rht_edenWitness ε
    have hc : |(if ε 0 0 then -1 else 1 : ℝ) / 4| = 1 / 4 := by
      cases ε 0 0 <;> norm_num [abs_of_pos, abs_of_neg]
    rw [msEden_const hc u ε hy]
    have hfun : (fun _ _ => (if ε 0 0 then -1 else 1 : ℝ) / 4 / 2 : Fin (2 ^ 0) → Fin 16 → ℝ) =
        fun a b => rht 0 ε edenWitness a b / 2 := by rw [hy]
    rw [hfun, rhtInv_div_two, rhtInv_rht]
    norm_num [edenWitness]
  have h0 := h (k := 0) (s := 21) (by norm_num) edenWitness 0 0
  rw [show (fun ε u => rhtInv 0 ε (msEden 0 21 edenWitness ε u) 0 0) = fun _ _ => (1 / 2 : ℝ)
    from funext fun ε => funext fun u => hpt ε u, mean_const] at h0
  norm_num [edenWitness] at h0

end Quartet
end Transformer
