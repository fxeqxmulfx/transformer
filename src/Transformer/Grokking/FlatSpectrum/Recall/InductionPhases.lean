module

public import Transformer.Grokking.FlatSpectrum.Recall.InductionExact

/-!
# The phases of the induction head: the copy needs `s₁ ≥ 1`, not `s₁ > 1`

Source: `review/flat_spectrum.md`, *Softmax attention escapes the cut* (added), item SA5, for
the attention scale `β_n = γ log n` of arXiv:2512.01868v4, §6 (*Long context Transformers*), on
the task of arXiv:2312.04927v1, §3.

`Recall.InductionExact` writes the induction head over `n = 2m + 2` tokens as one-hot attention
at the scale `s₂ a`, where `a = (n^{s₁} − 1) / (n^{s₁} + n − 1)` is the lead of the copy of
layer 1. Since `a = 1 − 1 / (n^{s₁ − 1} + 1 − 1/n)` (`copyLead_eq`), the lead tends to `0`,
`1/2` or `1` as `s₁` is below, at or above `1` (`tendsto_copyLead_of_lt`,
`tendsto_copyLead_one`, `tendsto_copyLead_of_gt`): the limits of the diagonal weight of the
equiangular model at `ρ = 0` (`Transformer.MeanField.tendsto_equiDiag_of_lt` and its two
neighbours). If the lead tends to `a`, the head recalls when `s₂ a > 1`
(`exists_forall_inductionHead_recall_of_tendsto`). When `s₂ a < 1` it fades: from some number
of keys on, the worst dictionary of values `±1` is answered with an error above every `ε < 3/2`
(`exists_lt_inductionHead_error_of_tendsto`), and never with `3/2` itself
(`isGreatest_inductionHead_error`).

So the condition `s₁ > 1` of `Recall.Induction` is not needed: at `s₁ = 1` the copy keeps half
its lead, and `s₂ > 2` recalls (`exists_forall_inductionHead_one_recall`). The condition
`s₁ ≥ 1` is needed: below it the copy loses its lead, and no `s₂` recalls
(`exists_lt_inductionHead_error_of_lt_one`). Above `s₁ = 1`, `s₂ < 1` fades as well. The boundary
`s₂ a = 1` is not treated here.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Recall

open Filter Topology

/-- The context of `m + 1` keys has `n = 2m + 2 → ∞` tokens. -/
lemma tendsto_two_mul_add_two : Tendsto (fun m : ℕ => (2 * m + 2 : ℝ)) atTop atTop :=
  tendsto_atTop_add_const_right _ 2 (tendsto_natCast_atTop_atTop.const_mul_atTop two_pos)

/-- The lead of the copy is `1 − 1 / (n^{s − 1} + 1 − 1/n)` over `n = 2m + 2` tokens. -/
lemma copyLead_eq (s : ℝ) (m : ℕ) :
    copyLead s m = 1 - 1 / ((2 * m + 2 : ℝ) ^ (s - 1) + 1 - (2 * m + 2 : ℝ)⁻¹) := by
  have hN : (0 : ℝ) < 2 * m + 2 := by positivity
  have hZ : (0 : ℝ) < (2 * m + 2 : ℝ) ^ s + 2 * m + 1 := by positivity
  rw [copyLead, Real.rpow_sub_one hN.ne', show (2 * m + 2 : ℝ) ^ s / (2 * m + 2) + 1 -
      (2 * m + 2 : ℝ)⁻¹ = ((2 * m + 2 : ℝ) ^ s + 2 * m + 1) / (2 * m + 2) by field_simp; ring,
    one_div_div, one_sub_div hZ.ne']
  ring_nf

/-- If `n^{s − 1}` tends to `p`, the lead of the copy tends to `1 − 1 / (p + 1)`. -/
lemma tendsto_copyLead_of_tendsto {s p : ℝ}
    (hp : Tendsto (fun m : ℕ => (2 * m + 2 : ℝ) ^ (s - 1)) atTop (𝓝 p)) (hp1 : p + 1 ≠ 0) :
    Tendsto (copyLead s) atTop (𝓝 (1 - 1 / (p + 1))) := by
  have hi := tendsto_inv_atTop_zero.comp tendsto_two_mul_add_two
  have h : Tendsto (fun m : ℕ => 1 - 1 / ((2 * m + 2 : ℝ) ^ (s - 1) + 1 - (2 * m + 2 : ℝ)⁻¹))
      atTop (𝓝 (1 - 1 / (p + 1 - 0))) :=
    tendsto_const_nhds.sub (tendsto_const_nhds.div ((hp.add_const 1).sub hi) (by simpa))
  rw [sub_zero] at h
  exact h.congr fun m => (copyLead_eq s m).symm

/-- At `s = 1`, `n^{s − 1} = 1`. -/
example := tendsto_copyLead_of_tendsto (s := 1) (p := 1)
  (by simp only [sub_self, Real.rpow_zero]; exact tendsto_const_nhds) (by norm_num)

/-- **Below `s₁ = 1` the copy loses its lead** (`review/flat_spectrum.md`, SA5). -/
theorem tendsto_copyLead_of_lt {s : ℝ} (hs : s < 1) : Tendsto (copyLead s) atTop (𝓝 0) := by
  have hp := (tendsto_rpow_neg_atTop (by linarith : 0 < 1 - s)).comp tendsto_two_mul_add_two
  rw [neg_sub] at hp
  simpa using tendsto_copyLead_of_tendsto hp (by norm_num)

/-- At `s = 1/2`. -/
example := tendsto_copyLead_of_lt (s := 1 / 2) (by norm_num)

/-- **At `s₁ = 1` the copy keeps half its lead** (`review/flat_spectrum.md`, SA5). -/
theorem tendsto_copyLead_one : Tendsto (copyLead 1) atTop (𝓝 (1 / 2)) := by
  convert tendsto_copyLead_of_tendsto (s := 1) (p := 1)
    (by simp only [sub_self, Real.rpow_zero]; exact tendsto_const_nhds) (by norm_num) using 2
  norm_num

/-- **Above `s₁ = 1` the copy keeps its whole lead** (`review/flat_spectrum.md`, SA5). -/
theorem tendsto_copyLead_of_gt {s : ℝ} (hs : 1 < s) : Tendsto (copyLead s) atTop (𝓝 1) := by
  have hp := (tendsto_rpow_atTop (by linarith : 0 < s - 1)).comp tendsto_two_mul_add_two
  have hi := tendsto_inv_atTop_zero.comp tendsto_two_mul_add_two
  have h := (tendsto_const_nhds (x := (1 : ℝ))).sub
    (hp.atTop_add ((tendsto_const_nhds (x := (1 : ℝ))).sub hi)).inv_tendsto_atTop
  rw [sub_zero] at h
  refine h.congr fun m => ?_
  simp only [Pi.inv_apply, Function.comp_apply]
  rw [copyLead_eq, one_div, add_sub_assoc]

/-- At `s = 2`. -/
example := tendsto_copyLead_of_gt (s := 2) (by norm_num)

/-- **The induction head recalls above its threshold** (`review/flat_spectrum.md`, SA5): if the
lead of the copy tends to `a` and `s₂ a > 1`, the induction head recalls values bounded by `C` to
every accuracy `ε > 0` from some number of keys on. -/
theorem exists_forall_inductionHead_recall_of_tendsto {s₁ s₂ a : ℝ}
    (ha : Tendsto (copyLead s₁) atTop (𝓝 a)) (h : 1 < s₂ * a) {ν : Type*} {φ : ν → ℝ} {C : ℝ}
    (hφ : ∀ v, |φ v| ≤ C) {ε : ℝ} (hε : 0 < ε) :
    ∃ M : ℕ, ∀ m ≥ M, ∀ (q : Fin (m + 1)) (π : Fin (m + 1) → ν),
      |φ (π q) - inductionHead s₁ s₂ φ π q| ≤ ε := by
  set σ := (1 + s₂ * a) / 2 with hσ
  obtain ⟨M₁, hM₁⟩ := eventually_atTop.1
    ((ha.const_mul s₂).eventually (le_mem_nhds (by linarith : σ < s₂ * a)))
  have h2 := ((tendsto_rpow_neg_atTop (by linarith : 0 < σ - 1)).comp
    tendsto_two_mul_add_two).const_mul (3 / 2 * C)
  rw [mul_zero] at h2
  obtain ⟨M₂, hM₂⟩ := eventually_atTop.1 (h2.eventually (ge_mem_nhds hε))
  refine ⟨max M₁ M₂, fun m hm q π => ?_⟩
  have hL := hM₁ m ((le_max_left _ _).trans hm)
  have hE := hM₂ m ((le_max_right _ _).trans hm)
  simp only [Function.comp_apply] at hE
  have hN : (0 : ℝ) < 2 * m + 2 := by positivity
  have hN1 : (1 : ℝ) ≤ 2 * m + 2 := by linarith [(Nat.cast_nonneg m : (0 : ℝ) ≤ m)]
  have hC : 0 ≤ C := (abs_nonneg _).trans (hφ (π q))
  have hD := Real.rpow_le_rpow_of_exponent_le hN1 hL
  have hX : 0 < (2 * m + 2 : ℝ) ^ σ := Real.rpow_pos_of_pos hN _
  rw [show -(σ - 1) = 1 - σ by ring, Real.rpow_sub hN, Real.rpow_one] at hE
  calc |φ (π q) - inductionHead s₁ s₂ φ π q|
      ≤ (3 * m + 1) * C / ((2 * m + 2 : ℝ) ^ (s₂ * copyLead s₁ m) + 2 * m + 1) :=
        abs_sub_inductionHead_le s₁ s₂ hφ π q
    _ ≤ (3 * m + 1) * C / (2 * m + 2 : ℝ) ^ σ :=
        div_le_div_of_nonneg_left (by positivity) hX (by linarith)
    _ ≤ (3 * m + 3) * C / (2 * m + 2 : ℝ) ^ σ :=
        div_le_div_of_nonneg_right (mul_le_mul_of_nonneg_right (by linarith) hC) hX.le
    _ = 3 / 2 * C * ((2 * m + 2) / (2 * m + 2 : ℝ) ^ σ) := by ring
    _ ≤ ε := hE

/-- At `s₁ = s₂ = 2`, values `±1` to accuracy `1/10` from some number of keys on. -/
example := exists_forall_inductionHead_recall_of_tendsto (tendsto_copyLead_of_gt (s := 2)
  (by norm_num)) (s₂ := 2) (by norm_num) (φ := fun b : Bool => if b then (1 : ℝ) else -1)
  (C := 1) (fun b => by cases b <;> simp) (ε := 1 / 10) (by norm_num)

/-- **The induction head fades below its threshold** (`review/flat_spectrum.md`, SA5): if the
lead of the copy tends to `a` and `s₂ a < 1`, then for every `ε < 3/2`, from some number of keys
on, every query has a dictionary of values `±1` answered with an error above `ε`. -/
theorem exists_lt_inductionHead_error_of_tendsto {s₁ s₂ a : ℝ}
    (ha : Tendsto (copyLead s₁) atTop (𝓝 a)) (h : s₂ * a < 1) {ε : ℝ} (hε : ε < 3 / 2) :
    ∃ M : ℕ, ∀ m ≥ M, ∀ q : Fin (m + 1), ∃ π : Fin (m + 1) → Bool,
      ε < |(if π q then (1 : ℝ) else -1) -
        inductionHead s₁ s₂ (fun b => if b then 1 else -1) π q| := by
  set σ := (1 + s₂ * a) / 2 with hσ
  obtain ⟨M₁, hM₁⟩ := eventually_atTop.1
    ((ha.const_mul s₂).eventually (ge_mem_nhds (by linarith : s₂ * a < σ)))
  have h3 := (((tendsto_rpow_neg_atTop (by linarith : 0 < 1 - σ)).comp
    tendsto_two_mul_add_two).const_mul 3).add (tendsto_inv_atTop_zero.comp tendsto_two_mul_add_two)
  rw [mul_zero, add_zero] at h3
  obtain ⟨M₂, hM₂⟩ := eventually_atTop.1 (h3.eventually (gt_mem_nhds (by linarith : 0 < 3 / 2 - ε)))
  refine ⟨max M₁ M₂, fun m hm q => ?_⟩
  obtain ⟨π, hπ⟩ := (isGreatest_inductionHead_error s₁ s₂ q).1
  refine ⟨π, lt_of_lt_of_eq ?_ hπ.symm⟩
  have hL := hM₁ m ((le_max_left _ _).trans hm)
  have hr := hM₂ m ((le_max_right _ _).trans hm)
  simp only [Function.comp_apply] at hr
  have hN : (0 : ℝ) < 2 * m + 2 := by positivity
  have hN1 : (1 : ℝ) ≤ 2 * m + 2 := by linarith [(Nat.cast_nonneg m : (0 : ℝ) ≤ m)]
  have hXD := Real.rpow_le_rpow_of_exponent_le hN1 hL
  have hD : 0 < (2 * m + 2 : ℝ) ^ (s₂ * copyLead s₁ m) := Real.rpow_pos_of_pos hN _
  rw [show -(1 - σ) = σ - 1 by ring, Real.rpow_sub_one hN.ne'] at hr
  set D := (2 * m + 2 : ℝ) ^ (s₂ * copyLead s₁ m)
  set X := (2 * m + 2 : ℝ) ^ σ
  have key : 3 / 2 - (3 * m + 1) / (D + 2 * m + 1) ≤ 3 * (X / (2 * m + 2)) + (2 * m + 2 : ℝ)⁻¹ := by
    rw [show 3 / 2 - (3 * m + 1) / (D + 2 * m + 1) = (3 * D + 1) / (2 * (D + 2 * m + 1)) by
        field_simp; ring,
      show 3 * (X / (2 * m + 2)) + (2 * m + 2 : ℝ)⁻¹ = (3 * X + 1) / (2 * m + 2) by field_simp,
      div_le_div_iff₀ (by positivity) hN]
    nlinarith [mul_le_mul_of_nonneg_right hXD (by positivity : (0 : ℝ) ≤ 2 * (D + 2 * m + 1))]
  linarith

/-- At `s₁ = 2` and `s₂ = 1/2`, above `s₁ = 1` but below `s₂ = 1`: errors above `1`. -/
example := exists_lt_inductionHead_error_of_tendsto (tendsto_copyLead_of_gt (s := 2)
  (by norm_num)) (s₂ := 1 / 2) (by norm_num) (ε := 1) (by norm_num)

/-- **`s₁ > 1` is not needed** (`review/flat_spectrum.md`, SA5): at `s₁ = 1` the copy keeps half
its lead, and the induction head recalls values bounded by `C` once `s₂ > 2`. -/
theorem exists_forall_inductionHead_one_recall {s₂ : ℝ} (hs₂ : 2 < s₂) {ν : Type*}
    {φ : ν → ℝ} {C : ℝ} (hφ : ∀ v, |φ v| ≤ C) {ε : ℝ} (hε : 0 < ε) :
    ∃ M : ℕ, ∀ m ≥ M, ∀ (q : Fin (m + 1)) (π : Fin (m + 1) → ν),
      |φ (π q) - inductionHead 1 s₂ φ π q| ≤ ε :=
  exists_forall_inductionHead_recall_of_tendsto tendsto_copyLead_one (by linarith) hφ hε

/-- At `s₁ = 1` and `s₂ = 3`, values `±1` to accuracy `1/10`. -/
example := exists_forall_inductionHead_one_recall (s₂ := 3) (by norm_num)
  (φ := fun b : Bool => if b then (1 : ℝ) else -1) (C := 1) (fun b => by cases b <;> simp)
  (ε := 1 / 10) (by norm_num)

/-- **`s₁ ≥ 1` is needed** (`review/flat_spectrum.md`, SA5): below it, at every `s₂` and `ε < 3/2`,
from some number of keys on every query has a dictionary of values `±1` missed by more than `ε`. -/
theorem exists_lt_inductionHead_error_of_lt_one {s₁ : ℝ} (hs₁ : s₁ < 1) (s₂ : ℝ) {ε : ℝ}
    (hε : ε < 3 / 2) :
    ∃ M : ℕ, ∀ m ≥ M, ∀ q : Fin (m + 1), ∃ π : Fin (m + 1) → Bool,
      ε < |(if π q then (1 : ℝ) else -1) -
        inductionHead s₁ s₂ (fun b => if b then 1 else -1) π q| :=
  exists_lt_inductionHead_error_of_tendsto (tendsto_copyLead_of_lt hs₁) (by simp) hε

/-- At `s₁ = 0` and `s₂ = 100`: errors above `1`. -/
example := exists_lt_inductionHead_error_of_lt_one (s₁ := 0) zero_lt_one 100 (ε := 1) (by norm_num)

end Transformer.Grokking.FlatSpectrum.Recall
