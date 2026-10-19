module

public import Transformer.Perspective.SyncKernelPieces
public import Mathlib.Analysis.Real.Pi.Bounds

/-!
# Certificates for `τ_β ∫ |f_β'''|₊ < 4`

Polyanskiy, Rigollet, Yao — arXiv:2507.22857v1, *Synchronization of mean-field
models on the circle*, §5, proof of `cor:conjecture`.

For `β` in an interval `[β₁, β₂]` of `(0, 1]`, `τ_β ∫_{-π}^{π} |f_β'''|₊ < 4`
follows from finitely many inequalities between numbers (`SmallBetaCert`,
`τ_mul_integral_lt_four_of_cert`):

* `-A ≤ f_β'' ≤ U` on `[0, π]`, so `∫ |f_β'''|₊ ≤ 2(U + A)`
  (`integral_posPart_f_β'''_le`), with `A` and `U` from the pieces of
  `Perspective.SyncKernelPieces` (`f_β''_mem_Icc_of_isChain`);
* `τ_β ≤ π/2 - k` whenever `k ≤ cos τ_β` (`τ_β_star_two_le`), and
  `k ≤ cos τ_β` whenever `β ≥ β₁` and `k ≤ β₁(1 - k²)` (`le_cosTauStar_two`).

They take the place of the values of `g¹`, `g²` and of `lemma:tau1` that the
source uses in `lemma:beta_0.75`, …, `lemma:beta_final`.
-/

@[expose] public section

open Real

namespace Transformer
namespace Perspective

/-- **`-A ≤ f_β'' ≤ U` on `[0, π]` from chains of pieces.**  For `0 ≤ β₁ ≤ β ≤ β₂`,
`β ≤ 1`, `1 ≤ A` and `0 ≤ U`: if consecutive points of `0 :: cs ++ [1]` satisfy
`NegPieceOK` and those of `0 :: us ++ [1]` satisfy `PosPieceOK`, then
`-A ≤ f_β'' ≤ U` on `[0, π]`. Where `cos x ≤ 0`, `-f_β'' ≤ 1`; where
`cos x ≥ 0`, `f_β'' ≤ 0`. -/
theorem f_β''_mem_Icc_of_isChain {β β₁ β₂ A U : ℝ} (cs us : List ℝ) (hβ₁ : 0 ≤ β₁)
    (hβ : β ∈ Set.Icc β₁ β₂) (hβ1 : β ≤ 1) (hA : 1 ≤ A) (hU : 0 ≤ U)
    (hcs : List.IsChain (NegPieceOK β₁ β₂ A) (0 :: (cs ++ [1])))
    (hus : List.IsChain (PosPieceOK β₁ β₂ U) (0 :: (us ++ [1]))) {x : ℝ}
    (hx : x ∈ Set.Icc 0 π) : -A ≤ f_β'' β x ∧ f_β'' β x ≤ U := by
  have hs := Real.sin_nonneg_of_nonneg_of_le_pi hx.1 hx.2
  have hβ0 : 0 ≤ β := hβ₁.trans hβ.1
  rcases le_total (Real.cos x) 0 with hc | hc
  · obtain ⟨u₁, u₂, h, hu⟩ := exists_piece_of_isChain us hus (-Real.cos x)
      ⟨by linarith, by linarith [Real.neg_one_le_cos x]⟩
    exact ⟨by linarith [neg_f_β''_le_one hβ0 hs hc], f_β''_le_of_posPieceOK hβ₁ hβ hU hs hu h⟩
  · obtain ⟨c₁, c₂, h, hc'⟩ := exists_piece_of_isChain cs hcs (Real.cos x)
      ⟨hc, Real.cos_le_one x⟩
    exact ⟨by linarith [neg_f_β''_le_of_negPieceOK hβ₁ hβ (by linarith) hs hc' h],
      (f_β''_nonpos hβ0 hβ1 hs hc).trans hU⟩

/-- `τ_β ≤ π/2 - k` whenever `k ≤ cos τ_β`: `τ_β = π/2 - arcsin(cos τ_β)` and
`arcsin c ≥ c` for `c ∈ [0, 1]`. -/
theorem τ_β_star_two_le {β k : ℝ} (hβ : 0 < β) (hk : k ≤ cosTauStar 2 β) :
    τ_β_star 2 β ≤ π / 2 - k := by
  have hc0 := cosTauStar_pos 2 β hβ (by norm_num)
  have hc1 := cosTauStar_le_one 2 β hβ (by norm_num)
  have h := Real.sin_le (Real.arcsin_nonneg.mpr hc0.le)
  rw [Real.sin_arcsin (by linarith) hc1] at h
  rw [τ_β_star, Real.arccos_eq_pi_div_two_sub_arcsin]
  linarith

/-- `k ≤ cos τ_β` whenever `0 ≤ β₁ ≤ β` and `0 ≤ k ≤ β₁(1 - k²)`: `c = cos τ_β`
solves `β(1 - c²) = c`, and `c ↦ c/(1 - c²)` increases on `[0, 1)`. -/
theorem le_cosTauStar_two {β β₁ k : ℝ} (hβ₁ : 0 ≤ β₁) (hβ : β₁ ≤ β) (hβ0 : 0 < β)
    (hk0 : 0 ≤ k) (hk : k ≤ β₁ * (1 - k ^ 2)) : k ≤ cosTauStar 2 β := by
  have hq := cosTauStar_two_quadratic hβ0
  have hc0 := cosTauStar_pos 2 β hβ0 (by norm_num)
  have hk1 : 0 ≤ 1 - k ^ 2 := by
    by_contra h
    push Not at h
    have hk' : k = 0 :=
      le_antisymm (hk.trans (mul_nonpos_of_nonneg_of_nonpos hβ₁ h.le)) hk0
    rw [hk'] at h
    norm_num at h
  by_contra h
  push Not at h
  have hlt : 0 < k ^ 2 - cosTauStar 2 β ^ 2 := by nlinarith
  nlinarith [mul_pos hβ0 hlt, mul_nonneg (sub_nonneg.mpr hβ) hk1]

/-- The numerical part of a certificate for `τ_β ∫ |f_β'''|₊ < 4` on `β ∈ [β₁, β₂]`:
`k ≤ cos τ_β` there, and `(1.5708 - k)(A + U) < 2` for the bounds `A` of `-f_β''`
and `U` of `f_β''` that the chains of pieces `cs` and `us` give. -/
def SmallBetaCert (β₁ β₂ k A U : ℝ) (cs us : List ℝ) : Prop :=
  0 ≤ β₁ ∧ 0 ≤ k ∧ k ≤ β₁ * (1 - k ^ 2) ∧ 1 ≤ A ∧ 0 ≤ U ∧
    (15708 / 10000 - k) * (A + U) < 2 ∧
    List.IsChain (NegPieceOK β₁ β₂ A) (0 :: (cs ++ [1])) ∧
    List.IsChain (PosPieceOK β₁ β₂ U) (0 :: (us ++ [1]))

/-- **`τ_β ∫ |f_β'''|₊ < 4` from a certificate on `[β₁, β₂]`**, for `0 < β ≤ 1`:
`τ_β ∫ |f_β'''|₊ ≤ (π/2 - k) · 2(U + A)` and `π/2 < 1.5708`. -/
theorem τ_mul_integral_lt_four_of_cert {β β₁ β₂ k A U : ℝ} {cs us : List ℝ} (hβ : 0 < β)
    (hβ1 : β ≤ 1) (hβI : β ∈ Set.Icc β₁ β₂) (h : SmallBetaCert β₁ β₂ k A U cs us) :
    τ_β_star 2 β * ∫ x in (-π)..π, max (f_β''' β x) 0 < 4 := by
  obtain ⟨hβ₁, hk0, hk, hA, hU, hT, hcs, hus⟩ := h
  have hb := fun x (hx : x ∈ Set.Icc 0 π) =>
    f_β''_mem_Icc_of_isChain cs us hβ₁ hβI hβ1 hA hU hcs hus hx
  have hint := integral_posPart_f_β'''_le hβ (fun x hx => (hb x hx).1) (fun x hx => (hb x hx).2)
  have hτ := τ_β_star_two_le hβ (le_cosTauStar_two hβ₁ hβI.1 hβ hk0 hk)
  have hτ0 := (τ_β_star_two_mem hβ).1
  have hπ := Real.pi_lt_d4
  calc τ_β_star 2 β * ∫ x in (-π)..π, max (f_β''' β x) 0
      ≤ τ_β_star 2 β * (2 * (U - -A)) := mul_le_mul_of_nonneg_left hint hτ0.le
    _ ≤ (15708 / 10000 - k) * (2 * (U - -A)) :=
        mul_le_mul_of_nonneg_right (by norm_num at hπ; linarith) (by linarith)
    _ < 4 := by linarith

/-- The hypotheses of `τ_mul_integral_lt_four_of_cert`, `τ_β_star_two_le` and
`le_cosTauStar_two` are satisfiable: the certificate of `β ∈ [0, 3/20]` at
`β = 1/10`, and `k = 0` at `β = 1`. -/
example : τ_β_star 2 (1 / 10) * ∫ x in (-π)..π, max (f_β''' (1 / 10) x) 0 < 4 ∧
    τ_β_star 2 1 ≤ π / 2 - 0 ∧ (0 : ℝ) ≤ cosTauStar 2 1 :=
  ⟨τ_mul_integral_lt_four_of_cert (β₁ := 0) (β₂ := 3 / 20) (k := 0) (A := 159 / 125)
      (U := 0) (cs := [3 / 5]) (us := []) (by norm_num) (by norm_num) ⟨by norm_num, by norm_num⟩
      (by norm_num [SmallBetaCert, NegPieceOK, PosPieceOK, expPoly, List.isChain_singleton]),
    τ_β_star_two_le one_pos (cosTauStar_pos 2 1 one_pos (by norm_num)).le,
    le_cosTauStar_two le_rfl zero_le_one one_pos le_rfl (by norm_num)⟩

/-- The hypotheses of `f_β''_mem_Icc_of_isChain` are satisfiable: `β = 0` with the
single pieces `[0, 1]`, at `x = 0`. -/
example : -1 ≤ f_β'' 0 0 ∧ f_β'' 0 0 ≤ 1 :=
  f_β''_mem_Icc_of_isChain (β₂ := 0) [] [] le_rfl ⟨le_rfl, le_rfl⟩ zero_le_one le_rfl
    zero_le_one (by norm_num [NegPieceOK, expPoly]) (by norm_num [PosPieceOK])
    ⟨le_rfl, Real.pi_pos.le⟩

end Perspective
end Transformer
