/-
# `τ_β ∫ |f_β'''|₊ < 4` for `0 < β < 1`

Polyanskiy, Rigollet, Yao — arXiv:2507.22857v1, *Synchronization of mean-field
models on the circle*, §5, proof of `cor:conjecture`; `lemma:beta_0.75`,
`lemma:beta_0.5`, `lemma:beta_0.33` and `lemma:beta_final`.

§5 proves `τ ⟨1, |f'''|₊⟩ < 4` for `f = f_β` and `τ = τ_β`, the criterion of
`thm:main`: `τ(f''(a) - f''(b)) < 2` for `1/3 < β < 1` (`lemma:beta_0.75`,
`lemma:beta_0.5`, `lemma:beta_0.33`) and `τ f''(a) > -2` for `0 < β ≤ 1/3`
(`lemma:beta_final`).  Here the bound follows for every `0 < β < 1` from the
certificates of `Perspective.SyncKernelCert` on nine intervals of `β`
(`τ_mul_integral_posPart_f_β'''_lt_four_of_lt_one`), and the four lemmas in
the form of the source follow from it (`beta_0_75`, `beta_0_5`, `beta_0_33`,
`beta_final`).
-/

import Transformer.Perspective.SyncKernelCert

open Real

namespace Transformer
namespace Perspective

/-- **`τ_β ⟨1, |f_β'''|₊⟩ < 4` for `0 < β < 1`**, the bound that §5 derives from
`lemma:beta_0.75`, `lemma:beta_0.5`, `lemma:beta_0.33` (`1/3 < β < 1`) and
`lemma:beta_final` (`0 < β ≤ 1/3`).

Source: arXiv:2507.22857v1, §5, proof of `cor:conjecture`: for `β > 1/3` "it
suffices to prove that `τ(f''(a) - f''(b)) < 2`", for `0 < β ≤ 1/3` that
`τ f''(a) > -2`; both are `τ ∫_{-π}^{π} |f'''|₊ < 4`
(`integral_posPart_f_β'''_of_one_third_lt`, `integral_posPart_f_β'''_of_le_one_third`).
The proof differs from the source: instead of the values of `g¹`, `g²` and of
`lemma:tau1` at the ends of the intervals `[0.75, 1)`, `[0.5, 0.75)`,
`(1/3, 0.5)`, and of `sin τ e^{β cos τ - β}` on five intervals of `(0, 1/3]`, it
bounds `f_β''` on `[0, π]` by pieces of `cos x` (`Perspective.SyncKernelPieces`)
and `τ_β` by `π/2 - cos τ_β`, on nine intervals of `(0, 1)`. -/
theorem τ_mul_integral_posPart_f_β'''_lt_four_of_lt_one {β : ℝ} (hβ : 0 < β)
    (hβ1 : β < 1) : τ_β_star 2 β * ∫ x in (-π)..π, max (f_β''' β x) 0 < 4 := by
  have key := fun {β₁ β₂ k A U : ℝ} {cs us : List ℝ} (hβI : β ∈ Set.Icc β₁ β₂)
    (h : SmallBetaCert β₁ β₂ k A U cs us) => τ_mul_integral_lt_four_of_cert hβ hβ1.le hβI h
  rcases le_or_gt β (3 / 20) with h₁ | h₁
  · exact key (k := 0) (A := 159 / 125) (U := 0) (cs := [3 / 5]) (us := []) ⟨hβ.le, h₁⟩
      (by norm_num [SmallBetaCert, NegPieceOK, PosPieceOK, expPoly, List.isChain_singleton])
  rcases le_or_gt β (3 / 10) with h₂ | h₂
  · exact key (k := 7 / 50) (A := 349 / 250) (U := 0) (cs := [14 / 25, 41 / 50]) (us := [])
      ⟨h₁.le, h₂⟩
      (by norm_num [SmallBetaCert, NegPieceOK, PosPieceOK, expPoly, List.isChain_singleton])
  rcases le_or_gt β (2 / 5) with h₃ | h₃
  · exact key (k := 27 / 100) (A := 359 / 250) (U := 1 / 10) (cs := [57 / 100, 19 / 25])
      (us := [81 / 100]) ⟨h₂.le, h₃⟩
      (by norm_num [SmallBetaCert, NegPieceOK, PosPieceOK, expPoly, List.isChain_singleton])
  rcases le_or_gt β (1 / 2) with h₄ | h₄
  · exact key (k := 7 / 20) (A := 1477 / 1000) (U := 4 / 25)
      (cs := [57 / 100, 18 / 25, 43 / 50]) (us := [13 / 20, 21 / 25]) ⟨h₃.le, h₄⟩
      (by norm_num [SmallBetaCert, NegPieceOK, PosPieceOK, expPoly, List.isChain_singleton])
  rcases le_or_gt β (3 / 5) with h₅ | h₅
  · exact key (k := 41 / 100) (A := 1511 / 1000) (U := 21 / 100)
      (cs := [57 / 100, 7 / 10, 79 / 100, 9 / 10]) (us := [27 / 50, 71 / 100, 21 / 25])
      ⟨h₄.le, h₅⟩
      (by norm_num [SmallBetaCert, NegPieceOK, PosPieceOK, expPoly, List.isChain_singleton])
  rcases le_or_gt β (7 / 10) with h₆ | h₆
  · exact key (k := 23 / 50) (A := 1549 / 1000) (U := 1 / 4)
      (cs := [29 / 50, 69 / 100, 19 / 25, 83 / 100, 23 / 25])
      (us := [9 / 20, 3 / 5, 71 / 100, 83 / 100]) ⟨h₅.le, h₆⟩
      (by norm_num [SmallBetaCert, NegPieceOK, PosPieceOK, expPoly, List.isChain_singleton])
  rcases le_or_gt β (4 / 5) with h₇ | h₇
  · exact key (k := 51 / 100) (A := 198 / 125) (U := 3 / 10)
      (cs := [59 / 100, 69 / 100, 3 / 4, 4 / 5, 17 / 20, 23 / 25])
      (us := [2 / 5, 11 / 20, 67 / 100, 81 / 100]) ⟨h₆.le, h₇⟩
      (by norm_num [SmallBetaCert, NegPieceOK, PosPieceOK, expPoly, List.isChain_singleton])
  rcases le_or_gt β (9 / 10) with h₈ | h₈
  · exact key (k := 11 / 20) (A := 809 / 500) (U := 17 / 50)
      (cs := [3 / 5, 7 / 10, 3 / 4, 79 / 100, 41 / 50, 43 / 50, 91 / 100])
      (us := [7 / 20, 1 / 2, 31 / 50, 77 / 100]) ⟨h₇.le, h₈⟩
      (by norm_num [SmallBetaCert, NegPieceOK, PosPieceOK, expPoly, List.isChain_singleton])
  exact key (k := 29 / 50) (A := 1667 / 1000) (U := 7 / 20)
    (cs := [61 / 100, 71 / 100, 19 / 25, 79 / 100, 41 / 50, 17 / 20, 22 / 25, 93 / 100])
    (us := [7 / 25, 41 / 100, 51 / 100, 61 / 100, 37 / 50]) ⟨h₈.le, hβ1.le⟩
    (by norm_num [SmallBetaCert, NegPieceOK, PosPieceOK, expPoly, List.isChain_singleton])

/-- The common form of `lemma:beta_0.75`, `lemma:beta_0.5` and `lemma:beta_0.33`
of arXiv:2507.22857v1, Appendix A: for `1/3 < β < 1` and `0 < b < a < π` of
`lemma:positiveregion`, `τ_β (f_β''(a) - f_β''(b)) < 2`. -/
theorem τ_mul_f_β''_sub_lt_two_of_lt_one {β : ℝ} (hβ : 1 / 3 < β) (hβ1 : β < 1) :
    ∃ b a, 0 < b ∧ b < a ∧ a < π ∧
      (∀ x ∈ Set.Ioc (-π) π, 0 < f_β''' β x ↔ x ∈ Set.Ioo (-a) (-b) ∪ Set.Ioo b a) ∧
      τ_β_star 2 β * (f_β'' β a - f_β'' β b) < 2 := by
  obtain ⟨b, a, hb, hba, ha, hreg, hint⟩ := integral_posPart_f_β'''_of_one_third_lt hβ
  refine ⟨b, a, hb, hba, ha, hreg, ?_⟩
  have h := τ_mul_integral_posPart_f_β'''_lt_four_of_lt_one (by linarith) hβ1
  rw [hint] at h
  linarith

/-- The hypotheses `1/3 < β < 1` hold at `β = 1/2`. -/
example : ∃ b a, 0 < b ∧ b < a ∧ a < π ∧
    (∀ x ∈ Set.Ioc (-π) π, 0 < f_β''' (1 / 2) x ↔ x ∈ Set.Ioo (-a) (-b) ∪ Set.Ioo b a) ∧
    τ_β_star 2 (1 / 2) * (f_β'' (1 / 2) a - f_β'' (1 / 2) b) < 2 :=
  τ_mul_f_β''_sub_lt_two_of_lt_one (by norm_num) (by norm_num)

/-- **`lemma:beta_0.75`.**  For `β ∈ [0.75, 1)`, `τ(f''(a) - f''(b)) < 2`, with
`f = f_β`, `τ = τ_β` and `b < a` of `lemma:positiveregion`.

Source: arXiv:2507.22857v1, Appendix A, `lemma:beta_0.75`. -/
theorem beta_0_75 {β : ℝ} (hβ : 3 / 4 ≤ β) (hβ1 : β < 1) :
    ∃ b a, 0 < b ∧ b < a ∧ a < π ∧
      (∀ x ∈ Set.Ioc (-π) π, 0 < f_β''' β x ↔ x ∈ Set.Ioo (-a) (-b) ∪ Set.Ioo b a) ∧
      τ_β_star 2 β * (f_β'' β a - f_β'' β b) < 2 :=
  τ_mul_f_β''_sub_lt_two_of_lt_one (by linarith) hβ1

/-- **`lemma:beta_0.5`.**  For `β ∈ [0.5, 0.75)`, `τ(f''(a) - f''(b)) < 2`, with
`f = f_β`, `τ = τ_β` and `b < a` of `lemma:positiveregion`.

Source: arXiv:2507.22857v1, Appendix A, `lemma:beta_0.5`. -/
theorem beta_0_5 {β : ℝ} (hβ : 1 / 2 ≤ β) (hβ1 : β < 3 / 4) :
    ∃ b a, 0 < b ∧ b < a ∧ a < π ∧
      (∀ x ∈ Set.Ioc (-π) π, 0 < f_β''' β x ↔ x ∈ Set.Ioo (-a) (-b) ∪ Set.Ioo b a) ∧
      τ_β_star 2 β * (f_β'' β a - f_β'' β b) < 2 :=
  τ_mul_f_β''_sub_lt_two_of_lt_one (by linarith) (by linarith)

/-- **`lemma:beta_0.33`.**  For `β ∈ (1/3, 0.5)`, `τ(f''(a) - f''(b)) < 2`, with
`f = f_β`, `τ = τ_β` and `b < a` of `lemma:positiveregion`.

Source: arXiv:2507.22857v1, Appendix A, `lemma:beta_0.33`. -/
theorem beta_0_33 {β : ℝ} (hβ : 1 / 3 < β) (hβ1 : β < 1 / 2) :
    ∃ b a, 0 < b ∧ b < a ∧ a < π ∧
      (∀ x ∈ Set.Ioc (-π) π, 0 < f_β''' β x ↔ x ∈ Set.Ioo (-a) (-b) ∪ Set.Ioo b a) ∧
      τ_β_star 2 β * (f_β'' β a - f_β'' β b) < 2 :=
  τ_mul_f_β''_sub_lt_two_of_lt_one hβ (by linarith)

/-- **`lemma:beta_final`.**  For `β ∈ (0, 1/3]`, `τ f''(a) > -2`, with `f = f_β`,
`τ = τ_β` and `a ∈ (0, π/2)` of `lemma:positiveregion2`.

Source: arXiv:2507.22857v1, Appendix B, `lemma:beta_final`. -/
theorem beta_final {β : ℝ} (hβ : 0 < β) (hβ' : β ≤ 1 / 3) :
    ∃ a ∈ Set.Ioo 0 (π / 2),
      (∀ x ∈ Set.Ico 0 (2 * π), 0 ≤ f_β''' β x ↔ x ∈ Set.Icc a (2 * π - a)) ∧
      -2 < τ_β_star 2 β * f_β'' β a := by
  obtain ⟨a, ha, hreg, hint⟩ := integral_posPart_f_β'''_of_le_one_third hβ hβ'
  refine ⟨a, ha, hreg, ?_⟩
  have h := τ_mul_integral_posPart_f_β'''_lt_four_of_lt_one hβ (by linarith)
  rw [hint] at h
  linarith

/-- The hypotheses above are satisfiable: `β = 1/2` for the bound, and `β = 0.8`,
`0.6`, `0.4`, `0.2` for the four lemmas. -/
example : τ_β_star 2 (1 / 2) * ∫ x in (-π)..π, max (f_β''' (1 / 2) x) 0 < 4 ∧
    (∃ b a, 0 < b ∧ b < a ∧ a < π ∧
      (∀ x ∈ Set.Ioc (-π) π, 0 < f_β''' (4 / 5) x ↔ x ∈ Set.Ioo (-a) (-b) ∪ Set.Ioo b a) ∧
      τ_β_star 2 (4 / 5) * (f_β'' (4 / 5) a - f_β'' (4 / 5) b) < 2) ∧
    (∃ b a, 0 < b ∧ b < a ∧ a < π ∧
      (∀ x ∈ Set.Ioc (-π) π, 0 < f_β''' (3 / 5) x ↔ x ∈ Set.Ioo (-a) (-b) ∪ Set.Ioo b a) ∧
      τ_β_star 2 (3 / 5) * (f_β'' (3 / 5) a - f_β'' (3 / 5) b) < 2) ∧
    (∃ b a, 0 < b ∧ b < a ∧ a < π ∧
      (∀ x ∈ Set.Ioc (-π) π, 0 < f_β''' (2 / 5) x ↔ x ∈ Set.Ioo (-a) (-b) ∪ Set.Ioo b a) ∧
      τ_β_star 2 (2 / 5) * (f_β'' (2 / 5) a - f_β'' (2 / 5) b) < 2) ∧
    ∃ a ∈ Set.Ioo 0 (π / 2),
      (∀ x ∈ Set.Ico 0 (2 * π), 0 ≤ f_β''' (1 / 5) x ↔ x ∈ Set.Icc a (2 * π - a)) ∧
      -2 < τ_β_star 2 (1 / 5) * f_β'' (1 / 5) a :=
  ⟨τ_mul_integral_posPart_f_β'''_lt_four_of_lt_one (by norm_num) (by norm_num),
    beta_0_75 (by norm_num) (by norm_num), beta_0_5 (by norm_num) (by norm_num),
    beta_0_33 (by norm_num) (by norm_num), beta_final (by norm_num) (by norm_num)⟩

end Perspective
end Transformer
