module

public import Transformer.Grokking.FlatSpectrum.FlatCapacity
public import Transformer.Grokking.FlatSpectrum.Factorization
public import Transformer.Grokking.FlatSpectrum.Roots
public import Transformer.Grokking.FlatSpectrum.TablePart.E6Pairing
public import Transformer.Grokking.FlatSpectrum.TablePart.E6Symmetric

/-!
# PT6: a table that beats itself with a constant fill

Source: `review/flat_spectrum.md`, *What a part of the table determines*, PT6. Let the kernel `K` of
a table correlate with the flat kernel `C` of `a³ + a b`, with `Re tr(ψ(24) Cᴴ K) ≥ 5900`. Since
`C / √97` is a contraction, `‖K‖_S1 ≥ 5900 / √97 > 598`. Let its seen cells also lean towards the
phase `ψ(23)`. Fill the unseen cells with the constant `t = ψ(23) / 2`. The fill is `t J` plus a
matrix `R` that lives on the seen cells, with `‖R‖²_F = 5880 − Σ_Ω Re ψ(f − 23) ≤ 2930`, so its norm
is at most `97 / 2 + √(97 · 2930) < 582`. So `K` is not a minimum (`not_returned_of_pairing`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.TablePart.E6

open Finset Matrix
open scoped ComplexOrder

attribute [local instance] fact_prime_97

/-- The table on the seen cells, `t` elsewhere. -/
noncomputable def fillTable (f : ZMod 97 → ZMod 97 → ZMod 97) (t : ℂ) :
    Matrix (Fin 1 → ZMod 97) (Fin 1 → ZMod 97) ℂ :=
  Matrix.of fun u v =>
    if halfMask.testBit (97 * (u 0).val + (v 0).val) then tableKernel f u v else t

/-- The matrix of ones. -/
noncomputable def onesLine : Matrix (Fin 1 → ZMod 97) (Fin 1 → ZMod 97) ℂ :=
  Matrix.of fun _ _ => 1

lemma nuclearNorm_onesLine_le : nuclearNorm onesLine ≤ 97 := by
  have h := nuclearNorm_le_of_eq_mul (X := onesLine)
    (Matrix.of fun (_ : Fin 1 → ZMod 97) (_ : Unit) => (1 : ℂ))
    (Matrix.of fun (_ : Fin 1 → ZMod 97) (_ : Unit) => (1 : ℂ)) (by
      ext u v
      simp [onesLine, Matrix.mul_apply])
  have hF : frobeniusSq (Matrix.of fun (_ : Fin 1 → ZMod 97) (_ : Unit) => (1 : ℂ)) = 97 := by
    simp [frobeniusSq, ZMod.card]
  rw [hF] at h
  linarith

lemma mem_half_line (u v : Fin 1 → ZMod 97) :
    (u, v) ∈ half ↔ halfMask.testBit (97 * (u 0).val + (v 0).val) = true := by
  have h : (line 97 (u 0).val, line 97 (v 0).val) ∈ half ↔ _ :=
    mem_half_iff (ZMod.val_lt (u 0)) (ZMod.val_lt (v 0))
  rwa [line_val, line_val] at h

lemma fill_sub_apply (f : ZMod 97 → ZMod 97 → ZMod 97) (t : ℂ) (u v : Fin 1 → ZMod 97) :
    (fillTable f t - t • onesLine) u v =
      if halfMask.testBit (97 * (u 0).val + (v 0).val) then
        (ZMod.stdAddChar (f (u 0) (v 0)) : ℂ) - t else 0 := by
  rw [Matrix.sub_apply, Matrix.smul_apply]
  by_cases h : halfMask.testBit (97 * (u 0).val + (v 0).val) = true <;>
    simp [fillTable, onesLine, tableKernel, h]

/-- The mask has 4704 set bits on the cells of the table. -/
lemma card_mask : (∑ a : ZMod 97, ∑ b : ZMod 97,
    if halfMask.testBit (97 * a.val + b.val) then (1 : ℕ) else 0) = 4704 := by
  decide +kernel

/-- The squared Frobenius norm of the part of the fill that lives on the seen cells. -/
lemma frobeniusSq_fill (f : ZMod 97 → ZMod 97 → ZMod 97) :
    frobeniusSq (fillTable f (((1 / 2 : ℝ) : ℂ) * ZMod.stdAddChar (23 : ZMod 97)) -
        (((1 / 2 : ℝ) : ℂ) * ZMod.stdAddChar (23 : ZMod 97)) • onesLine) =
      5880 - ∑ a : ZMod 97, ∑ b : ZMod 97, if halfMask.testBit (97 * a.val + b.val) then
        (ZMod.stdAddChar (f a b - 23) : ℂ).re else 0 := by
  have hpt (u v : Fin 1 → ZMod 97) :
      ‖(fillTable f (((1 / 2 : ℝ) : ℂ) * ZMod.stdAddChar (23 : ZMod 97)) -
        (((1 / 2 : ℝ) : ℂ) * ZMod.stdAddChar (23 : ZMod 97)) • onesLine) u v‖ ^ 2 =
      (if halfMask.testBit (97 * (u 0).val + (v 0).val) then (5 / 4 : ℝ) else 0) -
        if halfMask.testBit (97 * (u 0).val + (v 0).val) then
          (ZMod.stdAddChar (f (u 0) (v 0) - 23) : ℂ).re else 0 := by
    rw [fill_sub_apply]
    by_cases h : halfMask.testBit (97 * (u 0).val + (v 0).val) = true
    · simp only [h, ↓reduceIte]
      exact norm_sq_sub_half _ _
    · simp [h]
  have hA : ∑ a : ZMod 97, ∑ b : ZMod 97,
      (if halfMask.testBit (97 * a.val + b.val) then (5 / 4 : ℝ) else 0) = 5880 := by
    have h2 : ((∑ a : ZMod 97, ∑ b : ZMod 97,
        if halfMask.testBit (97 * a.val + b.val) then (1 : ℕ) else 0 : ℕ) : ℝ) = 4704 := by
      rw [card_mask]
      norm_num
    push_cast at h2
    calc ∑ a : ZMod 97, ∑ b : ZMod 97,
          (if halfMask.testBit (97 * a.val + b.val) then (5 / 4 : ℝ) else 0)
        = (5 / 4) * ∑ a : ZMod 97, ∑ b : ZMod 97,
          (if halfMask.testBit (97 * a.val + b.val) then (1 : ℝ) else 0) := by
          simp only [Finset.mul_sum, mul_ite, mul_one, mul_zero]
      _ = 5880 := by
          rw [h2]
          norm_num
  unfold frobeniusSq
  simp only [hpt, Finset.sum_sub_distrib]
  rw [sum_lines fun a b => if halfMask.testBit (97 * a.val + b.val) then (5 / 4 : ℝ) else 0,
    sum_lines fun a b => if halfMask.testBit (97 * a.val + b.val) then
      (ZMod.stdAddChar (f a b - 23) : ℂ).re else 0, hA]

/-- **A table that correlates with `a³ + a b` and leans towards one phase on its seen cells is not
returned by M.** -/
theorem not_returned_of_pairing {f : ZMod 97 → ZMod 97 → ZMod 97}
    (h1 : (5900 : ℝ) ≤ ∑ a : ZMod 97, ∑ b : ZMod 97,
      (ZMod.stdAddChar (f a b - (a ^ 3 + a * b) + 24) : ℂ).re)
    (h2 : (2950 : ℝ) ≤ ∑ a : ZMod 97, ∑ b : ZMod 97,
      if halfMask.testBit (97 * a.val + b.val) then
        (ZMod.stdAddChar (f a b - 23) : ℂ).re else 0) : NotReturned f := by
  intro hmin
  have hX : fillTable f (((1 / 2 : ℝ) : ℂ) * ZMod.stdAddChar (23 : ZMod 97)) ∈
      {X | ∀ ij ∈ (half : Set ((Fin 1 → ZMod 97) × (Fin 1 → ZMod 97))),
        X ij.1 ij.2 = tableKernel f ij.1 ij.2} := by
    intro ij hij
    have hb := (mem_half_line ij.1 ij.2).1 (Finset.mem_coe.1 hij)
    simp [fillTable, hb]
  have hle := isMinOn_iff.1 hmin _ hX
  have hlow := nuclearNorm_ge_pairing h1
  set t : ℂ := ((1 / 2 : ℝ) : ℂ) * ZMod.stdAddChar (23 : ZMod 97) with ht
  have htn : ‖t‖ = 1 / 2 := by
    rw [ht, norm_mul]
    simp
  set R := fillTable f t - t • onesLine with hR
  have hsum : fillTable f t = t • onesLine + R := by
    rw [hR]
    abel
  have hF : frobeniusSq R ≤ 2930 := by
    rw [hR, ht, frobeniusSq_fill]
    linarith
  have hrank : (R.rank : ℝ) ≤ 97 := by
    have h := Matrix.rank_le_card_width R
    simp only [Fintype.card_fun, Fintype.card_fin, ZMod.card, pow_one] at h
    exact_mod_cast h
  have hFnn : 0 ≤ frobeniusSq R := Finset.sum_nonneg fun _ _ => Finset.sum_nonneg fun _ _ => by
    positivity
  have hRsq : nuclearNorm R ^ 2 ≤ (533.2 : ℝ) ^ 2 := by
    refine (nuclearNorm_sq_le_rank_mul_frobeniusSq R).trans ?_
    calc (R.rank : ℝ) * frobeniusSq R ≤ 97 * 2930 :=
          mul_le_mul hrank hF hFnn (by norm_num)
      _ ≤ (533.2 : ℝ) ^ 2 := by norm_num
  have hRle : nuclearNorm R ≤ 533.2 := abs_le_of_sq_le_sq' hRsq (by norm_num) |>.2
  have hJ : nuclearNorm (t • onesLine) ≤ 97 / 2 := by
    rw [nuclearNorm_smul, htn]
    linarith [nuclearNorm_onesLine_le]
  have hup : nuclearNorm (fillTable f t) ≤ 97 / 2 + 533.2 := by
    rw [hsum]
    exact (nuclearNorm_add_le _ _).trans (add_le_add hJ hRle)
  have hspos : (0 : ℝ) < √97 := Real.sqrt_pos.2 (by norm_num)
  have hs : √97 ≤ 9.85 := Real.sqrt_le_iff.2 ⟨by norm_num, by norm_num⟩
  have hgap : (581.8 : ℝ) < (√97)⁻¹ * 5900 := by
    rw [inv_mul_eq_div, lt_div_iff₀ hspos]
    linarith
  linarith

end Transformer.Grokking.FlatSpectrum.TablePart.E6
