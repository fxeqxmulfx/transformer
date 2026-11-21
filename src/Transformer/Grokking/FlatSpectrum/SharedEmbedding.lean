module

public import Transformer.Grokking.FlatSpectrum.MulCharSpectrum
public import Transformer.Grokking.FlatSpectrum.FlatKernel
public import Transformer.Grokking.FlatSpectrum.JacobiSums
public import Transformer.Grokking.FlatSpectrum.ReadoutChoice

/-!
# What one layer writes through a shared embedding

Source: `review/flat_spectrum.md`, NV9. Both operands of a layer are read through one embedding
`E`. With the value maps complex-linear, positional heads, and the scalars of `R_D` and the
routers set aside, the part of a coordinate that couples `a` and `b` is `E C Eᵀ`, and the rest
is a function of `a`, a function of `b` and a constant (`IsSharedTable`). The columns and the
rows of such a table are functions on the same vocabulary, and together they span at most
`D + 3` dimensions (`IsSharedTable.finrank_col_sup_row_le`), so a table that needs more is out
of reach at any FFN width (`not_isSharedTable_of_lt`). The additive code of `a³ + a b` is one
(`not_isSharedTable_cubicKernel`). Its multiplicative code has rank only `(p − 1)/2`, and with
a small bias on class `0` the readout `Re (conj χ(c) χ(f))` decodes it exactly
(`mulCharLogit_lt`), so rank alone does not exclude it.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Matrix Module

section Shared

variable {K α ι : Type*} [Field K] [Fintype α] [Fintype ι]

/-- `X` couples its operands only through the shared embedding `E`: a bilinear form `E C Eᵀ`, a
function of the first operand, a function of the second and a constant. -/
def IsSharedTable (E : Matrix α ι K) (X : Matrix α α K) : Prop :=
  ∃ (C : Matrix ι ι K) (f g : α → K) (c : K),
    ∀ a b, X a b = (E * C * Eᵀ) a b + f a + g b + c

omit [Fintype α] in
/-- The transpose of a shared table is one, with the one-sided functions swapped. -/
lemma transpose_shared {E : Matrix α ι K} {C : Matrix ι ι K} {f g : α → K} {c : K}
    {X : Matrix α α K} (hX : ∀ a b, X a b = (E * C * Eᵀ) a b + f a + g b + c) (a b : α) :
    Xᵀ a b = (E * Cᵀ * Eᵀ) a b + g a + f b + c := by
  have hsym : (E * Cᵀ * Eᵀ) a b = (E * C * Eᵀ) b a := by
    rw [← transpose_apply (E * C * Eᵀ), transpose_mul, transpose_mul, transpose_transpose,
      Matrix.mul_assoc]
  rw [transpose_apply, hX, hsym]
  ring

example : (0 : Matrix (Fin 2) (Fin 2) ℚ)ᵀ 0 1 =
    ((1 : Matrix (Fin 2) (Fin 2) ℚ) * (0 : Matrix (Fin 2) (Fin 2) ℚ)ᵀ *
      (1 : Matrix (Fin 2) (Fin 2) ℚ)ᵀ) 0 1 + (0 : Fin 2 → ℚ) 0 + (0 : Fin 2 → ℚ) 1 + 0 :=
  transpose_shared (E := (1 : Matrix (Fin 2) (Fin 2) ℚ)) (C := 0) (f := 0) (g := 0) (c := 0)
    (by simp) 0 1

/-- Every column of a shared table lies in any subspace holding the columns of `E`, the
function of the first operand and the constant. -/
lemma mulVec_mem_of_shared {E : Matrix α ι K} {C : Matrix ι ι K} {f g : α → K} {c : K}
    {X : Matrix α α K} (hX : ∀ a b, X a b = (E * C * Eᵀ) a b + f a + g b + c)
    {W : Submodule K (α → K)} (hE : LinearMap.range E.mulVecLin ≤ W) (hf : f ∈ W)
    (h1 : (fun _ => 1 : α → K) ∈ W) (v : α → K) : X *ᵥ v ∈ W := by
  have h2 : E *ᵥ (C *ᵥ (Eᵀ *ᵥ v)) = (E * C * Eᵀ) *ᵥ v := by
    rw [mulVec_mulVec, mulVec_mulVec]
  have hv : X *ᵥ v = E *ᵥ (C *ᵥ (Eᵀ *ᵥ v)) + (∑ b, v b) • f +
      ((∑ b, g b * v b) + c * ∑ b, v b) • (fun _ => 1 : α → K) := by
    rw [h2]
    ext a
    simp only [mulVec, dotProduct, Pi.add_apply, Pi.smul_apply, smul_eq_mul, hX, add_mul,
      Finset.sum_add_distrib, mul_one, ← Finset.mul_sum]
    ring
  rw [hv]
  exact W.add_mem (W.add_mem (hE ⟨_, rfl⟩) (W.smul_mem _ hf)) (W.smul_mem _ h1)

example (v : Fin 2 → ℚ) : (0 : Matrix (Fin 2) (Fin 2) ℚ) *ᵥ v ∈ (⊤ : Submodule ℚ (Fin 2 → ℚ)) :=
  mulVec_mem_of_shared (E := (1 : Matrix (Fin 2) (Fin 2) ℚ)) (C := 0) (f := 0) (g := 0) (c := 0)
    (by simp) le_top trivial trivial v

/-- **The count.** The columns and the rows of a table written through a shared embedding of
width `D` span at most `D + 3` dimensions. -/
theorem IsSharedTable.finrank_col_sup_row_le {E : Matrix α ι K} {X : Matrix α α K}
    (h : IsSharedTable E X) :
    finrank K ↥(LinearMap.range X.mulVecLin ⊔ LinearMap.range Xᵀ.mulVecLin) ≤
      Fintype.card ι + 3 := by
  classical
  obtain ⟨C, f, g, c, hX⟩ := h
  set T : Finset (α → K) := {f, g, fun _ => 1}
  set W : Submodule K (α → K) :=
    LinearMap.range E.mulVecLin ⊔ Submodule.span K (T : Set (α → K))
  have hE : LinearMap.range E.mulVecLin ≤ W := le_sup_left
  have hT : ∀ t ∈ T, t ∈ W := fun t ht =>
    (le_sup_right : Submodule.span K (T : Set (α → K)) ≤ W) (Submodule.subset_span ht)
  have hcol : LinearMap.range X.mulVecLin ≤ W := by
    rintro _ ⟨v, rfl⟩
    exact mulVec_mem_of_shared hX hE (hT f (by simp [T])) (hT _ (by simp [T])) v
  have hrow : LinearMap.range Xᵀ.mulVecLin ≤ W := by
    rintro _ ⟨v, rfl⟩
    exact mulVec_mem_of_shared (transpose_shared hX) hE (hT g (by simp [T]))
      (hT _ (by simp [T])) v
  calc finrank K ↥(LinearMap.range X.mulVecLin ⊔ LinearMap.range Xᵀ.mulVecLin)
      ≤ finrank K W := Submodule.finrank_mono (sup_le hcol hrow)
    _ ≤ finrank K (LinearMap.range E.mulVecLin) +
        finrank K (Submodule.span K (T : Set (α → K))) :=
        Submodule.finrank_add_le_finrank_add_finrank _ _
    _ ≤ Fintype.card ι + 3 :=
        add_le_add E.rank_le_card_width
          ((finrank_span_finset_le_card T).trans Finset.card_le_three)

example : finrank ℚ ↥(LinearMap.range (0 : Matrix (Fin 2) (Fin 2) ℚ).mulVecLin ⊔
      LinearMap.range (0 : Matrix (Fin 2) (Fin 2) ℚ)ᵀ.mulVecLin) ≤ Fintype.card (Fin 2) + 3 :=
  IsSharedTable.finrank_col_sup_row_le (E := (1 : Matrix (Fin 2) (Fin 2) ℚ))
    ⟨0, 0, 0, 0, by simp⟩

/-- A table whose columns and rows span more than `D + 3` dimensions is out of reach. -/
theorem not_isSharedTable_of_lt {E : Matrix α ι K} {X : Matrix α α K}
    (h : Fintype.card ι + 3 <
      finrank K ↥(LinearMap.range X.mulVecLin ⊔ LinearMap.range Xᵀ.mulVecLin)) :
    ¬IsSharedTable E X :=
  fun hX => absurd hX.finrank_col_sup_row_le (not_le.2 h)

end Shared

section Codes

variable {p : ℕ}

/-- **The additive code of `a³ + a b` is out of reach** of a shared embedding of width
`D < p − 3`: it has rank `p`. -/
theorem not_isSharedTable_cubicKernel [NeZero p] {ι : Type*} [Fintype ι]
    (E : Matrix (ZMod p) ι ℂ) (h : Fintype.card ι + 3 < p) :
    ¬IsSharedTable E (cubicKernel p) := by
  refine not_isSharedTable_of_lt (lt_of_lt_of_le h ?_)
  have hp : (p : ℝ) ≠ 0 := Nat.cast_ne_zero.2 (NeZero.ne p)
  have hr := (isFlat_cubicKernel p).rank_eq hp
  calc p = (cubicKernel p).rank := by simpa [ZMod.card] using hr.symm
    _ ≤ _ := Submodule.finrank_mono le_sup_left

example (E : Matrix (ZMod 5) (Fin 1) ℂ) : ¬IsSharedTable E (cubicKernel 5) :=
  not_isSharedTable_cubicKernel E (by simp)

example (E : Matrix (ZMod 5) (Fin 1) ℂ) : ¬IsSharedTable E (cubicKernel 5) := by
  refine not_isSharedTable_of_lt ?_
  have hr := (isFlat_cubicKernel 5).rank_eq (by norm_num)
  have hle : (cubicKernel 5).rank ≤ finrank ℂ ↥(LinearMap.range (cubicKernel 5).mulVecLin ⊔
      LinearMap.range (cubicKernel 5)ᵀ.mulVecLin) := Submodule.finrank_mono le_sup_left
  have h5 : (cubicKernel 5).rank = 5 := by simpa [ZMod.card] using hr
  simp only [Fintype.card_fin]
  omega

variable [Fact p.Prime]

/-- The logit of class `c` when the code holds `χ f`: the readout `Re (conj χ(c) χ(f))`, with a
bias `β` on class `0`, whose code is `0`. -/
noncomputable def mulCharLogit (χ : MulChar (ZMod p) ℂ) (β : ℝ) (f c : ZMod p) : ℝ :=
  (star (χ c) * χ f).re + if c = 0 then β else 0

/-- **The multiplicative code decodes exactly.** For an injective `χ` and any bias in `(0, 1)`,
the true class has the strictly largest logit. -/
theorem mulCharLogit_lt {χ : MulChar (ZMod p) ℂ} (hχ : Function.Injective χ) {β : ℝ}
    (h0 : 0 < β) (h1 : β < 1) {f c : ZMod p} (hc : c ≠ f) :
    mulCharLogit χ β f c < mulCharLogit χ β f f := by
  by_cases hf : f = 0
  · subst hf
    simp [mulCharLogit, MulChar.map_zero, hc, h0]
  have hfn : ‖χ f‖ = 1 := norm_mulChar_eq_one χ hf
  have hself : (star (χ f) * χ f).re = 1 := by
    have := Complex.normSq_eq_norm_sq (χ f)
    rw [hfn, Complex.normSq_apply] at this
    simp only [Complex.star_def, Complex.mul_re, Complex.conj_re, Complex.conj_im]
    nlinarith
  by_cases hc0 : c = 0
  · subst hc0
    simp only [mulCharLogit, MulChar.map_zero, star_zero, zero_mul, Complex.zero_re, ↓reduceIte,
      hf, hself]
    linarith
  have hcn : ‖χ c‖ = 1 := norm_mulChar_eq_one χ hc0
  have hcs := Complex.normSq_eq_norm_sq (χ c)
  have hfs := Complex.normSq_eq_norm_sq (χ f)
  rw [hcn, Complex.normSq_apply] at hcs
  rw [hfn, Complex.normSq_apply] at hfs
  simp only [mulCharLogit, hc0, hf, ↓reduceIte, add_zero, hself]
  simp only [Complex.star_def, Complex.mul_re, Complex.conj_re, Complex.conj_im]
  by_contra hle
  have hre : (χ c).re = (χ f).re := by
    nlinarith [sq_nonneg ((χ c).re - (χ f).re), sq_nonneg ((χ c).im - (χ f).im)]
  have him : (χ c).im = (χ f).im := by
    nlinarith [sq_nonneg ((χ c).re - (χ f).re), sq_nonneg ((χ c).im - (χ f).im)]
  exact hc (hχ (Complex.ext hre him))

example : ∃ χ : MulChar (ZMod 3) ℂ, mulCharLogit χ (1 / 2) 1 2 < mulCharLogit χ (1 / 2) 1 1 := by
  obtain ⟨χ, hχ⟩ := exists_injective_mulChar (p := 3)
  exact ⟨χ, mulCharLogit_lt hχ (by norm_num) (by norm_num) (by decide)⟩

end Codes

end Transformer.Grokking.FlatSpectrum
