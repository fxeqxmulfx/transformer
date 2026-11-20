module

public import Transformer.RoPE.ReRoPE.Section3_Invariant

/-!
# ReRoPE, 计算 (Computation): the score of a complex head

Su, *Transformer升级之路：12、无限外推的ReRoPE？*, kexue.fm/archives/9708, section 计算
(Computation), applied to the attention score of a complex head. The pairs
`x_l = x_{(0,l)} + i x_{(1,l)}` of a query and a key give the Hermitian product
`Z = ∑_l conj(x_l) y_l`, and the head scores `g (Re Z + β (Im Z)² / D)`:

* `phaseScore`, `phaseScore_eq_re_im`: the score, with `Re Z = xᵀy` and `Im Z = −xᵀ J y`, where
  `J = rotateHalf` is the multiplication by `i`;
* `rotateHalf_rotary`, `phaseScore_rotary`: `J` commutes with `R^c`, and `R^c` is orthogonal, so
  a common rotation keeps the score;
* `rerope_phase_attend_eq_onlineMerge`, `leaky_phase_attend_eq_onlineMerge`: in such a head,
  ReRoPE and Leaky ReRoPE are one attention, one score per key (`Section3_Invariant`): from the
  query rotated by `i` and the key by `j` in the window `i − j < w`, and beyond it from the query
  rotated by `w` and the key unrotated (Leaky ReRoPE: by `(i − w)/k + w` and by `j/k`). Both
  the real and the imaginary part of `Z` take the rotations of the region of the key.

This is the score of gpt's main research model (architecture `complex-phase-score-lerope`, in no
paper): SSMax's gain `g = ln(i + 1)/D` at the query `i`, the real width `D` of the head and
`β = a/√(1 + a²)` with one learned `a` per layer and head; LeRoPE learns the frequencies
`θ_l = 10000^{−2l/D} e^{α_l}`, which the theorems leave arbitrary.
-/

@[expose] public section

open Matrix
open scoped ComplexConjugate
open Transformer.AttentionResiduals Transformer.Residual

namespace Transformer.RoPE.ReRoPE

open RoFormer

variable {p : ℕ} {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

/-- The score of a complex attention head, `g (Re Z + β (Im Z)² / D)` with `Re Z = xᵀy` and
`(Im Z)² = (xᵀ J y)²` (`phaseScore_eq_re_im`). -/
noncomputable def phaseScore (g β D : ℝ) (x y : Fin 2 × Fin p → ℝ) : ℝ :=
  g * (x ⬝ᵥ y + β * (x ⬝ᵥ rotateHalf y) ^ 2 / D)

/-- The pairs of `x` as complex numbers, `x_l = x_{(0,l)} + i x_{(1,l)}`. -/
noncomputable def toComplex (x : Fin 2 × Fin p → ℝ) (l : Fin p) : ℂ :=
  ⟨x (0, l), x (1, l)⟩

/-- `phaseScore` is `g (Re Z + β (Im Z)² / D)` for the Hermitian product
`Z = ∑_l conj(x_l) y_l` of the pairs. -/
theorem phaseScore_eq_re_im (g β D : ℝ) (x y : Fin 2 × Fin p → ℝ) :
    phaseScore g β D x y =
      g * ((∑ l, conj (toComplex x l) * toComplex y l).re +
        β * (∑ l, conj (toComplex x l) * toComplex y l).im ^ 2 / D) := by
  have hre : (∑ l, conj (toComplex x l) * toComplex y l).re = x ⬝ᵥ y := by
    simp [Complex.re_sum, toComplex, dotProduct, Fintype.sum_prod_type, Fin.sum_univ_two,
      Finset.sum_add_distrib]
  have him : (∑ l, conj (toComplex x l) * toComplex y l).im = -(x ⬝ᵥ rotateHalf y) := by
    simp [Complex.im_sum, toComplex, dotProduct, Fintype.sum_prod_type, Fin.sum_univ_two,
      rotateHalf, ← Finset.sum_add_distrib, ← Finset.sum_neg_distrib]
    refine Finset.sum_congr rfl fun l _ => ?_
    ring
  rw [phaseScore, hre, him, neg_sq]

/-- The multiplication by `i` commutes with the rotations: `J (R^c x) = R^c (J x)`. -/
theorem rotateHalf_rotary (θ : Fin p → ℝ) (c : ℝ) (x : Fin 2 × Fin p → ℝ) :
    rotateHalf (rotary θ c *ᵥ x) = rotary θ c *ᵥ rotateHalf x := by
  rw [rotary_mulVec, rotary_mulVec]
  funext ⟨r, l⟩
  fin_cases r <;> simp [rotateHalf]
  ring

/-- A common rotation keeps the score of a complex head. -/
theorem phaseScore_rotary (θ : Fin p → ℝ) (g β D c : ℝ) (x y : Fin 2 × Fin p → ℝ) :
    phaseScore g β D (rotary θ c *ᵥ x) (rotary θ c *ᵥ y) = phaseScore g β D x y := by
  rw [phaseScore, phaseScore, rotateHalf_rotary, dotProduct_rotary_rotary,
    dotProduct_rotary_rotary]

/-- Computation: ReRoPE's attention in a complex head with the score `phaseScore` is the online
merge of the scores of `R^i q_i` and `R^j k_j` over the window `i − j < w` and of `R^w q_i` and
`k_j` beyond it. -/
theorem rerope_phase_attend_eq_onlineMerge (θ : Fin p → ℝ) {w : ℕ} (hw : 0 < w) (i : ℕ)
    (g β D : ℝ) (q : Fin 2 × Fin p → ℝ) (key : ℕ → Fin 2 × Fin p → ℝ) (v : ℕ → E) (m₁ m₂ : ℝ) :
    attend (fun j : Fin (i + 1) =>
        phaseScore g β D q (rotary θ (-rectifiedPosition w ((i - j : ℕ) : ℝ)) *ᵥ key j))
        (fun j => v j) =
      onlineMerge
        (fun j : {j : Fin (i + 1) // i - (j : ℕ) < w} =>
          phaseScore g β D (rotary θ i *ᵥ q) (rotary θ (j.1 : ℕ) *ᵥ key j.1))
        (fun j : {j : Fin (i + 1) // ¬i - (j : ℕ) < w} =>
          phaseScore g β D (rotary θ w *ᵥ q) (key j.1))
        (fun j => v j.1) (fun j => v j.1) m₁ m₂ :=
  rerope_attend_eq_onlineMerge_of_invariant θ hw i (phaseScore_rotary θ g β D) q key v m₁ m₂

example (q : Fin 2 × Fin 16 → ℝ) (key : ℕ → Fin 2 × Fin 16 → ℝ) (v : ℕ → ℝ) :
    attend (fun j : Fin (127 + 1) =>
        phaseScore (Real.log 128 / 32) (1 / 2) 32 q
          (rotary (fun l : Fin 16 => frequency 10000 32 l)
            (-rectifiedPosition (16 : ℕ) ((127 - j : ℕ) : ℝ)) *ᵥ key j)) (fun j => v j) =
      onlineMerge
        (fun j : {j : Fin (127 + 1) // 127 - (j : ℕ) < 16} =>
          phaseScore (Real.log 128 / 32) (1 / 2) 32
            (rotary (fun l : Fin 16 => frequency 10000 32 l) (127 : ℕ) *ᵥ q)
            (rotary (fun l : Fin 16 => frequency 10000 32 l) (j.1 : ℕ) *ᵥ key j.1))
        (fun j : {j : Fin (127 + 1) // ¬127 - (j : ℕ) < 16} =>
          phaseScore (Real.log 128 / 32) (1 / 2) 32
            (rotary (fun l : Fin 16 => frequency 10000 32 l) (16 : ℕ) *ᵥ q) (key j.1))
        (fun j => v j.1) (fun j => v j.1) 0 0 :=
  rerope_phase_attend_eq_onlineMerge _ (by norm_num) 127 _ _ _ q key v 0 0

/-- Computation: Leaky ReRoPE's attention in a complex head with the score `phaseScore` is the
online merge of the scores of `R^i q_i` and `R^j k_j` over the window `i − j < w` and of
`R^{(i−w)/k+w} q_i` and `R^{j/k} k_j` beyond it. -/
theorem leaky_phase_attend_eq_onlineMerge (θ : Fin p → ℝ) {w : ℕ} (hw : 0 < w) (k : ℝ) (i : ℕ)
    (g β D : ℝ) (q : Fin 2 × Fin p → ℝ) (key : ℕ → Fin 2 × Fin p → ℝ) (v : ℕ → E) (m₁ m₂ : ℝ) :
    attend (fun j : Fin (i + 1) =>
        phaseScore g β D q (rotary θ (-leakyPosition w k ((i - j : ℕ) : ℝ)) *ᵥ key j))
        (fun j => v j) =
      onlineMerge
        (fun j : {j : Fin (i + 1) // i - (j : ℕ) < w} =>
          phaseScore g β D (rotary θ i *ᵥ q) (rotary θ (j.1 : ℕ) *ᵥ key j.1))
        (fun j : {j : Fin (i + 1) // ¬i - (j : ℕ) < w} =>
          phaseScore g β D (rotary θ ((i - w) / k + w) *ᵥ q) (rotary θ ((j.1 : ℕ) / k) *ᵥ key j.1))
        (fun j => v j.1) (fun j => v j.1) m₁ m₂ :=
  leaky_attend_eq_onlineMerge_of_invariant θ hw k i (phaseScore_rotary θ g β D) q key v m₁ m₂

example (q : Fin 2 × Fin 16 → ℝ) (key : ℕ → Fin 2 × Fin 16 → ℝ) (v : ℕ → ℝ) :
    attend (fun j : Fin (127 + 1) =>
        phaseScore (Real.log 128 / 32) (1 / 2) 32 q
          (rotary (fun l : Fin 16 => frequency 10000 32 l)
            (-leakyPosition (16 : ℕ) 16 ((127 - j : ℕ) : ℝ)) *ᵥ key j)) (fun j => v j) =
      onlineMerge
        (fun j : {j : Fin (127 + 1) // 127 - (j : ℕ) < 16} =>
          phaseScore (Real.log 128 / 32) (1 / 2) 32
            (rotary (fun l : Fin 16 => frequency 10000 32 l) (127 : ℕ) *ᵥ q)
            (rotary (fun l : Fin 16 => frequency 10000 32 l) (j.1 : ℕ) *ᵥ key j.1))
        (fun j : {j : Fin (127 + 1) // ¬127 - (j : ℕ) < 16} =>
          phaseScore (Real.log 128 / 32) (1 / 2) 32
            (rotary (fun l : Fin 16 => frequency 10000 32 l)
              (((127 : ℕ) - (16 : ℕ)) / 16 + (16 : ℕ)) *ᵥ q)
            (rotary (fun l : Fin 16 => frequency 10000 32 l) ((j.1 : ℕ) / 16) *ᵥ key j.1))
        (fun j => v j.1) (fun j => v j.1) 0 0 :=
  leaky_phase_attend_eq_onlineMerge _ (by norm_num) 16 127 _ _ _ q key v 0 0

end Transformer.RoPE.ReRoPE
