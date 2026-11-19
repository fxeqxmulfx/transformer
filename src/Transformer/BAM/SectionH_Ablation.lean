module

public import Transformer.BAM.Section2_Locality
public import Transformer.BAM.SectionC_Parametrization

/-!
# Appendix H.1: the initializations of GGD-BAM

Bianchessi, Aguirre, Barros, Kupssinskü, arXiv:2505.22842v4, Appendix H.1 (Ablation study,
initialization).

The ablation starts the implementation's bias `implBias` (Appendix C) from two settings.

* "A Laplacian that replicates ALiBi, setting `θ_β = 1`, different `θ_α` for each layer and
  `θ_μ = 0`": the bias is ALiBi's with slope `e^{θ_α}`, shifted by the constant `−e^{θ_α} ε`
  (`implBias_alibi`), which softmax ignores (`softmax_add_const`). With `θ_α = ln m` attention
  is ALiBi's with slope `m`, for every `ε` (`relAttention_implBias_alibi`).
* "Uniform distribution prior `θ_β = 0`", "equivalent to assigning a Uniform prior to all tokens
  in the context": the bias is the constant `−e^{θ_α}` (`implBias_shape_zero`), for every `θ_μ`,
  `θ_α` and `ε`, so attention is NoPE's (`relAttention_implBias_shape_zero`) and the position
  prior is `Uniform(1, i, j)` (`softmax_causalMask_add_implBias`, after Lemma 1). The paper's
  `θ_α = 0` is not needed.
* "Which is a middle ground between ALiBi's Laplacian and `θ_β < 0`": for `ε > 0`, at every
  position the bias with `θ_β = 0` lies between those with `θ_β ≤ 0` and `θ_β ≥ 0`
  (`implBias_shape_zero_between`). In shape, the flat prior lies between one that falls with the
  distance from `μ` (`β > 0`, `ggdWeight_anti`) and one that rises with it (`β < 0`,
  `weight_ggdBias_mono`).
-/

@[expose] public section

namespace Transformer.BAM

open Finset

variable {L : ℕ}

/-- Adding a real constant to every score leaves softmax unchanged. -/
theorem softmax_add_const {ι : Type*} [Fintype ι] (s : ι → EReal) (c : ℝ) (j : ι) :
    softmax (fun z => s z + c) j = softmax s j := by
  simp only [softmax, partition, weight_add, weight_coe, ← sum_mul]
  exact mul_div_mul_right _ _ (Real.exp_ne_zero c)

/-- With `θ_μ = 0` and `θ_β = 1` the implementation's bias is ALiBi's with slope `e^{θ_α}`,
shifted by the constant `−e^{θ_α} ε` (Appendix H.1). -/
theorem implBias_alibi (θα ε x : ℝ) :
    (implBias 0 θα 1 ε x : EReal) =
      alibi (Real.exp θα) x + ((-(Real.exp θα * ε) : ℝ) : EReal) := by
  rw [implBias, alibi, ← EReal.coe_add, EReal.coe_eq_coe_iff]
  simp only [neg_zero, Real.exp_zero, sub_self, sub_zero, Real.rpow_one]
  ring

/-- **Appendix H.1: the initialization `θ_β = 1`, `θ_μ = 0` "replicates ALiBi"**: with
`θ_α = ln m` attention is ALiBi's with slope `m`, for every `ε`. -/
theorem relAttention_implBias_alibi {m : ℝ} (hm : 0 < m) (ε : ℝ) (f : Fin L → ℝ) (i j : Fin L) :
    relAttention (fun x => (implBias 0 (Real.log m) 1 ε x : EReal)) f i j =
      relAttention (alibi m) f i j := by
  simp only [relAttention, implBias_alibi, Real.exp_log hm, ← add_assoc]
  exact softmax_add_const _ _ j

/-- The hypothesis of `relAttention_implBias_alibi` holds for the slope `m = 1/2` and the
paper's `ε = 10^{−5}`. -/
example (f : Fin 3 → ℝ) (i j : Fin 3) :
    relAttention (fun x => (implBias 0 (Real.log (1 / 2)) 1 (1 / 100000) x : EReal)) f i j =
      relAttention (alibi (1 / 2)) f i j :=
  relAttention_implBias_alibi (by norm_num) _ f i j

/-- With `θ_β = 0` the implementation's bias is the constant `−e^{θ_α}` (Appendix H.1). -/
theorem implBias_shape_zero (θμ θα ε x : ℝ) : implBias θμ θα 0 ε x = -Real.exp θα := by
  rw [implBias, Real.rpow_zero, mul_one]

/-- **Appendix H.1: the initialization `θ_β = 0` assigns a uniform prior**: for every `θ_μ`,
`θ_α` and `ε`, attention is NoPE's. -/
theorem relAttention_implBias_shape_zero (θμ θα ε : ℝ) (f : Fin L → ℝ) (i j : Fin L) :
    relAttention (fun x => (implBias θμ θα 0 ε x : EReal)) f i j =
      relAttention (fun _ => 0) f i j := by
  simp only [relAttention, implBias_shape_zero, add_zero]
  exact softmax_add_const _ _ j

/-- **Appendix H.1: with `θ_β = 0` the position prior is `Uniform(1, i, j)`** "to all tokens in
the context", the prior of the causal mask (Lemma 1). -/
theorem softmax_causalMask_add_implBias (θμ θα ε : ℝ) (i j : Fin L) :
    softmax (fun z => causalMask i z + (implBias θμ θα 0 ε (relPos i z) : EReal)) j =
      uniform 1 ((i : ℕ) + 1) ((j : ℕ) + 1) := by
  simp only [implBias_shape_zero]
  rw [softmax_add_const (causalMask i), softmax_causalMask]

/-- **Appendix H.1: `θ_β = 0` is "a middle ground between ALiBi's Laplacian and `θ_β < 0`"**:
for `ε > 0`, at every position the bias with `θ_β = 0` lies between those with `θ_β = b₁ ≤ 0`
and `θ_β = b₂ ≥ 0`; ALiBi's shape is `b₂ = 1`. -/
theorem implBias_shape_zero_between {θμ θα ε b₁ b₂ : ℝ} (hε : 0 < ε) (h₁ : b₁ ≤ 0)
    (h₂ : 0 ≤ b₂) (x : ℝ) :
    min (implBias θμ θα b₁ ε x) (implBias θμ θα b₂ ε x) ≤ implBias θμ θα 0 ε x ∧
      implBias θμ θα 0 ε x ≤ max (implBias θμ θα b₁ ε x) (implBias θμ θα b₂ ε x) := by
  rw [implBias_shape_zero]
  simp only [implBias]
  set d := |x - (Real.exp θμ - Real.exp (-θμ))| + ε
  have hd0 : 0 < d := add_pos_of_nonneg_of_pos (abs_nonneg _) hε
  have he := Real.exp_pos θα
  rcases le_total 1 d with h | h
  · have k₁ : d ^ b₁ ≤ 1 := Real.rpow_le_one_of_one_le_of_nonpos h h₁
    have k₂ : 1 ≤ d ^ b₂ := Real.one_le_rpow h h₂
    exact ⟨min_le_of_right_le (by nlinarith), le_max_of_le_left (by nlinarith)⟩
  · have k₁ : 1 ≤ d ^ b₁ := Real.one_le_rpow_of_pos_of_le_one_of_nonpos hd0 h h₁
    have k₂ : d ^ b₂ ≤ 1 := Real.rpow_le_one hd0.le h h₂
    exact ⟨min_le_of_left_le (by nlinarith), le_max_of_le_right (by nlinarith)⟩

/-- The hypotheses of `implBias_shape_zero_between` hold for the paper's `ε = 10^{−5}`,
`θ_β = −1` and ALiBi's `θ_β = 1`. -/
example (x : ℝ) :
    min (implBias 0 0 (-1) (1 / 100000) x) (implBias 0 0 1 (1 / 100000) x) ≤
        implBias 0 0 0 (1 / 100000) x ∧
      implBias 0 0 0 (1 / 100000) x ≤
        max (implBias 0 0 (-1) (1 / 100000) x) (implBias 0 0 1 (1 / 100000) x) :=
  implBias_shape_zero_between (by norm_num) (by norm_num) zero_le_one x

end Transformer.BAM
