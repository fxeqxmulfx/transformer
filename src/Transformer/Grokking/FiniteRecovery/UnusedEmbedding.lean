module

public import Transformer.Grokking.FiniteRecovery.ArgandBatchDifferentiability

/-! A token absent from the input can be changed at the tied output head
without affecting any hidden state. The resulting directional derivative
rules out a nonuniform stationary prediction on a singleton batch. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.Argand

open Transformer.PhasorV2 Transformer.AMSGradW
open scoped InnerProductSpace ComplexConjugate

variable {A : Type*} {H d layers T : ℕ}

private def embeddingDirection [DecidableEq A] (a : A) (v : CSpace (H * d)) : Weights A H d layers
  | .inl (.embedding b j, im) => if b = a then (if im then (v j).im else (v j).re) else 0
  | _ => 0

private theorem embedding_along [DecidableEq A] (w : Weights A H d layers) (b a : A)
    (v : CSpace (H * d)) (s : ℝ) :
    (Model.fromWeights (w + s • embeddingDirection b v)).embedding a =
      (Model.fromWeights w).embedding a + if a = b then (s : ℂ) • v else 0 := by
  by_cases h : a = b
  · subst a
    ext j
    simp [Model.fromWeights, complexEntry, embeddingDirection, Pi.add_apply, Pi.smul_apply,
      PiLp.add_apply, PiLp.smul_apply, smul_eq_mul]
    apply Complex.ext
    · simp [Complex.mul_re]
    · simp [Complex.mul_im]
  · ext j
    simp [Model.fromWeights, complexEntry, embeddingDirection, Pi.add_apply, Pi.smul_apply, h]

private theorem hidden_along [DecidableEq A] (w : Weights A H d layers) (b : A)
    (v : CSpace (H * d)) (s : ℝ) (repeats : ℕ) (word : Word A T)
    (habsent : ∀ j, word j ≠ b) (i : Fin T) :
    (Model.fromWeights (w + s • embeddingDirection b v)).hidden repeats word i =
      (Model.fromWeights w).hidden repeats word i := by
  let w' := w + s • embeddingDirection b v
  have hLayer : (Model.fromWeights w').layer = (Model.fromWeights w).layer := by
    funext j
    simp [Model.fromWeights, w', complexEntry, embeddingDirection, Pi.add_apply, Pi.smul_apply, smul_eq_mul]
  have hAlpha : (Model.fromWeights w').alpha = (Model.fromWeights w).alpha := by
    funext j
    simp [Model.fromWeights, w', embeddingDirection, Pi.add_apply, Pi.smul_apply, smul_eq_mul]
  have hInput : (Model.fromWeights w').embedding ∘ word = (Model.fromWeights w).embedding ∘ word := by
    funext j
    simp only [Function.comp_apply, w', embedding_along, ite_eq_right (habsent j), add_zero]
  change (Model.fromWeights w').hidden repeats word i = _
  simp only [Model.hidden, Model.hiddenWith]
  rw [hLayer, hAlpha]
  have hi (j : Fin T) := congrFun hInput j
  simp only [Function.comp_apply] at hi
  simp only [hi]
  rfl

private theorem logit_along [DecidableEq A] (w : Weights A H d layers) (b : A)
    (repeats : ℕ) (word : Word A T) (habsent : ∀ j, word j ≠ b) (i : Fin T) (s : ℝ) (a : A) :
    let h := (Model.fromWeights w).hidden repeats word i
    (Model.fromWeights (w + s • embeddingDirection b h)).logits repeats word i a =
      (Model.fromWeights w).logits repeats word i a + if a = b then s * (‖h‖ ^ 2 / 2) else 0 := by
  let h := (Model.fromWeights w).hidden repeats word i
  change (Model.fromWeights (w + s • embeddingDirection b h)).logits repeats word i a = _
  rw [Model.logits, hidden_along w b h s repeats word habsent i, Model.readout, embedding_along]
  by_cases ha : a = b
  · subst a
    rw [ite_eq_left rfl, ite_eq_left rfl, inner_add_left, inner_smul_left,
      Complex.conj_ofReal, Complex.add_re, Complex.mul_re, Complex.ofReal_re, Complex.ofReal_im,
      zero_mul, sub_zero]
    change (_ + s * (⟪h, h⟫_ℂ).re) / 2 = _ + _
    have he : (⟪h, h⟫_ℂ).re = ‖h‖ ^ 2 := inner_self_eq_norm_sq (𝕜 := ℂ) h
    rw [he]
    change (_ + s * ‖h‖ ^ 2) / 2 = _ / 2 + s * (‖h‖ ^ 2 / 2)
    ring
  · simp only [ite_eq_right ha, add_zero]
    rfl

variable [Fintype A]

/-- When a singleton target has a competitor absent from the input word,
stationarity forces a zero final hidden vector. The derivative in the
absent embedding direction is the competitor probability times half the
squared norm of the final hidden vector. -/
theorem stationary_singleton_hidden_eq_zero (w : Weights A H d layers) (repeats : ℕ)
    (word : Word A (T + 1)) (i : Fin (T + 1)) (a b : A) (hba : b ≠ a)
    (habsent : ∀ j, word j ≠ b)
    (hzero : batchGradient repeats [⟨T, word, i, a⟩] w = 0) :
    (Model.fromWeights w).hidden repeats word i = 0 := by
  classical
  let E := Kernite.Space (Parameter A H d layers)
  let coords := PiLp.continuousLinearEquiv 2 ℝ (fun _ : Parameter A H d layers => ℝ)
  let batch : List (LabeledPosition A) := [⟨T, word, i, a⟩]
  let h := (Model.fromWeights w).hidden repeats word i
  let u : Weights A H d layers := embeddingDirection b h
  let z := (Model.fromWeights w).logits repeats word i
  let K := ‖h‖ ^ 2 / 2
  let C := ∑ k ∈ Finset.univ.erase b, Real.exp (z k)
  have hC : 0 ≤ C := Finset.sum_nonneg fun k _ => (Real.exp_pos (z k)).le
  have hden (s : ℝ) : 0 < C + Real.exp (z b + s * K) :=
    add_pos_of_nonneg_of_pos hC (Real.exp_pos _)
  have hlog (s : ℝ) :
      batchLoss repeats batch (w + s • u) = Real.log (C + Real.exp (z b + s * K)) - z a := by
    simp only [batch, batchLoss, List.map_singleton, List.sum_singleton, List.length_singleton, Nat.cast_one,
      div_one, Model.probability]
    have he (c : A) : (Model.fromWeights (w + s • u)).logits repeats word i c =
        z c + if c = b then s * K else 0 := logit_along w b repeats word habsent i s c
    change -Real.log (Real.exp ((Model.fromWeights (w + s • u)).logits repeats word i a) /
        ∑ c, Real.exp ((Model.fromWeights (w + s • u)).logits repeats word i c)) = _
    simp only [he, ite_eq_right hba.symm, add_zero]
    have hS : (∑ c, Real.exp (z c + if c = b then s * K else 0)) =
        C + Real.exp (z b + s * K) := by
      rw [← Finset.sum_erase_add Finset.univ _ (Finset.mem_univ b), ite_eq_left rfl]
      congr 1
      exact Finset.sum_congr rfl fun c hc => by
        rw [ite_eq_right (Finset.mem_erase.mp hc).1, add_zero]
    rw [hS, Real.log_div (Real.exp_ne_zero _) (hden s).ne', Real.log_exp, neg_sub]
  have hz : gradient (fun x : E => batchLoss repeats batch (coords x)) (coords.symm w) = 0 := by
    apply WithLp.ofLp_injective
    exact hzero
  have hfC : ContDiff ℝ 1 (fun x : E => batchLoss repeats batch (coords x)) :=
    (contDiff_batchLoss repeats batch).comp coords.contDiff
  have hf := hfC.differentiable one_ne_zero (coords.symm w)
  have heq : HasFDerivAt (fun x : E => batchLoss repeats batch (coords x))
      (0 : E →L[ℝ] ℝ) (coords.symm w) := by
    have hh := hf.hasGradientAt.hasFDerivAt
    rw [hz, map_zero] at hh
    exact hh
  have hline : HasDerivAt (fun s : ℝ => coords.symm w + s • (coords.symm u)) (coords.symm u) 0 := by
    simpa only [id_eq, one_smul] using
      ((hasDerivAt_id (0 : ℝ)).smul_const (coords.symm u)).const_add (coords.symm w)
  have hd0 : HasDerivAt (fun s : ℝ => batchLoss repeats batch (w + s • u)) 0 0 := by
    have heq' : HasFDerivAt (fun x : E => batchLoss repeats batch (coords x)) (0 : E →L[ℝ] ℝ)
        (coords.symm w + (0 : ℝ) • (coords.symm u)) := by simpa only [zero_smul, add_zero] using heq
    have hd := HasFDerivAt.comp_hasDerivAt (0 : ℝ) heq' hline
    simpa only [Function.comp_def, map_add, map_smul, ContinuousLinearEquiv.apply_symm_apply, zero_apply] using hd
  have hQ : HasDerivAt (fun s : ℝ => C + Real.exp (z b + s * K)) (Real.exp (z b) * K) 0 := by
    simpa only [id_eq, one_mul, zero_mul, add_zero] using
      (((hasDerivAt_id (0 : ℝ)).mul_const K).const_add (z b)).exp.const_add C
  have hdf := (hQ.log (hden 0).ne').sub_const (z a)
  simp only [zero_mul, add_zero] at hdf
  have hd : HasDerivAt (fun s : ℝ => batchLoss repeats batch (w + s • u))
      (Real.exp (z b) * K / (C + Real.exp (z b))) 0 := by
    simp_rw [hlog]
    exact hdf
  change h = 0
  by_contra hn
  have hK : 0 < K := div_pos (pow_pos (norm_pos_iff.mpr hn) _) (by norm_num)
  have hpos : 0 < Real.exp (z b) * K / (C + Real.exp (z b)) :=
    div_pos (mul_pos (Real.exp_pos _) hK) (add_pos_of_nonneg_of_pos hC (Real.exp_pos _))
  exact hpos.ne' (hd.deriv.symm.trans hd0.deriv)

end Transformer.Grokking.FiniteRecovery.Argand
