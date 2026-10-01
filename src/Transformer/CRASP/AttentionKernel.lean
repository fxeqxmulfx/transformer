/-
# Attention evaluated by integer state counts

arXiv:2506.16055v3, Appendix B.1, Equation `eq:att`, and B.2's
numerator/denominator construction in `thm:rtfr_to_TLCl`.
-/

import Transformer.CRASP.TransformerStateSums

namespace Transformer.CRASP.RTfr

universe u
variable {σ : Type u} {p s d k : ℕ}

/-- The attention part of a rounded layer (Appendix B.1, `eq:att`). -/
noncomputable def attention (T : RTfr σ p s d k) (ℓ : ℕ) {n : ℕ}
    (h : Fin n → Fin d → Fx p s) (i : Fin n) (c : Fin d) : Fx p s :=
  let score := fun j => ∑ t : Fin d, (T.WQ ℓ (h i) t).val * (T.WK ℓ (h j) t).val
  let den := ∑ j ∈ masked i, (Fx.round p s (Real.exp (score j))).val
  if den = 0 then
    Fx.round p s ((∑ j ∈ masked i, (T.WV ℓ (h j) c).val) / ((masked i).card : ℝ))
  else
    Fx.round p s
      ((∑ j ∈ masked i, (Fx.round p s (Real.exp (score j) * (T.WV ℓ (h j) c).val)).val) / den)

/-- Attention, rounded residual addition, and feed-forward form the layer (B.1). -/
theorem layer_eq_attention (T : RTfr σ p s d k) (ℓ : ℕ) {n : ℕ}
    (h : Fin n → Fin d → Fx p s) (i : Fin n) :
    T.layer ℓ h i = T.ff ℓ (fun c => Fx.add (T.attention ℓ h i c) (h i c)) := rfl

/-- The logit for a fixed query state and a possible source state (B.2). -/
noncomputable def stateLogit (T : RTfr σ p s d k) (ℓ : ℕ)
    (q a : Fin d → Fx p s) : ℝ :=
  ∑ c : Fin d, (T.WQ ℓ q c).val * (T.WK ℓ a c).val

/-- A denominator summand, as a function of the source state (Appendix B.2). -/
noncomputable def stateWeight (T : RTfr σ p s d k) (ℓ : ℕ)
    (q a : Fin d → Fx p s) : Fx p s := Fx.round p s (Real.exp (T.stateLogit ℓ q a))

/-- A numerator summand, rounded before summing, as the model requires (B.2). -/
noncomputable def stateWeightedValue (T : RTfr σ p s d k) (ℓ : ℕ)
    (q : Fin d → Fx p s) (c : Fin d) (a : Fin d → Fx p s) : Fx p s :=
  Fx.round p s (Real.exp (T.stateLogit ℓ q a) * (T.WV ℓ a c).val)

/-- Attention reconstructed from integer sums of the finite source states (B.2). -/
noncomputable def countAttention (T : RTfr (Option σ) p s d k) (ℓ : ℕ)
    (ψ : (Fin d → Fx p s) → Form σ) (q : Fin d → Fx p s)
    [DecidableEq σ] (w : List σ) (i : ℕ) (c : Fin d) : Fx p s :=
  let D := (T.stateSum ℓ ψ (T.stateWeight ℓ q)).val w i
  let N := (T.stateSum ℓ ψ (T.stateWeightedValue ℓ q c)).val w i
  let V := (T.stateSum ℓ ψ (fun a => T.WV ℓ a c)).val w i
  if D = 0 then Fx.round p s (((V : ℝ) / (i + 1 : ℕ)) / 2 ^ s)
  else Fx.round p s ((N : ℝ) / (D : ℝ))

variable [DecidableEq σ]

/-- Every rounded exponential contributes a nonnegative denominator (B.1). -/
theorem stateSum_weight_nonneg (T : RTfr (Option σ) p s d k) (ℓ : ℕ)
    (ψ : (Fin d → Fx p s) → Form σ) (q : Fin d → Fx p s) (w : List σ) (i : ℕ) :
    0 ≤ (T.stateSum ℓ ψ (T.stateWeight ℓ q)).val w i := by
  classical
  apply LinearCount.val_states_nonneg
  intro a
  exact Fx.m_round_nonneg _ (Real.exp_pos _).le

/-- The counted attention is exactly the model's rounded attention.
Source: arXiv:2506.16055v3, Appendix B.2, construction of `A` and `B`. -/
theorem attention_eq_countAttention (T : RTfr (Option σ) p s d k) (ℓ : ℕ)
    {ψ : (Fin d → Fx p s) → Form σ} (hψ : T.StateDefinition ℓ ψ)
    (w : List σ) (i : Fin (bos w).length) (q : Fin d → Fx p s)
    (hq : T.actAt w ℓ i.val = q) (c : Fin d) :
    T.attention ℓ (T.act (bos w) ℓ) i c = T.countAttention ℓ ψ q w i.val c := by
  have hc : T.act (bos w) ℓ i = q := by rw [← actAt_valid]; exact hq
  have hD := T.sum_masked_stateSum ℓ hψ w i (T.stateWeight ℓ q)
  have hN := T.sum_masked_stateSum ℓ hψ w i (T.stateWeightedValue ℓ q c)
  have hV := T.sum_masked_stateSum ℓ hψ w i (fun a => T.WV ℓ a c)
  have hs : (2 : ℝ) ^ s ≠ 0 := by positivity
  simp only [attention, hc]
  change (if (∑ j ∈ masked i, (T.stateWeight ℓ q (T.act (bos w) ℓ j)).val) = 0 then
      Fx.round p s ((∑ j ∈ masked i, (T.WV ℓ (T.act (bos w) ℓ j) c).val) /
        ((masked i).card : ℝ))
    else Fx.round p s
      ((∑ j ∈ masked i, (T.stateWeightedValue ℓ q c (T.act (bos w) ℓ j)).val) /
        ∑ j ∈ masked i, (T.stateWeight ℓ q (T.act (bos w) ℓ j)).val)) = _
  rw [hD, hV, hN, card_masked i]
  unfold countAttention
  by_cases hz : (T.stateSum ℓ ψ (T.stateWeight ℓ q)).val w i.val = 0
  · simp only [hz, Int.cast_zero, zero_div, ite_true]
    congr 1
    ring
  · have hzr : ((T.stateSum ℓ ψ (T.stateWeight ℓ q)).val w i.val : ℝ) ≠ 0 := by
      exact_mod_cast hz
    simp only [hz, div_eq_zero_iff, hzr, hs, or_self, ite_false]
    congr 1
    field_simp

/-- The source-state and query-agreement hypotheses have witnesses (B.2). -/
example : ∃ T : RTfr (Option Bool) 2 0 1 0,
    T.StateDefinition 0 T.initialFormula ∧
      T.actAt [true] 0 1 = (fun _ => 0) := by
  refine ⟨{
    E := fun _ _ => 0
    WQ := fun _ _ => 0
    WK := fun _ _ => 0
    WV := fun _ _ => 0
    ff := fun _ _ => 0
    Wout := fun _ => 0 }, initialFormula_defines _, ?_⟩
  simp [actAt, act]

end Transformer.CRASP.RTfr
