/-
# Masked attention sums from finite state predicates

arXiv:2506.16055v3, Appendix B.2, the weighted numerator and
denominator in `thm:rtfr_to_TLCl`. The BOS contribution is a constant;
the remaining contributions are past counts of activation states.
-/

import Transformer.CRASP.TransformerStates
import Transformer.CRASP.StateCounts

namespace Transformer.CRASP.RTfr

universe u
variable {σ : Type u} [DecidableEq σ] {p s d k : ℕ}

/-- The affine count representing a sum of fixed-precision values (B.2). -/
noncomputable def stateSum (T : RTfr (Option σ) p s d k) (ℓ : ℕ)
    (ψ : (Fin d → Fx p s) → Form σ) (f : (Fin d → Fx p s) → Fx p s) :
    LinearCount σ := LinearCount.states ψ (T.actAt [] ℓ 0) fun q => (f q).m

omit [DecidableEq σ] in
/-- A state sum inherits the bound of the state formulas (Appendix B.2). -/
theorem stateSum_good (T : RTfr (Option σ) p s d k) (ℓ j : ℕ)
    {ψ : (Fin d → Fx p s) → Form σ} (hψ : ∀ q, ψ q ∈ TLCl σ j)
    (f : (Fin d → Fx p s) → Fx p s) : (T.stateSum ℓ ψ f).Good j :=
  LinearCount.good_states hψ _ _

/-- Summing fixed-precision values agrees with the weighted state count.
Source: arXiv:2506.16055v3, Appendix B.2, the terms `A` and `B`. -/
theorem sum_masked_stateSum (T : RTfr (Option σ) p s d k) (ℓ : ℕ)
    {ψ : (Fin d → Fx p s) → Form σ} (hψ : T.StateDefinition ℓ ψ)
    (w : List σ) (i : Fin (bos w).length) (f : (Fin d → Fx p s) → Fx p s) :
    (∑ j ∈ masked i, (f (T.act (bos w) ℓ j)).val) =
      ((T.stateSum ℓ ψ f).val w i.val : ℝ) / 2 ^ s := by
  classical
  rw [T.sum_masked_actAt w ℓ i (fun a => (f a).val)]
  have hs := LinearCount.val_states (ψ := ψ) (T.actAt [] ℓ 0) (fun q => (f q).m)
    w i.val (T.actAt w ℓ) (by
      intro q j hj
      have hh := Finset.mem_Icc.mp hj
      apply hψ w j
      exact ⟨by have := i.isLt; simp only [length_bos] at this; omega, Or.inl (by omega)⟩)
  change (f (T.actAt [] ℓ 0)).val + ∑ j ∈ Finset.Icc 1 i.val, (f (T.actAt w ℓ j)).val =
    ((LinearCount.states ψ (T.actAt [] ℓ 0) (fun q => (f q).m)).val w i.val : ℝ) / 2 ^ s
  rw [hs]
  simp only [Fx.val]
  push_cast
  rw [add_div, Finset.sum_div]

omit [DecidableEq σ] in
/-- A masked prefix contains its query and all preceding positions (B.1). -/
theorem card_masked {n : ℕ} (i : Fin n) : (masked i).card = i.val + 1 := by
  have hn : 0 < n := lt_of_le_of_lt (Nat.zero_le _) i.isLt
  have hm : masked i = Finset.Icc (⟨0, hn⟩ : Fin n) i := by
    ext j
    simp only [masked, Finset.mem_filter, Finset.mem_univ, true_and,
      Finset.mem_Icc, Fin.le_def]
    omega
  rw [hm, Fin.card_Icc]
  simp

/-- Constant-zero state predicates meet the semantic hypothesis (B.2). -/
example : ∃ T : RTfr (Option Bool) 2 0 1 0,
    T.StateDefinition 0 T.initialFormula :=
  ⟨{ E := fun _ _ => 0, WQ := fun _ _ => 0, WK := fun _ _ => 0,
     WV := fun _ _ => 0, ff := fun _ _ => 0, Wout := fun _ => 0 },
    initialFormula_defines _⟩

end Transformer.CRASP.RTfr
