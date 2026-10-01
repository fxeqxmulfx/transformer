/-
# Finite attention tables for periodic position encodings

arXiv:2506.16055v3, Appendix F, `thm:rtfr_to_TLClmod`.
The source state includes its activation vector and the residue of its
position. The rounded numerator and denominator are functions on this
finite set, even when the unrounded rotation entries are irrational.
-/

import Transformer.CRASP.PositionalPrefix
import Transformer.CRASP.PositionalStateInduction
import Transformer.CRASP.PrefixSums

namespace Transformer.CRASP.PeriodicAttention

universe u
variable {σ : Type u} {p s d k M : ℕ}

/-- Activation states decorated by a position residue (Appendix F). -/
abbrev State (p s d M : ℕ) := Fin M × (Fin d → Fx p s)

/-- The finite residue of a natural position (Appendix F). -/
def phase (M : ℕ) (hM : 0 < M) (i : ℕ) : Fin M := ⟨i % M, Nat.mod_lt _ hM⟩

/-- A decorated activation at a natural input position (Appendix F). -/
noncomputable def state (T : PTfr (Option σ) p s d k) (hM : 0 < M)
    (ℓ : ℕ) (w : List σ) (i : ℕ) : State p s d M :=
  (phase M hM i, T.actAt w ℓ i)

/-- The decorated BOS state is independent of the ordinary input (F/B.2). -/
theorem state_bos (T : PTfr (Option σ) p s d k) (hM : 0 < M) (ℓ : ℕ) (w : List σ) :
    state T hM ℓ w 0 = state T hM ℓ [] 0 := by
  simp [state, PTfr.actAt_bos]

/-- A denominator summand indexed by finite query/source states (Appendix F). -/
noncomputable def weight (T : PTfr (Option σ) p s d k) (ℓ : ℕ)
    (q a : State p s d M) : Fx p s :=
  Fx.round p s (Real.exp (T.pe.logit q.1.val a.1.val (T.WQ ℓ q.2) (T.WK ℓ a.2)))

/-- A rounded numerator summand indexed by finite states (Appendix F). -/
noncomputable def numerator (T : PTfr (Option σ) p s d k) (ℓ : ℕ)
    (q : State p s d M) (c : Fin d) (a : State p s d M) : Fx p s :=
  Fx.round p s (Real.exp (T.pe.logit q.1.val a.1.val (T.WQ ℓ q.2) (T.WK ℓ a.2)) *
    (T.WV ℓ a.2 c).val)

/-- The unweighted source value for the denominator-zero fallback (F/B.1). -/
noncomputable def value (T : PTfr (Option σ) p s d k) (ℓ : ℕ)
    (c : Fin d) (a : State p s d M) : Fx p s := T.WV ℓ a.2 c

/-- The finite residual/feed-forward update retains the query residue (F). -/
noncomputable def update (T : PTfr (Option σ) p s d k) (ℓ : ℕ)
    (q : State p s d M) (a : Fin d → Fx p s) : State p s d M :=
  (q.1, T.ff ℓ (fun c => Fx.add (a c) (q.2 c)))

/-- All rounded exponential weights are nonnegative (Appendix F/B.1). -/
theorem weight_nonneg (T : PTfr (Option σ) p s d k) (ℓ : ℕ)
    (q a : State p s d M) : 0 ≤ (weight T ℓ q a).m :=
  Fx.m_round_nonneg _ (Real.exp_pos _).le

/-- The positional attention operation before residual addition (F/B.1). -/
noncomputable def attention (T : PTfr (Option σ) p s d k) (ℓ : ℕ) {n : ℕ}
    (h : Fin n → Fin d → Fx p s) (i : Fin n) (c : Fin d) : Fx p s :=
  let score := fun j => T.pe.logit i.val j.val (T.WQ ℓ (h i)) (T.WK ℓ (h j))
  let D := ∑ j ∈ RTfr.masked i, (Fx.round p s (Real.exp (score j))).val
  if D = 0 then Fx.round p s
    ((∑ j ∈ RTfr.masked i, (T.WV ℓ (h j) c).val) / ((RTfr.masked i).card : ℝ))
  else Fx.round p s ((∑ j ∈ RTfr.masked i,
    (Fx.round p s (Real.exp (score j) * (T.WV ℓ (h j) c).val)).val) / D)

/-- The layer is attention followed by its original residual and FFN (F/B.1). -/
theorem layer_eq_attention (T : PTfr (Option σ) p s d k) (ℓ : ℕ) {n : ℕ}
    (h : Fin n → Fin d → Fx p s) (i : Fin n) :
    T.layer ℓ h i = T.ff ℓ (fun c => Fx.add (attention T ℓ h i c) (h i c)) := rfl

/-- A masked table sum is its integer count sum in fixed-point units (F/B.2). -/
theorem sum_states (T : PTfr (Option σ) p s d k) (hM : 0 < M) (ℓ : ℕ)
    (w : List σ) (i : Fin (bos w).length) (f : State p s d M → Fx p s) :
    (∑ j ∈ RTfr.masked i, (f (phase M hM j.val, T.act (bos w) ℓ j)).val) =
      (CountCells.sum (state T hM ℓ [] 0) f (state T hM ℓ w) i.val : ℝ) / 2 ^ s := by
  have hs : (∑ j ∈ RTfr.masked i, (f (phase M hM j.val, T.act (bos w) ℓ j)).val) =
      ∑ j ∈ RTfr.masked i, (f (state T hM ℓ w j.val)).val := by
    apply Finset.sum_congr rfl
    intro j hj
    rw [state, PTfr.actAt_valid]
  rw [hs, RTfr.sum_masked_nat i (fun j => (f (state T hM ℓ w j)).val), state_bos]
  simp only [Fx.val, CountCells.sum, Int.cast_add, Int.cast_sum]
  rw [← Finset.sum_div]
  ring

/-- The positive period used by these finite tables has a witness (Appendix F). -/
example : 0 < 1 := one_pos

end Transformer.CRASP.PeriodicAttention
