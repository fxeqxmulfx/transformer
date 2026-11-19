module

public import Transformer.CRASP.GramsRead
public import Transformer.CRASP.AttentionKernel
public import Transformer.CRASP.QueryFree

/-!
# Positional logits read the last tokens: the running layers

In no paper; the construction is `Transformer.CRASP.Grams`, for the rounded transformers of
arXiv:2506.16055v3 (Appendix B.1, `def:transformer`, Equation `eq:att`), with a logit `L` of the
positions and the dot product (`Transformer.CRASP.LTfr`).

A running layer gives every pair the dot product `-1` (`dot_run`). `ReadsPrev L b` sends the logits
of each query all below `0`, so that every weight `round(exp z)` is `0` and attention falls back on
the average of the values, as `def:transformer` prescribes when the denominator vanishes, or all to
`0`, so that every weight is `1` and attention is the same average (`attention_run`). When `M`
scores every pair `0`, its weights are `round(exp 0) = 1` and its attention is that average too.
So on the first `d` coordinates the running layer is the layer of `M` (`lo_layer_run`).
-/

@[expose] public section

namespace Transformer.CRASP.Grams

universe u v
variable {σ : Type u} {Γ : Type v} {n p d k : ℕ} [DecidableEq σ] [Fintype σ]
  (L : ℕ → ℕ → ℝ → ℝ) (b : ℕ) (M : RTfr (Option Γ) p 0 d k) (κ : Gram σ n → Γ)

/-- The dot product of a running layer is `-1`. -/
theorem dot_run (hp : 3 ≤ p) {ℓ : ℕ} (hℓ : n ≤ ℓ) (x y : Fin (width σ n d) → Fx p 0) :
    ∑ t, ((model L b M κ).WQ ℓ x t).val * ((model L b M κ).WK ℓ y t).val = -1 := by
  have hq : (model L b M κ).WQ ℓ x = mk 0 (onehot fun _ => none) := by
    simp [model, Nat.not_lt.mpr hℓ]
  have hk : (model L b M κ).WK ℓ y =
      mk 0 fun c => Fx.round p 0 (-(onehot (p := p) (fun _ => none) c).val) := by
    simp [model, Nat.not_lt.mpr hℓ]
  have h1 : ∀ c : Gram σ n, (onehot (p := p) (fun _ => none) c).val *
      (Fx.round p 0 (-(onehot (p := p) (fun _ => none) c).val)).val =
        if c = fun _ => none then -1 else 0 := fun c => by
    rw [val_onehot hp]
    split_ifs
    · rw [one_mul]; exact val_round_small hp (by norm_num) (by norm_num)
    · rw [zero_mul]
  rw [hq, hk, sum_width]
  simp only [mk_castAdd, mk_natAdd, Pi.zero_apply, Fx.val_zero, mul_zero,
    Finset.sum_const_zero, zero_add, h1, Finset.sum_ite_eq', Finset.mem_univ, ite_true]

/-- The hypotheses of `dot_run` hold at precision `3` in the layer after one reading layer. -/
example (M : RTfr (Option (Gram Bool 1)) 3 0 0 0) (x y : Fin (width Bool 1 0) → Fx 3 0) :=
  dot_run (alibiLogit (1 : ℕ)) 1 M id le_rfl (ℓ := 1) le_rfl x y

/-- **A running layer averages the values**: the weights of a query are all `0`, and attention
falls back on the average, or all `1`. -/
theorem attention_run (hp : 3 ≤ p) (hL : ReadsPrev L b) {ℓ : ℕ} (hℓ : n ≤ ℓ) {N : ℕ}
    (h : Fin N → Fin (width σ n d) → Fx p 0) (i : Fin N) (t : Fin (width σ n d)) :
    (model L b M κ).attention ℓ h i t = Fx.round p 0
      ((∑ j ∈ RTfr.masked i, ((model L b M κ).WV ℓ (h j) t).val) /
        ((RTfr.masked i).card : ℝ)) := by
  simp only [LTfr.attention, dot_run L b M κ hp hℓ]
  rcases hL.run i.val with hneg | hzero
  · have hD : ∑ j ∈ RTfr.masked i,
        (Fx.round p 0 (Real.exp ((model L b M κ).logit i.val j.val (-1)))).val = 0 := by
      refine Finset.sum_eq_zero fun j hj => ?_
      have hle : j ≤ i := by simpa [RTfr.masked] using hj
      exact val_round_exp_neg (hneg j.val (Fin.le_def.mp hle))
    rw [hD, ite_eq_left rfl]
  · have hz : ∀ j ∈ RTfr.masked i, (model L b M κ).logit i.val j.val (-1) = 0 := fun j hj => by
      have hle : j ≤ i := by simpa [RTfr.masked] using hj
      exact hzero j.val (Fin.le_def.mp hle)
    have h1 : (Fx.round p 0 1).val = 1 := val_round_small hp rfl (by norm_num)
    have hcard : ((RTfr.masked i).card : ℝ) ≠ 0 := by
      exact_mod_cast (Finset.card_pos.mpr ⟨i, RTfr.self_mem_masked i⟩).ne'
    have hD : ∑ j ∈ RTfr.masked i,
        (Fx.round p 0 (Real.exp ((model L b M κ).logit i.val j.val (-1)))).val =
          (RTfr.masked i).card := by
      rw [Finset.sum_congr rfl fun j hj => by rw [hz j hj, Real.exp_zero, h1],
        Finset.sum_const, nsmul_eq_mul, mul_one]
    have hN : ∑ j ∈ RTfr.masked i, (Fx.round p 0 (Real.exp ((model L b M κ).logit i.val j.val
        (-1)) * ((model L b M κ).WV ℓ (h j) t).val)).val =
          ∑ j ∈ RTfr.masked i, ((model L b M κ).WV ℓ (h j) t).val :=
      Finset.sum_congr rfl fun j hj => by rw [hz j hj, Real.exp_zero, one_mul, Fx.round_val]
    rw [hD, hN, ite_eq_right hcard]

example (M : RTfr (Option (Gram Bool 1)) 3 0 0 0) (h : Fin 2 → Fin (width Bool 1 0) → Fx 3 0)
    (i : Fin 2) (t : Fin (width Bool 1 0)) :=
  attention_run (alibiLogit (1 : ℕ)) 1 M id le_rfl (readsPrev_alibi le_rfl) (ℓ := 1) le_rfl
    h i t

/-- The attention of `M` when it scores every pair `0`: the average of the values. -/
theorem attention_uniform (hp : 3 ≤ p) (hM : ∀ ℓ x y, M.score ℓ x y = 0) (ℓ : ℕ) {N : ℕ}
    (g : Fin N → Fin d → Fx p 0) (i : Fin N) (t : Fin d) :
    M.attention ℓ g i t = Fx.round p 0
      ((∑ j ∈ RTfr.masked i, (M.WV ℓ (g j) t).val) / ((RTfr.masked i).card : ℝ)) := by
  have hM' : ∀ ℓ x y, ∑ c, (M.WQ ℓ x c).val * (M.WK ℓ y c).val = 0 := hM
  have hcard : ((RTfr.masked i).card : ℝ) ≠ 0 := by
    exact_mod_cast (Finset.card_pos.mpr ⟨i, RTfr.self_mem_masked i⟩).ne'
  have h1 : (Fx.round p 0 1).val = 1 := val_round_small hp rfl (by norm_num)
  simp only [RTfr.attention, hM', Real.exp_zero, h1, one_mul, Fx.round_val, Finset.sum_const,
    nsmul_eq_mul, mul_one, hcard, ite_false]

/-- The hypotheses of `attention_uniform` hold for the transformer of `thm:TLCl_to_rtfr` of a
formula of capacity `1`. -/
example (g : Fin 2 → Fin _ → Fx _ 0) (i : Fin 2) (t : Fin _) :=
  attention_uniform (TemporalProgram.model (Form.lt .one (.add .one .one) : Form Bool) 0)
    (by decide) (fun ℓ x y => by simp [RTfr.score, TemporalProgram.model]) 0 g i t

/-- **A running layer runs `M`**: on the first `d` coordinates it is the layer `ℓ - n` of `M`,
when `M` scores every pair `0`. -/
theorem lo_layer_run (hp : 3 ≤ p) (hL : ReadsPrev L b) (hM : ∀ ℓ x y, M.score ℓ x y = 0)
    {ℓ : ℕ} (hℓ : n ≤ ℓ) {N : ℕ} (h : Fin N → Fin (width σ n d) → Fx p 0) (i : Fin N) :
    lo ((model L b M κ).layer ℓ h i) = M.layer (ℓ - n) (fun j => lo (h j)) i := by
  have hff : ∀ x, (model L b M κ).ff ℓ x = mk (M.ff (ℓ - n) (lo x)) 0 := fun x => by
    simp [model, Nat.not_lt.mpr hℓ]
  have hWV : ∀ j (t : Fin d), (model L b M κ).WV ℓ (h j) (Fin.castAdd _ t) =
      M.WV (ℓ - n) (lo (h j)) t := fun j t => by
    simp [model, Nat.not_lt.mpr hℓ]
  have hatt : ∀ t : Fin d, (model L b M κ).attention ℓ h i (Fin.castAdd _ t) =
      M.attention (ℓ - n) (fun j => lo (h j)) i t := fun t => by
    rw [attention_run L b M κ hp hL hℓ, attention_uniform M hp hM]
    simp only [hWV]
  simp only [LTfr.layer]
  rw [RTfr.layer_eq_attention, hff, lo_mk]
  congr 1
  funext t
  exact congrArg (fun a => Fx.add a (h i (Fin.castAdd _ t))) (hatt t)

/-- The hypotheses of `lo_layer_run` hold for ALiBi of slope `1` and the transformer of
`thm:TLCl_to_rtfr` of a formula of capacity `1`, after one reading layer. -/
example (h : Fin 2 → Fin (width Bool 0 _) → Fx _ 0) (i : Fin 2) :
    lo ((model (alibiLogit (1 : ℕ)) 1
      (TemporalProgram.model (Form.lt .one (.add .one .one) : Form Bool) 0)
      (fun c : Gram Bool 0 => decide (c 0 = some true))).layer 0 h i) =
      (TemporalProgram.model (Form.lt .one (.add .one .one) : Form Bool) 0).layer 0
        (fun j => lo (h j)) i :=
  lo_layer_run _ 1 _ _ (by decide) (readsPrev_alibi le_rfl)
    (fun ℓ x y => by simp [RTfr.score, TemporalProgram.model]) le_rfl h i

end Transformer.CRASP.Grams
