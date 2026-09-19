/-
# A layer above a collapse computes one vector

If every position of an input carries the same activation `v`, then a layer of
a future-masked rounded transformer (`Transformer.CRASP.RTfr`) gives every
position the same activation again, and that common value is a function of `v`
and of the layer index alone — it does not depend on the position, nor on the
length of the input.

The reason is the one `eq:att` makes visible: with all keys, queries and
values equal, every score `s_{ij}` is the same number `S`, so the attention
average of `m` identical rounded terms over `m` identical rounded weights is
that one term over that one weight, whatever `m = |{j ≤ i}|` is.  The
`den = 0` branch, which averages the values themselves, collapses the same
way.

`constLayer` is that common value, `above` iterates it, and
`act_add_of_const` runs the iteration along the layers above a collapse.
`act_eq_of_fields` records that the activations read only the weights, not the
declared depth, which is what lets the layers below a collapse be reused in a
shallower transformer.

Source: arXiv:2506.16055v3, Appendix B.1, `def:transformer`, `eq:att`.
-/

import Transformer.CRASP.Transformers

namespace Transformer
namespace CRASP

universe u

variable {σ : Type u}

namespace RTfr

variable {p s d k : ℕ}

/-- The activation layer `ℓ` gives every position when every position carries
the same activation `v`. -/
noncomputable def constLayer (T : RTfr σ p s d k) (ℓ : ℕ) (v : Fin d → Fx p s) :
    Fin d → Fx p s :=
  let score : ℝ := ∑ c : Fin d, (T.WQ ℓ v c).val * (T.WK ℓ v c).val
  let den : ℝ := (Fx.round p s (Real.exp score)).val
  let att : Fin d → Fx p s := fun c =>
    if den = 0 then Fx.round p s ((T.WV ℓ v c).val)
    else Fx.round p s
      ((Fx.round p s (Real.exp score * (T.WV ℓ v c).val)).val / den)
  T.ff ℓ fun c => Fx.add (att c) (v c)

/-- **A layer turns a constant activation into a constant activation.**  The
value is `constLayer`, so it depends neither on the position nor on the length
of the input. -/
theorem layer_of_const (T : RTfr σ p s d k) (ℓ : ℕ) {n : ℕ}
    {h : Fin n → Fin d → Fx p s} {v : Fin d → Fx p s} (hv : ∀ i, h i = v)
    (i : Fin n) :
    T.layer ℓ h i = T.constLayer ℓ v := by
  classical
  have hcard : 0 < (masked i).card :=
    Finset.card_pos.mpr ⟨i, self_mem_masked i⟩
  have hcardR : ((masked i).card : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr hcard.ne'
  simp only [layer, constLayer, hv]
  congr 1
  funext c
  congr 1
  rw [Finset.sum_const, nsmul_eq_mul, Finset.sum_const, nsmul_eq_mul,
    Finset.sum_const, nsmul_eq_mul]
  simp only [mul_eq_zero, hcardR, false_or]
  split_ifs with hz
  · rw [mul_div_cancel_left₀ _ hcardR]
  · rw [mul_div_mul_left _ _ hcardR]

/-- The activation `m` layers above a constant one `v`, starting at layer
`L`. -/
noncomputable def above (T : RTfr σ p s d k) (L : ℕ) :
    ℕ → (Fin d → Fx p s) → (Fin d → Fx p s)
  | 0, v => v
  | m + 1, v => T.constLayer (L + m) (T.above L m v)

/-- **Above a constant layer, every activation is `above`.**  If layer `L` is
constant with value `v` on `w`, then layer `L + m` is constant with value
`above L m v`. -/
theorem act_add_of_const (T : RTfr σ p s d k) (w : List σ) (L : ℕ)
    {v : Fin d → Fx p s} (hv : ∀ i, T.act w L i = v) (m : ℕ)
    (i : Fin w.length) :
    T.act w (L + m) i = T.above L m v := by
  induction m generalizing i with
  | zero => exact hv i
  | succ m ih =>
      have hstep : T.act w (L + m + 1) = T.layer (L + m) (T.act w (L + m)) := rfl
      have : L + (m + 1) = L + m + 1 := rfl
      rw [this, hstep]
      exact layer_of_const T (L + m) (fun j => ih j) i

/-- **A layer reads only the weights.**  Two transformers sharing the
projections and the feed-forward networks run the same layer, whatever depths
they are declared with. -/
theorem layer_eq_of_fields {k' : ℕ} (T : RTfr σ p s d k) (T' : RTfr σ p s d k')
    (hQ : T.WQ = T'.WQ) (hK : T.WK = T'.WK) (hV : T.WV = T'.WV)
    (hf : T.ff = T'.ff) (ℓ : ℕ) {n : ℕ} (h : Fin n → Fin d → Fx p s) :
    T.layer ℓ h = T'.layer ℓ h := by
  funext i
  simp only [layer, hQ, hK, hV, hf]

/-- **The activations read only the weights**, not the declared depth: the
layers below a collapse can be reused in a shallower transformer. -/
theorem act_eq_of_fields {k' : ℕ} (T : RTfr σ p s d k) (T' : RTfr σ p s d k')
    (hE : T.E = T'.E) (hQ : T.WQ = T'.WQ) (hK : T.WK = T'.WK) (hV : T.WV = T'.WV)
    (hf : T.ff = T'.ff) (w : List σ) (ℓ : ℕ) :
    T.act w ℓ = T'.act w ℓ := by
  induction ℓ with
  | zero =>
      have h0 : T.act w 0 = fun i c => T.E w[i] c := rfl
      have h0' : T'.act w 0 = fun i c => T'.E w[i] c := rfl
      rw [h0, h0', hE]
  | succ ℓ ih =>
      have h1 : T.act w (ℓ + 1) = T.layer ℓ (T.act w ℓ) := rfl
      have h2 : T'.act w (ℓ + 1) = T'.layer ℓ (T'.act w ℓ) := rfl
      rw [h1, h2, ih, layer_eq_of_fields T T' hQ hK hV hf]

/-- **The collapsed transformer.**  Runs the first `L` layers of `T` and reads
its output off the common activation of layer `L`, carried `M` layers up by
`above`.  It has depth `L`, and `out_collapse` says it computes what `T`
computes whenever layer `L` of `T` is constant. -/
noncomputable def collapse {L M : ℕ} (T : RTfr σ p s d (L + M)) :
    RTfr σ p s d L where
  E := T.E
  WQ := T.WQ
  WK := T.WK
  WV := T.WV
  ff := T.ff
  Wout := fun v => T.Wout (T.above L M v)

/-- **The collapsed transformer computes what the deep one computes**, on
every input whose layer-`L` activations agree across positions. -/
theorem out_collapse {L M : ℕ} (T : RTfr σ p s d (L + M)) (w : List σ)
    (hconst : ∀ i j : Fin w.length, T.act w L i = T.act w L j) :
    (T.collapse : RTfr σ p s d L).out w = T.out w := by
  rw [out, out]
  split_ifs with hlen
  · have hact : (T.collapse : RTfr σ p s d L).act w L = T.act w L :=
      (act_eq_of_fields T (T.collapse : RTfr σ p s d L) rfl rfl rfl rfl rfl w L).symm
    have hup := act_add_of_const T w L
      (fun i => hconst i ⟨w.length - 1, by omega⟩) M ⟨w.length - 1, by omega⟩
    show T.Wout (T.above L M
        ((T.collapse : RTfr σ p s d L).act w L ⟨w.length - 1, by omega⟩))
      = T.Wout (T.act w (L + M) ⟨w.length - 1, by omega⟩)
    rw [hact, hup]
  · rfl

end RTfr

end CRASP
end Transformer
