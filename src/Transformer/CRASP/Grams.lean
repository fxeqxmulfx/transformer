module

public import Mathlib.Data.Fin.Tuple.Basic
public import Transformer.CRASP.LogitTransformers

/-!
# Positional logits read the last tokens: the construction

In no paper. arXiv:2506.16055v3, Appendix F (`sec:alibi_pes`), rounds every rounded transformer
with ALiBi into `TL[◁#, Y]` (`thm:rtfr_to_TLCly`), where `Y` reads the previous position, and
compiles `TL[◁#]` into rounded transformers without positions (`thm:TLCl_to_rtfr`); it does not
compile `Y`. Here a logit of the positions computes it: ALiBi, KERPLE, either under SSMax, with or
without a division by `√d` or by `d` (`Transformer.CRASP.LTfr`). The first `n` layers bring to
every position of `⊲ · w` its *gram* (`window`): the tokens at it and at the `n` positions before
it, newest first, with `⊲` before the start. The next `k` layers run a rounded transformer `M`
with uniform attention over the letters `κ` of the grams (`model`). The maps `W_Q`, `W_K`, `W_V`
and `f` of `def:transformer` are arbitrary maps `𝔽^d → 𝔽^d`, as in `Transformer.CRASP.RTfr`.

A reading layer gives a key the dot product `b` when its gram differs from the gram of the query
and `0` when it is the same. `ReadsPrev L b` asks the logit `L` to send the query position and, if
its gram differs, the previous one to `0`, and every other position to at most `-1`. Then those
one or two positions get the weight `round(exp 0) = 1` and the others `0`. With the value `2` at
the coordinate of the gram, the average is the sum of the two one-hot grams in either case. The
layers of `M` give every key the dot product `-1`; `ReadsPrev` asks that the logits of each query
be then all negative, so that every weight rounds to `0` and attention falls back on the average
of all values, or all `0`, so that every weight is `1`. Either way attention averages, as the
uniform attention of `M` does.

`Transformer.CRASP.GramsLayer` states `ReadsPrev` and checks it for ALiBi,
`Transformer.CRASP.GramsRead` computes the reading layers, `Transformer.CRASP.GramsRun` the
running layers, `Transformer.CRASP.GramsCorrect` the transformer, and
`Transformer.CRASP.GramsLogits` checks `ReadsPrev` for ALiBi and KERPLE with a divisor, and for
SSMax.
-/

@[expose] public section

namespace Transformer.CRASP.Grams

universe u v
variable {σ : Type u} {Γ : Type v} {n p d k : ℕ}

/-- The gram of a position: the tokens at it and at the `n` positions before it, newest first.
`none` stands for `⊲` and for the positions before it. -/
abbrev Gram (σ : Type u) (n : ℕ) := Fin (n + 1) → Option σ

/-- The token at the position `j` of `⊲ · w`; `none` at `⊲` and off the word. -/
def tok (w : List σ) (j : ℕ) : Option σ := ((bos w)[j]?).join

@[simp] theorem tok_zero (w : List σ) : tok w 0 = none := by simp [tok, bos]

@[simp] theorem tok_succ (w : List σ) (j : ℕ) : tok w (j + 1) = w[j]? := by
  simp only [tok, bos, List.getElem?_cons_succ, List.getElem?_map]
  cases w[j]? <;> rfl

/-- After `ℓ` reading layers the position `j` holds the tokens at `j, j - 1, …, j - ℓ`. The
subtraction is truncated, so `⊲` stands before the start. -/
def window (w : List σ) (ℓ j : ℕ) : Gram σ n := fun m =>
  if m.val ≤ ℓ then tok w (j - m.val) else none

/-- A gram from the token of a position and the gram of the position before it. -/
def push (c c' : Gram σ n) : Gram σ n := fun m =>
  if h : m.val = 0 then c 0 else c' ⟨m.val - 1, by have := m.isLt; omega⟩

/-- One more layer pushes the token of the position onto the gram of the position before it. -/
theorem window_succ (w : List σ) (ℓ j : ℕ) :
    (window w (ℓ + 1) j : Gram σ n) = push (window w ℓ j) (window w ℓ (j - 1)) := by
  funext m
  by_cases hm : m.val = 0
  · have h0 : m = 0 := Fin.ext hm
    subst h0
    simp [window, push]
  · simp only [window, push, dite_eq_right hm]
    rw [show j - 1 - (m.val - 1) = j - m.val by omega]
    by_cases hl : m.val ≤ ℓ + 1
    · rw [ite_eq_left hl, ite_eq_left (show m.val - 1 ≤ ℓ by omega)]
    · rw [ite_eq_right hl, ite_eq_right (show ¬ m.val - 1 ≤ ℓ by omega)]

/-- The hypothesis of `window_succ` holds at the first position of a one-letter word. -/
example : (window [true] 1 1 : Gram Bool 1) = push (window [true] 0 1) (window [true] 0 0) :=
  window_succ [true] 0 1

/-- The letter of a gram: `none` at `⊲`, else `κ` of the gram. -/
def letter (κ : Gram σ n → Γ) (c : Gram σ n) : Option Γ := (c 0).map fun _ => κ c

/-- The word of the letters of the grams at the positions of `w`. -/
def grams (n : ℕ) (κ : Gram σ n → Γ) (w : List σ) : List Γ :=
  List.ofFn fun i : Fin w.length => κ (window w n (i.val + 1))

@[simp] theorem length_grams (κ : Gram σ n → Γ) (w : List σ) :
    (grams n κ w).length = w.length := List.length_ofFn

/-- The position `j` of `⊲ · grams n κ w` carries the letter of the gram of `j`. -/
theorem getElem_bos_grams (κ : Gram σ n → Γ) (w : List σ) (j : ℕ)
    (hj : j < (bos (grams n κ w)).length) :
    (bos (grams n κ w))[j] = letter κ (window w n j) := by
  cases j with
  | zero => simp [bos, letter, window]
  | succ j =>
      have hj' : j < w.length := by simpa using hj
      simp [bos, grams, letter, window, List.getElem?_eq_getElem hj']

example := getElem_bos_grams (n := 1) id [true] 1 (by simp)

/-- A gram with the property `P`, if there is one. -/
noncomputable def pick (P : Gram σ n → Prop) : Gram σ n := by
  classical exact if h : ∃ c, P c then h.choose else fun _ => none

/-- The only gram with the property is picked. -/
theorem pick_eq {P : Gram σ n → Prop} {c₀ : Gram σ n} (hP : ∀ c, P c ↔ c = c₀) :
    pick P = c₀ := by
  have h : ∃ c, P c := ⟨c₀, (hP c₀).2 rfl⟩
  classical
  simp only [pick, dite_eq_left h]
  exact (hP _).1 h.choose_spec

/-- The hypothesis of `pick_eq` holds for the property of being a given gram. -/
example (c₀ : Gram Bool 0) : pick (· = c₀) = c₀ := pick_eq fun _ => Iff.rfl

section Coordinates

variable [Fintype σ]

/-- The width: the `d` coordinates of `M`, then one per gram. -/
abbrev width (σ : Type u) [Fintype σ] (n d : ℕ) : ℕ := d + Fintype.card (Gram σ n)

/-- The grams, numbered. -/
noncomputable def idx : Gram σ n ≃ Fin (Fintype.card (Gram σ n)) := Fintype.equivFin _

/-- The vector with `x` on the first `d` coordinates and `y c` at the coordinate of each gram. -/
noncomputable def mk (x : Fin d → Fx p 0) (y : Gram σ n → Fx p 0) :
    Fin (width σ n d) → Fx p 0 :=
  Fin.append x (y ∘ idx.symm)

/-- The first `d` coordinates. -/
def lo (h : Fin (width σ n d) → Fx p 0) : Fin d → Fx p 0 := fun t => h (Fin.castAdd _ t)

/-- The coordinate of each gram. -/
noncomputable def hi (h : Fin (width σ n d) → Fx p 0) : Gram σ n → Fx p 0 :=
  fun c => h (Fin.natAdd d (idx c))

@[simp] theorem mk_castAdd (x : Fin d → Fx p 0) (y : Gram σ n → Fx p 0) (t : Fin d) :
    mk x y (Fin.castAdd _ t) = x t := Fin.append_left _ _ t

@[simp] theorem mk_natAdd (x : Fin d → Fx p 0) (y : Gram σ n → Fx p 0) (c : Gram σ n) :
    mk x y (Fin.natAdd d (idx c)) = y c := by
  simp [mk]

@[simp] theorem lo_mk (x : Fin d → Fx p 0) (y : Gram σ n → Fx p 0) : lo (mk x y) = x :=
  funext fun t => mk_castAdd x y t

@[simp] theorem hi_mk (x : Fin d → Fx p 0) (y : Gram σ n → Fx p 0) : hi (mk x y) = y :=
  funext fun c => mk_natAdd x y c

/-- A sum over the coordinates: those of `M`, then those of the grams. -/
theorem sum_width (f : Fin (width σ n d) → ℝ) :
    ∑ t, f t = ∑ t : Fin d, f (Fin.castAdd _ t) + ∑ c : Gram σ n, f (Fin.natAdd d (idx c)) := by
  rw [Fin.sum_univ_add, Equiv.sum_comp idx (fun i => f (Fin.natAdd d i))]

variable [DecidableEq σ]

/-- `1` at the coordinate of `c`, `0` at the others. -/
noncomputable def onehot (c : Gram σ n) : Gram σ n → Fx p 0 :=
  fun c' => Fx.round p 0 (if c' = c then 1 else 0)

/-- The state of a position whose gram is `c`: the embedding of its letter by `M`, and `c`
one-hot. -/
noncomputable def enc (M : RTfr (Option Γ) p 0 d k) (κ : Gram σ n → Γ) (c : Gram σ n) :
    Fin (width σ n d) → Fx p 0 :=
  mk (M.E (letter κ c)) (onehot c)

/-- **The rounded transformer with the logit `L`.** The layers `ℓ < n` read the gram of the
previous position, with the dot product `b` between differing grams; the layers `n + ℓ` run the
layer `ℓ` of `M` on the first `d` coordinates, with every dot product `-1`. -/
noncomputable def model (L : ℕ → ℕ → ℝ → ℝ) (b : ℕ) (M : RTfr (Option Γ) p 0 d k)
    (κ : Gram σ n → Γ) : LTfr (Option σ) p 0 (width σ n d) (n + k) where
  E a := enc M κ fun m => if m.val = 0 then a else none
  WQ ℓ h := if ℓ < n then mk 0 fun c => Fx.round p 0 (b * (1 - (hi h c).val))
    else mk 0 (onehot fun _ => none)
  WK ℓ h := if ℓ < n then mk 0 (hi h)
    else mk 0 fun c => Fx.round p 0 (-(onehot (p := p) (fun _ => none) c).val)
  WV ℓ h := if ℓ < n then mk 0 fun c => Fx.round p 0 (2 * (hi h c).val)
    else mk (M.WV (ℓ - n) (lo h)) 0
  ff ℓ x := if ℓ < n then
      enc M κ (push (pick fun c => 2 ≤ (hi x c).val)
        (pick fun c => (hi x c).val = 1 ∨ (hi x c).val = 3))
    else mk (M.ff (ℓ - n) (lo x)) 0
  Wout h := M.Wout (lo h)
  logit := L

end Coordinates

end Transformer.CRASP.Grams
