module

public import Transformer.CausalRecall.Recurrent
public import Transformer.CRASP.QueryFreeRecall

/-!
# Aligned recall through query-free summaries

In no paper: the index argument of `Transformer.CRASP.QueryFreeRecall` (Zoology,
arXiv:2312.04927v1, Appendix `app:retnet-proof`, `cor: space-ar`, there for
RetNet's state) for every causal model of `Transformer.CausalRecall.Summary`,
and for answers right on only some keys.

A model with Boolean outputs accepts a word when its output at the last
position is `true` (`Model.Accepts`); accepting exactly when `b` holds is
outputting `b` there (`Model.accepts_iff_getLast?`). On the instance
`CRASP.indexInstance` of a set `S` of `m` keys and a query `q`, the first `n`
rows write `S` and do not depend on `q`, which the last row alone carries, and
`v` answers the query exactly when `S` holds `q`. A model that decides at the
last row whether `v` answers thus outputs `S q` from the summaries of the
first `n` rows (`Model.accepts_indexInstance_iff`). Over `c` tokens and
`n + 1 ≥ c` rows, deciding every consistent instance needs

    2^{c-1} ≤ |R|^L        (`Model.two_pow_le_of_recall`),

as `Transformer.CRASP.two_pow_le_of_queryFree` bounds a transformer whose
attention ignores the query by its `k (2d + 1)` integers. Over `m + 1` tokens
and for weights `y ≤ x`, the sets `S` on which a model decides `a` of the `m`
queries right number at most `|R|^L (x + y)^m / (x^a y^(m - a))`
(`Model.card_recall_mul_le`).
-/

@[expose] public section

namespace Transformer.CausalRecall

open Finset Zoology CRASP

/-- Whether a row before `i` binds the key `i` queries to `v` is decidable. -/
instance decidablePriorAnswer {n c : ℕ} (x : MQARInstance n c) (i : Fin n) (v : Fin c) :
    Decidable (PriorAnswer x i v) :=
  inferInstanceAs (Decidable (∃ j : Fin n, j < i ∧ x.key j = x.query i ∧ x.value j = v))

namespace Model

variable {X Z R : Type*}

/-- A model with Boolean outputs accepts a word when its output at the last
position is `true`. -/
def Accepts (T : Model X Z R Bool) (xs : List X) : Prop :=
  (T.outputs xs).getLast? = some true

instance decidableAccepts (T : Model X Z R Bool) (xs : List X) : Decidable (T.Accepts xs) :=
  inferInstanceAs (Decidable ((T.outputs xs).getLast? = some true))

/-- A model has `1 ≤ |R|^L`: a layer's summary of no states lies in `R`. -/
theorem one_le_card_pow [Fintype R] {Y : Type*} (T : Model X Z R Y) :
    1 ≤ Fintype.card R ^ T.layers.length := by
  cases h : T.layers with
  | nil => simp
  | cons f fs =>
    have : Nonempty R := ⟨f.read []⟩
    exact Nat.one_le_pow _ _ Fintype.card_pos

/-- **Deciding is outputting**: a model accepts `xs ++ [x]` exactly when `b`
holds if and only if its output at `x` is `b`. -/
theorem accepts_iff_getLast? (T : Model X Z R Bool) (xs : List X) (x : X) (b : Bool) :
    (T.Accepts (xs ++ [x]) ↔ b = true) ↔ (T.outputs (xs ++ [x])).getLast? = some b := by
  rw [Accepts, getLast?_outputs_append_singleton]
  cases T.reply xs.length (T.summary xs) x <;> cases b <;> simp

variable {m : ℕ}

/-- **Deciding the index instance is recalling `S q`**: a model decides at the
last row whether `v` answers exactly when it outputs `S q` after the first
`n ≥ m` rows, which do not depend on `q`. -/
theorem accepts_indexInstance_iff (T : Model (Fin (m + 1) × Fin (m + 1) × Fin (m + 1)) Z R Bool)
    (hm : 0 < m) (v : Fin (m + 1)) (S : Fin m → Bool) {n : ℕ} (hn : m ≤ n) (q : Fin m) :
    (T.Accepts (word (indexInstance hm v S n q)) ↔
        PriorAnswer (indexInstance hm v S n q) (Fin.last n) v) ↔
      (T.outputs (List.ofFn (fun j : Fin n => indexRow hm v S j) ++ [(0, 0, q.succ)])).getLast? =
        some (S q) := by
  rw [priorAnswer_indexInstance hm v S hn q, word_indexInstance, accepts_iff_getLast?]

/-- **A model of the class recalls over `c` tokens only if `2^{c-1} ≤ |R|^L`**:
if, on the consistent instances of `n + 1 ≥ c` rows, it accepts exactly those
whose last query `v` answers, then `2^{c-1} ≤ |R|^L` (Zoology's `cor: space-ar`,
there for RetNet, by its proof). -/
theorem two_pow_le_of_recall [Fintype R] {c n : ℕ} (T : Model (Fin c × Fin c × Fin c) Z R Bool)
    (v : Fin c) (hn : c ≤ n + 1)
    (hT : ∀ x : MQARInstance (n + 1) c, Consistent x →
      (T.Accepts (word x) ↔ PriorAnswer x (Fin.last n) v)) :
    2 ^ (c - 1) ≤ Fintype.card R ^ T.layers.length := by
  obtain _ | _ | m := c
  · simpa using T.one_le_card_pow
  · simpa using T.one_le_card_pow
  have hm : 0 < m + 1 := by omega
  have h := T.card_pow_le
    (fun S : Fin (m + 1) → Bool => List.ofFn fun j : Fin n => indexRow hm v S j)
    (fun q => (0, 0, q.succ)) (fun S => List.length_ofFn) fun S q =>
      (T.accepts_indexInstance_iff hm v S (by omega) q).mp
        (hT _ (consistent_indexInstance hm v S n q))
  simpa using h

/-- **A model of the class decides few sets on many queries**: over `m + 1`
tokens and for weights `y ≤ x`, the sets `S` on whose index instances of
`n + 1 > m` rows it decides `a` of the `m` queries right number at most
`|R|^L (x + y)^m / (x^a y^(m - a))`. -/
theorem card_recall_mul_le [Fintype R]
    (T : Model (Fin (m + 1) × Fin (m + 1) × Fin (m + 1)) Z R Bool)
    (hm : 0 < m) (v : Fin (m + 1)) {n : ℕ} (hn : m ≤ n) {x y : ℕ} (hyx : y ≤ x) (a : ℕ) :
    #{S : Fin m → Bool | a ≤ #{q | T.Accepts (word (indexInstance hm v S n q)) ↔
        PriorAnswer (indexInstance hm v S n q) (Fin.last n) v}} * (x ^ a * y ^ (m - a)) ≤
      Fintype.card R ^ T.layers.length * (x + y) ^ m := by
  have h := T.card_answered_mul_le
    (fun S : Fin m → Bool => List.ofFn fun j : Fin n => indexRow hm v S j)
    (fun q => (0, 0, q.succ)) (fun S => List.length_ofFn) hyx a
  simp only [Fintype.card_fin, Fintype.card_bool, Nat.add_one_sub_one, one_mul] at h
  simpa only [T.accepts_indexInstance_iff hm v _ hn] using h

end Model

/-- The hypotheses of `Model.two_pow_le_of_recall` are satisfiable: over two
tokens and two rows, a recurrent network keeps the first key and value and
the last query. -/
example : 2 ^ (2 - 1) ≤ Fintype.card (Option (Fin 2 × Fin 2) × Fin 2) ^
    (recurrent (X := Fin 2 × Fin 2 × Fin 2) (fun s r => (s.1.or (some (r.1, r.2.1)), r.2.2))
      ((none : Option (Fin 2 × Fin 2)), (0 : Fin 2))
      fun s => decide (s.1 = some (s.2, 0))).layers.length :=
  Model.two_pow_le_of_recall _ 0 le_rfl fun x _ => by
    have hw : word x = [(x.key 0, x.value 0, x.query 0)] ++ [(x.key 1, x.value 1, x.query 1)] := by
      simp [word, List.ofFn_succ]
    rw [hw, Model.Accepts, getLast?_outputs_recurrent]
    simp [PriorAnswer, Fin.last]

/-- The hypotheses of `Model.card_recall_mul_le` are satisfiable: one key, one
row, the weights `3` and `1`, and a network that never changes its state. -/
example : #{S : Fin 1 → Bool | 1 ≤ #{q | (recurrent (X := Fin 2 × Fin 2 × Fin 2)
      (fun s _ => s) false id).Accepts (word (indexInstance (by norm_num) 0 S 1 q)) ↔
        PriorAnswer (indexInstance (by norm_num) 0 S 1 q) (Fin.last 1) 0}} *
        (3 ^ 1 * 1 ^ (1 - 1)) ≤
      Fintype.card Bool ^
          (recurrent (X := Fin 2 × Fin 2 × Fin 2) (fun s _ => s) false id).layers.length *
        (3 + 1) ^ 1 :=
  Model.card_recall_mul_le _ (by norm_num) 0 le_rfl (by norm_num) 1

end Transformer.CausalRecall
