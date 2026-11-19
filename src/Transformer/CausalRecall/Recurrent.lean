module

public import Transformer.CausalRecall.Summary

/-!
# A recurrent network reads the past through its state

In no paper. A recurrent network over a set `S` of states moves its state with
each token and reads its output off the state. It is a model of
`Transformer.CausalRecall.Summary` with one layer, whose summary of the
positions before is the state they reach (`recurrent`,
`getLast?_outputs_recurrent`). Over a finite `S` it therefore recalls a
dictionary only if `|V|^|K| ≤ |S|` (`card_pow_le_of_recurrent`): Zoology's
`cor: space-ar` (arXiv:2312.04927v1, Appendix `app:retnet-proof`) for every
finite-state recurrence and exact recall, where the paper argues it for
RetNet's state.
-/

@[expose] public section

namespace Transformer.CausalRecall

open Finset

variable {X Y : Type*}

/-- A recurrent network over the states `S`: from `s₀`, the token `x` moves the
state `s` to `step s x`, and `o` reads the output off the state. As a model, one
layer whose summary of the positions before is the state they reach. -/
def recurrent {S : Type*} (step : S → X → S) (s₀ : S) (o : S → Y) : Model X (X × S) S Y where
  embed _ x := (x, s₀)
  layers := [⟨fun zs => (zs.map Prod.fst).foldl step s₀, fun _ z s => (z.1, step s z.1)⟩]
  out z := o z.2

theorem map_fst_mapIdx {S : Type*} (s₀ : S) (xs : List X) :
    (xs.mapIdx fun _ x => (x, s₀)).map Prod.fst = xs := by
  induction xs with
  | nil => rfl
  | cons y ys ih => simp [List.mapIdx_cons, ih]

/-- **The model `recurrent` is the recurrent network**: its output at the last
token is `o` of the state that the tokens reach. -/
theorem getLast?_outputs_recurrent {S : Type*} (step : S → X → S) (s₀ : S) (o : S → Y)
    (xs : List X) (x : X) :
    ((recurrent step s₀ o).outputs (xs ++ [x])).getLast? =
      some (o ((xs ++ [x]).foldl step s₀)) := by
  rw [Model.getLast?_outputs_append_singleton]
  simp [recurrent, Model.reply, Model.summary, summaries, finish, map_fst_mapIdx]

variable {K V : Type*} [Fintype K] [Fintype V]

/-- **A recurrent network recalls a dictionary only if it has `|V|^|K|`
states**: if `o` of the state reached after each context `ctx D` of `n` tokens
and the token `qry q` is `D q`, then `|V|^|K| ≤ |S|`. -/
theorem card_pow_le_of_recurrent {S : Type*} [Fintype S] (step : S → X → S) (s₀ : S)
    (o : S → V) (ctx : (K → V) → List X) (qry : K → X) {n : ℕ}
    (hlen : ∀ D, (ctx D).length = n) (h : ∀ D q, o ((ctx D ++ [qry q]).foldl step s₀) = D q) :
    Fintype.card V ^ Fintype.card K ≤ Fintype.card S := by
  simpa [recurrent] using (recurrent step s₀ o).card_pow_le ctx qry hlen fun D q => by
    rw [getLast?_outputs_recurrent, h]

/-- The hypotheses of `card_pow_le_of_recurrent` are satisfiable: two states
keep the bit of the first token, and the query token `none` leaves it. -/
example : Fintype.card Bool ^ Fintype.card (Fin 1) ≤ Fintype.card Bool :=
  card_pow_le_of_recurrent (X := Option Bool) (fun s x => x.getD s) false id
    (fun D => [some (D 0)]) (fun _ => none) (n := 1) (fun _ => rfl)
    fun D q => by fin_cases q; rfl

/-- The hypotheses of `Model.card_pow_le` are satisfiable, by the same network. -/
example : Fintype.card Bool ^ Fintype.card (Fin 1) ≤
    Fintype.card Bool ^
      (recurrent (X := Option Bool) (fun s x => x.getD s) false id).layers.length :=
  Model.card_pow_le _ (fun D : Fin 1 → Bool => [some (D 0)]) (fun _ => none) (n := 1)
    (fun _ => rfl) fun D q => by rw [getLast?_outputs_recurrent]; fin_cases q; rfl

/-- The hypotheses of `Model.card_answered_mul_le` are satisfiable: one-token
contexts and the weights `3` and `1`. -/
example : #{D : Fin 1 → Bool | 1 ≤ #{q | ((recurrent (X := Option Bool)
      (fun s x => x.getD s) false id).outputs ([some (D 0)] ++ [none])).getLast? = some (D q)}} *
        (3 ^ 1 * 1 ^ (Fintype.card (Fin 1) - 1)) ≤
      Fintype.card Bool ^
          (recurrent (X := Option Bool) (fun s x => x.getD s) false id).layers.length *
        (3 + (Fintype.card Bool - 1) * 1) ^ Fintype.card (Fin 1) :=
  Model.card_answered_mul_le _ (fun D : Fin 1 → Bool => [some (D 0)]) (fun _ => none) (n := 1)
    (fun _ => rfl) (by norm_num) 1

end Transformer.CausalRecall
