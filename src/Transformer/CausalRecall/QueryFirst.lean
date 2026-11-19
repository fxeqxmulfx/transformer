module

public import Transformer.CausalRecall.Recurrent

/-!
# A query read before the context

In no paper. `Transformer.CausalRecall.Recurrent` bounds a recurrent network
that reads the context and then the query: it recalls every dictionary only
with `|V|^|K|` states (`card_pow_le_of_recurrent`), since its state must answer
every key before it knows which one is asked. This module reverses the order on
the same tokens.

A token is a query `Sum.inl q` or a binding `Sum.inr (k, v)` of the key `k` to
the value `v`, and `bindingTokens ks D` writes the dictionary `D` along the
list of keys `ks`. The network `queryFirstStep` holds the key it waits for or
the value it has found: a query sets the key, a binding of that key replaces it
by its value, and a found value stays. Read after the query, the bindings leave
in it the value `D q` whenever `q` is among the keys written
(`queryFirstStep_recall`), so `|K| + |V|` states recall every dictionary.

No network that reads the query first does with fewer than `max(|K|, |V|)`
states: it must send different queries to different states, and its output
must take every value (`max_card_le_of_queryFirst_recall`). So `queryFirstStep`
has at most twice the states needed. Over the same tokens the order therefore
decides the states recall needs: `|K| + |V|` with the query first, at least
`|V|^|K|` with it last (`queryFirst_and_queryLast`). The state of the first
network is formed with the query known; that of the second is not.
-/

@[expose] public section

namespace Transformer.CausalRecall

universe u

variable {K V : Type*}

/-- The context that writes the dictionary `D` along the keys `ks`: the binding
`Sum.inr (k, D k)` for each key `k` of `ks`, in order. -/
def bindingTokens (ks : List K) (D : K → V) : List (K ⊕ (K × V)) :=
  ks.map fun k => Sum.inr (k, D k)

@[simp] theorem bindingTokens_nil (D : K → V) : bindingTokens [] D = [] := rfl

@[simp] theorem bindingTokens_cons (k : K) (ks : List K) (D : K → V) :
    bindingTokens (k :: ks) D = Sum.inr (k, D k) :: bindingTokens ks D := rfl

@[simp] theorem length_bindingTokens (ks : List K) (D : K → V) :
    (bindingTokens ks D).length = ks.length := by
  simp [bindingTokens]

/-- **With the query first, recall needs at least `|K|` states and at least
`|V|`**: a recurrent network that reads the query and then a context of the
dictionary, and answers every query on every dictionary, sends different
queries to different states, and its output takes every value. -/
theorem max_card_le_of_queryFirst_recall [Fintype K] [Fintype V] [Nonempty K] [Nontrivial V]
    {X S : Type*} [Fintype S] (step : S → X → S) (s₀ : S) (o : S → V)
    (ctx : (K → V) → List X) (qry : K → X)
    (h : ∀ D q, o ((qry q :: ctx D).foldl step s₀) = D q) :
    max (Fintype.card K) (Fintype.card V) ≤ Fintype.card S := by
  classical
  refine max_le (Fintype.card_le_of_injective (fun q => step s₀ (qry q)) fun q₁ q₂ hq => ?_)
    (Fintype.card_le_of_surjective o fun v => ?_)
  · obtain ⟨a, b, hab⟩ := exists_pair_ne V
    by_contra hne
    have h₁ := h (fun k => if k = q₁ then a else b) q₁
    have h₂ := h (fun k => if k = q₁ then a else b) q₂
    have hq' : step s₀ (qry q₁) = step s₀ (qry q₂) := hq
    rw [List.foldl_cons, hq'] at h₁
    rw [List.foldl_cons, h₁] at h₂
    exact hab (by simpa [Ne.symm hne] using h₂)
  · obtain ⟨q⟩ := ‹Nonempty K›
    exact ⟨_, h (fun _ => v) q⟩

variable [DecidableEq K]

/-- The network that reads the query first: its state is the key `Sum.inl q` it
waits for or the value `Sum.inr a` it has found. A query sets the key; a binding
of the key waited for replaces it by its value; any other binding leaves the
state. -/
def queryFirstStep : K ⊕ V → K ⊕ (K × V) → K ⊕ V
  | _, Sum.inl q => Sum.inl q
  | Sum.inl q, Sum.inr (k, b) => if k = q then Sum.inr b else Sum.inl q
  | Sum.inr a, Sum.inr _ => Sum.inr a

/-- A query sets the key waited for. -/
theorem queryFirstStep_inl (s : K ⊕ V) (q : K) : queryFirstStep s (Sum.inl q) = Sum.inl q := by
  cases s <;> rfl

/-- A binding of the key waited for replaces it by its value; another leaves it. -/
theorem queryFirstStep_inl_inr (q k : K) (b : V) :
    queryFirstStep (Sum.inl q) (Sum.inr (k, b)) = if k = q then Sum.inr b else Sum.inl q :=
  rfl

/-- A found value stays. -/
theorem queryFirstStep_inr_inr (a : V) (k : K) (b : V) :
    queryFirstStep (Sum.inr a) (Sum.inr (k, b)) = Sum.inr a := rfl

/-- A found value stays through the bindings of every dictionary. -/
@[simp] theorem foldl_queryFirstStep_inr (D : K → V) (ks : List K) (a : V) :
    (bindingTokens ks D).foldl queryFirstStep (Sum.inr a) = Sum.inr a := by
  induction ks with
  | nil => rfl
  | cons k ks ih => rw [bindingTokens_cons, List.foldl_cons, queryFirstStep_inr_inr, ih]

/-- After the query `q`, the bindings of a dictionary leave the value `D q` if
`q` is written, and the key `q` still waited for otherwise. -/
theorem foldl_queryFirstStep_inl (D : K → V) (q : K) (ks : List K) :
    (bindingTokens ks D).foldl queryFirstStep (Sum.inl q) =
      if q ∈ ks then Sum.inr (D q) else Sum.inl q := by
  induction ks with
  | nil => simp
  | cons k ks ih =>
    rw [bindingTokens_cons, List.foldl_cons, queryFirstStep_inl_inr]
    by_cases hk : k = q
    · subst hk
      simp
    · simp [hk, Ne.symm hk, ih]

/-- **With the query first, `|K| + |V|` states recall every dictionary**: from
any state, the network `queryFirstStep` reading the query `q` and then the
bindings of `D` along keys that include `q` holds the value `D q`. -/
theorem queryFirstStep_recall (s₀ : K ⊕ V) (D : K → V) {q : K} {ks : List K} (hq : q ∈ ks) :
    (Sum.inl q :: bindingTokens ks D).foldl queryFirstStep s₀ = Sum.inr (D q) := by
  rw [List.foldl_cons, queryFirstStep_inl, foldl_queryFirstStep_inl, ite_eq_left hq]

/-- The hypothesis of `queryFirstStep_recall` is satisfiable: two keys with
values in `Bool`, the second key asked. -/
example : (Sum.inl 1 :: bindingTokens (List.finRange 2) ![true, false]).foldl queryFirstStep
    (Sum.inl 0) = Sum.inr (![true, false] 1) :=
  queryFirstStep_recall _ _ (List.mem_finRange 1)

/-- The hypotheses of `max_card_le_of_queryFirst_recall` hold for
`queryFirstStep` on two keys with values in `Bool`, read by the value it has
found: it has `4` states, and at least `2` are needed. -/
example : max (Fintype.card (Fin 2)) (Fintype.card Bool) ≤ Fintype.card (Fin 2 ⊕ Bool) :=
  max_card_le_of_queryFirst_recall queryFirstStep (Sum.inl 0) (Sum.elim (fun _ => false) id)
    (bindingTokens (List.finRange 2)) Sum.inl fun D q =>
      congrArg (Sum.elim (fun _ => false) id) (queryFirstStep_recall _ D (List.mem_finRange q))

/-- **The order of the query and the context decides the states recall needs**:
over the same tokens and keys `ks` that list every key, `queryFirstStep`, of
`|K| + |V|` states, holds the answer when the query comes first, and every
recurrent network that recalls every dictionary when the query comes last has
at least `|V|^|K|` states (`card_pow_le_of_recurrent`). -/
theorem queryFirst_and_queryLast [Fintype K] [Fintype V] (ks : List K) (hks : ∀ k, k ∈ ks) :
    (∀ (s₀ : K ⊕ V) (D : K → V) (q : K),
      (Sum.inl q :: bindingTokens ks D).foldl queryFirstStep s₀ = Sum.inr (D q)) ∧
    ∀ {S : Type u} [Fintype S] (step : S → K ⊕ (K × V) → S) (s₀ : S) (o : S → V),
      (∀ D q, o ((bindingTokens ks D ++ [Sum.inl q]).foldl step s₀) = D q) →
        Fintype.card V ^ Fintype.card K ≤ Fintype.card S :=
  ⟨fun s₀ D q => queryFirstStep_recall s₀ D (hks q), fun step s₀ o h =>
    card_pow_le_of_recurrent step s₀ o (bindingTokens ks) Sum.inl (n := ks.length)
      (fun D => length_bindingTokens ks D) h⟩

/-- The hypothesis of `queryFirst_and_queryLast` is satisfiable: the keys of
`Fin 2` listed in order, values in `Bool`. -/
example := queryFirst_and_queryLast (K := Fin 2) (V := Bool) (List.finRange 2) List.mem_finRange

/-- With ten keys and values in `Bool` the query first needs between `10` and
`12` states, and the query last at least `1024`. -/
example :
    Fintype.card (Fin 10 ⊕ Bool) = 12 ∧ Fintype.card Bool ^ Fintype.card (Fin 10) = 1024 := by
  simp

end Transformer.CausalRecall
