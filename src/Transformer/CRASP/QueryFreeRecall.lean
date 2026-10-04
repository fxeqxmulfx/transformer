/-
# Recall that ignores the query needs a width growing with the vocabulary

In no paper: Zoology's index argument (arXiv:2312.04927v1, Appendix
`app:retnet-proof`, `cor: space-ar`, there for RetNet's state) for the
future-masked rounded transformers of arXiv:2506.16055v3 whose attention
ignores the query (`RTfr.QueryFree`), on the aligned recall of
`CRASP.AlignedRecall`.

Over `m + 1` tokens, the first `n ≥ m` rows of an instance bind each key
`i + 1` to the value `v` or to another, as a set `S` of `i < m` chooses; the
last row binds key `0`, which no other row has, and queries a key `q + 1`
(`indexInstance`).  The dictionary is consistent (`consistent_indexInstance`),
and `v` answers the query exactly when `S` holds `q`
(`priorAnswer_indexInstance`).  A transformer deciding at the last row whether
`v` answers thus solves the index problem on `m` bits.  If its attention
ignores the query, it reads the first `n` rows through a message of
`k (2d + 1)` integers, each one of `2^p (n + 1) + 1` (`RTfr.actAt_last`,
`RTfr.msg_mem`), and different sets need different messages: over `c` tokens,

    2^{c-1} ≤ (2^p (n + 1) + 1)^{k (2d + 1)}        (`two_pow_le_of_queryFree`),

that is `c - 1 ≤ k (2d + 1) log₂(2^p (n + 1) + 1)`.  At the fewest rows,
`n = c - 1`, this is `c - 1 ≤ k (2d + 1) (p + 1 + log₂ c)`: at a fixed depth
and precision the width grows with the vocabulary, linearly up to a logarithm.
The transformer of `exists_rtfr_answers` ignores the query, so the bound holds
for it.  Zoology argues from the randomized index bound; an exact recognizer
needs only the deterministic one, a pigeonhole
(`Transformer.Zoology.exact_index_requires_bits` states it for bits).  Below
`c - 1` rows nothing is claimed.  Attention comparing a query with a key
recalls in width `5` at `p = O(log c + log n)` bits (`matcher_answers`), so at
every depth and width the two separate (`exists_matcher_not_queryFree`).
-/

import Transformer.CRASP.AlignedRecall

namespace Transformer.CRASP

variable {m : ℕ}

/-- A value other than `v`, over `m + 1 ≥ 2` tokens. -/
def otherValue (hm : 0 < m) (v : Fin (m + 1)) : Fin (m + 1) :=
  ⟨if v.val = 0 then 1 else 0, by split <;> omega⟩

theorem otherValue_ne (hm : 0 < m) (v : Fin (m + 1)) : otherValue hm v ≠ v := by
  intro h
  have h' := congrArg Fin.val h
  simp only [otherValue] at h'
  split at h' <;> omega

/-- The hypothesis of `otherValue_ne` is satisfiable. -/
example : otherValue (by norm_num : 0 < 1) 0 ≠ 0 := otherValue_ne _ 0

/-- Row `j` of the first `n`: key `j mod m + 1`, bound to `v` when `S` holds
`j mod m` and to another value otherwise, querying its own key. -/
def indexRow (hm : 0 < m) (v : Fin (m + 1)) (S : Fin m → Bool) (j : ℕ) :
    Fin (m + 1) × Fin (m + 1) × Fin (m + 1) :=
  let i : Fin m := ⟨j % m, Nat.mod_lt _ hm⟩
  (i.succ, if S i then v else otherValue hm v, i.succ)

/-- The instance of the index problem for `S`: the `n` rows of `indexRow`, then a
last row binding key `0`, which no other row has, and querying `q + 1`
(Appendix `sec: intro-general-ar`). -/
def indexInstance (hm : 0 < m) (v : Fin (m + 1)) (S : Fin m → Bool) (n : ℕ) (q : Fin m) :
    Zoology.MQARInstance (n + 1) (m + 1) where
  key j := if j.val < n then (indexRow hm v S j).1 else 0
  value j := if j.val < n then (indexRow hm v S j).2.1 else 0
  query j := if j.val < n then (indexRow hm v S j).2.2 else q.succ

/-- Its string is the `n` rows, which do not depend on `q`, then
`(0, 0, q + 1)`. -/
theorem word_indexInstance (hm : 0 < m) (v : Fin (m + 1)) (S : Fin m → Bool) (n : ℕ)
    (q : Fin m) : word (indexInstance hm v S n q) =
      List.ofFn (fun j : Fin n => indexRow hm v S j) ++ [(0, 0, q.succ)] := by
  rw [word, List.ofFn_succ_last]
  simp [indexInstance]

/-- The dictionary binds each key to one value (`Zoology.Consistent`). -/
theorem consistent_indexInstance (hm : 0 < m) (v : Fin (m + 1)) (S : Fin m → Bool) (n : ℕ)
    (q : Fin m) : Zoology.Consistent (indexInstance hm v S n q) := by
  intro j l h
  by_cases hj : j.val < n <;> by_cases hl : l.val < n <;>
    simp only [indexInstance, indexRow, hj, hl, ite_true, ite_false, Fin.succ_inj] at h ⊢
  · rw [h]
  · exact absurd h (Fin.succ_ne_zero _)
  · exact absurd h.symm (Fin.succ_ne_zero _)

/-- **The value `v` answers the last query exactly when `S` holds it**, once
every key but `0` has a row (`m ≤ n`). -/
theorem priorAnswer_indexInstance (hm : 0 < m) (v : Fin (m + 1)) (S : Fin m → Bool) {n : ℕ}
    (hn : m ≤ n) (q : Fin m) :
    Zoology.PriorAnswer (indexInstance hm v S n q) (Fin.last n) v ↔ S q = true := by
  constructor
  · rintro ⟨j, hj, hkey, hval⟩
    have hjn : j.val < n := hj
    simp only [indexInstance, indexRow, hjn, Fin.val_last, lt_irrefl, ite_true, ite_false,
      Fin.succ_inj] at hkey hval
    subst hkey
    by_contra hS
    simp only [hS, ite_false] at hval
    exact otherValue_ne hm v hval
  · intro hS
    refine ⟨⟨q.val, by omega⟩, Fin.mk_lt_of_lt_val (by simp; omega), ?_, ?_⟩ <;>
      simp [indexInstance, indexRow, show q.val < n by omega, Nat.mod_eq_of_lt q.isLt, hS,
        Fin.ext_iff]

/-- The hypotheses of `priorAnswer_indexInstance` are satisfiable: two tokens, one
row binding key `1` to `0`. -/
example : Zoology.PriorAnswer (indexInstance (by norm_num : 0 < 1) 0 (fun _ => true) 1 0)
    (Fin.last 1) 0 :=
  (priorAnswer_indexInstance _ 0 _ le_rfl 0).mpr rfl

/-- **A transformer whose attention ignores the query recalls over `c`
tokens only if `c - 1 ≤ k (2d + 1) log₂(2^p (n + 1) + 1)`**: if, on the
consistent instances of `n + 1 ≥ c` rows, it accepts exactly those whose last
query `v` answers, then `2^{c-1} ≤ (2^p (n + 1) + 1)^{k (2d + 1)}` (Zoology's
`cor: space-ar`, there for RetNet, by its proof). -/
theorem two_pow_le_of_queryFree {c p s d k n : ℕ}
    (T : RTfr (Option (Fin c × Fin c × Fin c)) p s d k) (hQ : T.QueryFree) (v : Fin c)
    (hn : c ≤ n + 1)
    (hT : ∀ x : Zoology.MQARInstance (n + 1) c, Zoology.Consistent x →
      (T.Accepts (bos (word x)) ↔ Zoology.PriorAnswer x (Fin.last n) v)) :
    2 ^ (c - 1) ≤ (2 ^ p * (n + 1) + 1) ^ (k * (2 * d + 1)) := by
  classical
  obtain _ | _ | m := c
  · exact Nat.one_le_pow _ _ (by omega)
  · exact Nat.one_le_pow _ _ (by omega)
  have hm : 0 < m + 1 := by omega
  let alice : (Fin (m + 1) → Bool) → List (Fin (m + 2) × Fin (m + 2) × Fin (m + 2)) :=
    fun S => List.ofFn fun j : Fin n => indexRow hm v S j
  have hlen : ∀ S, (alice S).length = n := fun S => by simp [alice]
  have hacc : ∀ S (q : Fin (m + 1)),
      S q = true ↔ 0 < (T.Wout (T.bob n (T.msg (alice S)) (0, 0, q.succ) k)).val := by
    intro S q
    have hb := T.actAt_last hQ (alice S) (0, 0, q.succ) k
    rw [hlen S] at hb
    rw [← priorAnswer_indexInstance hm v S (n := n) (by omega) q,
      ← hT _ (consistent_indexInstance hm v S n q), RTfr.Accepts, word_indexInstance,
      RTfr.out_bos, List.length_append, List.length_singleton]
    change 0 < (T.Wout (T.actAt (alice S ++ [(0, 0, q.succ)]) k ((alice S).length + 1))).val ↔ _
    rw [hlen S, hb]
  have hmem : ∀ S ℓ o, 0 ≤ T.msg (alice S) ℓ o + (n + 1) * 2 ^ (p - 1) ∧
      T.msg (alice S) ℓ o + (n + 1) * 2 ^ (p - 1) < ((2 ^ p * (n + 1) + 1 : ℕ) : ℤ) := by
    intro S ℓ o
    have h := T.msg_mem (alice S) ℓ o
    rw [hlen S] at h
    push_cast
    exact h
  let code : (Fin (m + 1) → Bool) → Fin k → Option (Fin d ⊕ Fin d) →
      Fin (2 ^ p * (n + 1) + 1) :=
    fun S ℓ o => ⟨(T.msg (alice S) ℓ o + (n + 1) * 2 ^ (p - 1)).toNat,
      (Int.toNat_lt (hmem S ℓ o).1).mpr (hmem S ℓ o).2⟩
  have hinj : Function.Injective code := by
    intro S S' h
    have hM : ∀ ℓ < k, T.msg (alice S) ℓ = T.msg (alice S') ℓ := by
      intro ℓ hℓ
      funext o
      have h₁ := congrArg (fun f => ((f ⟨ℓ, hℓ⟩ o : Fin _) : ℕ)) h
      have h₂ := congrArg (fun m : ℕ => (m : ℤ)) h₁
      simp only [code, Int.toNat_of_nonneg (hmem S ℓ o).1,
        Int.toNat_of_nonneg (hmem S' ℓ o).1] at h₂
      linarith
    funext q
    rw [Bool.eq_iff_iff, hacc, hacc, T.bob_congr n (0, 0, q.succ) hM]
  have hcard := Fintype.card_le_of_injective code hinj
  simp only [Fintype.card_fun, Fintype.card_bool, Fintype.card_fin, Fintype.card_option,
    Fintype.card_sum] at hcard
  rw [Nat.add_sub_cancel, mul_comm k, pow_mul, two_mul]
  exact hcard

/-- The hypotheses of `two_pow_le_of_queryFree` are satisfiable: over two
tokens, at two rows, by the transformer of `exists_rtfr_answers`. -/
example : ∃ (p s d : ℕ) (T : RTfr (Option (Fin 2 × Fin 2 × Fin 2)) p s d 1), T.QueryFree ∧
    ∀ x : Zoology.MQARInstance (1 + 1) 2, Zoology.Consistent x →
      (T.Accepts (bos (word x)) ↔ Zoology.PriorAnswer x (Fin.last 1) 0) := by
  obtain ⟨p, s, d, T, hQ, hT⟩ := exists_rtfr_answers (0 : Fin 2)
  exact ⟨p, s, d, T, hQ, fun x _ => hT x⟩

end Transformer.CRASP
