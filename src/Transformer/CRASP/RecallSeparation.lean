/-
# Comparing a query with a key separates recall

In no paper.  At every depth `k` and width `d`, aligned recall over `c = 2^t`
tokens and `c` rows, at `p = 4t + 5` bits of which `s = t + 1` are fractional,
is recognized by the matcher of `CRASP.MatchingRecallAnswers`, one layer of
width `5` comparing a query with a key (`matcher_answers`), and by no
transformer of depth `k` and width `d` whose attention ignores the query
(`two_pow_le_of_queryFree`), once `t = 2 (k (2d + 1) + 7)`
(`exists_matcher_not_queryFree`): there `2^{c-1} ≤ (2^p c + 1)^{k (2d + 1)}`
fails, as `2^t` outgrows `(5t + 6) k (2d + 1) + 1` (`mul_lt_two_pow_sub_one`).
At the same precision, logarithmic in the vocabulary, attention comparing a
query with a key recalls in constant width, and attention ignoring the query
does not at any fixed width and depth.

Up to a constant factor, no transformer recalls with fewer bits a position
than the matcher, at any depth and whatever its attention compares.  One
recognizing recall of `v` on the consistent instances of `n + 1 ≥ 2` rows
embeds the `c` triples `(a, v, a)` apart (`card_le_of_recall`), since a word
reaches the output only through its embeddings (`out_eq_of_map`) and the
instance whose rows all read `(a, v, a)` recalls `v` while the one whose
earlier rows read `(a', v, a')` does not (`priorAnswer_collide`).  So
`c ≤ 2^{p d}` at `p ≥ 1` bits and width `d` (`le_two_pow_of_recall`):
`p d ≥ log₂ c`, which the matcher's `5 p = O(log c + log n)` meets up to a
constant factor once `n ≤ c`.
-/

import Transformer.CRASP.MatchingRecallAnswers

namespace Transformer.CRASP

/-- `11 b ≤ 2^b` from `b = 7` on. -/
theorem eleven_mul_le_two_pow {b : ℕ} (hb : 7 ≤ b) : 11 * b ≤ 2 ^ b := by
  induction b, hb using Nat.le_induction with
  | base => norm_num
  | succ b hb ih => rw [pow_succ]; omega

/-- The hypothesis of `eleven_mul_le_two_pow` is satisfiable: `b = 7`. -/
example : 11 * 7 ≤ 2 ^ 7 := eleven_mul_le_two_pow le_rfl

/-- `2^t` outgrows `(5t + 6) K + 1` at `t = 2 (K + 7)`. -/
theorem mul_lt_two_pow_sub_one (K : ℕ) :
    (5 * (2 * (K + 7)) + 6) * K < 2 ^ (2 * (K + 7)) - 1 := by
  have h₁ := eleven_mul_le_two_pow (b := K + 7) (by omega)
  have h₂ : K + 2 ≤ 2 ^ (K + 7) := by
    have := Nat.lt_two_pow_self (n := K + 7)
    omega
  have h₃ := Nat.mul_le_mul h₁ h₂
  rw [two_mul (K + 7), pow_add]
  have h₄ : (5 * (K + 7 + (K + 7)) + 6) * K + 1 < 11 * (K + 7) * (K + 2) := by nlinarith
  omega

/-- **Comparing a query with a key separates recall**: at every depth `k` and
width `d` there is `t` such that, over `2^t` tokens and `2^t` rows at `4t + 5`
bits of which `t + 1` are fractional, the matcher of one layer and width `5`
recalls every value `v`, and no transformer of depth `k` and width `d` whose
attention ignores the query does, even on consistent instances alone. -/
theorem exists_matcher_not_queryFree (k d : ℕ) :
    ∃ t : ℕ, ∀ v : Fin (2 ^ t),
      (∀ x : Zoology.MQARInstance (2 ^ t - 1 + 1) (2 ^ t),
        (matcher v : RTfr _ (4 * t + 5) (t + 1) 5 1).Accepts (bos (word x)) ↔
          Zoology.PriorAnswer x (Fin.last _) v) ∧
      ∀ T : RTfr (Option (Fin (2 ^ t) × Fin (2 ^ t) × Fin (2 ^ t))) (4 * t + 5) (t + 1) d k,
        T.QueryFree → ¬ ∀ x : Zoology.MQARInstance (2 ^ t - 1 + 1) (2 ^ t), Zoology.Consistent x →
          (T.Accepts (bos (word x)) ↔ Zoology.PriorAnswer x (Fin.last _) v) := by
  obtain ⟨t, ht⟩ : ∃ t, t = 2 * (k * (2 * d + 1) + 7) := ⟨_, rfl⟩
  have h₁ : 1 ≤ 2 ^ t := Nat.one_le_two_pow
  have hp : 2 * (t + 1 + 1) * (2 ^ t) ^ 2 * 2 ^ (t + 1) < 2 ^ (4 * t + 5 - 1) := by
    have e₁ : 2 * (t + 1 + 1) * (2 ^ t) ^ 2 * 2 ^ (t + 1) = (t + 2) * 2 ^ (3 * t + 2) := by ring
    have e₂ : 2 ^ (4 * t + 5 - 1) = 2 ^ (t + 2) * 2 ^ (3 * t + 2) := by
      rw [← pow_add]
      congr 1
      omega
    rw [e₁, e₂]
    exact Nat.mul_lt_mul_of_pos_right Nat.lt_two_pow_self (by positivity)
  have hs : 2 ^ t - 1 + 2 ≤ 2 ^ (t + 1) := by
    rw [pow_succ]
    omega
  refine ⟨t, fun v => ⟨fun x => matcher_answers hp hs v x, fun T hQ hT => ?_⟩⟩
  have h := two_pow_le_of_queryFree T hQ v (by omega) hT
  have hb : 2 ^ (4 * t + 5) * (2 ^ t - 1 + 1) + 1 ≤ 2 ^ (5 * t + 6) := by
    rw [Nat.sub_add_cancel h₁, ← pow_add, show 5 * t + 6 = 4 * t + 5 + t + 1 by omega, pow_succ]
    have := Nat.one_le_two_pow (n := 4 * t + 5 + t)
    omega
  have hK := Nat.pow_le_pow_left hb (k * (2 * d + 1))
  rw [← pow_mul] at hK
  have hlt : 2 ^ ((5 * t + 6) * (k * (2 * d + 1))) < 2 ^ (2 ^ t - 1) :=
    Nat.pow_lt_pow_right (by norm_num) (ht ▸ mul_lt_two_pow_sub_one (k * (2 * d + 1)))
  omega

variable {σ : Type*} {c p s d k : ℕ}

/-- **A word reaches the output only through its embeddings**: words whose
letters embed alike, one by one, have one output (Definition
`def:transformer`). -/
theorem RTfr.out_eq_of_map (T : RTfr σ p s d k) {w w' : List σ}
    (h : w.map T.E = w'.map T.E) : T.out w = T.out w' := by
  have hl : w.length = w'.length := by simpa using congrArg List.length h
  have hE (i : Fin w.length) : T.E w[i] = T.E w'[Fin.cast hl i] := by
    simpa using List.getElem_of_eq h (i := i.1) (by simp)
  have hc {n m : ℕ} (hnm : n = m) (ℓ : ℕ) (g : Fin m → Fin d → Fx p s) :
      T.layer ℓ (fun i => g (Fin.cast hnm i)) = fun i => T.layer ℓ g (Fin.cast hnm i) := by
    subst hnm
    rfl
  have hact (ℓ : ℕ) : T.act w ℓ = fun i => T.act w' ℓ (Fin.cast hl i) := by
    induction ℓ with
    | zero => exact funext fun i => funext fun c => congrFun (hE i) c
    | succ ℓ ih => exact (congrArg (T.layer ℓ) ih).trans (hc hl ℓ _)
  unfold RTfr.out
  rw [hact]
  by_cases hw : 0 < w.length
  · rw [dite_eq_left hw, dite_eq_left (hl ▸ hw)]
    exact congrArg T.Wout (congrArg (T.act w' k) (Fin.ext (by simp [hl])))
  · rw [dite_eq_right hw, dite_eq_right (hl ▸ hw)]

/-- The hypothesis of `out_eq_of_map` is satisfiable: a word and itself. -/
example (T : RTfr Bool 2 0 1 0) : T.out [true] = T.out [true] :=
  T.out_eq_of_map rfl

/-- `n` rows `(b, v, b)`, then the row `(a, v, a)`. -/
def collide (n : ℕ) (v b a : Fin c) : Zoology.MQARInstance (n + 1) c where
  key j := if j = Fin.last n then a else b
  value _ := v
  query j := if j = Fin.last n then a else b

/-- One value throughout: every `collide` instance is consistent. -/
theorem consistent_collide (n : ℕ) (v b a : Fin c) : Zoology.Consistent (collide n v b a) :=
  fun _ _ _ => rfl

/-- The last row of `collide` recalls `v` exactly when `b = a`. -/
theorem priorAnswer_collide {n : ℕ} (hn : 0 < n) (v b a : Fin c) :
    Zoology.PriorAnswer (collide n v b a) (Fin.last n) v ↔ b = a := by
  constructor
  · rintro ⟨j, hj, hk, -⟩
    simpa [collide, hj.ne] using hk
  · rintro rfl
    exact ⟨⟨0, by omega⟩, Fin.mk_lt_mk.mpr hn, by simp [collide], rfl⟩

/-- The hypothesis of `priorAnswer_collide` is satisfiable: `n = 1`. -/
example : Zoology.PriorAnswer (collide 1 (0 : Fin 1) 0 0) (Fin.last 1) 0 :=
  (priorAnswer_collide one_pos 0 0 0).mpr rfl

/-- **Recall embeds its keys apart**: a transformer recognizing recall of `v`
on the consistent instances of `n + 1 ≥ 2` rows, at any depth and whatever
its attention compares, embeds the triples `(a, v, a)` injectively, so the
`c` keys are at most the states of a position. -/
theorem card_le_of_recall (T : RTfr (Option (Fin c × Fin c × Fin c)) p s d k) {n : ℕ}
    (hn : 0 < n) (v : Fin c)
    (hT : ∀ x : Zoology.MQARInstance (n + 1) c, Zoology.Consistent x →
      (T.Accepts (bos (word x)) ↔ Zoology.PriorAnswer x (Fin.last n) v)) :
    c ≤ Fintype.card (Fin d → Fx p s) := by
  have hinj : Function.Injective fun a : Fin c => T.E (some (a, v, a)) := by
    intro a a' (hE : T.E (some (a, v, a)) = T.E (some (a', v, a')))
    by_contra hne
    have hw : (bos (word (collide n v a a))).map T.E =
        (bos (word (collide n v a' a))).map T.E := by
      simp only [bos, word, List.map_cons, List.map_ofFn, List.cons.injEq,
        List.ofFn_inj, true_and]
      funext j
      by_cases hj : j = Fin.last n <;> simp [collide, hj, hE]
    have h₁ := (hT _ (consistent_collide n v a a)).mpr ((priorAnswer_collide hn v a a).mpr rfl)
    rw [RTfr.Accepts, T.out_eq_of_map hw] at h₁
    exact hne ((priorAnswer_collide hn v a' a).mp ((hT _ (consistent_collide n v a' a)).mp h₁)).symm
  simpa using Fintype.card_le_of_injective _ hinj

/-- The hypotheses of `card_le_of_recall` are satisfiable: the matcher over
one token at `6` bits, two of them fractional, recalls on two rows. -/
example : 1 ≤ Fintype.card (Fin 5 → Fx 6 2) :=
  card_le_of_recall (matcher 0) one_pos 0 fun x _ => matcher_answers (by norm_num) (by norm_num) 0 x

/-- **Recall needs `log₂ c` bits a position**: a transformer of width `d` at
`p ≥ 1` bits recognizing recall of `v` on the consistent instances of
`n + 1 ≥ 2` rows, at any depth and whatever its attention compares, has
`c ≤ 2^{p d}`. -/
theorem le_two_pow_of_recall (hp : 0 < p) (T : RTfr (Option (Fin c × Fin c × Fin c)) p s d k)
    {n : ℕ} (hn : 0 < n) (v : Fin c)
    (hT : ∀ x : Zoology.MQARInstance (n + 1) c, Zoology.Consistent x →
      (T.Accepts (bos (word x)) ↔ Zoology.PriorAnswer x (Fin.last n) v)) :
    c ≤ 2 ^ (p * d) := by
  have h := card_le_of_recall T hn v hT
  rwa [Fintype.card_fun, Fintype.card_fin, Fx.card_eq, ← pow_succ', Nat.sub_add_cancel hp,
    ← pow_mul] at h

/-- The hypotheses of `le_two_pow_of_recall` are satisfiable: the matcher
over one token at `6` bits, two of them fractional, recalls on two rows. -/
example : 1 ≤ 2 ^ (6 * 5) :=
  le_two_pow_of_recall (by norm_num) (matcher (0 : Fin 1) : RTfr _ 6 2 5 1) one_pos 0
    fun x _ => matcher_answers (by norm_num) (by norm_num) 0 x

end Transformer.CRASP
