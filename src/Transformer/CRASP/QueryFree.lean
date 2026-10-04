/-
# Attention that ignores the query reads the past through sums

In no paper: the step of Zoology's index argument (arXiv:2312.04927v1,
Appendix `app:retnet-proof`, `cor: space-ar`, there for RetNet) for the
future-masked rounded transformers of arXiv:2506.16055v3 (Appendix B.1,
`def:transformer`).

A transformer *ignores the query* (`RTfr.QueryFree`) when no score
`sᵢⱼ = qᵢ · kⱼ` of a layer depends on the query `qᵢ`, as when `W_Q` or `W_K`
is constant; the transformer that `thm:TLCl_to_rtfr` builds from a formula of
`TL[◁#]` uses uniform attention, `W_Q = W_K = 0` (`queryFree_model`).  A query-free
layer reads the positions up to `i` through the totals of `2d + 1` parts
(`part`): their rounded weights `round(exp sⱼ)`, their weighted values, and
their values, which the fallback of Equation `eq:att` averages
(`layer_queryFree`).  A transformer does not look ahead (`actAt_append`), so
at the last position of `⊲ · u · b` the totals over `⊲ · u` are a message
from `u`, `msg`, of `2d + 1` integers per layer, the sums of the mantissas,
and the last activation is a function of that message and of `b`
(`actAt_last`); each integer is one of `2^p (|u| + 1) + 1` (`msg_mem`).
-/

import Transformer.CRASP.Transformers
import Transformer.CRASP.Locality

namespace Transformer.CRASP.RTfr

universe u
variable {σ : Type u} {p s d k : ℕ}

/-- The score `q · k` of layer `ℓ` between a query state `x` and a key state
`y` (Equation `eq:att`). -/
noncomputable def score (T : RTfr σ p s d k) (ℓ : ℕ) (x y : Fin d → Fx p s) : ℝ :=
  ∑ c : Fin d, (T.WQ ℓ x c).val * (T.WK ℓ y c).val

/-- No score of any layer depends on the query state. -/
def QueryFree (T : RTfr σ p s d k) : Prop :=
  ∀ ℓ x x' y, T.score ℓ x y = T.score ℓ x' y

/-- **The transformer of `thm:TLCl_to_rtfr` ignores the query**: "use
uniform attention", its `W_Q` and `W_K` vanish (`TemporalProgram.model`). -/
theorem queryFree_model [DecidableEq σ] (φ : Form σ) (k : ℕ) :
    (TemporalProgram.model φ k).QueryFree := by
  intro ℓ x x' y
  simp [score, TemporalProgram.model]

/-- What a position with state `y` contributes to a query-free layer: its
rounded weight (`none`), its weighted value at `c` (`inl c`), and its value at
`c` (`inr c`), which the fallback averages (Equation `eq:att`). -/
noncomputable def part (T : RTfr σ p s d k) (ℓ : ℕ) :
    Option (Fin d ⊕ Fin d) → (Fin d → Fx p s) → Fx p s
  | none, y => Fx.round p s (Real.exp (T.score ℓ 0 y))
  | some (.inl c), y => Fx.round p s (Real.exp (T.score ℓ 0 y) * (T.WV ℓ y c).val)
  | some (.inr c), y => T.WV ℓ y c

/-- A query-free layer at a position with state `y` and `len` positions up to
it, from the totals `tot` of the parts over them (Equation `eq:att`). -/
noncomputable def mix (T : RTfr σ p s d k) (ℓ : ℕ) (tot : Option (Fin d ⊕ Fin d) → ℝ)
    (len : ℕ) (y : Fin d → Fx p s) : Fin d → Fx p s :=
  T.ff ℓ fun c => Fx.add
    (if tot none = 0 then Fx.round p s (tot (some (.inr c)) / len)
      else Fx.round p s (tot (some (.inl c)) / tot none)) (y c)

/-- **A query-free layer reads the positions up to `i` through the totals of
their parts.** -/
theorem layer_queryFree (T : RTfr σ p s d k) (hT : T.QueryFree) (ℓ : ℕ) {n : ℕ}
    (h : Fin n → Fin d → Fx p s) (i : Fin n) :
    T.layer ℓ h i =
      T.mix ℓ (fun o => ∑ j ∈ masked i, (T.part ℓ o (h j)).val) (i.val + 1) (h i) := by
  have hs : ∀ j, ∑ c : Fin d, (T.WQ ℓ (h i) c).val * (T.WK ℓ (h j) c).val =
      T.score ℓ 0 (h j) := fun j => hT ℓ (h i) 0 (h j)
  simp only [layer, mix, part, hs, card_masked]
  rfl

/-- The hypothesis of `layer_queryFree` holds for the transformer of
`thm:TLCl_to_rtfr`. -/
example [DecidableEq σ] (φ : Form σ) (k ℓ n : ℕ) (h : Fin n → Fin _ → Fx _ 0) (i : Fin n) :
    (TemporalProgram.model φ k).layer ℓ h i = (TemporalProgram.model φ k).mix ℓ
      (fun o => ∑ j ∈ masked i, ((TemporalProgram.model φ k).part ℓ o (h j)).val)
      (i.val + 1) (h i) :=
  layer_queryFree _ (queryFree_model φ k) ℓ h i

/-- Alice's message: for each layer and part, the sum of the mantissas of the
part over the positions of `⊲ · u`. -/
noncomputable def msg (T : RTfr (Option σ) p s d k) (u : List σ) (ℓ : ℕ)
    (o : Option (Fin d ⊕ Fin d)) : ℤ :=
  (T.part ℓ o (T.actAt [] ℓ 0)).m + ∑ j ∈ Finset.Icc 1 u.length, (T.part ℓ o (T.actAt u ℓ j)).m

/-- Bob's activations at position `n + 1`, from a message `M` about the `n + 1`
positions before it and his letter `b`. -/
noncomputable def bob (T : RTfr (Option σ) p s d k) (n : ℕ)
    (M : ℕ → Option (Fin d ⊕ Fin d) → ℤ) (b : σ) : ℕ → Fin d → Fx p s
  | 0 => T.E (some b)
  | ℓ + 1 => T.mix ℓ (fun o => (M ℓ o : ℝ) / 2 ^ s + (T.part ℓ o (T.bob n M b ℓ)).val) (n + 2)
      (T.bob n M b ℓ)

/-- Bob's activation at depth `ℓ` reads the message only below `ℓ`. -/
theorem bob_congr (T : RTfr (Option σ) p s d k) (n : ℕ) {M M' : ℕ → Option (Fin d ⊕ Fin d) → ℤ}
    (b : σ) {ℓ : ℕ} (h : ∀ ℓ' < ℓ, M ℓ' = M' ℓ') : T.bob n M b ℓ = T.bob n M' b ℓ := by
  induction ℓ with
  | zero => rfl
  | succ ℓ ih => rw [bob, bob, ih fun ℓ' hℓ' => h ℓ' (by omega), h ℓ (by omega)]

/-- The hypothesis of `bob_congr` is satisfiable: at depth `0` it is empty. -/
example (T : RTfr (Option σ) p s d k) (n : ℕ) (M M' : ℕ → Option (Fin d ⊕ Fin d) → ℤ)
    (b : σ) : T.bob n M b 0 = T.bob n M' b 0 :=
  T.bob_congr n b fun _ h => absurd h (Nat.not_lt_zero _)

/-- **Each integer of the message is one of `2^p (|u| + 1) + 1`**: shifted by
`(|u| + 1) 2^{p-1}`, a sum of `|u| + 1` mantissas lies in `[0, 2^p (|u| + 1)]`. -/
theorem msg_mem (T : RTfr (Option σ) p s d k) (u : List σ) (ℓ : ℕ) (o : Option (Fin d ⊕ Fin d)) :
    0 ≤ T.msg u ℓ o + (u.length + 1) * 2 ^ (p - 1) ∧
      T.msg u ℓ o + (u.length + 1) * 2 ^ (p - 1) < 2 ^ p * (u.length + 1) + 1 := by
  set A : ℤ := 2 ^ (p - 1)
  have hA : 2 * A ≤ 2 ^ p + 1 := by
    cases p with
    | zero => simp [A]
    | succ p => simp only [A, Nat.add_sub_cancel, pow_succ]; omega
  have hlo := Finset.card_nsmul_le_sum (Finset.Icc 1 u.length)
    (fun j => (T.part ℓ o (T.actAt u ℓ j)).m) (-A) fun j _ => (T.part ℓ o (T.actAt u ℓ j)).lo
  have hhi := Finset.sum_le_card_nsmul (Finset.Icc 1 u.length)
    (fun j => (T.part ℓ o (T.actAt u ℓ j)).m) (A - 1) fun j _ => by
      have := (T.part ℓ o (T.actAt u ℓ j)).hi; omega
  have h₀ := (T.part ℓ o (T.actAt [] ℓ 0)).lo
  have h₁ := (T.part ℓ o (T.actAt [] ℓ 0)).hi
  simp only [Nat.card_Icc, Nat.add_sub_cancel, nsmul_eq_mul] at hlo hhi
  simp only [msg]
  constructor <;> nlinarith

variable [Fintype σ] [DecidableEq σ]

/-- **A transformer does not look ahead**: at a position `1 ≤ i ≤ |u|` of
`⊲ · u · v` its activation is that of `⊲ · u` (Appendix B.1, the future mask),
since the formulas of `thm:rtfr_to_TLCl` that define it look only back. -/
theorem actAt_append (T : RTfr (Option σ) p s d k) (u v : List σ) (ℓ : ℕ) {i : ℕ}
    (h₁ : 1 ≤ i) (h₂ : i ≤ u.length) : T.actAt (u ++ v) ℓ i = T.actAt u ℓ i := by
  obtain ⟨ψ, hψ, hdef⟩ := T.exists_stateFormulas ℓ
  have hu : (ψ (T.actAt u ℓ i)).sat u i = true := by
    rw [hdef u i ⟨h₂, Or.inl h₁⟩]
    simp
  rw [← Form.sat_append u v _ (hψ _).1 (hψ _).2.1 i h₁ h₂,
    hdef (u ++ v) i ⟨by rw [List.length_append]; omega, Or.inl h₁⟩] at hu
  simpa using hu

/-- The hypotheses of `actAt_append` are satisfiable: the one position of
`[a]`. -/
example (T : RTfr (Option σ) p s d k) (a : σ) (v : List σ) (ℓ : ℕ) :
    T.actAt ([a] ++ v) ℓ 1 = T.actAt [a] ℓ 1 :=
  T.actAt_append [a] v ℓ le_rfl le_rfl

/-- **A query-free transformer's last activation on `⊲ · u · b` is Bob's**,
computed from Alice's message about `u` and from `b`. -/
theorem actAt_last (T : RTfr (Option σ) p s d k) (hT : T.QueryFree) (u : List σ) (b : σ)
    (ℓ : ℕ) : T.actAt (u ++ [b]) ℓ (u.length + 1) = T.bob u.length (T.msg u) b ℓ := by
  have hi : u.length + 1 < (bos (u ++ [b])).length := by simp
  induction ℓ with
  | zero => simp [actAt, act, bos, bob]
  | succ ℓ ih =>
      rw [actAt, dite_eq_left hi, act, T.layer_queryFree hT, bob, ← ih,
        ← actAt_valid T (u ++ [b]) ℓ ⟨u.length + 1, hi⟩]
      congr 1
      funext o
      rw [T.sum_masked_actAt (u ++ [b]) ℓ ⟨u.length + 1, hi⟩ (fun a => (T.part ℓ o a).val),
        Finset.sum_Icc_succ_top (by omega), msg,
        Finset.sum_congr rfl fun j hj => by
          rw [T.actAt_append u [b] ℓ (Finset.mem_Icc.mp hj).1 (Finset.mem_Icc.mp hj).2]]
      simp only [Fx.val]
      push_cast
      rw [add_div, Finset.sum_div]
      ring

/-- The hypothesis of `actAt_last` holds for the transformer of
`thm:TLCl_to_rtfr`. -/
example (φ : Form σ) (k ℓ : ℕ) (u : List σ) (b : σ) :
    (TemporalProgram.model φ k).actAt (u ++ [b]) ℓ (u.length + 1) =
      (TemporalProgram.model φ k).bob u.length ((TemporalProgram.model φ k).msg u) b ℓ :=
  actAt_last _ (queryFree_model φ k) u b ℓ

end Transformer.CRASP.RTfr
