/-
# Aligned recall over a finite vocabulary is counting at depth 1

In no paper: the remark behind the lab's choice of a recall vocabulary larger
than either model's width (`python/src/lab/domain/basis.py`).

Zoology's MQAR (arXiv:2312.04927v1, Appendix `sec: intro-general-ar`, Setup;
`Transformer.Zoology.MQARInstance`) asks at each row `i` for the value `v_j`
of an earlier row `j < i` whose key is the query `q_i`.  Its attention
solution (§4, `prop: attention-ar`, proved as `prop: app-attention`) spends
its first layer shifting each value onto its key; after it, the input is a
string of triples (key, value, query) over a vocabulary of `c` tokens
(`word`).  Over a finite vocabulary the answer then needs no comparison of a
query with a key: "the value `v` answers the query at `i`" is

    ⋁_{q < c} (query = q ∧ ∃ j < i [key = q ∧ value = v]),

a formula of `TL[◁#]` of depth 1 (`answers_mem`) that is true exactly where
`v` answers (`sat_answers`), and that counts nothing but the `c` formulas
`key = q ∧ value = v` (`countSubs_answers`); the `c` values together take
`c²` counts, one per pair.  By `thm:transformer_equivalence` of
arXiv:2506.16055v3 a future-masked rounded transformer of one layer, whose
attention in that construction ignores the query (`RTfr.QueryFree`),
recognizes the instances whose last query `v` answers (`exists_rtfr_answers`):
over a fixed vocabulary, aligned recall is counting, with no attention from a
query to a key.  At `n + 1 ≥ c` rows such a transformer needs
`c - 1 ≤ k (2d + 1) log₂(2^p (n + 1) + 1)`, a width growing with `c`
(`two_pow_le_of_queryFree`, `CRASP.QueryFreeRecall`).
-/

import Transformer.CRASP.BoundedExists
import Transformer.CRASP.FormulaBounds
import Transformer.CRASP.Locality
import Transformer.CRASP.QueryFree
import Transformer.CRASP.Transformers
import Transformer.Zoology.Section3_MQAR

namespace Transformer
namespace CRASP

variable {c : ℕ}

/-- The string of an aligned instance: position `j + 1` carries the triple
(key, value, query) of row `j` (Appendix `sec: intro-general-ar`, Setup). -/
def word {n : ℕ} (x : Zoology.MQARInstance n c) : List (Fin c × Fin c × Fin c) :=
  List.ofFn fun j => (x.key j, x.value j, x.query j)

/-- Every triple over `Fin c`. -/
def triples (c : ℕ) : List (Fin c × Fin c × Fin c) :=
  (List.finRange c).flatMap fun k =>
    (List.finRange c).flatMap fun u => (List.finRange c).map fun r => (k, u, r)

theorem mem_triples (t : Fin c × Fin c × Fin c) : t ∈ triples c := by
  obtain ⟨k, u, r⟩ := t
  simp [triples]

/-- `⋁ Q_t` over the triples `t` with `p t`: a test of depth `0`. -/
def holds (p : Fin c × Fin c × Fin c → Bool) : Form (Fin c × Fin c × Fin c) :=
  Form.any (((triples c).filter p).map Form.sym)

theorem sat_holds (p : Fin c × Fin c × Fin c → Bool) (w : List (Fin c × Fin c × Fin c))
    (i : ℕ) : (holds p).sat w i = true ↔ ∃ t, w[i - 1]? = some t ∧ p t = true := by
  simp only [holds, Form.sat_any, List.mem_map, List.mem_filter]
  constructor
  · rintro ⟨_, ⟨t, ⟨-, ht⟩, rfl⟩, hsat⟩
    exact ⟨t, by simpa [Form.sat] using hsat, ht⟩
  · rintro ⟨t, hw, ht⟩
    exact ⟨_, ⟨t, ⟨mem_triples t, ht⟩, rfl⟩, by simp [Form.sat, hw]⟩

theorem holds_mem (p : Fin c × Fin c × Fin c → Bool) : holds p ∈ TLCl _ 0 :=
  Form.any_mem _ fun φ hφ => by
    obtain ⟨t, -, rfl⟩ := List.mem_map.mp hφ
    exact ⟨rfl, rfl, le_rfl⟩

/-- `⋁_{q < c} (query = q ∧ ∃ j < i [key = q ∧ value = v])`: an earlier row
binds the current query to `v`. -/
def answers (v : Fin c) : Form (Fin c × Fin c × Fin c) :=
  Form.any ((List.finRange c).map fun q =>
    .and (holds fun t => t.2.2 = q) (Form.exBefore (holds fun t => t.1 = q && t.2.1 = v)))

/-- It is a formula of `TL[◁#]_1`. -/
theorem answers_mem (v : Fin c) : answers v ∈ TLCl _ 1 :=
  Form.any_mem _ fun φ hφ => by
    obtain ⟨q, -, rfl⟩ := List.mem_map.mp hφ
    have h := holds_mem (c := c) fun t => t.1 = q && t.2.1 = v
    refine Form.and_mem (Form.mem_mono (holds_mem _) (Nat.zero_le 1)) ⟨?_, ?_, ?_⟩
    · rw [Form.past_exBefore]; exact h.1
    · rw [Form.pnpFree_exBefore]; exact h.2.1
    · rw [Form.depth_exBefore]; exact Nat.succ_le_succ h.2.2

/-- **`answers v` holds at a row exactly when `v` answers its query**, by the
appendix's relation `PriorAnswer`: some earlier row has the query as its key
and `v` as its value. -/
theorem sat_answers {n : ℕ} (x : Zoology.MQARInstance n c) (i : Fin n) (v : Fin c) :
    (answers v).sat (word x) (i.val + 1) = true ↔ Zoology.PriorAnswer x i v := by
  have hget : ∀ j : Fin n, (word x)[j.val]? = some (x.key j, x.value j, x.query j) :=
    fun j => by simp [word]
  simp only [answers, Form.sat_any, List.mem_map, List.mem_finRange, true_and]
  constructor
  · rintro ⟨_, ⟨q, rfl⟩, hsat⟩
    simp only [Form.sat, Bool.and_eq_true, sat_holds, Form.sat_exBefore] at hsat
    obtain ⟨⟨t, ht, hq⟩, j, hj₁, hj₂, t', ht', hkv⟩ := hsat
    rw [Nat.add_sub_cancel, hget i, Option.some.injEq] at ht
    have hj : j - 1 < n := by omega
    rw [show j - 1 = (⟨j - 1, hj⟩ : Fin n).val from rfl, hget, Option.some.injEq] at ht'
    subst ht ht'
    rw [decide_eq_true_eq] at hq
    rw [decide_eq_true_eq, decide_eq_true_eq] at hkv
    exact ⟨⟨j - 1, hj⟩, Fin.mk_lt_of_lt_val (by omega), hkv.1.trans hq.symm, hkv.2⟩
  · rintro ⟨j, hji, hkey, hval⟩
    refine ⟨_, ⟨x.query i, rfl⟩, ?_⟩
    simp only [Form.sat, Bool.and_eq_true, sat_holds, Form.sat_exBefore]
    refine ⟨⟨_, by rw [Nat.add_sub_cancel]; exact hget i, by simp⟩,
      j.val + 1, by omega, by have := Fin.lt_def.mp hji; omega, _, by
        rw [Nat.add_sub_cancel]; exact hget j, by simp [hkey, hval]⟩

/-- `Form.any` counts what its members count. -/
theorem countSubs_any {σ : Type*} (L : List (Form σ)) :
    (Form.any L).countSubs = L.flatMap Form.countSubs := by
  induction L with
  | nil => simp [Form.any, Form.countSubs, Term.countSubs]
  | cons φ L ih => simp [Form.any, Form.or, Form.countSubs, ← ih]

/-- A test of depth `0` counts nothing. -/
theorem countSubs_holds (p : Fin c × Fin c × Fin c → Bool) : (holds p).countSubs = [] := by
  simp [holds, countSubs_any, List.flatMap_map, Form.countSubs]

/-- **`answers v` counts only the `c` formulas `key = q ∧ value = v`**: over
all `c` values, the `c²` pairs. -/
theorem countSubs_answers (v : Fin c) :
    ∀ ψ ∈ (answers v).countSubs, ∃ q, ψ = holds fun t => t.1 = q && t.2.1 = v := by
  intro ψ hψ
  simp only [answers, countSubs_any, List.flatMap_map, List.mem_flatMap, List.mem_finRange,
    true_and] at hψ
  obtain ⟨q, hq⟩ := hψ
  refine ⟨q, ?_⟩
  simpa [Form.countSubs, Form.exBefore, Form.or, Form.le, Term.countSubs, countSubs_holds] using hq

/-- The hypothesis of `countSubs_answers` is satisfiable: over one token, the
formula counts `key = 0 ∧ value = 0`. -/
example : (holds fun t : Fin 1 × Fin 1 × Fin 1 => t.1 = 0 && t.2.1 = 0) ∈
    (answers (0 : Fin 1)).countSubs := by
  simp [answers, countSubs_any, List.finRange, Form.countSubs, Form.exBefore, Form.or, Form.le,
    Term.countSubs, countSubs_holds]

/-- **A one-layer future-masked rounded transformer whose attention ignores
the query recognizes aligned recall over `c` tokens**: for each value `v`, one
accepts exactly the instances whose last query `v` answers
(`thm:TLCl_to_rtfr` of arXiv:2506.16055v3 applied to `answers v`, whose
transformer has uniform attention, `W_Q = W_K = 0`). -/
theorem exists_rtfr_answers (v : Fin c) :
    ∃ (p s d : ℕ) (T : RTfr (Option (Fin c × Fin c × Fin c)) p s d 1), T.QueryFree ∧
      ∀ {n : ℕ} (x : Zoology.MQARInstance (n + 1) c),
        T.Accepts (bos (word x)) ↔ Zoology.PriorAnswer x (Fin.last n) v := by
  refine ⟨_, _, _, TemporalProgram.model (answers v) 1, RTfr.queryFree_model _ _,
    fun {n} x => ?_⟩
  rw [TemporalProgram.model_recognizes _ 1 (answers_mem v), ← sat_answers x (Fin.last n) v]
  change (answers v).sat (word x) (word x).length = true ↔ _
  simp [word]

/-- An instance where the formula answers: the third row queries key `0`,
which the first row bound to value `1`. -/
example : (answers (1 : Fin 2)).sat [(0, 1, 1), (1, 0, 0), (0, 0, 0)] 3 = true := by decide

end CRASP
end Transformer
