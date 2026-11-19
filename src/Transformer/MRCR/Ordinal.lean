module

public import Transformer.CRASP.AlignedRecall
public import Transformer.MRCR.Counting

/-!
# The ordinal counted from the start: a formula of depth two

Source: MRCR of arXiv:2409.12640v2, `eval_framework.tex`, lines 35–37: for a topic and a format
the context holds "a list of model outputs, in the order that they were presented in the
context", and when that list has more than one entry the query asks for "the entry with the
desired index" ("we may ask for the $5^{th}$ poem about penguins"); MRCR-V2 of
arXiv:2507.06261v6, Appendix, line 75 of `8_appendix/1_evaluation_additional_details.tex`, has
eight needles. The logic `TL[◁#]` and the rounded transformers are those of arXiv:2506.16055v3,
whose `thm:TLCl_to_rtfr` compiles a formula of depth `k` to a transformer of `k` layers.

A row carries a key, a value and a query (`Zoology.MQARInstance`) and an ordinal `o i < R`; as in
`Transformer.CRASP.exists_rtfr_answers`, an ask and its answer share one token. The row `i` asks
for the needle of its query with `o i` needles of that key before it: `v` answers it
(`NthAnswer`) when an earlier row `j` has the key `query i`, the value `v` and exactly `o i` rows
of that key before it. Over the string of the rows (`ordWord`) the formula

    ⋁_{q < c, r < R} (query = q ∧ ordinal = r ∧ ∃ j < i [key = q ∧ value = v ∧ #(key = q) = r + 1])

has depth two (`nthAnswers_mem`) and holds at a row exactly when `v` answers it
(`sat_nthAnswers`): the count `#(key = q)` at the row `j` counts the rows up to `j`, `j`
included (`val_countL_eq_card`). So a rounded transformer of two layers, whose attention ignores
the query, recognizes it at every length (`exists_rtfr_nthAnswers`): the first layer counts, the
second compares the count with the ordinal of the query.

MRCR counts its ordinal from the start. Counted from the end the task is out of reach already at
the last needle: no rounded transformer recognizes the latest binding of the query
(`Transformer.CRASP.LatestAnswer`) at every length (`Transformer.CRASP.not_recallsLatest`). One layer that scores the needles by content and
distance does not return an inner needle at every placement
(`Transformer.MRCR.exists_spaced_logScaleAttention_indicator_nonpos`).
-/

@[expose] public section

namespace Transformer.MRCR

open Finset Transformer.CRASP

variable {c R : ℕ}

/-- **The needle with the desired index** (MRCR, arXiv:2409.12640v2, `eval_framework.tex`): `v`
answers the row `i` at the ordinal `o i` when an earlier row has the key `query i`, the value `v`
and exactly `o i` rows of that key before it. -/
def NthAnswer {n : ℕ} (x : Zoology.MQARInstance n c) (o : Fin n → Fin R) (i : Fin n)
    (v : Fin c) : Prop :=
  ∃ j < i, x.key j = x.query i ∧ x.value j = v ∧
    (univ.filter fun j' => j' < j ∧ x.key j' = x.query i).card = o i

/-- The string of the rows: position `j + 1` carries the row `j` and its ordinal. -/
def ordWord {n : ℕ} (x : Zoology.MQARInstance n c) (o : Fin n → Fin R) :
    List ((Fin c × Fin c × Fin c) × Fin R) :=
  List.ofFn fun j => ((x.key j, x.value j, x.query j), o j)

/-- Every row with every ordinal below `R`. -/
def quads (c R : ℕ) : List ((Fin c × Fin c × Fin c) × Fin R) :=
  (triples c).flatMap fun t => (List.finRange R).map fun r => (t, r)

theorem mem_quads (a : (Fin c × Fin c × Fin c) × Fin R) : a ∈ quads c R :=
  List.mem_flatMap.2 ⟨a.1, mem_triples a.1, List.mem_map.2 ⟨a.2, List.mem_finRange _, rfl⟩⟩

/-- `⋁_{q, r} (query = q ∧ ordinal = r ∧ ∃ j < i [key = q ∧ value = v ∧ #(key = q) = r + 1])`:
an earlier row binds the query to `v`, with as many rows of that key up to it, itself included,
as the ordinal plus one. -/
def nthAnswers (v : Fin c) : Form ((Fin c × Fin c × Fin c) × Fin R) :=
  Form.any ((List.finRange c).flatMap fun q => (List.finRange R).map fun r =>
    .and (holdsIn (quads c R) fun a => a.1.2.2 = q && a.2 = r)
      (Form.exBefore (.and (holdsIn (quads c R) fun a => a.1.1 = q && a.1.2.1 = v)
        (Form.eq (.countL (holdsIn (quads c R) fun a => a.1.1 = q)) (Term.ofPos r)))))

/-- It is a formula of `TL[◁#]_2`. -/
theorem nthAnswers_mem (v : Fin c) : nthAnswers (R := R) v ∈ TLCl _ 2 :=
  Form.any_mem _ fun φ hφ => by
    obtain ⟨q, -, hq⟩ := List.mem_flatMap.mp hφ
    obtain ⟨r, -, rfl⟩ := List.mem_map.mp hq
    have h := Form.and_mem (Form.mem_mono (holdsIn_mem (quads c R)
      fun a => a.1.1 = q && a.1.2.1 = v) (Nat.zero_le 1))
      (eq_countL_ofPos_mem (holdsIn_mem (quads c R) fun a => a.1.1 = q) r)
    refine Form.and_mem (Form.mem_mono (holdsIn_mem _ _) (Nat.zero_le 2)) ⟨?_, ?_, ?_⟩
    · rw [Form.past_exBefore]; exact h.1
    · rw [Form.pnpFree_exBefore]; exact h.2.1
    · rw [Form.depth_exBefore]; exact Nat.succ_le_succ h.2.2

/-- **`nthAnswers v` holds at a row exactly when `v` answers it at its ordinal** (MRCR,
arXiv:2409.12640v2, `eval_framework.tex`). -/
theorem sat_nthAnswers {n : ℕ} (x : Zoology.MQARInstance n c) (o : Fin n → Fin R) (i : Fin n)
    (v : Fin c) : (nthAnswers v).sat (ordWord x o) (i.val + 1) = true ↔ NthAnswer x o i v := by
  have hget : ∀ j : Fin n, (ordWord x o)[j.val]? = some ((x.key j, x.value j, x.query j), o j) :=
    fun j => by simp [ordWord]
  have hψ (q : Fin c) (j : Fin n) : (holdsIn (quads c R) fun a => a.1.1 = q).sat (ordWord x o)
      (j.val + 1) = true ↔ x.key j = q := by
    rw [sat_holdsIn mem_quads, Nat.add_sub_cancel, hget]
    simp
  have hcount (q : Fin c) (j : Fin n) :
      (Term.countL (holdsIn (quads c R) fun a => a.1.1 = q)).val (ordWord x o) (j.val + 1) =
        (univ.filter fun j' => j' < j ∧ x.key j' = q).card + if x.key j = q then 1 else 0 := by
    rw [val_countL_succ, val_countL_eq_card _ _ (fun j' => x.key j' = q) (hψ q) j.val j.isLt.le]
    simp only [hψ q j, Fin.lt_def]
  simp only [nthAnswers, Form.sat_any, List.mem_flatMap, List.mem_map, List.mem_finRange,
    true_and]
  constructor
  · rintro ⟨_, ⟨q, r, rfl⟩, hsat⟩
    simp only [Form.sat, Bool.and_eq_true, sat_holdsIn mem_quads, Form.sat_exBefore,
      Form.sat_eq, decide_eq_true_eq, Term.val_ofPos] at hsat
    obtain ⟨⟨a, ha, hq, hr⟩, j, hj₁, hj₂, ⟨b, hb, hk, hv⟩, hcnt⟩ := hsat
    rw [Nat.add_sub_cancel, hget i, Option.some.injEq] at ha
    have hj : j - 1 < n := by omega
    rw [show j - 1 = (⟨j - 1, hj⟩ : Fin n).val from rfl, hget, Option.some.injEq] at hb
    subst ha hb
    rw [show j = (⟨j - 1, hj⟩ : Fin n).val + 1 by dsimp; omega, hcount, ite_eq_left hk] at hcnt
    refine ⟨⟨j - 1, hj⟩, Fin.mk_lt_of_lt_val (by omega), hk.trans hq.symm, hv, ?_⟩
    dsimp only at hq hr
    rw [hq, hr]
    omega
  · rintro ⟨j, hji, hk, hv, hc⟩
    refine ⟨_, ⟨x.query i, o i, rfl⟩, ?_⟩
    simp only [Form.sat, Bool.and_eq_true, sat_holdsIn mem_quads, Form.sat_exBefore,
      Form.sat_eq, decide_eq_true_eq, Term.val_ofPos]
    refine ⟨⟨_, by rw [Nat.add_sub_cancel]; exact hget i, rfl, rfl⟩, j.val + 1, by omega,
      by have := Fin.lt_def.mp hji; omega,
      ⟨_, by rw [Nat.add_sub_cancel]; exact hget j, hk, hv⟩, ?_⟩
    rw [hcount, ite_eq_left hk, hc]

/-- Three rows over two keys: the last asks for the second needle of the key `0`, whose value is
`0`. -/
example : (nthAnswers (R := 2) (0 : Fin 2)).sat [((0, 1, 0), 0), ((0, 0, 0), 0), ((1, 1, 0), 1)]
    3 = true := by decide

/-- **Two layers return the needle with the desired index** (MRCR, arXiv:2409.12640v2,
`eval_framework.tex`; `thm:TLCl_to_rtfr` of arXiv:2506.16055v3 applied to `nthAnswers v`): for
each value `v` a rounded transformer of two layers, whose attention ignores the query, accepts
exactly the strings of rows whose last row `v` answers at its ordinal, at every length. -/
theorem exists_rtfr_nthAnswers (v : Fin c) :
    ∃ (p s d : ℕ) (T : RTfr (Option ((Fin c × Fin c × Fin c) × Fin R)) p s d 2), T.QueryFree ∧
      ∀ {n : ℕ} (x : Zoology.MQARInstance (n + 1) c) (o : Fin (n + 1) → Fin R),
        T.Accepts (bos (ordWord x o)) ↔ NthAnswer x o (Fin.last n) v := by
  refine ⟨_, _, _, TemporalProgram.model (nthAnswers v) 2, RTfr.queryFree_model _ _,
    fun {n} x o => ?_⟩
  rw [TemporalProgram.model_recognizes _ 2 (nthAnswers_mem v),
    ← sat_nthAnswers x o (Fin.last n) v]
  change (nthAnswers v).sat (ordWord x o) (ordWord x o).length = true ↔ _
  simp [ordWord]

end Transformer.MRCR
