/-
# Recall of the latest value needs positions

In no paper: the hard recall of the lab's basis (`python/src/lab/domain/basis.py`)
rebinds keys and answers a query with the value its key was bound to last
(`LatestAnswer`), which Zoology's Setup leaves open
(`Zoology.unrestricted_mqar_ambiguous`).  No future-masked rounded transformer
recalls it at every length over two values or more, whatever its depth, width
and precision (`not_recallsLatest`), while the matcher recalls a prior value of
the query in one layer of width five at every length (`matcher_answers`).

Over one key, written and queried at every row, the latest value at the last
row is the value of the row before it (`latestAnswer_oneKey_iff`): binding the
key by the bits of a string, the answer is whether the string lies in `Σ*bΣ`.
A transformer over the triples (key, value, query) read through that binding
(`RTfr.comap`) is one over `{a, b}`, which would then agree with `Σ*bΣ` on the
nonempty strings, against `not_models_iff_secondLast`.  Those instances rebind
the key to the value it holds, which the lab's generator never does; that recall
stays out of reach when every rebinding changes the value is not proved here.
-/

import Transformer.CRASP.MatchingRecallAnswers
import Transformer.CRASP.SecondLast

namespace Transformer
namespace CRASP

variable {σ τ : Type*} {c p s d k : ℕ}

/-- `T` reading `τ` through `g`: the embedding of `t` is that of `g t`, and the
layers and the output are those of `T`. -/
def RTfr.comap (T : RTfr σ p s d k) (g : τ → σ) : RTfr τ p s d k := { T with E := T.E ∘ g }

/-- Reading through `g` is reading the image under `g`. -/
theorem RTfr.out_comap (T : RTfr σ p s d k) (g : τ → σ) (w : List τ) :
    (T.comap g).out w = T.out (w.map g) := by
  have hl : (w.map g).length = w.length := List.length_map g
  have hc {n m : ℕ} (hnm : n = m) (ℓ : ℕ) (h : Fin m → Fin d → Fx p s) :
      T.layer ℓ (fun i => h (Fin.cast hnm i)) = fun i => T.layer ℓ h (Fin.cast hnm i) := by
    subst hnm
    rfl
  have hact (ℓ : ℕ) : T.act (w.map g) ℓ = fun i => (T.comap g).act w ℓ (Fin.cast hl i) := by
    induction ℓ with
    | zero => exact funext fun i => funext fun c => by simp [RTfr.act, RTfr.comap]
    | succ ℓ ih => exact (congrArg (T.layer ℓ) ih).trans (hc hl ℓ _)
  unfold RTfr.out
  rw [hact]
  by_cases hw : 0 < w.length
  · rw [dite_eq_left hw, dite_eq_left (hl ▸ hw)]
    exact congrArg T.Wout (congrArg ((T.comap g).act w k) (Fin.ext (by simp)))
  · rw [dite_eq_right hw, dite_eq_right (hl ▸ hw)]

/-- Reading `⊲ · w` through `g`, fixing `⊲`, is reading `⊲ · g(w)`. -/
theorem RTfr.accepts_comap_bos (T : RTfr (Option σ) p s d k) (g : τ → σ) (w : List τ) :
    (T.comap (Option.map g)).Accepts (bos w) ↔ T.Accepts (bos (w.map g)) := by
  rw [RTfr.Accepts, RTfr.Accepts, RTfr.out_comap]
  simp [bos, Function.comp_def]

/-- The value bound last to the query of row `i`: an earlier row `j` binds the
query to `v`, and no row between binds the query again.  The lab's recall
settles a rebinding so (`python/src/lab/domain/tasks.py`, `MQAR` with
`overwrites`); Zoology's `PriorAnswer` accepts any earlier binding. -/
def LatestAnswer {n : ℕ} (x : Zoology.MQARInstance n c) (i : Fin n) (v : Fin c) : Prop :=
  ∃ j : Fin n, j < i ∧ x.key j = x.query i ∧ x.value j = v ∧
    ∀ j' : Fin n, j < j' → j' < i → x.key j' ≠ x.query i

/-- **On a consistent dictionary the latest value is the prior one**: the two
recalls differ only where a key is rebound.  Of the earlier rows holding the
query the latest exists when one does, and consistency gives it that value. -/
theorem latestAnswer_iff_priorAnswer {n : ℕ} {x : Zoology.MQARInstance n c}
    (hx : Zoology.Consistent x) (i : Fin n) (v : Fin c) :
    LatestAnswer x i v ↔ Zoology.PriorAnswer x i v := by
  refine ⟨fun ⟨j, hj, hk, hv, _⟩ => ⟨j, hj, hk, hv⟩, fun ⟨j, hj, hk, hv⟩ => ?_⟩
  classical
  let S := Finset.univ.filter fun j' : Fin n => j' < i ∧ x.key j' = x.query i
  have hS : S.Nonempty := ⟨j, Finset.mem_filter.mpr ⟨Finset.mem_univ _, hj, hk⟩⟩
  obtain ⟨-, hj₀, hk₀⟩ := Finset.mem_filter.mp (S.max'_mem hS)
  refine ⟨S.max' hS, hj₀, hk₀, (hx _ _ (hk₀.trans hk.symm)).trans hv, fun j' h₁ h₂ hk' => ?_⟩
  exact (S.le_max' j' (Finset.mem_filter.mpr ⟨Finset.mem_univ _, h₂, hk'⟩)).not_gt h₁

/-- One key `v`, written and queried at every row, bound to `v` where `b` holds
and to `v'` elsewhere. -/
def oneKey {n : ℕ} (v v' : Fin c) (b : Fin n → Bool) : Zoology.MQARInstance n c where
  key _ := v
  value j := if b j then v else v'
  query _ := v

/-- The hypothesis of `latestAnswer_iff_priorAnswer` is satisfiable: one key
bound to its one value. -/
example : LatestAnswer (oneKey (0 : Fin 1) 0 fun _ : Fin 2 => true) 1 0 ↔
    Zoology.PriorAnswer (oneKey (0 : Fin 1) 0 fun _ : Fin 2 => true) 1 0 :=
  latestAnswer_iff_priorAnswer (fun _ _ _ => Subsingleton.elim _ _) 1 0

theorem word_oneKey {n : ℕ} (v v' : Fin c) (b : Fin n → Bool) :
    word (oneKey v v' b) = (List.ofFn b).map fun t => (v, if t then v else v', v) := by
  simp [word, oneKey, Function.comp_def]

/-- **Over one key the latest value is the second-to-last bit.**  The instance
binding `v` by the bits of `w` recalls `v` last at its last row iff
`w ∈ Σ*bΣ`. -/
theorem latestAnswer_oneKey_iff {n : ℕ} {v v' : Fin c} (hv : v' ≠ v) (w : List Bool)
    (hn : w.length = n + 1) :
    LatestAnswer (oneKey v v' fun j : Fin (n + 1) => w[j.1]) (Fin.last n) v ↔ w ∈ secondLast := by
  constructor
  · rintro ⟨j, hj, -, hval, hlast⟩
    have h₁ : j.1 < n := hj
    have hjn : j.1 + 1 = n := by
      by_contra hne
      exact hlast ⟨j.1 + 1, by omega⟩ (Fin.lt_def.mpr (Nat.lt_succ_self _))
        (Fin.lt_def.mpr (by simp; omega)) rfl
    have hb : w[j.1] = true := by
      by_contra hb
      exact hv (by simpa [oneKey, hb] using hval)
    have e := eq_take_append_pair (w := w) (m := j.1) (by omega)
    rw [hb] at e
    exact ⟨_, _, e⟩
  · rintro ⟨u, x, rfl⟩
    simp only [List.length_append, List.length_cons, List.length_nil] at hn
    refine ⟨⟨u.length, by omega⟩, Fin.lt_def.mpr (by simp; omega), rfl, by simp [oneKey],
      fun j' h₁ h₂ => absurd (Fin.lt_def.mp h₂) ?_⟩
    have := Fin.lt_def.mp h₁
    simp only [Fin.val_last] at this ⊢
    omega

/-- The hypotheses of `latestAnswer_oneKey_iff` are satisfiable: `ba` over two
values. -/
example : LatestAnswer (oneKey (0 : Fin 2) 1 fun j : Fin (1 + 1) => [true, false][j.1])
    (Fin.last 1) 0 :=
  (latestAnswer_oneKey_iff (by decide) [true, false] rfl).2 ⟨[], false, rfl⟩

/-- **No future-masked rounded transformer recalls the latest value** at every
length over two values or more, whatever its depth, width and precision: read
through one key bound by bits (`latestAnswer_oneKey_iff`), it would be a
transformer over `{a, b}` agreeing with `Σ*bΣ` on the nonempty strings, which
by `thm:rtfr_to_TLCl` of arXiv:2506.16055v3 no formula of `TL[◁#]` does
(`not_models_iff_secondLast`). -/
theorem not_recallsLatest (hc : 2 ≤ c) (v : Fin c)
    (T : RTfr (Option (Fin c × Fin c × Fin c)) p s d k) :
    ¬ ∀ n (x : Zoology.MQARInstance (n + 1) c),
      (T.Accepts (bos (word x)) ↔ LatestAnswer x (Fin.last n) v) := by
  intro hT
  have := Fin.nontrivial_iff_two_le.mpr hc
  obtain ⟨v', hv⟩ := exists_ne v
  obtain ⟨φ, hφ, hL⟩ := exists_mem_TLCl_of_rtfr
    (T.comap (Option.map fun t : Bool => (v, if t then v else v', v)))
  refine not_models_iff_secondLast k φ hφ fun w hw => ?_
  obtain ⟨n, hn⟩ : ∃ n, w.length = n + 1 :=
    ⟨w.length - 1, by have := List.length_pos_iff.mpr hw; omega⟩
  have hw' : List.ofFn (fun j : Fin (n + 1) => w[j.1]) = w :=
    List.ext_getElem (by simp [hn]) fun _ h _ => List.getElem_ofFn h
  rw [← latestAnswer_oneKey_iff hv w hn, ← hT n, word_oneKey, hw', ← RTfr.accepts_comap_bos]
  exact Set.ext_iff.1 hL w

/-- The matcher, which recalls a prior value at every length, recalls the
latest at none: the hypothesis of `not_recallsLatest` holds over two values. -/
example : (∀ n (x : Zoology.MQARInstance (n + 1) 2),
      ((matcher 0 : RTfr _ 5 0 5 1).Accepts (bos (word x)) ↔
        Zoology.PriorAnswer x (Fin.last n) 0)) ∧
    ¬ ∀ n (x : Zoology.MQARInstance (n + 1) 2),
      ((matcher 0 : RTfr _ 5 0 5 1).Accepts (bos (word x)) ↔ LatestAnswer x (Fin.last n) 0) :=
  ⟨fun _ x => matcher_answers (by norm_num) 0 x, not_recallsLatest le_rfl 0 _⟩

end CRASP
end Transformer
