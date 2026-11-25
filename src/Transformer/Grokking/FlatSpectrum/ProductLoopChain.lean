module

public import Transformer.Grokking.FlatSpectrum.ProductLoopDoubling

/-!
# The walk as one map: doubling meets the bound

Source: `review/flat_spectrum.md`, *Loops*, item LP3. The walk through the `2K` tables
`x₀, y₀, x₁, y₁, …` on the keys `γ` is one map on the keys `Fin (K + 1) × Bool × γ` (`chainMap`):
from `(t, false, c)` it follows `x t` to `(t, true, x t c)`, from `(t, true, c)` it follows `y t`
to `(t + 1, false, y t c)`, and level `K` is absorbing. After `2K` steps the start `(0, false, a)`
stands at `(K, false, w)`, with `w` the end of the walk (`chainMap_iterate`), and stays there.

An entry of the table of this map reads at most one of the `2K` tables, so a loop on it is a loop
on features of order one (`chainTable_mem_orderSpan`), and LP1's bound applies. Doubling answers
the walk after pass `n + 1` if and only if `1 + ⌈log₂ K⌉ ≤ n + 1` (`doubling_walk_iff`): it meets
the bound exactly. The scan from the one-hot start answers after pass `n + 1` if and only if
`2K ≤ n + 1` (`scan_walk_iff`), one pass more than the order allows a scan layer.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Finset

variable {γ : Type}

/-- **The walk as one map**: on level `t < K` the key `(t, false, c)` follows `x t` and the key
`(t, true, c)` follows `y t` to the next level; level `K` stands still. -/
def chainMap {K : ℕ} (x y : Fin K → γ → γ) (v : Fin (K + 1) × Bool × γ) :
    Fin (K + 1) × Bool × γ :=
  if h : (v.1 : ℕ) < K then
    if v.2.1 then (⟨v.1 + 1, by omega⟩, false, y ⟨v.1, h⟩ v.2.2) else (v.1, true, x ⟨v.1, h⟩ v.2.2)
  else v

lemma chainMap_lt {K : ℕ} (x y : Fin K → γ → γ) {t : Fin (K + 1)} (h : (t : ℕ) < K) (b : Bool)
    (c : γ) : chainMap x y (t, b, c) =
      if b then (⟨t + 1, by omega⟩, false, y ⟨t, h⟩ c) else (t, true, x ⟨t, h⟩ c) := by
  simp [chainMap, h]

lemma chainMap_last {K : ℕ} (x y : Fin K → γ → γ) (b : Bool) (c : γ) :
    chainMap x y (Fin.last K, b, c) = (Fin.last K, b, c) := by
  simp [chainMap]

/-- The walk through `K + 1` pairs above level `0` is the walk through the last `K`, one level
up. -/
lemma chainMap_succ {K : ℕ} (x y : Fin (K + 1) → γ → γ) (v : Fin (K + 1) × Bool × γ) :
    chainMap x y (Prod.map Fin.succ id v) =
      Prod.map Fin.succ id (chainMap (Fin.tail x) (Fin.tail y) v) := by
  obtain ⟨t, b, c⟩ := v
  show chainMap x y (t.succ, b, c) = _
  by_cases h : (t : ℕ) < K
  · rw [chainMap_lt x y (t := t.succ) (by simpa using h), chainMap_lt _ _ h]
    cases b <;> simp [Fin.tail, Fin.succ_mk]
  · have ht : t = Fin.last K := Fin.ext (by have := t.2; simp; omega)
    subst ht
    simp only [chainMap_last, Fin.succ_last]
    rfl

/-- **After `2K` steps the start stands at the end of the walk.** -/
theorem chainMap_iterate : ∀ (K : ℕ) (x y : Fin K → γ → γ) (a : γ),
    (chainMap x y)^[2 * K] (0, false, a) = (Fin.last K, false, pairWalk K x y a)
  | 0, _, _, _ => rfl
  | K + 1, x, y, a => by
    have hs : Function.Semiconj (Prod.map Fin.succ id) (chainMap (Fin.tail x) (Fin.tail y))
        (chainMap x y) := fun v => (chainMap_succ x y v).symm
    have h0 : ((0 : Fin (K + 2)) : ℕ) < K + 1 := by simp
    have h2 : (chainMap x y)^[2] (0, false, a) = Prod.map Fin.succ id (0, false, y 0 (x 0 a)) := by
      show chainMap x y (chainMap x y (0, false, a)) = _
      rw [chainMap_lt x y h0, ite_eq_right Bool.false_ne_true, chainMap_lt x y h0, ite_eq_left rfl]
      simp
    rw [show 2 * (K + 1) = 2 * K + 2 by ring, Function.iterate_add_apply, h2,
      ← hs.iterate_right (2 * K) (0, false, y 0 (x 0 a)), chainMap_iterate K]
    simp [pairWalk]

/-- From `2K` steps on, the start stays at the end of the walk. -/
lemma chainMap_iterate_of_le {K : ℕ} (x y : Fin K → γ → γ) (a : γ) {M : ℕ} (hM : 2 * K ≤ M) :
    (chainMap x y)^[M] (0, false, a) = (Fin.last K, false, pairWalk K x y a) := by
  obtain ⟨j, rfl⟩ : ∃ j, M = j + 2 * K := ⟨M - 2 * K, by omega⟩
  rw [Function.iterate_add_apply, chainMap_iterate]
  exact Function.iterate_fixed (chainMap_last x y _ _) j

/-- The level of a key: two per pair of tables behind it. -/
def chainLevel {K : ℕ} (v : Fin (K + 1) × Bool × γ) : ℕ := 2 * v.1 + if v.2.1 then 1 else 0

lemma chainLevel_iterate_le {K : ℕ} (x y : Fin K → γ → γ) (v : Fin (K + 1) × Bool × γ) :
    ∀ j, chainLevel ((chainMap x y)^[j] v) ≤ chainLevel v + j
  | 0 => le_rfl
  | j + 1 => by
    rw [Function.iterate_succ_apply']
    refine le_trans ?_ (Nat.succ_le_succ (chainLevel_iterate_le x y v j))
    obtain ⟨t, b, c⟩ := (chainMap x y)^[j] v
    by_cases h : (t : ℕ) < K
    · rw [chainMap_lt x y h]
      cases b
      · simp [chainLevel]
      · simp [chainLevel]
        omega
    · simp [chainMap, h]

variable [DecidableEq γ]

/-- **Every entry of the table of the walk reads at most one table.** -/
lemma chainTable_mem_orderSpan {K : ℕ} (u v : Fin (K + 1) × Bool × γ) :
    (fun x y : Fin K → γ → γ => funMatrix (chainMap x y) u v) ∈ orderSpan 1 := by
  obtain ⟨t, b, c⟩ := u
  by_cases h : (t : ℕ) < K
  · cases b
    · refine mem_orderSpan_of_readsOnly (S := {Sum.inl ⟨t, h⟩}) (by simp)
        fun x x' y y' hx _ => ?_
      simp [funMatrix, chainMap_lt _ _ h, hx ⟨t, h⟩ (mem_singleton_self _)]
    · refine mem_orderSpan_of_readsOnly (S := {Sum.inr ⟨t, h⟩}) (by simp)
        fun x x' y y' _ hy => ?_
      simp [funMatrix, chainMap_lt _ _ h, hy ⟨t, h⟩ (mem_singleton_self _)]
  · exact mem_orderSpan_of_readsOnly (S := ∅) (by simp) fun _ _ _ _ _ _ => by
      simp [funMatrix, chainMap, h]

variable [Fintype γ]

/-- The readout of the doubling loop: the entry of the written table from the start to the end
key `(K, false, a)`. -/
def chainEntry {K : ℕ} (a : γ) : (DoublingSlot (Fin (K + 1) × Bool × γ) → ℂ) →ₗ[ℂ] ℂ :=
  LinearMap.proj (R := ℂ) (φ := fun _ => ℂ) (Sum.inr ((0, false, a), (Fin.last K, false, a)))

/-- **Doubling meets the bound exactly**: the doubling loop on the table of the walk answers
whether `2K` tables lead from `a` back to `a` after pass `n + 1`, for all tables, if and only if
`1 + ⌈log₂ K⌉ ≤ n + 1`. -/
theorem doubling_walk_iff [Nontrivial γ] {K : ℕ} (hK : 0 < K) (a : γ) (n : ℕ) :
    (∀ x y : Fin K → γ → γ, chainEntry a (loopState doublingCore doublingRead
        (doublingState (funMatrix (chainMap x y)) 0) n) = if pairWalk K x y a = a then 1 else 0) ↔
      1 + Nat.clog 2 K ≤ n + 1 := by
  constructor
  · intro hval
    obtain ⟨b, hb⟩ := exists_ne a
    have h := le_passes_of_loop_walk hK (a := a) (b := b) (fun c => if c = a then 1 else 0)
      (by simp [hb]) (fun x y => doublingState (funMatrix (chainMap x y)) 0)
      (fun c => by
        rcases c with ⟨u, v⟩ | t
        · exact chainTable_mem_orderSpan u v
        · exact (orderSpan 1).zero_mem)
      (IsProductNet.id_net 0 0) (IsProductNet.of_layer isProductLayer_doublingCore) doublingRead
      (IsProductNet.id_net 0 0) (chainEntry a) n hval
    simpa using h
  · intro hn x y
    have h2 : 2 * K ≤ 2 ^ (n + 1) := by
      calc 2 * K ≤ 2 * 2 ^ Nat.clog 2 K := by
            have := Nat.le_pow_clog (by norm_num : 1 < 2) K
            omega
        _ = 2 ^ (1 + Nat.clog 2 K) := by ring
        _ ≤ 2 ^ (n + 1) := Nat.pow_le_pow_right (by norm_num) hn
    rw [loopState_doublingCore_funMatrix]
    simp [chainEntry, doublingState, funMatrix, chainMap_iterate_of_le x y a h2]

/-- **The scan pays one pass per table**: the scan from the one-hot start answers the same
question after pass `n + 1`, for all tables, if and only if `2K ≤ n + 1`. -/
theorem scan_walk_iff {K : ℕ} (a : γ) (n : ℕ) :
    (∀ x y : Fin K → γ → γ, (loopState scanCore scanRead (scanState (funMatrix (chainMap x y))
        (Pi.single (0, false, a) 1) 0) n) (Sum.inr (Sum.inr (Fin.last K, false, a))) =
          if pairWalk K x y a = a then 1 else 0) ↔ 2 * K ≤ n + 1 := by
  constructor
  · intro hval
    have h := hval (fun _ => id) (fun _ => id)
    have hw : pairWalk K (fun _ => id) (fun _ => id) a = a := by
      simpa using pairWalk_iterate id K (fun _ => id) (fun _ => id) (fun _ => 0) (fun _ => 0)
        (fun _ => rfl) (fun _ => rfl) a
    rw [loopState_scanCore_funMatrix, hw, ite_eq_left rfl] at h
    have he : (Fin.last K, false, a) = (chainMap (fun _ => id) (fun _ => id))^[n + 1]
        ((0 : Fin (K + 1)), false, a) := by
      simp only [scanState, Sum.elim_inr, Pi.single_apply] at h
      split_ifs at h with he
      · exact he
      · exact absurd h zero_ne_one
    have hl := chainLevel_iterate_le (fun _ => id) (fun _ => id) ((0 : Fin (K + 1)), false, a)
      (n + 1)
    rw [← he] at hl
    simpa [chainLevel] using hl
  · intro hn x y
    rw [loopState_scanCore_funMatrix, chainMap_iterate_of_le x y a hn]
    by_cases hw : pairWalk K x y a = a
    · simp [scanState, hw]
    · simp [scanState, hw, Ne.symm hw]

/-- **Eight links on `Bool`**: three passes of doubling answer, two do not. -/
example : (∀ x y : Fin 4 → Bool → Bool, chainEntry false (loopState doublingCore doublingRead
      (doublingState (funMatrix (chainMap x y)) 0) 2) =
        if pairWalk 4 x y false = false then 1 else 0) ∧
    ¬∀ x y : Fin 4 → Bool → Bool, chainEntry false (loopState doublingCore doublingRead
      (doublingState (funMatrix (chainMap x y)) 0) 1) =
        if pairWalk 4 x y false = false then 1 else 0 := by
  rw [doubling_walk_iff (K := 4) (by norm_num) false 2, doubling_walk_iff (K := 4) (by norm_num),
    show Nat.clog 2 4 = 2 by simpa using Nat.clog_pow 2 2 (by norm_num)]
  norm_num

end Transformer.Grokking.FlatSpectrum
