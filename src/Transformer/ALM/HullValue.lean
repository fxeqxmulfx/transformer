/-
# The value the walk hands back, wherever it starts

`Transformer.ALM.HullResolve` prices one end of the merge walk and
`Transformer.ALM.HullCost` the other, but they meet only in prose: `resolve`
was described at an arbitrary index of an arbitrary `argmaxSet`, and the
trichotomy at the index the search returns.

`argmaxSet_resolve` joins them, and joins them at *any* maximizer rather than
at `hullProbe`: which of the three cases holds decides what `resolve` returns
under `TieBreak::AVERAGE` — the payload of the line the search found when it
is the only maximizer, the componentwise mean of two payloads when the walk
merged the neighbour on one side.  `argmaxSet_resolveLatest` does the same for
the other tie-break mode, where the order the walk merges in would matter and
`scanCombined_comm` says it does not.

Stating them at an arbitrary maximizer is what lets
`Transformer.ALM.FloatResolve` re-use them at the index the *floating-point*
search returns, which at a tie need not be `hullProbe`.  The corollaries here
are the real-arithmetic case.

Source: `transformer_vm/attention/hull2d_cht.h`, lines 276-306 (the walk) and
70-83 (`resolve`).
-/

import Transformer.ALM.HullCost

namespace Transformer
namespace ALM

variable {n : ℕ}

/-! ### At an arbitrary maximizer -/

/-- **The whole query, value and all, under `AVERAGE`.**  For any index the
tie set contains — whatever arithmetic found it — the set is that index alone,
or it together with one neighbour, and `resolve` returns the corresponding
payload or mean. -/
theorem argmaxSet_resolve (K : ℕ → ℝ) (q : ℝ) (N : ℕ) (hstep : ∀ j, K j < K (j + 1))
    {b : ℕ} (hb : b ∈ argmaxSet K q N)
    (M : ℕ → Meta) (V : ℕ → ℝ × ℝ) (s : ℕ → ℤ) (hs : ∀ j, 0 ≤ s j)
    (hM : ∀ j, M j = Meta.empty.add (V j) (s j)) :
    (argmaxSet K q N = {b} ∧ (scanBest M b).resolveAverage = V b) ∨
      (argmaxSet K q N = {b, b + 1} ∧
        (scanCombined M b (b + 1)).resolveAverage
          = (((V b).1 + (V (b + 1)).1) / 2, ((V b).2 + (V (b + 1)).2) / 2)) ∨
      (argmaxSet K q N = {b - 1, b} ∧
        (scanCombined M b (b - 1)).resolveAverage
          = (((V b).1 + (V (b - 1)).1) / 2, ((V b).2 + (V (b - 1)).2) / 2)) := by
  have hlast : ∀ j, 0 ≤ (M j).lastSeq := by
    intro j
    rw [hM j]
    simp only [Meta.add, Meta.empty]
    exact le_max_of_le_left (hs j)
  rcases argmaxSet_trichotomy K q N hstep hb with h | h | h
  · refine Or.inl ⟨h, ?_⟩
    rw [scanBest_eq M b (hlast b), hM b]
    exact (Meta.resolve_eq_of_single (V b) (s b) (hs b)).1
  · exact Or.inr (Or.inl ⟨h, scanCombined_resolveAverage M b (b + 1) (V b) (V (b + 1))
      (s b) (s (b + 1)) (hs b) (hs (b + 1)) (hM b) (hM (b + 1))⟩)
  · exact Or.inr (Or.inr ⟨h, scanCombined_resolveAverage M b (b - 1) (V b) (V (b - 1))
      (s b) (s (b - 1)) (hs b) (hs (b - 1)) (hM b) (hM (b - 1))⟩)

/-- **And under `LATEST`.**  The neighbour is named rather than placed,
because once the sequence numbers differ — which for an append-only log they
always do — `scanCombined_comm` makes the side it lies on irrelevant. -/
theorem argmaxSet_resolveLatest (K : ℕ → ℝ) (q : ℝ) (N : ℕ) (hstep : ∀ j, K j < K (j + 1))
    {b : ℕ} (hb : b ∈ argmaxSet K q N)
    (M : ℕ → Meta) (V : ℕ → ℝ × ℝ) (s : ℕ → ℤ) (hs : ∀ j, 0 ≤ s j)
    (hsinj : Function.Injective s)
    (hM : ∀ j, M j = Meta.empty.add (V j) (s j)) :
    (argmaxSet K q N = {b} ∧ (scanBest M b).resolveLatest = V b) ∨
      ∃ c, c ≠ b ∧
        (argmaxSet K q N = {b, c} ∨ argmaxSet K q N = {c, b}) ∧
        (scanCombined M b c).resolveLatest = if s b < s c then V c else V b := by
  have hsingle : (scanBest M b).resolveLatest = V b := by
    have hlast : 0 ≤ (M b).lastSeq := by
      rw [hM b]
      simp only [Meta.add, Meta.empty]
      exact le_max_of_le_left (hs b)
    rw [scanBest_eq M b hlast, hM b]
    exact (Meta.resolve_eq_of_single (V b) (s b) (hs b)).2
  rcases argmaxSet_trichotomy K q N hstep hb with h | h | h
  · exact Or.inl ⟨h, hsingle⟩
  · exact Or.inr ⟨b + 1, by omega, Or.inl h,
      scanCombined_resolveLatest_of_ne M b (b + 1) (V b) (V (b + 1)) (s b) (s (b + 1))
        (hs b) (hs (b + 1)) (fun he => absurd (hsinj he) (by omega)) (hM b) (hM (b + 1))⟩
  · rcases Nat.eq_zero_or_pos b with hb0 | hbpos
    · refine Or.inl ⟨?_, hsingle⟩
      have hself : b - 1 = b := by omega
      rw [h, hself]
      exact Finset.insert_eq_self.mpr (Finset.mem_singleton_self b)
    · exact Or.inr ⟨b - 1, by omega, Or.inr h,
        scanCombined_resolveLatest_of_ne M b (b - 1) (V b) (V (b - 1)) (s b) (s (b - 1))
          (hs b) (hs (b - 1)) (fun he => absurd (hsinj he) (by omega)) (hM b) (hM (b - 1))⟩

/-- The hypotheses are satisfiable: aggregates built one value per line, in
insertion order, are what `HullMeta::add` produces, an append-only log numbers
its lines `0, 1, 2, …`, and that numbering is injective. -/
example : (∀ j : ℕ, (0 : ℤ) ≤ (j : ℤ)) ∧ Function.Injective (fun i : ℕ => (i : ℤ)) ∧
    ∀ j : ℕ, (fun i : ℕ => Meta.empty.add ((i : ℝ), 0) (i : ℤ)) j
      = Meta.empty.add ((fun i : ℕ => ((i : ℝ), (0 : ℝ))) j) ((fun i : ℕ => (i : ℤ)) j) :=
  ⟨fun j => Int.natCast_nonneg j, fun a b h => by simpa using h, fun _ => rfl⟩

/-! ### At the index the exact search returns -/

/-- **The real-arithmetic query, value and all.**  `argmaxSet_resolve` at
`hullProbe`, the index `HullHalf::query` starts its two loops from when the
breakpoints are computed exactly. -/
theorem hullQuery_resolve [Nonempty (Fin n)] (K : Fin n → ℝ) (q : ℝ)
    (M : ℕ → Meta) (V : ℕ → ℝ × ℝ) (s : ℕ → ℤ) (hs : ∀ j, 0 ≤ s j)
    (hM : ∀ j, M j = Meta.empty.add (V j) (s j)) :
    (argmaxSet (sortedKey K) q (keyCard K - 1) = {hullProbe K q} ∧
        (scanBest M (hullProbe K q)).resolveAverage = V (hullProbe K q)) ∨
      (argmaxSet (sortedKey K) q (keyCard K - 1)
          = {hullProbe K q, hullProbe K q + 1} ∧
        (scanCombined M (hullProbe K q) (hullProbe K q + 1)).resolveAverage
          = (((V (hullProbe K q)).1 + (V (hullProbe K q + 1)).1) / 2,
             ((V (hullProbe K q)).2 + (V (hullProbe K q + 1)).2) / 2)) ∨
      (argmaxSet (sortedKey K) q (keyCard K - 1)
          = {hullProbe K q - 1, hullProbe K q} ∧
        (scanCombined M (hullProbe K q) (hullProbe K q - 1)).resolveAverage
          = (((V (hullProbe K q)).1 + (V (hullProbe K q - 1)).1) / 2,
             ((V (hullProbe K q)).2 + (V (hullProbe K q - 1)).2) / 2)) :=
  argmaxSet_resolve (sortedKey K) q (keyCard K - 1) (sortedKey_lt_succ K)
    (hullProbe_mem_argmaxSet K q) M V s hs hM

/-- **The same query under `LATEST`.** -/
theorem hullQuery_resolveLatest [Nonempty (Fin n)] (K : Fin n → ℝ) (q : ℝ)
    (M : ℕ → Meta) (V : ℕ → ℝ × ℝ) (s : ℕ → ℤ) (hs : ∀ j, 0 ≤ s j)
    (hsinj : Function.Injective s)
    (hM : ∀ j, M j = Meta.empty.add (V j) (s j)) :
    (argmaxSet (sortedKey K) q (keyCard K - 1) = {hullProbe K q} ∧
        (scanBest M (hullProbe K q)).resolveLatest = V (hullProbe K q)) ∨
      ∃ c, c ≠ hullProbe K q ∧
        (argmaxSet (sortedKey K) q (keyCard K - 1) = {hullProbe K q, c} ∨
          argmaxSet (sortedKey K) q (keyCard K - 1) = {c, hullProbe K q}) ∧
        (scanCombined M (hullProbe K q) c).resolveLatest
          = if s (hullProbe K q) < s c then V c else V (hullProbe K q) :=
  argmaxSet_resolveLatest (sortedKey K) q (keyCard K - 1) (sortedKey_lt_succ K)
    (hullProbe_mem_argmaxSet K q) M V s hs hsinj hM

end ALM
end Transformer
