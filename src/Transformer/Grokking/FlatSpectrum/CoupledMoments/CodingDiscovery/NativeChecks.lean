module

public import Transformer.Grokking.FlatSpectrum.CoupledMoments.CodingDiscovery.Witness

/-!
# The native checks beside the learner

Source: `review/flat_spectrum.md`, CM17, *The native checks*. Native reference at 414431c7:
`discover` in `experiments/convex_stress_relabel/src/coding.rs`, `propose` in
`experiments/convex_stress_relabel/phase_start/src/lib.rs` and `analyze` in
`experiments/convex_stress_relabel/graph_audit/src/graph.rs`.

The learner of `Learner` is stated by what the native checks compute. It differs from the
native code in these places.

* **The rectangles.** For two rows `row < other` the native elimination calls their first
  shared column the anchor and keeps the rectangles through it. For every rule that gives two
  such rows with a shared column a column that both show, the kept rectangles have the same
  solutions as all rectangles (`mem_rectangleCodes_iff_anchored`), when a cell has one label.
* **A repeated cell** is an error of the native code ("invalid or repeated categorical input
  cell"). The answer of the learner does not depend on repetitions (`discover_congr`).
* **A class that no sample shows** is refused by the native code before the elimination
  (`unobserved_output_classes`). The learner has no such answer: for `p ≠ 2` the pinned codings
  of such a training set do not have dimension one
  (`finrank_pinnedCodes_ne_one_of_unobserved`), and it answers `ambiguous`
  (`discover_eq_ambiguous_of_unobserved`).
* **A training set without a perfect matching** is refused by the native constructor
  (`NoPerfectMatching`), which admits the case in which CM13 gives one minimizer. The learner
  needs no matching to return the table: `discover_eq_accept` has no such hypothesis. The
  matching is a hypothesis of the uniqueness of the minimizer (`discovered_argmin`).
* **A composite vocabulary** is refused by the native code
  (`composite_vocabulary_unsupported`); the modulus of the learner is prime.

The elimination itself and the searches of the graph are not formalized.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.CoupledMoments.CodingDiscovery

section Anchored

variable {p : ℕ} {C n S : Type*} [LinearOrder n] {cell : S → n × n} {labels : S → C}

/-- **The rectangles that the native elimination keeps have the same solutions as all
rectangles** (CM17). The native loop takes two rows `row < other`, calls their first shared
column the anchor, and keeps the rectangle of the anchor and of every other shared column.
Here `anchor` is any rule under which two rows `row < other` that share a column are both
observed at their anchor column (`hanchor`), and a cell has one label (`hlab`), as in a native
input, which may not repeat a cell. -/
theorem mem_rectangleCodes_iff_anchored {z : C → ZMod p}
    (hlab : ∀ t t', cell t = cell t' → labels t = labels t') (anchor : n → n → n)
    (hanchor : ∀ t t', (cell t).1 < (cell t').1 → (cell t).2 = (cell t').2 →
      ∃ t₀ t₀', cell t₀ = ((cell t).1, anchor (cell t).1 (cell t').1) ∧
        cell t₀' = ((cell t').1, anchor (cell t).1 (cell t').1)) :
    z ∈ rectangleCodes p cell labels ↔ ∀ t₁ t₂ t₃ t₄, Rectangle cell t₁ t₂ t₃ t₄ →
      (cell t₁).1 < (cell t₃).1 → (cell t₂).2 = anchor (cell t₁).1 (cell t₃).1 →
      (cell t₁).2 ≠ (cell t₂).2 →
      z (labels t₁) + z (labels t₄) = z (labels t₂) + z (labels t₃) := by
  refine ⟨fun hz t₁ t₂ t₃ t₄ hr _ _ _ => hz t₁ t₂ t₃ t₄ hr, fun hz => ?_⟩
  -- the kept rectangles, and the degenerate ones that lie in the anchor column
  have hkept : ∀ t₁ t₂ t₃ t₄, Rectangle cell t₁ t₂ t₃ t₄ → (cell t₁).1 < (cell t₃).1 →
      (cell t₂).2 = anchor (cell t₁).1 (cell t₃).1 →
      z (labels t₁) + z (labels t₄) = z (labels t₂) + z (labels t₃) := by
    intro t₁ t₂ t₃ t₄ hr hrow ha
    by_cases hc : (cell t₁).2 = (cell t₂).2
    · obtain ⟨h12, h34, h13, h24⟩ := hr
      rw [hlab t₁ t₂ (Prod.ext h12 hc),
        hlab t₃ t₄ (Prod.ext h34 (h13.symm.trans (hc.trans h24)))]
    · exact hz t₁ t₂ t₃ t₄ hr hrow ha hc
  -- every rectangle of two rows in their order: the difference of two kept ones
  have hlt : ∀ t₁ t₂ t₃ t₄, Rectangle cell t₁ t₂ t₃ t₄ → (cell t₁).1 < (cell t₃).1 →
      z (labels t₁) + z (labels t₄) = z (labels t₂) + z (labels t₃) := by
    rintro t₁ t₂ t₃ t₄ ⟨h12, h34, h13, h24⟩ hrow
    obtain ⟨t₀, t₀', h0, h0'⟩ := hanchor t₁ t₃ hrow h13
    have hcol : (cell t₀).2 = (cell t₀').2 :=
      (congrArg Prod.snd h0).trans (congrArg Prod.snd h0').symm
    have hA := hkept t₁ t₀ t₃ t₀' ⟨(congrArg Prod.fst h0).symm, (congrArg Prod.fst h0').symm,
      h13, hcol⟩ hrow (congrArg Prod.snd h0)
    have hB := hkept t₂ t₀ t₄ t₀' ⟨h12.symm.trans (congrArg Prod.fst h0).symm,
      h34.symm.trans (congrArg Prod.fst h0').symm, h24, hcol⟩ (by rw [← h12, ← h34]; exact hrow)
      (by rw [← h12, ← h34]; exact congrArg Prod.snd h0)
    exact rectangle_of_anchored hA.symm hB.symm
  rintro t₁ t₂ t₃ t₄ ⟨h12, h34, h13, h24⟩
  rcases lt_trichotomy (cell t₁).1 (cell t₃).1 with h | h | h
  · exact hlt t₁ t₂ t₃ t₄ ⟨h12, h34, h13, h24⟩ h
  · rw [hlab t₃ t₁ (Prod.ext h.symm h13.symm),
      hlab t₄ t₂ (Prod.ext (h34.symm.trans (h.symm.trans h12)) h24.symm)]
    exact add_comm _ _
  · have hrev := hlt t₃ t₄ t₁ t₂ ⟨h34, h12, h13.symm, h24.symm⟩ h
    linear_combination -hrev

end Anchored

section Unobserved

variable {p : ℕ} [Fact p.Prime] {C n S : Type*} {cell : S → n × n} {labels : S → C}

/-- **A class that no training sample shows fails the test of the elimination** (CM17), for
`p ≠ 2`: the indicator of that class satisfies every rectangle and is no affine gauge of the
coding. The native code refuses such a training set before the elimination
(`unobserved_output_classes`). -/
theorem finrank_pinnedCodes_ne_one_of_unobserved (e : C ≃ ZMod p) (r s : n → ZMod p)
    (hreal : ∀ t, e (labels t) = r (cell t).1 + s (cell t).2) (c₀ : C) {c : C}
    (hc : ∀ t, labels t ≠ c) (hp : p ≠ 2) :
    Module.finrank (ZMod p) (pinnedCodes p cell labels c₀) ≠ 1 := by
  classical
  intro hrank
  have hmem : (Pi.single c 1 : C → ZMod p) ∈ rectangleCodes p cell labels :=
    fun t₁ t₂ t₃ t₄ _ => by
      rw [Pi.single_eq_of_ne (hc t₁), Pi.single_eq_of_ne (hc t₂), Pi.single_eq_of_ne (hc t₃),
        Pi.single_eq_of_ne (hc t₄)]
  obtain ⟨α, β, h⟩ := (finrank_pinnedCodes_eq_one_iff e r s hreal c₀).1 hrank _ hmem
  have htwo : (2 : ZMod p) ≠ 0 := fun h2 =>
    hp ((Nat.prime_dvd_prime_iff_eq (Fact.out : p.Prime) Nat.prime_two).1
      ((ZMod.natCast_eq_zero_iff 2 p).1 (by exact_mod_cast h2)))
  have hne : ∀ d : ZMod p, d ≠ 0 → e.symm (e c + d) ≠ c := fun d hd heq =>
    hd (by linear_combination e.symm_apply_eq.1 heq)
  have h1 := h (e.symm (e c + 1))
  have h2 := h (e.symm (e c + 2))
  have h0 := h c
  rw [Pi.single_eq_of_ne (hne 1 one_ne_zero), e.apply_symm_apply] at h1
  rw [Pi.single_eq_of_ne (hne 2 htwo), e.apply_symm_apply] at h2
  rw [Pi.single_eq_same] at h0
  have hα : α = 0 := by linear_combination h1 - h2
  exact (one_ne_zero : (1 : ZMod p) ≠ 0) (by linear_combination h0 - h1 - hα)

/-- **The learner refuses a training set that does not show every class** (CM17), for `p ≠ 2`:
its answer is `ambiguous`. -/
theorem discover_eq_ambiguous_of_unobserved (e : C ≃ ZMod p) (r s : n → ZMod p)
    (hreal : ∀ t, e (labels t) = r (cell t).1 + s (cell t).2) (c₀ : C) {c : C}
    (hc : ∀ t, labels t ≠ c) (hp : p ≠ 2) :
    discover (n := n) p cell labels c₀ = .ambiguous :=
  discover_eq_ambiguous e r s hreal c₀
    (finrank_pinnedCodes_ne_one_of_unobserved e r s hreal c₀ hc hp)

end Unobserved

/-! The witness: its first eight samples with the column `0` as the anchor of the rows `0` and
`1`, and its first two samples, which show two of the five classes. -/

attribute [local instance] fact_prime_five

/-- The rows `0` and `1` of the witness at the columns `0`–`3`, anchored at the column `0`. -/
example := mem_rectangleCodes_iff_anchored (p := 5) (z := fiveCode)
  (cell := fiveCells ∘ Fin.castLE (by norm_num : 8 ≤ 12))
  (labels := fiveLabels ∘ Fin.castLE (by norm_num : 8 ≤ 12)) (by decide) (fun _ _ => 0)
  (by decide)

/-- The first two samples of the witness: the cells `(0, 0)`, `(0, 1)` with the classes `1`,
`0`. -/
def twoCells : Fin 2 → Fin 5 × Fin 5 := fiveCells ∘ Fin.castLE (by norm_num)

/-- The labels of the first two samples of the witness. -/
def twoLabels : Fin 2 → ZMod 5 := fiveLabels ∘ Fin.castLE (by norm_num)

lemma two_real (t : Fin 2) :
    fiveCode (twoLabels t) = fivePhase (twoCells t).1 + fivePhase (twoCells t).2 :=
  five_real (Fin.castLE (by norm_num) t)

example : Module.finrank (ZMod 5) (pinnedCodes 5 twoCells twoLabels 0) ≠ 1 :=
  finrank_pinnedCodes_ne_one_of_unobserved fiveCode fivePhase fivePhase two_real 0 (c := 2)
    (by decide) (by norm_num)

/-- **The learner refuses the two samples**: the classes `2`, `3`, `4` are not shown. -/
theorem two_ambiguous : discover (n := Fin 5) 5 twoCells twoLabels 0 = .ambiguous :=
  discover_eq_ambiguous_of_unobserved fiveCode fivePhase fivePhase two_real 0 (c := 2)
    (by decide) (by norm_num)

end Transformer.Grokking.FlatSpectrum.CoupledMoments.CodingDiscovery
