module

public import Transformer.Grokking.FlatSpectrum.CoupledMoments.CodingDiscovery.TwoRows

/-!
# A nonzero coding of the training rectangles is the coding of the table up to its gauge

Source: `review/flat_spectrum.md`, CM17, *What the test gives*. Native
reference: `experiments/convex_stress_relabel/src/coding.rs` at 414431c7: `unique_direction`
accepts when `rank + 1 == p`, then `validate` checks that the direction is a bijection and
`consistent_edges` that it has phase potentials on the observed cells.

Let the training labels be those of a table `e⁻¹ (r i + s j)` for a bijection `e : C ≃ ZMod p`,
`p` prime, and phases `r`, `s`, none of them given to the learner. The learner computes the
space of the pinned codings and tests that its dimension is one. Then, for every nonzero pinned
coding `z`, whichever the elimination returns:

* `z = α · (e − e c₀)` with `α ≠ 0` (`exists_gauge`): the coding is found up to a factor and a
  constant, which do not change the table;
* `z` is a bijection of the classes (`bijective_of_mem_pinnedCodes`);
* `z` has row and column phases on the whole table, so on the training cells
  (`exists_phases_of_mem_pinnedCodes`).

The last two are what the native code checks after the elimination. Here they are proved: on a
table of this kind the two checks cannot fail, and the bijection is not assumed.

The test itself says that the rectangles leave only the affine gauges `α · e + β`
(`finrank_pinnedCodes_eq_one_iff`); it does not depend on the pinned class
(`finrank_pinnedCodes_eq_one_pin`), it holds under the coverage of `TwoRows`
(`finrank_pinnedCodes_eq_one_of_two_rows`), and the pinned space is never zero
(`finrank_pinnedCodes_ne_zero`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.CoupledMoments.CodingDiscovery

open Schatten

variable {p : ℕ} [Fact p.Prime] {C n S : Type*} {cell : S → n × n} {labels : S → C}

/-- The coding of the table, shifted to vanish at `c₀`, is a pinned coding. -/
lemma code_sub_mem_pinnedCodes (e : C ≃ ZMod p) (r s : n → ZMod p)
    (hreal : ∀ t, e (labels t) = r (cell t).1 + s (cell t).2) (c₀ : C) :
    (fun c => e c - e c₀) ∈ pinnedCodes p cell labels c₀ :=
  ⟨mem_rectangleCodes_of_fit (ρ := fun i => r i - e c₀) (σ := s) fun t => by
    rw [hreal t]
    ring, sub_self _⟩

/-- A bijection onto `ZMod p` shifted to vanish at `c₀` is not zero: it takes the value `1`. -/
lemma code_sub_ne_zero (e : C ≃ ZMod p) (c₀ : C) : (fun c => e c - e c₀) ≠ 0 := by
  intro h
  have h1 := congrFun h (e.symm (e c₀ + 1))
  simp at h1

/-- On a table of this kind the pinned codings are never zero alone: the elimination cannot
report that no coding exists. -/
lemma finrank_pinnedCodes_ne_zero (e : C ≃ ZMod p) (r s : n → ZMod p)
    (hreal : ∀ t, e (labels t) = r (cell t).1 + s (cell t).2) (c₀ : C) :
    Module.finrank (ZMod p) (pinnedCodes p cell labels c₀) ≠ 0 := by
  have : Finite C := Finite.of_equiv _ e.symm
  intro h0
  exact code_sub_ne_zero e c₀ ((Submodule.eq_bot_iff _).1 (Submodule.finrank_eq_zero.1 h0) _
    (code_sub_mem_pinnedCodes e r s hreal c₀))

/-- **A nonzero pinned coding is the coding of the table up to a factor** (CM17). The test of
the elimination, that the pinned codings have dimension one, gives for every nonzero pinned
coding `z` a factor `α ≠ 0` with `z = α · (e − e c₀)`. -/
theorem exists_gauge (e : C ≃ ZMod p) (r s : n → ZMod p)
    (hreal : ∀ t, e (labels t) = r (cell t).1 + s (cell t).2) (c₀ : C)
    (hrank : Module.finrank (ZMod p) (pinnedCodes p cell labels c₀) = 1)
    {z : C → ZMod p} (hz : z ∈ pinnedCodes p cell labels c₀) (hz0 : z ≠ 0) :
    ∃ α : ZMod p, α ≠ 0 ∧ ∀ c, z c = α * (e c - e c₀) := by
  have hv : (⟨_, code_sub_mem_pinnedCodes e r s hreal c₀⟩ : pinnedCodes p cell labels c₀) ≠ 0 :=
    fun h => code_sub_ne_zero e c₀ (congrArg Subtype.val h)
  obtain ⟨α, hα⟩ := (finrank_eq_one_iff_of_nonzero' _ hv).1 hrank ⟨z, hz⟩
  have hzα : ∀ c, z c = α * (e c - e c₀) := fun c =>
    (congrFun (congrArg Subtype.val hα) c).symm
  exact ⟨α, fun h0 => hz0 (funext fun c => by rw [hzα c, h0, zero_mul]; rfl), hzα⟩

/-- **A nonzero pinned coding is a bijection of the classes** (CM17): the native check of the
bijection cannot fail on a table of this kind. -/
theorem bijective_of_mem_pinnedCodes (e : C ≃ ZMod p) (r s : n → ZMod p)
    (hreal : ∀ t, e (labels t) = r (cell t).1 + s (cell t).2) (c₀ : C)
    (hrank : Module.finrank (ZMod p) (pinnedCodes p cell labels c₀) = 1)
    {z : C → ZMod p} (hz : z ∈ pinnedCodes p cell labels c₀) (hz0 : z ≠ 0) :
    Function.Bijective z := by
  obtain ⟨α, hα, hzα⟩ := exists_gauge e r s hreal c₀ hrank hz hz0
  have hfun : z = ⇑(e.trans ((Equiv.subRight (e c₀)).trans (Equiv.mulLeft₀ α hα))) :=
    funext fun c => by simp [hzα c]
  rw [hfun]
  exact Equiv.bijective _

/-- **A nonzero pinned coding has row and column phases on the whole table** (CM17), so at every
training sample: the native check of the observed cycles cannot fail on a table of this
kind. -/
theorem exists_phases_of_mem_pinnedCodes (e : C ≃ ZMod p) (r s : n → ZMod p)
    (hreal : ∀ t, e (labels t) = r (cell t).1 + s (cell t).2) (c₀ : C)
    (hrank : Module.finrank (ZMod p) (pinnedCodes p cell labels c₀) = 1)
    {z : C → ZMod p} (hz : z ∈ pinnedCodes p cell labels c₀) (hz0 : z ≠ 0) :
    ∃ ρ σ : n → ZMod p, (∀ i j, z (phaseLabels e r s i j) = ρ i + σ j) ∧
      ∀ t, z (labels t) = ρ (cell t).1 + σ (cell t).2 := by
  obtain ⟨α, -, hzα⟩ := exists_gauge e r s hreal c₀ hrank hz hz0
  refine ⟨fun i => α * (r i - e c₀), fun j => α * s j, fun i j => ?_, fun t => ?_⟩
  · rw [hzα, phaseLabels, e.apply_symm_apply]
    ring
  · rw [hzα, hreal t]
    ring

/-- **The test of the elimination says that the rectangles leave only the affine gauges**
(CM17): the pinned codings have dimension one exactly if every coding of the training
rectangles is `α · e + β`. -/
theorem finrank_pinnedCodes_eq_one_iff (e : C ≃ ZMod p) (r s : n → ZMod p)
    (hreal : ∀ t, e (labels t) = r (cell t).1 + s (cell t).2) (c₀ : C) :
    Module.finrank (ZMod p) (pinnedCodes p cell labels c₀) = 1 ↔
      ∀ z ∈ rectangleCodes p cell labels, ∃ α β : ZMod p, ∀ c, z c = α * e c + β := by
  constructor
  · intro hrank z hz
    by_cases hz0 : (fun c => z c - z c₀) = 0
    · refine ⟨0, z c₀, fun c => ?_⟩
      have h : z c - z c₀ = 0 := congrFun hz0 c
      linear_combination h
    · obtain ⟨α, -, hα⟩ := exists_gauge e r s hreal c₀ hrank
        (z := fun c => z c - z c₀) ⟨sub_const_mem_rectangleCodes hz _, sub_self _⟩ hz0
      exact ⟨α, z c₀ - α * e c₀, fun c => by linear_combination hα c⟩
  · intro haff
    have hv : (⟨_, code_sub_mem_pinnedCodes e r s hreal c₀⟩ : pinnedCodes p cell labels c₀) ≠ 0 :=
      fun h => code_sub_ne_zero e c₀ (congrArg Subtype.val h)
    rw [finrank_eq_one_iff_of_nonzero' _ hv]
    rintro ⟨w, hw⟩
    obtain ⟨α, β, hαβ⟩ := haff w hw.1
    have h0 : α * e c₀ + β = 0 := (hαβ c₀).symm.trans hw.2
    refine ⟨α, Subtype.ext (funext fun c => ?_)⟩
    change α * (e c - e c₀) = w c
    rw [hαβ c]
    linear_combination -h0

/-- The test of the elimination does not depend on the pinned class. -/
theorem finrank_pinnedCodes_eq_one_pin (e : C ≃ ZMod p) (r s : n → ZMod p)
    (hreal : ∀ t, e (labels t) = r (cell t).1 + s (cell t).2) (c₀ c₁ : C) :
    Module.finrank (ZMod p) (pinnedCodes p cell labels c₀) = 1 ↔
      Module.finrank (ZMod p) (pinnedCodes p cell labels c₁) = 1 :=
  (finrank_pinnedCodes_eq_one_iff e r s hreal c₀).trans
    (finrank_pinnedCodes_eq_one_iff e r s hreal c₁).symm

/-- **The coverage of `TwoRows` passes the test of the elimination** (CM17): two rows that
share `p − 1` columns with distinct labels, and differ in one of them. -/
theorem finrank_pinnedCodes_eq_one_of_two_rows (e : C ≃ ZMod p) (r s : n → ZMod p)
    (hreal : ∀ t, e (labels t) = r (cell t).1 + s (cell t).2) (c₀ : C)
    {K : Type*} [Fintype K] (a b : K → S)
    (hrowa : ∀ k k', (cell (a k)).1 = (cell (a k')).1)
    (hrowb : ∀ k k', (cell (b k)).1 = (cell (b k')).1)
    (hcol : ∀ k, (cell (a k)).2 = (cell (b k)).2)
    (hinj : Function.Injective fun k => labels (a k)) (hK : p - 1 ≤ Fintype.card K)
    (hdiff : ∃ k, labels (a k) ≠ labels (b k)) :
    Module.finrank (ZMod p) (pinnedCodes p cell labels c₀) = 1 :=
  (finrank_pinnedCodes_eq_one_iff e r s hreal c₀).2 fun _ hz =>
    affine_of_two_rows e r s hreal a b hrowa hrowb hcol hinj hK hdiff hz

/-! The witness: its pinned codings have dimension one, whatever the pinned class. -/

attribute [local instance] fact_prime_five

/-- The witness passes the test of the elimination: its rows `0` and `1` share four columns
with distinct labels. -/
lemma five_rank (c₀ : ZMod 5) :
    Module.finrank (ZMod 5) (pinnedCodes 5 fiveCells fiveLabels c₀) = 1 :=
  finrank_pinnedCodes_eq_one_of_two_rows fiveCode fivePhase fivePhase five_real c₀
    (fun k : Fin 4 => Fin.castLE (by norm_num) k) (fun k : Fin 4 => Fin.natAdd 4 (k.castLE
      (by norm_num) : Fin 8)) (by decide) (by decide) (by decide) (by decide) (by decide)
    (by decide)

example := (finrank_pinnedCodes_eq_one_iff fiveCode fivePhase fivePhase five_real 0).1
  (five_rank 0)

/-- The pinned coding of the witness that its own coding gives. -/
lemma five_mem : (fun c => fiveCode c - fiveCode 0) ∈ pinnedCodes 5 fiveCells fiveLabels 0 :=
  code_sub_mem_pinnedCodes fiveCode fivePhase fivePhase five_real 0

example := finrank_pinnedCodes_ne_zero fiveCode fivePhase fivePhase five_real 0
example := exists_gauge fiveCode fivePhase fivePhase five_real 0 (five_rank 0) five_mem
  (code_sub_ne_zero fiveCode 0)
example := bijective_of_mem_pinnedCodes fiveCode fivePhase fivePhase five_real 0 (five_rank 0)
  five_mem (code_sub_ne_zero fiveCode 0)
example := exists_phases_of_mem_pinnedCodes fiveCode fivePhase fivePhase five_real 0
  (five_rank 0) five_mem (code_sub_ne_zero fiveCode 0)
example := (finrank_pinnedCodes_eq_one_pin fiveCode fivePhase fivePhase five_real 0 3).1
  (five_rank 0)

end Transformer.Grokking.FlatSpectrum.CoupledMoments.CodingDiscovery
