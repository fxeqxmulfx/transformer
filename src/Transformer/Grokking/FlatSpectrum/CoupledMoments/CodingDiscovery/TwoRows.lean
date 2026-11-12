module

public import Transformer.Grokking.FlatSpectrum.CoupledMoments.CodingDiscovery.Invariance

/-!
# Two rows that share `p − 1` labels leave only the affine gauges of the coding

Source: `review/flat_spectrum.md`, CM17, *An observable sufficient coverage*. Native reference:
`experiments/convex_stress_relabel/src/coding.rs` at 414431c7, `unique_direction`
(`rank + 1 == p`).

Let the training labels be those of a table `e⁻¹ (r i + s j)` for a bijection `e : C ≃ ZMod p`,
`p` prime, and phases `r`, `s`; the learner knows none of the three. Take two rows and `p − 1`
columns observed in both, such that the labels of the first row in these columns are distinct
and the two rows differ in one of them. All of this is read off the training set.

Then every coding `z` of the training rectangles is `α · e + β` (`affine_of_two_rows`). With
`w = z ∘ e⁻¹` and the difference `δ ≠ 0` of the phases of the two rows, the rectangles of the
two rows say that `w (u + δ) − w u` is one constant `κ` at the `p − 1` phases `u` of the shared
columns. The sum of `w (u + δ) − w u` over all `u` is zero and `p · κ = 0`, so the increment is
`κ` at the last phase as well; and a function on `ZMod p` whose increment over a nonzero step is
constant is affine (`affine_of_step`), as the step generates the group.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.CoupledMoments.CodingDiscovery

open Finset

variable {p : ℕ} [Fact p.Prime] {C n S : Type*}

/-- A function on `ZMod p`, `p` prime, whose increment over a nonzero step `δ` is one constant
`κ` is affine: the step generates the group. -/
lemma affine_of_step {w : ZMod p → ZMod p} {δ κ : ZMod p} (hδ : δ ≠ 0)
    (h : ∀ u, w (u + δ) = w u + κ) (v : ZMod p) : w v = κ * δ⁻¹ * v + w 0 := by
  have hk : ∀ k : ℕ, w ((k : ZMod p) * δ) = w 0 + (k : ZMod p) * κ := by
    intro k
    induction k with
    | zero => simp
    | succ k ih =>
      rw [Nat.cast_succ, add_mul, one_mul, h, ih]
      ring
  have hv := hk (v * δ⁻¹).val
  rw [ZMod.natCast_zmod_val, inv_mul_cancel_right₀ hδ] at hv
  rw [hv]
  ring

/-- **Two rows that share `p − 1` labels leave only affine gauges** (CM17). The samples `a k`
lie in one row and the samples `b k` in another, `a k` and `b k` in one column; the labels of
the `a k` are distinct, there are at least `p − 1` of them, and some `a k`, `b k` have different
labels. If the labels are those of a table `e⁻¹ (r i + s j)`, every coding of the training
rectangles is `α · e + β`. -/
theorem affine_of_two_rows (e : C ≃ ZMod p) (r s : n → ZMod p) {cell : S → n × n}
    {labels : S → C} (hreal : ∀ t, e (labels t) = r (cell t).1 + s (cell t).2)
    {K : Type*} [Fintype K] (a b : K → S)
    (hrowa : ∀ k k', (cell (a k)).1 = (cell (a k')).1)
    (hrowb : ∀ k k', (cell (b k)).1 = (cell (b k')).1)
    (hcol : ∀ k, (cell (a k)).2 = (cell (b k)).2)
    (hinj : Function.Injective fun k => labels (a k)) (hK : p - 1 ≤ Fintype.card K)
    (hdiff : ∃ k, labels (a k) ≠ labels (b k))
    {z : C → ZMod p} (hz : z ∈ rectangleCodes p cell labels) :
    ∃ α β : ZMod p, ∀ c, z c = α * e c + β := by
  obtain ⟨k₀, hk₀⟩ := hdiff
  obtain ⟨δ, hδdef⟩ : ∃ δ, δ = r (cell (b k₀)).1 - r (cell (a k₀)).1 := ⟨_, rfl⟩
  obtain ⟨κ, hκdef⟩ : ∃ κ, κ = z (labels (b k₀)) - z (labels (a k₀)) := ⟨_, rfl⟩
  have hshift : ∀ k, e (labels (b k)) = e (labels (a k)) + δ := by
    intro k
    rw [hreal, hreal, hrowb k k₀, hrowa k k₀, hcol k, hδdef]
    ring
  have hδ : δ ≠ 0 := fun h0 => hk₀ (e.injective (by rw [hshift k₀, h0, add_zero])).symm
  have hpair : ∀ k, z (labels (b k)) - z (labels (a k)) = κ := by
    intro k
    have h := hz (a k) (a k₀) (b k) (b k₀) ⟨hrowa k k₀, hrowb k k₀, hcol k, hcol k₀⟩
    linear_combination -h - hκdef
  -- The increment over `δ` is `κ` at the phases of the shared columns.
  obtain ⟨U, hUdef⟩ : ∃ U : Finset (ZMod p), U = univ.image fun k => e (labels (a k)) :=
    ⟨_, rfl⟩
  have hUcard : p - 1 ≤ U.card := by
    rw [hUdef, card_image_of_injective _
      (show Function.Injective fun k => e (labels (a k)) from e.injective.comp hinj), card_univ]
    exact hK
  have hmem : ∀ u ∈ U, z (e.symm (u + δ)) - z (e.symm u) = κ := by
    intro u hu
    rw [hUdef] at hu
    obtain ⟨k, -, rfl⟩ := mem_image.1 hu
    rw [← hshift k, e.symm_apply_apply, e.symm_apply_apply]
    exact hpair k
  have hsum : ∑ u : ZMod p, (z (e.symm (u + δ)) - z (e.symm u) - κ) = 0 := by
    rw [sum_sub_distrib, sum_sub_distrib, sum_const, card_univ, ZMod.card, nsmul_eq_mul,
      ZMod.natCast_self, zero_mul, sub_zero, sub_eq_zero]
    exact Equiv.sum_comp (Equiv.addRight δ) fun u => z (e.symm u)
  -- At most one phase is missing, and the sum gives the increment there.
  have hall : ∀ u, z (e.symm (u + δ)) = z (e.symm u) + κ := by
    intro u
    by_cases hu : u ∈ U
    · linear_combination hmem u hu
    · have hone : ∀ v ∈ (univ : Finset (ZMod p)), v ≠ u →
          z (e.symm (v + δ)) - z (e.symm v) - κ = 0 := by
        intro v _ hvu
        by_contra hv
        have hvU : v ∉ U := fun hvU => hv (sub_eq_zero.2 (hmem v hvU))
        have hsub : U ⊆ (univ.erase u).erase v := fun x hx =>
          mem_erase.2 ⟨fun hxv => hvU (hxv ▸ hx),
            mem_erase.2 ⟨fun hxu => hu (hxu ▸ hx), mem_univ x⟩⟩
        have hc := card_le_card hsub
        rw [card_erase_of_mem (mem_erase.2 ⟨hvu, mem_univ v⟩), card_erase_of_mem (mem_univ u),
          card_univ, ZMod.card] at hc
        have hp := (Fact.out : p.Prime).two_le
        omega
      have hsingle := sum_eq_single u hone fun h => absurd (mem_univ u) h
      rw [hsum] at hsingle
      linear_combination -hsingle
  refine ⟨κ * δ⁻¹, z (e.symm 0), fun c => ?_⟩
  have hc := affine_of_step (w := fun u => z (e.symm u)) hδ hall (e c)
  rwa [e.symm_apply_apply] at hc

/-! The witness: the rows `0` and `1` of the table share the columns `0`–`3`, with the four
distinct labels `1`, `0`, `2`, `3` in the row `0`. -/

local instance fact_prime_five : Fact (Nat.Prime 5) := ⟨by norm_num⟩

/-- In the witness the training rectangles leave only the affine gauges of its coding. -/
lemma five_affine {z : ZMod 5 → ZMod 5}
    (hz : z ∈ rectangleCodes 5 fiveCells fiveLabels) :
    ∃ α β : ZMod 5, ∀ c, z c = α * fiveCode c + β :=
  affine_of_two_rows fiveCode fivePhase fivePhase five_real
    (fun k : Fin 4 => Fin.castLE (by norm_num) k) (fun k : Fin 4 => Fin.natAdd 4 (k.castLE
      (by norm_num) : Fin 8)) (by decide) (by decide) (by decide) (by decide) (by decide)
    (by decide) hz

example := five_affine (mem_rectangleCodes_of_fit five_real)

example := affine_of_step (w := fun u : ZMod 5 => 4 * u + 1) (δ := 2) (κ := 3) (by decide)
  (by decide) 4

end Transformer.Grokking.FlatSpectrum.CoupledMoments.CodingDiscovery
