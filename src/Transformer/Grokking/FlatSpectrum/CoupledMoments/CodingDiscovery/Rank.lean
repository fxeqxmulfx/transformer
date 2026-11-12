module

public import Transformer.Grokking.FlatSpectrum.CoupledMoments.CodingDiscovery.Gauge
public import Mathlib.LinearAlgebra.Matrix.Rank

/-!
# The test of the elimination is `rank + 1 = p`

Source: `review/flat_spectrum.md`, CM17, *The linear system*. Native reference:
`experiments/convex_stress_relabel/src/coding.rs` at 414431c7: `discover` puts the gauge row
`z[0] = 0` and the rows `z[a] + z[d] − z[b] − z[c]` of the anchored rectangles into an echelon
system over the field of `p` elements, and `unique_direction` accepts when `rank + 1 == p`.

`constraintMatrix p cell labels c₀` has the gauge row of the pinned class and, for every
quadruple of training samples, the row of its rectangle, or the zero row if the quadruple is no
rectangle. Its kernel is the space of the pinned codings (`pinnedCodes_eq_ker`), so by the
rank–nullity theorem its rank and the dimension of the pinned codings add up to the number of
classes (`rank_add_finrank_pinnedCodes`): the pinned codings have dimension one exactly if
`rank + 1 = p` (`finrank_pinnedCodes_eq_one_iff_rank`).

The matrix has more rows than the native system, which keeps the rectangles through the first
shared column of each pair of rows; these give the others (`rectangle_of_anchored`), so the row
space is the same. That the native elimination computes the rank of its rows is the usual
correctness of an echelon form and is not formalized.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.CoupledMoments.CodingDiscovery

open Matrix

variable {p : ℕ} {C n S : Type*} [Fintype C] [DecidableEq C] [DecidableEq n]

/-- **The linear system of the elimination** (CM17): the gauge row of the pinned class `c₀`,
and for every quadruple of samples the row `z[a] + z[d] − z[b] − z[c]` of its labels if it is a
rectangle, the zero row if it is not. -/
def constraintMatrix (p : ℕ) (cell : S → n × n) (labels : S → C) (c₀ : C) :
    Matrix (Option (S × S × S × S)) C (ZMod p)
  | none => Pi.single c₀ 1
  | some (t₁, t₂, t₃, t₄) =>
    if Rectangle cell t₁ t₂ t₃ t₄ then
      Pi.single (labels t₁) 1 + Pi.single (labels t₄) 1 - Pi.single (labels t₂) 1 -
        Pi.single (labels t₃) 1
    else 0

/-- The gauge row reads the pinned class. -/
lemma constraintMatrix_mulVec_none (cell : S → n × n) (labels : S → C) (c₀ : C)
    (z : C → ZMod p) : (constraintMatrix p cell labels c₀ *ᵥ z) none = z c₀ := by
  change (Pi.single c₀ 1 : C → ZMod p) ⬝ᵥ z = z c₀
  rw [single_dotProduct, one_mul]

/-- The row of a quadruple of samples reads its rectangle, or nothing. -/
lemma constraintMatrix_mulVec_some (cell : S → n × n) (labels : S → C) (c₀ : C)
    (z : C → ZMod p) (t₁ t₂ t₃ t₄ : S) :
    (constraintMatrix p cell labels c₀ *ᵥ z) (some (t₁, t₂, t₃, t₄)) =
      if Rectangle cell t₁ t₂ t₃ t₄ then
        z (labels t₁) + z (labels t₄) - z (labels t₂) - z (labels t₃) else 0 := by
  change (if Rectangle cell t₁ t₂ t₃ t₄ then
    (Pi.single (labels t₁) 1 + Pi.single (labels t₄) 1 - Pi.single (labels t₂) 1 -
      Pi.single (labels t₃) 1 : C → ZMod p) else 0) ⬝ᵥ z = _
  split_ifs
  · simp only [sub_dotProduct, add_dotProduct, single_dotProduct, one_mul]
  · exact zero_dotProduct z

/-- **The pinned codings are the kernel of the linear system of the elimination.** -/
theorem pinnedCodes_eq_ker (cell : S → n × n) (labels : S → C) (c₀ : C) :
    pinnedCodes p cell labels c₀ = LinearMap.ker (constraintMatrix p cell labels c₀).mulVecLin := by
  ext z
  rw [mem_pinnedCodes, LinearMap.mem_ker, mulVecLin_apply]
  constructor
  · rintro ⟨hz, h0⟩
    funext k
    rcases k with _ | ⟨t₁, t₂, t₃, t₄⟩
    · rw [constraintMatrix_mulVec_none, h0]
      rfl
    · rw [constraintMatrix_mulVec_some]
      split_ifs with h
      · have := hz t₁ t₂ t₃ t₄ h
        change _ = (0 : ZMod p)
        linear_combination this
      · rfl
  · intro h
    refine ⟨fun t₁ t₂ t₃ t₄ hr => ?_, ?_⟩
    · have := congrFun h (some (t₁, t₂, t₃, t₄))
      rw [constraintMatrix_mulVec_some, ite_eq_left hr] at this
      have h0 : z (labels t₁) + z (labels t₄) - z (labels t₂) - z (labels t₃) = 0 := this
      linear_combination h0
    · have := congrFun h none
      rwa [constraintMatrix_mulVec_none] at this

variable [Fact p.Prime]

/-- **The rank of the system and the dimension of the pinned codings add up to the number of
classes**: the rank–nullity theorem. -/
theorem rank_add_finrank_pinnedCodes (cell : S → n × n) (labels : S → C) (c₀ : C) :
    (constraintMatrix p cell labels c₀).rank +
      Module.finrank (ZMod p) (pinnedCodes p cell labels c₀) = Fintype.card C := by
  rw [pinnedCodes_eq_ker, Matrix.rank,
    LinearMap.finrank_range_add_finrank_ker (constraintMatrix p cell labels c₀).mulVecLin,
    Module.finrank_fintype_fun_eq_card]

/-- **The test of the elimination is `rank + 1 = p`** (CM17), when the classes are as many as
the residues, as a bijective coding needs. -/
theorem finrank_pinnedCodes_eq_one_iff_rank (cell : S → n × n) (labels : S → C) (c₀ : C)
    (hC : Fintype.card C = p) :
    Module.finrank (ZMod p) (pinnedCodes p cell labels c₀) = 1 ↔
      (constraintMatrix p cell labels c₀).rank + 1 = p := by
  have h := rank_add_finrank_pinnedCodes (p := p) cell labels c₀
  omega

/-! The witness: its system has rank four. -/

attribute [local instance] fact_prime_five

example : (constraintMatrix 5 fiveCells fiveLabels 0).rank + 1 = 5 :=
  (finrank_pinnedCodes_eq_one_iff_rank fiveCells fiveLabels 0 (ZMod.card 5)).1 (five_rank 0)

example (z : ZMod 5 → ZMod 5) : (constraintMatrix 5 fiveCells fiveLabels 0 *ᵥ z) none = z 0 :=
  constraintMatrix_mulVec_none fiveCells fiveLabels 0 z

/-- The row of the samples `0`, `1`, `4`, `5` of the witness, with the labels `1`, `0`, `0`,
`2`. -/
example (z : ZMod 5 → ZMod 5) :
    (constraintMatrix 5 fiveCells fiveLabels 0 *ᵥ z) (some (0, 1, 4, 5)) =
      z 1 + z 2 - z 0 - z 0 := by
  rw [constraintMatrix_mulVec_some, ite_eq_left (by decide)]
  rfl

end Transformer.Grokking.FlatSpectrum.CoupledMoments.CodingDiscovery
