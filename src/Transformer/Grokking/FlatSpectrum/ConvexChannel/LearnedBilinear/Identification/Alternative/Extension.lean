module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedBilinear.Identification.Alternative.Recession
public import Mathlib.LinearAlgebra.Isomorphisms

/-!
# Positive extension of a functional on finitely many observed coordinates

Source: `review/flat_spectrum.md`, CM87. Apply Mathlib's M. Riesz extension
theorem to the range of a linear observation map. A strictly positive range
vector provides the required cofinality. A kernel argument constructs the
partial functional, so no well-definedness or identification is assumed.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedBilinear.Identification.Alternative

open scoped BigOperators

variable {P J : Type*} [AddCommGroup P] [Module ℝ P] [Fintype J]

/-- Nonnegativity along a linear observation map and a positive range vector
give a positive functional on every observation coordinate, CM87. -/
theorem positive_extension (B : P →ₗ[ℝ] (J → ℝ)) (f : P →ₗ[ℝ] ℝ)
    (hpos : ∀ p, 0 ≤ B p → 0 ≤ f p)
    (p₀ : P) (h₀ : ∀ j, 1 ≤ B p₀ j) :
    ∃ g : (J → ℝ) →ₗ[ℝ] ℝ, (∀ p, g (B p) = f p) ∧
      ∀ y, 0 ≤ y → 0 ≤ g y := by
  classical
  have hk : B.ker ≤ f.ker := by
    intro p hp
    have hb : B p = 0 := hp
    have h := hpos p (by rw [hb])
    have hn := hpos (-p) (by rw [map_neg, hb, neg_zero])
    rw [map_neg] at hn
    change f p = 0
    linarith
  let d : (J → ℝ) →ₗ.[ℝ] ℝ :=
    ⟨B.range, ((B.ker).liftQ f hk).comp B.quotKerEquivRange.symm.toLinearMap⟩
  have hd (p : P) : d ⟨B p, B.mem_range_self p⟩ = f p := by
    simp [d, LinearMap.quotKerEquivRange_symm_apply_image]
  have hnonneg : ∀ y : d.domain, (y : J → ℝ) ∈ PointedCone.positive ℝ (J → ℝ) →
      0 ≤ d y := by
    rintro ⟨y, p, rfl⟩ hy
    rw [hd]
    exact hpos p hy
  have hdense : ∀ y : J → ℝ, ∃ x : d.domain,
      (x : J → ℝ) + y ∈ PointedCone.positive ℝ (J → ℝ) := by
    intro y
    let r : ℝ := ∑ j, |y j|
    have hr : 0 ≤ r := Finset.sum_nonneg (fun j _ => abs_nonneg _)
    refine ⟨⟨B (r • p₀), B.mem_range_self _⟩, ?_⟩
    change ∀ j, 0 ≤ (B (r • p₀) + y) j
    intro j
    simp only [map_smul, Pi.add_apply, Pi.smul_apply, smul_eq_mul]
    have hj : |y j| ≤ r := Finset.single_le_sum (fun k _ => abs_nonneg (y k))
      (Finset.mem_univ j)
    have hm := mul_le_mul_of_nonneg_left (h₀ j) hr
    have ha := neg_abs_le (y j)
    nlinarith
  obtain ⟨g, hg, hgpos⟩ := riesz_extension (PointedCone.positive ℝ (J → ℝ)) d hnonneg hdense
  refine ⟨g, ?_, ?_⟩
  · intro p
    exact (hg ⟨B p, B.mem_range_self p⟩).trans (hd p)
  · intro y hy
    exact hgpos y hy

omit [Fintype J] in
/-- A finite positive functional has nonnegative coefficients on every
coordinate, CM87. The entire finite set is retained. -/
theorem positive_coordinate [DecidableEq J] (g : (J → ℝ) →ₗ[ℝ] ℝ)
    (hg : ∀ y, 0 ≤ y → 0 ≤ g y) (j : J) : 0 ≤ g (Pi.single j 1) := by
  apply hg
  intro k
  by_cases h : j = k <;> simp [h]

/-- Every finite linear functional is the sum of its full coordinate
coefficients, CM87. No positive or negative coefficient is omitted. -/
theorem coordinate_expansion [DecidableEq J] (g : (J → ℝ) →ₗ[ℝ] ℝ) (y : J → ℝ) :
    g y = ∑ j, y j * g (Pi.single j 1) := by
  have he : y = ∑ j, y j • Pi.single j 1 := by
    funext k
    simp [Finset.sum_apply, Pi.single_apply]
  calc
    g y = g (∑ j, y j • Pi.single j 1) := congrArg g he
    _ = _ := by simp only [map_sum, map_smul, smul_eq_mul]

example : ∃ g : (Unit → ℝ) →ₗ[ℝ] ℝ,
    (∀ p : Unit → ℝ, g p = p ()) ∧
      ∀ y, 0 ≤ y → 0 ≤ g y :=
  positive_extension (LinearMap.id : (Unit → ℝ) →ₗ[ℝ] (Unit → ℝ)) (LinearMap.proj ())
    (by intro p hp; exact hp ()) (fun _ => 1) (by simp)

example : 0 ≤ (LinearMap.proj () : (Unit → ℝ) →ₗ[ℝ] ℝ) (Pi.single () 1) :=
  positive_coordinate (LinearMap.proj ()) (by intro y hy; exact hy ()) ()

end Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedBilinear.Identification.Alternative
