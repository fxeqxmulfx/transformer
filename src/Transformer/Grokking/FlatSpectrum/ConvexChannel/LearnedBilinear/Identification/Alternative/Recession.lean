module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedBilinear.Identification.Basic
public import Mathlib.Analysis.Convex.Cone.Extension

/-!
# Homogeneous consequences of consistent observed unit margins

Source: `review/convex/moments.md`, CM87. A lower bound on a nonempty unit-margin
polyhedron is nonnegative on its recession directions. Homogenization then
preserves nonnegativity, including the zero scaling boundary. No unproved
identification result or attained optimizer is used.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedBilinear.Identification.Alternative

variable {P I : Type*} [AddCommGroup P] [Module ℝ P]

/-- A valid lower bound cannot decrease along any feasible recession ray,
CM87. The concrete feasible point excludes inconsistent-data vacuity. -/
theorem recession_nonneg (A : I → P →ₗ[ℝ] ℝ) (f : P →ₗ[ℝ] ℝ) (bound : ℝ)
    (p₀ : P) (h₀ : ∀ i, 1 ≤ A i p₀)
    (hforced : ∀ p, (∀ i, 1 ≤ A i p) → bound ≤ f p)
    (v : P) (hv : ∀ i, 0 ≤ A i v) : 0 ≤ f v := by
  by_contra hn
  have hn' : f v < 0 := lt_of_not_ge hn
  have hb := hforced p₀ h₀
  let r : ℝ := (f p₀ - bound + 1) / (-f v)
  have hr : 0 ≤ r := div_nonneg (by linarith) (by linarith)
  have hfit : ∀ i, 1 ≤ A i (p₀ + r • v) := by
    intro i
    rw [map_add, map_smul, smul_eq_mul]
    have hm := mul_nonneg hr (hv i)
    linarith [h₀ i]
  have h := hforced (p₀ + r • v) hfit
  rw [map_add, map_smul, smul_eq_mul] at h
  have he : r * f v = -(f p₀ - bound + 1) := by
    dsimp [r]
    field_simp [hn'.ne]
  linarith

/-- A positive scale reduces to the observed unit-margin hypothesis; zero
scale is handled by the recession theorem, CM87. -/
theorem homogeneous_nonneg (A : I → P →ₗ[ℝ] ℝ) (f : P →ₗ[ℝ] ℝ) (bound : ℝ)
    (p₀ : P) (h₀ : ∀ i, 1 ≤ A i p₀)
    (hforced : ∀ p, (∀ i, 1 ≤ A i p) → bound ≤ f p)
    (p : P) (t : ℝ) (ht : 0 ≤ t) (hp : ∀ i, t ≤ A i p) :
    0 ≤ f p - bound * t := by
  rcases ht.eq_or_lt with ht | ht
  · subst t
    simpa using recession_nonneg A f bound p₀ h₀ hforced p hp
  · have hfit : ∀ i, 1 ≤ A i (t⁻¹ • p) := by
      intro i
      rw [map_smul, smul_eq_mul]
      have h := mul_le_mul_of_nonneg_left (hp i) (inv_nonneg.mpr ht.le)
      simpa only [inv_mul_cancel₀ ht.ne'] using h
    have h := hforced (t⁻¹ • p) hfit
    rw [map_smul, smul_eq_mul] at h
    have h' := mul_le_mul_of_nonneg_left h ht.le
    rw [← mul_assoc, mul_inv_cancel₀ ht.ne', one_mul] at h'
    nlinarith

example : (0 : ℝ) ≤ 2 :=
  recession_nonneg (fun _ : Unit => LinearMap.id) LinearMap.id 1 1 (by simp)
    (by intro p hp; exact hp ()) 2 (by simp)

example : (0 : ℝ) ≤ 3 - 1 * 2 :=
  homogeneous_nonneg (fun _ : Unit => LinearMap.id) LinearMap.id 1 1 (by simp)
    (by intro p hp; exact hp ()) 3 2 (by norm_num) (by norm_num)

end Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedBilinear.Identification.Alternative
