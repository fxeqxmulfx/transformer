module

public import Transformer.Landscape.Neural.Products
public import Mathlib.Topology.Instances.Matrix

/-!
# Finite genuine SiLU gates approach separating triangular features

Source: `review/flat_spectrum.md`, *Finite sequence capacity and information limits*, F2.
The limit is used only to prove existence of a finite invertible evaluation
matrix. The deployed activation remains SiLU; it is never replaced by ReLU.
-/

@[expose] public section
noncomputable section

namespace Transformer.SequenceModel.Attention.Finite

open Filter Landscape.Neural
open scoped Topology

/-- SiLU with its common gain divided out. -/
def softHinge (gain x : ℝ) : ℝ := x / (1 + Real.exp (-gain * x))

/-- F2: this is the actual SiLU activation at every nonzero finite gain. -/
theorem softHinge_silu (gain x : ℝ) (h : gain ≠ 0) :
    softHinge gain x = silu (gain * x) / gain := by
  unfold softHinge silu
  rw [neg_mul]
  field_simp

/-- F2: the odd part is exact even at zero gain. -/
theorem softHinge_sub_neg (gain x : ℝ) :
    softHinge gain x - softHinge gain (-x) = x := by
  by_cases h : gain = 0
  · simp [softHinge, h]; ring
  · rw [softHinge_silu gain x h, softHinge_silu gain (-x) h,
      ← sub_div, mul_neg, silu_sub_neg]
    exact mul_div_cancel_left₀ x h

/-- F2: a positive input converges to itself at large positive gain. -/
theorem tendsto_softHinge_pos (x : ℝ) (hx : 0 < x) :
    Tendsto (fun gain => softHinge gain x) atTop (𝓝 x) := by
  have ht : Tendsto (fun gain : ℝ => x * gain) atTop atTop :=
    (tendsto_const_mul_atTop_of_pos hx).mpr tendsto_id
  have he := Real.tendsto_exp_neg_atTop_nhds_zero.comp ht
  have h := (tendsto_const_nhds (x := x)).div (tendsto_const_nhds.add he)
    (by norm_num : (1 : ℝ) + 0 ≠ 0)
  simp only [add_zero, div_one] at h
  convert! h using 1
  ext gain
  simp only [softHinge, Function.comp_def, Pi.div_apply, neg_mul, mul_comm x]

/-- F2: the normalized SiLU family converges pointwise to the positive part. -/
theorem tendsto_softHinge (x : ℝ) :
    Tendsto (fun gain => softHinge gain x) atTop (𝓝 (max x 0)) := by
  rcases lt_trichotomy 0 x with hx | hx | hx
  · simpa only [max_eq_left hx.le] using tendsto_softHinge_pos x hx
  · subst x
    simp [softHinge]
  · have ht := (tendsto_softHinge_pos (-x) (by linarith)).add_const x
    have he (gain : ℝ) : softHinge gain (-x) + x = softHinge gain x := by
      linarith [softHinge_sub_neg gain x]
    simpa only [he, neg_add_cancel, max_eq_right hx.le] using ht

/-- Three finite gates forming a smooth localized feature. -/
def softTriangle (gain δ x : ℝ) : ℝ :=
  softHinge gain (x + δ) - 2 * softHinge gain x + softHinge gain (x - δ)

/-- The limiting piecewise-linear feature; this is not the model activation. -/
def triangle (δ x : ℝ) : ℝ := max (x + δ) 0 - 2 * max x 0 + max (x - δ) 0

/-- F2: three actual gates converge to a triangular feature. -/
theorem tendsto_softTriangle (δ x : ℝ) :
    Tendsto (fun gain => softTriangle gain δ x) atTop (𝓝 (triangle δ x)) := by
  exact ((tendsto_softHinge (x + δ)).sub ((tendsto_softHinge x).const_mul 2)).add
    (tendsto_softHinge (x - δ))

/-- F2: the limiting feature has a nonzero value at its center. -/
theorem triangle_zero {δ : ℝ} (hδ : 0 < δ) : triangle δ 0 = δ := by
  simp [triangle, max_eq_left hδ.le, max_eq_right (by linarith : -δ ≤ 0)]

/-- F2: separated centers give exact zero off-diagonal in the limiting
evaluation matrix; the finite-gain matrix will only need invertibility. -/
theorem triangle_outside {δ x : ℝ} (hδ : 0 < δ) (hx : δ ≤ |x|) : triangle δ x = 0 := by
  rcases le_total 0 x with h | h
  · rw [abs_of_nonneg h] at hx
    rw [triangle, max_eq_left (by linarith), max_eq_left h, max_eq_left (by linarith)]
    ring
  · rw [abs_of_nonpos h] at hx
    rw [triangle, max_eq_right (by linarith), max_eq_right h, max_eq_right (by linarith)]
    ring

/-- F2: a scalar feature is exactly one bias-free SwiGLU gate when a
positive common channel supplies the offset and value factor. -/
theorem hinge_swiglu (gain l u t : ℝ) (hg : gain ≠ 0) (hu : u ≠ 0) :
    u ^ 2 * softHinge (gain * u) (l / u - t) =
      swiglu (gain * (l - t * u)) u / gain := by
  have he : -(gain * u) * (l / u - t) = -(gain * (l - t * u)) := by
    field_simp
  unfold softHinge swiglu silu
  rw [he]
  field_simp

/-- F2: three genuine gates realize each finite-gain triangular feature
exactly, including the row-dependent common channel. -/
theorem triangle_swiglu (gain δ l u β : ℝ) (hg : gain ≠ 0) (hu : u ≠ 0) :
    (swiglu (gain * (l - (β - δ) * u)) u / gain -
        2 * (swiglu (gain * (l - β * u)) u / gain) +
        swiglu (gain * (l - (β + δ) * u)) u / gain) =
      u ^ 2 * softTriangle (gain * u) δ (l / u - β) := by
  rw [← hinge_swiglu gain l u (β - δ) hg hu,
    ← hinge_swiglu gain l u β hg hu, ← hinge_swiglu gain l u (β + δ) hg hu]
  have h₁ : l / u - (β - δ) = l / u - β + δ := by ring
  have h₂ : l / u - (β + δ) = l / u - β - δ := by ring
  rw [h₁, h₂, softTriangle]
  ring

example := softHinge_silu 10 (-3) (by norm_num)
example := tendsto_softHinge_pos (3 / 7) (by norm_num)
example := triangle_zero (δ := 1 / 10) (by norm_num)
example := triangle_outside (δ := 1 / 10) (x := -2) (by norm_num) (by norm_num)
example := hinge_swiglu 10 2 3 1 (by norm_num) (by norm_num)
example := triangle_swiglu 10 (1 / 10) 2 3 1 (by norm_num) (by norm_num)

end Transformer.SequenceModel.Attention.Finite
