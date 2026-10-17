/-
# Appendix A — the configurations staying near a saddle form a null set

Geshkovski, Letrouit, Polyanskiy, Rigollet — arXiv:2312.10794v5,
*A mathematical perspective on Transformers*, Appendix A, `l:nosaddleconv`.

`stayNear e i₀ r` is the set of initial configurations whose trajectory under
`e:gradfl` stays within `r` of the saddle `signedTuple e {i₀}` for all `t ≥ 0`:
the local center-stable set of the source.  Any two points of its closure have
`q(x - y) ≤ 0` (`Perspective.AppendixA_Cone`), while a perturbation of the
particle `i₀` alone has `q > 0`.  So the slices of the closure along `i₀` have
at most one point, and the closure is null for the uniform law
(`measure_closure_stayNear`).  A trajectory converging to the saddle of `e`
enters `stayNear e' i₀ r` at an integer time for every `e'` near `e`
(`exists_nat_mem_stayNear`), which is how the countably many local sets cover
the global one.
-/

import Transformer.Perspective.AppendixA_Cone
import Transformer.Perspective.UniformSlices

open Set Filter MeasureTheory
open scoped Topology

namespace Transformer
namespace Perspective

variable {d n : ℕ}

/-- The configurations whose trajectory under `e:gradfl` stays within `r` of the
saddle `signedTuple e {i₀}` for all `t ≥ 0`: the local center-stable set of
`l:nosaddleconv`. -/
def stayNear (e : SSphere d) (i₀ : Idx n) (r : ℝ) : Set (SphereTuple d n) :=
  {X₀ | ∃ X : ℝ → SphereTuple d n, X 0 = X₀ ∧ E0GradientAscent d n X ∧
    ∀ t : ℝ, 0 ≤ t → ∀ k : Idx n, ‖(X t k : EucSpace d) - signedTuple e {i₀} k‖ ≤ r}

/-- A time shift of a solution of `e:gradfl` solves `e:gradfl`. -/
theorem E0GradientAscent.comp_const_add {X : ℝ → SphereTuple d n}
    (hX : E0GradientAscent d n X) (T : ℝ) : E0GradientAscent d n fun s => X (T + s) :=
  fun t i => (hX (T + t) i).comp_const_add T t

/-- Two saddles of the same shape are as far apart, particle by particle, as
their axes. -/
theorem norm_signedTuple_sub (e e' : SSphere d) (S : Finset (Idx n)) (k : Idx n) :
    ‖(signedTuple e S k : EucSpace d) - signedTuple e' S k‖ = ‖(e : EucSpace d) - e'‖ := by
  rw [coe_signedTuple, coe_signedTuple]
  split_ifs
  · rw [neg_sub_neg, norm_sub_rev]
  · rfl

/-- **A trajectory converging to a saddle eventually stays near it.** If a
solution of `e:gradfl` converges to `signedTuple e {i₀}`, then for every `e'`
with `‖e - e'‖ < r / 2` some integer time `T` has `X T ∈ stayNear e' i₀ r`. -/
theorem exists_nat_mem_stayNear {X : ℝ → SphereTuple d n} (hX : E0GradientAscent d n X)
    (e : SSphere d) (i₀ : Idx n)
    (hlim : ∀ i : Idx n,
      Tendsto (fun t => (X t i : EucSpace d) - signedTuple e {i₀} i) atTop (𝓝 0))
    {r : ℝ} (e' : SSphere d) (he' : ‖(e : EucSpace d) - e'‖ < r / 2) :
    ∃ T : ℕ, X T ∈ stayNear e' i₀ r := by
  have hr : 0 < r / 2 := lt_of_le_of_lt (norm_nonneg _) he'
  have hev : ∀ᶠ t in atTop,
      ∀ i : Idx n, ‖(X t i : EucSpace d) - signedTuple e {i₀} i‖ < r / 2 :=
    eventually_all.mpr fun i => (hlim i).norm.eventually (gt_mem_nhds (by simpa using hr))
  obtain ⟨T₀, hT₀⟩ := eventually_atTop.mp hev
  refine ⟨⌈T₀⌉₊, fun s => X (⌈T₀⌉₊ + s), by simp, hX.comp_const_add _, fun t ht k => ?_⟩
  have h1 := hT₀ (⌈T₀⌉₊ + t) (by linarith [Nat.le_ceil T₀]) k
  have h2 := norm_sub_le_norm_sub_add_norm_sub (X (⌈T₀⌉₊ + t) k : EucSpace d)
    (signedTuple e {i₀} k) (signedTuple e' {i₀} k)
  rw [norm_signedTuple_sub] at h2
  linarith

/-- The hypotheses of `exists_nat_mem_stayNear` hold for the constant trajectory
at the saddle, `e' = e`, `r = 1`. -/
example (e : SSphere 2) : E0GradientAscent 2 3 (fun _ => signedTuple e {0}) ∧
    (∀ i : Idx 3, Tendsto (fun _ : ℝ => (signedTuple e {0} i : EucSpace 2)
      - signedTuple e {0} i) atTop (𝓝 0)) ∧ ‖(e : EucSpace 2) - e‖ < 1 / 2 :=
  ⟨E0GradientAscent_const _ (isCriticalE0_signedTuple e {0}),
    fun i => by simp, by norm_num⟩

/-- The cone form of the difference of two tuples depends continuously on them. -/
theorem continuous_coneForm_sub (e : EucSpace d) (i₀ : Idx n) :
    Continuous fun p : SphereTuple d n × SphereTuple d n =>
      coneForm e i₀ (fun k => (p.1 k : EucSpace d) - p.2 k) := by
  unfold coneForm coneW proj
  fun_prop

/-- The closure of `stayNear e i₀ r` stays within `r` of the saddle. -/
theorem closure_stayNear_subset (e : SSphere d) (i₀ : Idx n) (r : ℝ) :
    closure (stayNear e i₀ r) ⊆
      {x : SphereTuple d n | ∀ k : Idx n, ‖(x k : EucSpace d) - signedTuple e {i₀} k‖ ≤ r} := by
  refine closure_minimal ?_ ?_
  · rintro _ ⟨X, rfl, -, hXr⟩
    exact hXr 0 le_rfl
  · simp only [ofPred_forall]
    exact isClosed_iInter fun k => isClosed_le (by fun_prop) continuous_const

/-- **The closure of `stayNear` lies in the stable cone:** any two of its points
`x, y` have `q(x - y) ≤ 0`. -/
theorem coneForm_nonpos_of_mem_closure (hn : 3 ≤ n) (e : SSphere d) (i₀ : Idx n) {r : ℝ}
    (hr : 60 * (n : ℝ) * r ≤ 1) {x y : SphereTuple d n}
    (hx : x ∈ closure (stayNear e i₀ r)) (hy : y ∈ closure (stayNear e i₀ r)) :
    coneForm (e : EucSpace d) i₀ (fun k => (x k : EucSpace d) - y k) ≤ 0 := by
  have hsub : stayNear e i₀ r ×ˢ stayNear e i₀ r ⊆
      (fun p : SphereTuple d n × SphereTuple d n =>
        coneForm (e : EucSpace d) i₀ (fun k => (p.1 k : EucSpace d) - p.2 k)) ⁻¹' Iic 0 := by
    rintro ⟨x₀, y₀⟩ ⟨⟨X, hX0, hX, hXr⟩, ⟨Y, hY0, hY, hYr⟩⟩
    have h := coneForm_nonpos_of_stay hn e i₀ hr X Y hX hY hXr hYr
    rw [hX0, hY0] at h
    exact h
  have h := closure_minimal hsub
    (isClosed_Iic.preimage (continuous_coneForm_sub (e : EucSpace d) i₀))
  rw [closure_prod_eq] at h
  have h2 := h (mk_mem_prod hx hy)
  rwa [mem_preimage, mem_Iic] at h2

/-- The saddle itself stays near the saddle: it lies in `stayNear e i₀ r` for
every `r ≥ 0`. -/
theorem signedTuple_mem_stayNear (e : SSphere d) (i₀ : Idx n) {r : ℝ} (hr : 0 ≤ r) :
    signedTuple e {i₀} ∈ stayNear e i₀ r :=
  ⟨fun _ => signedTuple e {i₀}, rfl,
    E0GradientAscent_const _ (isCriticalE0_signedTuple e {i₀}), fun _ _ _ => by simpa using hr⟩

/-- The hypotheses of `coneForm_nonpos_of_mem_closure` hold for `x = y` the saddle,
`n = 3`, `r = 0`. -/
example (e : SSphere 2) : 3 ≤ 3 ∧ 60 * ((3 : ℕ) : ℝ) * 0 ≤ 1 ∧
    signedTuple e {(0 : Idx 3)} ∈ closure (stayNear e 0 0) :=
  ⟨le_rfl, by norm_num, subset_closure (signedTuple_mem_stayNear e 0 le_rfl)⟩

/-- **The slices of the closure of `stayNear` along the particle `i₀` have at
most one point:** two points of a slice differ only in the particle `i₀`, where
`q(x - y) = (n - 2)/n ‖Proj_e (x_{i₀} - y_{i₀})‖² ≤ 0` forces
`Proj_e (x_{i₀} - y_{i₀}) = 0`, and near the saddle the projection controls the
whole difference. -/
theorem subsingleton_slice_closure_stayNear (hn : 3 ≤ n) (e : SSphere d) (i₀ : Idx n)
    {r : ℝ} (hr : 60 * (n : ℝ) * r ≤ 1) (X : SphereTuple d n) :
    {x : SSphere d | Function.update X i₀ x ∈ closure (stayNear e i₀ r)}.Subsingleton := by
  intro x hx y hy
  have hN : (3 : ℝ) ≤ n := by exact_mod_cast hn
  have hr2 : r ≤ 1 / 2 := by nlinarith
  have hq := coneForm_nonpos_of_mem_closure hn e i₀ hr hx hy
  rw [coneForm_single (by omega) _ i₀ _ fun j hj => by simp [Function.update_of_ne hj]] at hq
  simp only [Function.update_self] at hq
  have hc : 0 < ((n : ℝ) - 2) / n := div_pos (by linarith) (by linarith)
  have ha : ‖proj d e ((x : EucSpace d) - y)‖ ^ 2 = 0 :=
    le_antisymm (by nlinarith) (sq_nonneg _)
  have h2 := norm_sub_le_two_mul_norm_proj e {i₀} hr2 (Function.update X i₀ x)
    (Function.update X i₀ y) i₀ (closure_stayNear_subset e i₀ r hx i₀)
    (closure_stayNear_subset e i₀ r hy i₀)
  simp only [Function.update_self] at h2
  rw [norm_eq_zero.mp (pow_eq_zero_iff two_ne_zero |>.mp ha), norm_zero, mul_zero] at h2
  exact Subtype.ext (sub_eq_zero.mp (norm_le_zero_iff.mp h2))

/-- The hypotheses of `subsingleton_slice_closure_stayNear` hold for `n = 3`,
`r = 0`. -/
example : 3 ≤ 3 ∧ 60 * ((3 : ℕ) : ℝ) * 0 ≤ 1 := ⟨le_rfl, by norm_num⟩

/-- **The closure of `stayNear` is null for the uniform law** (`d ≥ 2`, `n ≥ 3`,
`60 n r ≤ 1`): its slices along the particle `i₀` have at most one point. -/
theorem measure_closure_stayNear (hd : 2 ≤ d) (hn : 3 ≤ n) (P : Measure (SphereTuple d n))
    (hP : UniformTuple d n P) (e : SSphere d) (i₀ : Idx n) {r : ℝ}
    (hr : 60 * (n : ℝ) * r ≤ 1) : P (closure (stayNear e i₀ r)) = 0 :=
  measure_eq_zero_of_slices hd P hP isClosed_closure.measurableSet i₀
    (subsingleton_slice_closure_stayNear hn e i₀ hr)

/-- The dimension and size hypotheses of `measure_closure_stayNear` hold for
`d = 2`, `n = 3`, `r = 0`. -/
example : 2 ≤ 2 ∧ 3 ≤ 3 ∧ 60 * ((3 : ℕ) : ℝ) * 0 ≤ 1 := ⟨le_rfl, le_rfl, by norm_num⟩

end Perspective
end Transformer
