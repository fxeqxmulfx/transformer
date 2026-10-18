/-
# Saddle avoidance — the local gradient inequality of `Lojasiewicz`

`Lojasiewicz` states the Łojasiewicz gradient inequality at a point `z` in
the form of the hypothesis `hlocal` of
`Landscape.Lojasiewicz.lojasiewicz_modulated_of_local_gradient_inequality`: for some
`0 < α < 1`, `k > 0` and an open `V ∋ z`, `|E y - E z| ^ α ≤ k ‖∇E(y)‖` for
every `y ∈ V`.  That form gives `IsLojasiewiczAt E z α` with `m = 1/k`
(`isLojasiewiczAt_of_forall_mem`), so it supplies the hypothesis `hloj` of
the convergence theorems of this directory at `z`
(`exists_isLojasiewiczAt_of_local_gradient_inequality`).  Every case of the
local inequality that `Lojasiewicz` proves therefore plugs in, the general
analytic case `Landscape.Lojasiewicz.analytic_gradient_inequality` included:
Łojasiewicz's theorem, degenerate critical points included.

`IsLojasiewiczAt` depends only on the germ of the function (`isLojasiewiczAt_congr`)
and linear isometries preserve it (`isLojasiewiczAt_comp_linearIsometryEquiv_symm_iff`),
so Łojasiewicz's theorem carries from `EucSpace N` to every finite-dimensional real
inner product space (`exists_isLojasiewiczAt_of_analyticAt`; for `hloj`,
`exists_isLojasiewiczAt_of_analyticOnNhd`).  In dimension at most one an elementary
proof gives it at the critical points (`exists_isLojasiewiczAt_of_analyticAt_low_dimension`).
-/

import Transformer.Landscape.SaddleAvoidance.Lojasiewicz
import Transformer.Landscape.Lojasiewicz.AnalyticMonomialCharts
import Transformer.Landscape.Lojasiewicz.LojasiewiczInnerProduct
import Transformer.Landscape.Lojasiewicz.OneDimensionalGradientInequality

open Filter
open scoped Topology

namespace Transformer.Landscape.SaddleAvoidance

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [CompleteSpace E]
  {f : E → ℝ} {z : E}

/-- **The local gradient inequality on an open set gives the Łojasiewicz
inequality**: if `|f y - f z| ^ a ≤ k ‖∇f(y)‖` for every `y` in an open set
`V ∋ z`, with `k > 0`, then `IsLojasiewiczAt f z a`, with `m = 1/k` and
`ε = 1`.  The hypothesis asks the inequality on both sides of the level
`f z`; `IsLojasiewiczAt` asks it only above. -/
theorem isLojasiewiczAt_of_forall_mem {a k : ℝ} {V : Set E} (hk : 0 < k) (hV : IsOpen V)
    (hzV : z ∈ V) (h : ∀ y ∈ V, |f y - f z| ^ a ≤ k * ‖gradient f y‖) :
    IsLojasiewiczAt f z a := by
  refine ⟨1 / k, by positivity, 1, one_pos, ?_⟩
  filter_upwards [hV.mem_nhds hzV] with y hy _ _
  calc 1 / k * |f y - f z| ^ a ≤ 1 / k * (k * ‖gradient f y‖) :=
        mul_le_mul_of_nonneg_left (h y hy) (by positivity)
    _ = ‖gradient f y‖ := by rw [← mul_assoc, one_div_mul_cancel hk.ne', one_mul]

/-- **The form `hlocal` of `Lojasiewicz` gives the form `hloj`.**  The
local gradient inequality at `z` as the hypothesis `hlocal` of
`Landscape.Lojasiewicz.lojasiewicz_modulated_of_local_gradient_inequality` states it
(Appendix D.1, `lem: loj`, of arXiv:2510.22026v2) gives an exponent
`0 ≤ a < 1` with `IsLojasiewiczAt f z a`, which is what the hypothesis
`hloj` of `ae_exists_tendsto_isLocalMin` asks at each critical point. -/
theorem exists_isLojasiewiczAt_of_local_gradient_inequality
    (h : ∃ alpha k : ℝ, ∃ V : Set E, 0 < alpha ∧ alpha < 1 ∧ 0 < k ∧ IsOpen V ∧ z ∈ V ∧
      ∀ y ∈ V, |f y - f z| ^ alpha ≤ k * ‖gradient f y‖) :
    ∃ a, 0 ≤ a ∧ a < 1 ∧ IsLojasiewiczAt f z a := by
  obtain ⟨alpha, k, V, ha0, ha1, hk, hV, hzV, hineq⟩ := h
  exact ⟨alpha, ha0.le, ha1, isLojasiewiczAt_of_forall_mem hk hV hzV hineq⟩

/-- **The Łojasiewicz inequality depends only on the germ of `f`**: a function equal
to `f` near `z` has the same value at `z` and, near `z`, the same gradient
(`Filter.EventuallyEq.gradient`), so it satisfies the inequality with the same `m`,
`ε` and exponent. -/
theorem IsLojasiewiczAt.congr_of_eventuallyEq {g : E → ℝ} {a : ℝ}
    (h : IsLojasiewiczAt f z a) (hfg : f =ᶠ[𝓝 z] g) : IsLojasiewiczAt g z a := by
  obtain ⟨m, hm, ε, hε, hev⟩ := h
  refine ⟨m, hm, ε, hε, ?_⟩
  have hz : f z = g z := hfg.eq_of_nhds
  filter_upwards [hev, hfg, hfg.gradient] with y hy hfy hgy
  rwa [hfy, hz, hgy] at hy

/-- Functions with the same germ at `z` satisfy the Łojasiewicz inequality at `z`
together, with the same exponent. -/
theorem isLojasiewiczAt_congr {g : E → ℝ} {a : ℝ} (hfg : f =ᶠ[𝓝 z] g) :
    IsLojasiewiczAt f z a ↔ IsLojasiewiczAt g z a :=
  ⟨fun h => h.congr_of_eventuallyEq hfg, fun h => h.congr_of_eventuallyEq hfg.symm⟩

section Isometry

variable {F : Type*} [NormedAddCommGroup F] [InnerProductSpace ℝ F] [CompleteSpace F]

/-- **Linear isometries preserve the Łojasiewicz inequality**: if it holds for `f`
at `z`, it holds for `f ∘ e⁻¹` at `e z` with the same `m`, `ε` and exponent, since
`∇(f ∘ e⁻¹)(y) = e ∇f(e⁻¹ y)` (`Landscape.Lojasiewicz.gradient_comp_linearIsometryEquiv_symm`)
has the norm of `∇f(e⁻¹ y)`. -/
theorem IsLojasiewiczAt.comp_linearIsometryEquiv_symm {a : ℝ} (h : IsLojasiewiczAt f z a)
    (e : E ≃ₗᵢ[ℝ] F) : IsLojasiewiczAt (f ∘ e.symm) (e z) a := by
  obtain ⟨m, hm, ε, hε, hev⟩ := h
  refine ⟨m, hm, ε, hε, ?_⟩
  filter_upwards [(e.symm.continuous.tendsto' _ _ (e.symm_apply_apply z)).eventually hev]
    with y hy
  rw [Landscape.Lojasiewicz.gradient_comp_linearIsometryEquiv_symm, LinearIsometryEquiv.norm_map]
  simpa only [Function.comp_apply, LinearIsometryEquiv.symm_apply_apply] using hy

/-- `f ∘ e⁻¹` satisfies the Łojasiewicz inequality at `e z` exactly when `f` does at
`z`, with the same exponent. -/
theorem isLojasiewiczAt_comp_linearIsometryEquiv_symm_iff {a : ℝ} (e : E ≃ₗᵢ[ℝ] F) :
    IsLojasiewiczAt (f ∘ e.symm) (e z) a ↔ IsLojasiewiczAt f z a := by
  refine ⟨fun h => ?_, fun h => h.comp_linearIsometryEquiv_symm e⟩
  simpa [Function.comp_def] using h.comp_linearIsometryEquiv_symm e.symm

end Isometry

/-- **Analytic functions in dimension at most one satisfy the Łojasiewicz
inequality at every critical point**, with an exponent `0 ≤ a < 1`,
degenerate critical points included.  The local inequality is
`Landscape.Lojasiewicz.analytic_gradient_inequality_low_dimension`. -/
theorem exists_isLojasiewiczAt_of_analyticAt_low_dimension {N : ℕ} (hN : N ≤ 1)
    {g : EucSpace N → ℝ} {w : EucSpace N} (hg : AnalyticAt ℝ g w) (hw : gradient g w = 0) :
    ∃ a, 0 ≤ a ∧ a < 1 ∧ IsLojasiewiczAt g w a :=
  exists_isLojasiewiczAt_of_local_gradient_inequality
    (Landscape.Lojasiewicz.analytic_gradient_inequality_low_dimension hN g w hg hw)

/-- **Łojasiewicz's theorem: analytic functions satisfy the Łojasiewicz
inequality at every point** of a finite-dimensional real inner product space, with
an exponent `0 ≤ a < 1`, degenerate critical points included
(`lojasiewicz1963propriete`).  On `EucSpace N` the local inequality is
`Landscape.Lojasiewicz.analytic_gradient_inequality`, from local uniformization; the
isometry `(stdOrthonormalBasis ℝ E).repr` onto `EucSpace (finrank ℝ E)` carries it to `E`
(`isLojasiewiczAt_comp_linearIsometryEquiv_symm_iff`), as in
`Landscape.Lojasiewicz.lojasiewicz_modulated_of_finiteDimensional`. -/
theorem exists_isLojasiewiczAt_of_analyticAt [FiniteDimensional ℝ E] (hf : AnalyticAt ℝ f z) :
    ∃ a, 0 ≤ a ∧ a < 1 ∧ IsLojasiewiczAt f z a := by
  let e : E ≃ₗᵢ[ℝ] EucSpace (Module.finrank ℝ E) := (stdOrthonormalBasis ℝ E).repr
  obtain ⟨a, ha0, ha1, h⟩ := exists_isLojasiewiczAt_of_local_gradient_inequality
    (Landscape.Lojasiewicz.analytic_gradient_inequality (f ∘ e.symm) (e z)
      (hf.comp_of_eq (e.symm.analyticAt (e z)) (e.symm_apply_apply z)))
  exact ⟨a, ha0, ha1, (isLojasiewiczAt_comp_linearIsometryEquiv_symm_iff e).1 h⟩

/-- **A function analytic everywhere satisfies the Łojasiewicz inequality at every
point**, with an exponent `0 ≤ a < 1`: `fun w _ => exists_isLojasiewiczAt_of_analyticOnNhd
hf w` is the hypothesis `hloj` of the convergence theorems of this directory. -/
theorem exists_isLojasiewiczAt_of_analyticOnNhd [FiniteDimensional ℝ E]
    (hf : AnalyticOnNhd ℝ f Set.univ) : ∀ w, ∃ a, 0 ≤ a ∧ a < 1 ∧ IsLojasiewiczAt f w a :=
  fun w => exists_isLojasiewiczAt_of_analyticAt (hf w (Set.mem_univ w))

/-- The hypotheses of `isLojasiewiczAt_of_forall_mem` and
`exists_isLojasiewiczAt_of_local_gradient_inequality` hold for `f(x) = ‖x‖² / 2` on
the line at `z = 0`, with `a = 1/2`, `k = 1` and `V = ℝ`. -/
example : let f := fun x : ℝ => (1 / 2 : ℝ) * ‖x‖ ^ 2
    (0 : ℝ) < 1 ∧ IsOpen (Set.univ : Set ℝ) ∧ (0 : ℝ) ∈ Set.univ ∧
      (∀ y ∈ Set.univ, |f y - f 0| ^ (1 / 2 : ℝ) ≤ 1 * ‖gradient f y‖) ∧
      (∃ alpha k : ℝ, ∃ V : Set ℝ, 0 < alpha ∧ alpha < 1 ∧ 0 < k ∧ IsOpen V ∧ 0 ∈ V ∧
        ∀ y ∈ V, |f y - f 0| ^ alpha ≤ k * ‖gradient f y‖) := by
  intro f
  have hineq : ∀ y ∈ Set.univ, |f y - f 0| ^ (1 / 2 : ℝ) ≤ 1 * ‖gradient f y‖ := by
    intro y _
    have h0 : f y - f 0 = 1 / 2 * ‖y‖ ^ 2 := by
      simp only [f, norm_zero]
      ring
    rw [h0, abs_of_nonneg (by positivity), one_mul, gradient_half_mul_norm_sq, id]
    exact half_mul_norm_sq_rpow_le y
  exact ⟨one_pos, isOpen_univ, Set.mem_univ 0, hineq,
    1 / 2, 1, Set.univ, by norm_num, by norm_num, one_pos, isOpen_univ, Set.mem_univ 0, hineq⟩

/-- The hypotheses of `exists_isLojasiewiczAt_of_analyticAt_low_dimension` and
`exists_isLojasiewiczAt_of_analyticAt` hold for `g(y) = y₀⁴` on `EucSpace 1` at its
degenerate critical point `0`. -/
example : let p : EucSpace 1 →L[ℝ] ℝ := EuclideanSpace.proj (0 : Fin 1)
    1 ≤ 1 ∧ AnalyticAt ℝ (fun y => p y ^ 4) 0 ∧ gradient (fun y => p y ^ 4) 0 = 0 := by
  intro p
  refine ⟨le_rfl, (p.analyticAt 0).fun_pow 4, ?_⟩
  have h := (p.hasFDerivAt (x := 0)).pow 4
  rw [map_zero, zero_pow (by norm_num), smul_zero, zero_smul] at h
  refine HasGradientAt.gradient ?_
  rw [hasGradientAt_iff_hasFDerivAt, map_zero]
  exact h

/-- The hypotheses of `IsLojasiewiczAt.congr_of_eventuallyEq` and `isLojasiewiczAt_congr`
hold for `f(x) = ‖x‖² / 2` on the line at `0`, with `a = 1/2`, and the function equal
to `f` on `(-1, 1)` and to `0` elsewhere. -/
example : let f := fun x : ℝ => (1 / 2 : ℝ) * ‖x‖ ^ 2
    IsLojasiewiczAt f 0 (1 / 2) ∧ f =ᶠ[𝓝 0] fun x => if ‖x‖ < 1 then f x else 0 := by
  intro f
  refine ⟨isLojasiewiczAt_half_mul_norm_sq, ?_⟩
  filter_upwards [Metric.ball_mem_nhds (0 : ℝ) one_pos] with x hx
  rw [mem_ball_zero_iff] at hx
  exact (ite_eq_left hx).symm

/-- The hypotheses of `IsLojasiewiczAt.comp_linearIsometryEquiv_symm` and
`isLojasiewiczAt_comp_linearIsometryEquiv_symm_iff` hold for `‖x‖² / 2` on `ℝ × ℝ`
with the `ℓ²` norm at `0`, with `a = 1/2`, and an isometry onto `EucSpace 2`. -/
example : IsLojasiewiczAt (fun x : WithLp 2 (ℝ × ℝ) => (1 / 2 : ℝ) * ‖x‖ ^ 2) 0 (1 / 2) ∧
    Nonempty (WithLp 2 (ℝ × ℝ) ≃ₗᵢ[ℝ] EucSpace (Module.finrank ℝ (WithLp 2 (ℝ × ℝ)))) :=
  ⟨isLojasiewiczAt_half_mul_norm_sq, ⟨(stdOrthonormalBasis ℝ _).repr⟩⟩

/-- The hypotheses of `exists_isLojasiewiczAt_of_analyticAt` and
`exists_isLojasiewiczAt_of_analyticOnNhd` hold on `ℝ × ℝ` with the `ℓ²` norm, a
finite-dimensional space other than `EucSpace N`, for `x ↦ ⟪v, x⟫⁴`, which is
analytic everywhere. -/
example (v : WithLp 2 (ℝ × ℝ)) : FiniteDimensional ℝ (WithLp 2 (ℝ × ℝ)) ∧
    AnalyticOnNhd ℝ (fun x : WithLp 2 (ℝ × ℝ) => inner (𝕜 := ℝ) v x ^ 4) Set.univ :=
  ⟨inferInstance, ((innerSL ℝ v).analyticOnNhd _).fun_pow 4⟩

end Transformer.Landscape.SaddleAvoidance
