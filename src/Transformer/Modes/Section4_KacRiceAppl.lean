import Transformer.Modes.Section2_MainForm

/-
# The number of modes of a Gaussian KDE — applying Kac–Rice to `F_n`

§4.1 of arXiv:2412.09080v3, `sec: kac.rice.application`: `lem: pt.bdd`, that
`thm:kac-rice` applies to `F_n` on `T`, and `lem:kr-appl`, the formula it gives.

**What the source says and what is carried here.**

* `lem: pt.bdd` claims a continuous density `p_t` of `(F_n(t), F_n'(t))`
  vanishing at infinity, "conditions 1, 2, 4" of `thm:kac-rice`, and "thus
  `thm:kac-rice` applies".  Applying it needs condition 3 as well — the joint
  density, continuous in `(t, x, y)` — so what is stated is the whole of
  `IsKacRiceField`, together with continuity and decay of each `p_t`.

* `lem:kr-appl` evaluates `p_t` on the line `x = 0`, a null set.  A density is
  determined only almost everywhere, so the formula is stated for a
  *continuous* density, which is unique; for an arbitrary one it would say
  nothing.

Source: arXiv:2412.09080v3, §4.1, `lem: pt.bdd`, `lem:kr-appl`.
-/

open MeasureTheory Filter
open scoped ENNReal Topology

namespace Transformer
namespace Modes

/-- **Proposition (lem: pt.bdd).**  For `β > 0` and `n ≥ 5`, the field `F_n`
satisfies the hypotheses of `thm:kac-rice` at level `0` on `T`, and the joint
density `p_t` of `(F_n(t), F_n'(t))` is continuous and vanishes at infinity for
every `t ∈ T`.

Not proved here.  See the module docstring for condition 3.

Source: arXiv:2412.09080v3, `lem: pt.bdd`. -/
theorem pt_bdd {β : ℝ} (hβ : 0 < β) {n : ℕ} (hn : 5 ≤ n) (w : ℝ) :
    ∃ (p1 : ℝ → ℝ → ℝ) (p : ℝ → ℝ → ℝ → ℝ),
      IsKacRiceField (gaussianSample n) (fun X => fieldF β X) 0 (intervalT n β w) p1 p ∧
      ∀ t ∈ intervalT n β w, Continuous (fun z : ℝ × ℝ => p t z.1 z.2) ∧
        Tendsto (fun z : ℝ × ℝ => p t z.1 z.2) (cocompact (ℝ × ℝ)) (𝓝 0) := by
  sorry

/-- **Lemma (lem:kr-appl).**  For `β > 0` and `n ≥ 5`, the law of
`(F_n(t), F_n'(t))` has a continuous density `p_t` for every `t ∈ T`, and

  `𝔼 U₀(F_n, T) = ∫_T ∫_0^∞ y p_t(0, y) dy dt`.

Not proved here; the source deduces it from `lem: pt.bdd` and `thm:kac-rice`.

Source: arXiv:2412.09080v3, `lem:kr-appl`. -/
theorem kr_appl {β : ℝ} (hβ : 0 < β) {n : ℕ} (hn : 5 ≤ n) (w : ℝ) :
    ∃ p : ℝ → ℝ → ℝ → ℝ,
      (∀ t ∈ intervalT n β w, Continuous (fun z : ℝ × ℝ => p t z.1 z.2) ∧
        Measure.map (fun X => (fieldF β X t, deriv (fieldF β X) t)) (gaussianSample n)
          = volume.withDensity fun z : ℝ × ℝ => ENNReal.ofReal (p t z.1 z.2)) ∧
      expectedUpcrossings (gaussianSample n) (fun X => fieldF β X) 0 (intervalT n β w)
        = ∫⁻ t in intervalT n β w, ∫⁻ y in Set.Ioi (0 : ℝ), ENNReal.ofReal (y * p t 0 y) := by
  sorry

/-- The hypotheses of `pt_bdd` and `kr_appl` are satisfiable. -/
example : (0 : ℝ) < 1 ∧ 5 ≤ 5 := ⟨one_pos, le_rfl⟩

end Modes
end Transformer
