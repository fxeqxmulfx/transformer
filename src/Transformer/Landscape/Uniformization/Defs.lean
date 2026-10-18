/-
# Local uniformization: definitions

Coordinate monomials `x^p` on `ℝⁿ`, monomial maps `s ↦ (s^{E i})_i`, functions
with normal crossings (an analytic unit times a coordinate monomial centred at
the point), and finite families of analytic charts with compact domains in
which a function has normal crossings at every point
(`HasMonomialCharts`).  `Landscape.Uniformization.Induction` proves that every real
analytic germ that does not vanish identically has such charts; this is the
local uniformization behind the gradient inequality of Appendix D.1,
`lem: loj`, of arXiv:2510.22026v2.
-/

import Transformer.Landscape.AnalyticPreparation.Basic
import Mathlib.Analysis.Analytic.Constructions
import Mathlib.Analysis.Analytic.Linear

open Filter Set
open scoped BigOperators Topology

namespace Transformer.Landscape.Uniformization

open Transformer.Landscape.AnalyticPreparation (Base)

/-- The coordinate monomial `x^p = ∏ i, x_i ^ p_i` on `ℝⁿ`. -/
def monomial {n : ℕ} (p : Fin n → ℕ) (x : Base n) : ℝ := ∏ i, x i ^ p i

/-- The monomial map `s ↦ (s^{E i})_i` with exponent matrix `E`. -/
def monomialMap {m n : ℕ} (E : Fin n → Fin m → ℕ) (s : Base m) : Base n :=
  fun i => monomial (E i) s

/-- `f` has normal crossings at `w`: near `w` it is an analytic function
not vanishing at `w` times a coordinate monomial centred at `w`. -/
def NormalCrossingAt {n : ℕ} (f : Base n → ℝ) (w : Base n) : Prop :=
  ∃ (p : Fin n → ℕ) (u : Base n → ℝ), AnalyticAt ℝ u w ∧ u w ≠ 0 ∧
    ∀ᶠ x in 𝓝 w, f x = u x * monomial p (x - w)

/-- A monomial chart for `f`: a map analytic at every point of its compact
domain, in which `f` has normal crossings at every point of the domain. -/
def IsMonomialChart {n : ℕ} (f : Base n → ℝ) (c : (Base n → Base n) × Set (Base n)) :
    Prop :=
  IsCompact c.2 ∧ AnalyticOnNhd ℝ c.1 c.2 ∧ ∀ w ∈ c.2, NormalCrossingAt (f ∘ c.1) w

/-- `f` has monomial charts at `z`: finitely many monomial charts for `f`
whose images of the domains cover a neighbourhood of `z`. -/
def HasMonomialCharts {n : ℕ} (f : Base n → ℝ) (z : Base n) : Prop :=
  ∃ S : Set ((Base n → Base n) × Set (Base n)), S.Finite ∧
    (∀ c ∈ S, IsMonomialChart f c) ∧ (⋃ c ∈ S, c.1 '' c.2) ∈ 𝓝 z

variable {k m n : ℕ}

theorem monomial_zero_exp (x : Base n) : monomial 0 x = 1 := by
  simp [monomial]

theorem monomial_add_exp (p q : Fin n → ℕ) (x : Base n) :
    monomial (p + q) x = monomial p x * monomial q x := by
  simp only [monomial, Pi.add_apply, pow_add, Finset.prod_mul_distrib]

theorem monomial_smul_exp (k : ℕ) (p : Fin n → ℕ) (x : Base n) :
    monomial (k • p) x = monomial p x ^ k := by
  simp only [monomial, Pi.smul_apply, smul_eq_mul, ← Finset.prod_pow, ← pow_mul,
    mul_comm k]

theorem monomial_mul (p : Fin n → ℕ) (x y : Base n) :
    monomial p (x * y) = monomial p x * monomial p y := by
  simp only [monomial, Pi.mul_apply, mul_pow, Finset.prod_mul_distrib]

theorem monomial_ne_zero_iff (p : Fin n → ℕ) (x : Base n) :
    monomial p x ≠ 0 ↔ ∀ i, p i ≠ 0 → x i ≠ 0 := by
  simp only [monomial, Finset.prod_ne_zero_iff, Finset.mem_univ, true_implies, ne_eq,
    pow_eq_zero_iff', not_and, not_not]
  exact forall_congr' fun i => ⟨fun h hp hx => hp (h hx), fun h hx => by_contra fun hp =>
    h hp hx⟩

/-- The monomial of a unit exponent is the coordinate. -/
theorem monomial_single (i : Fin n) (x : Base n) : monomial (Pi.single i 1) x = x i := by
  classical
  simp [monomial, Pi.single_apply, pow_ite]

/-- A monomial vanishes at the origin unless its exponent is zero. -/
theorem monomial_zero_point (p : Fin n → ℕ) (hp : p ≠ 0) : monomial p 0 = 0 := by
  obtain ⟨i, hi⟩ : ∃ i, p i ≠ 0 := by
    by_contra h
    push Not at h
    exact hp (funext h)
  by_contra h
  exact (monomial_ne_zero_iff p 0).1 h i hi rfl

/-- A monomial in a monomial map is a monomial: `(s^E)^p = s^{Σ_i p_i E_i}`. -/
theorem monomial_monomialMap (E : Fin n → Fin m → ℕ) (p : Fin n → ℕ) (s : Base m) :
    monomial p (monomialMap E s) = monomial (∑ i, p i • E i) s := by
  simp only [monomial, monomialMap, ← Finset.prod_pow, ← pow_mul]
  rw [Finset.prod_comm]
  refine Finset.prod_congr rfl fun k _ => ?_
  rw [Finset.prod_pow_eq_pow_sum, Finset.sum_apply]
  simp only [Pi.smul_apply, smul_eq_mul, mul_comm]

/-- Monomial maps compose to monomial maps. -/
theorem monomialMap_comp (E : Fin n → Fin m → ℕ) (F : Fin m → Fin k → ℕ) (s : Base k) :
    monomialMap E (monomialMap F s) = monomialMap (fun i => ∑ j, E i j • F j) s := by
  funext i
  exact monomial_monomialMap F (E i) s

theorem analyticAt_monomial (p : Fin n → ℕ) (x : Base n) :
    AnalyticAt ℝ (monomial p) x := by
  unfold monomial
  exact Finset.analyticAt_fun_prod _ fun i _ =>
    ((ContinuousLinearMap.proj i : Base n →L[ℝ] ℝ).analyticAt x).fun_pow _

theorem analyticAt_monomialMap (E : Fin n → Fin m → ℕ) (s : Base m) :
    AnalyticAt ℝ (monomialMap E) s :=
  analyticAt_pi_iff.2 fun i => analyticAt_monomial (E i) s

theorem continuous_monomial (p : Fin n → ℕ) : Continuous (monomial p) := by
  unfold monomial
  fun_prop

theorem continuous_monomialMap (E : Fin n → Fin m → ℕ) : Continuous (monomialMap E) :=
  continuous_pi fun i => continuous_monomial (E i)

end Transformer.Landscape.Uniformization
