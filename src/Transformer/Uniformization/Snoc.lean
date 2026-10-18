/-
# Local uniformization: the last variable

`ℝⁿ⁺¹` as `ℝⁿ × ℝ` through `Fin.init` and `Fin.snoc`: analyticity, monomials,
normal crossings of functions of the first `n` variables, and products of
charts on `ℝⁿ` with an interval in the last variable. Auxiliary for
Appendix D.1, `lem: loj`, of arXiv:2510.22026v2.
-/

import Transformer.Uniformization.Charts

open Filter Set
open scoped BigOperators Topology

namespace Transformer.Uniformization

open Transformer.AnalyticPreparation (Base)

variable {n : ℕ}

theorem analyticAt_init (x : Base (n + 1)) :
    AnalyticAt ℝ (fun x : Base (n + 1) => (Fin.init x : Base n)) x :=
  analyticAt_pi_iff.2 fun i =>
    (ContinuousLinearMap.proj i.castSucc : Base (n + 1) →L[ℝ] ℝ).analyticAt x

theorem analyticAt_last (x : Base (n + 1)) :
    AnalyticAt ℝ (fun x : Base (n + 1) => x (Fin.last n)) x :=
  (ContinuousLinearMap.proj (Fin.last n) : Base (n + 1) →L[ℝ] ℝ).analyticAt x

theorem AnalyticAt.snoc {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E] {f : E → Base n}
    {g : E → ℝ} {x : E} (hf : AnalyticAt ℝ f x) (hg : AnalyticAt ℝ g x) :
    AnalyticAt ℝ (fun y => (Fin.snoc (f y) (g y) : Base (n + 1))) x := by
  refine analyticAt_pi_iff.2 fun i => ?_
  induction i using Fin.lastCases with
  | last => simpa only [Fin.snoc_last] using hg
  | cast i => simpa only [Fin.snoc_castSucc] using analyticAt_pi_iff.1 hf i

theorem continuous_snoc :
    Continuous (fun p : Base n × ℝ => (Fin.snoc p.1 p.2 : Base (n + 1))) := by
  refine continuous_pi fun i => ?_
  induction i using Fin.lastCases with
  | last => simpa only [Fin.snoc_last] using continuous_snd
  | cast i =>
    simp only [Fin.snoc_castSucc]
    exact (continuous_apply i).comp continuous_fst

theorem init_sub (x y : Base (n + 1)) : Fin.init (x - y) = Fin.init x - Fin.init y := rfl

theorem snoc_sub_snoc (x' w' : Base n) (y y₀ : ℝ) :
    (Fin.snoc x' y : Base (n + 1)) - Fin.snoc w' y₀ = Fin.snoc (x' - w') (y - y₀) := by
  funext i
  induction i using Fin.lastCases with
  | last => simp
  | cast i => simp

theorem snoc_add_snoc (x' w' : Base n) (y y₀ : ℝ) :
    (Fin.snoc x' y : Base (n + 1)) + Fin.snoc w' y₀ = Fin.snoc (x' + w') (y + y₀) := by
  funext i
  induction i using Fin.lastCases with
  | last => simp
  | cast i => simp

theorem snoc_zero_zero : (Fin.snoc (0 : Base n) (0 : ℝ) : Base (n + 1)) = 0 := by
  funext i
  induction i using Fin.lastCases with
  | last => simp
  | cast i => simp

/-- A monomial on `ℝⁿ⁺¹` splits off the last variable. Auxiliary for
Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem monomial_succ (p : Fin (n + 1) → ℕ) (x : Base (n + 1)) :
    monomial p x = monomial (Fin.init p) (Fin.init x) * x (Fin.last n) ^ p (Fin.last n) := by
  simp only [monomial, Fin.prod_univ_castSucc, Fin.init]

theorem monomial_snoc (p : Fin n → ℕ) (e : ℕ) (x : Base (n + 1)) :
    monomial (Fin.snoc p e : Fin (n + 1) → ℕ) x =
      monomial p (Fin.init x) * x (Fin.last n) ^ e := by
  rw [monomial_succ, Fin.init_snoc, Fin.snoc_last]

/-- A function of the first `n` variables with normal crossings has normal
crossings on `ℝⁿ⁺¹`. Auxiliary for Appendix D.1, `lem: loj`, of
arXiv:2510.22026v2. -/
theorem NormalCrossingAt.comp_init {F : Base n → ℝ} {w : Base (n + 1)}
    (h : NormalCrossingAt F (Fin.init w)) : NormalCrossingAt (fun x => F (Fin.init x)) w := by
  obtain ⟨p, u, hu, huw, hF⟩ := h
  refine ⟨Fin.snoc p 0, fun x => u (Fin.init x), hu.comp (analyticAt_init w), huw, ?_⟩
  have ht : Tendsto (fun x : Base (n + 1) => (Fin.init x : Base n)) (𝓝 w) (𝓝 (Fin.init w)) :=
    (analyticAt_init w).continuousAt.tendsto
  filter_upwards [ht.eventually hF] with x hx
  rw [hx, monomial_snoc, init_sub, pow_zero, mul_one]

/-- The product of a set of `ℝⁿ` and an interval of the last variable. -/
def prodIcc (L : Set (Base n)) (a b : ℝ) : Set (Base (n + 1)) :=
  (fun p : Base n × ℝ => (Fin.snoc p.1 p.2 : Base (n + 1))) '' (L ×ˢ Icc a b)

theorem mem_prodIcc {L : Set (Base n)} {a b : ℝ} {x : Base (n + 1)} :
    x ∈ prodIcc L a b ↔ Fin.init x ∈ L ∧ x (Fin.last n) ∈ Icc a b := by
  constructor
  · rintro ⟨p, ⟨hp1, hp2⟩, rfl⟩
    simpa only [Fin.init_snoc, Fin.snoc_last] using And.intro hp1 hp2
  · rintro ⟨h1, h2⟩
    exact ⟨(Fin.init x, x (Fin.last n)), ⟨h1, h2⟩, Fin.snoc_init_self x⟩

theorem IsCompact.prodIcc {L : Set (Base n)} (hL : IsCompact L) (a b : ℝ) :
    IsCompact (prodIcc L a b) :=
  (hL.prod isCompact_Icc).image continuous_snoc

/-- A neighbourhood of `init z` times a neighbourhood interval of `z last` is
a neighbourhood of `z`. Auxiliary for Appendix D.1, `lem: loj`, of
arXiv:2510.22026v2. -/
theorem prodIcc_mem_nhds {z : Base (n + 1)} {L : Set (Base n)}
    (hL : L ∈ 𝓝 (Fin.init z : Base n))
    {δ : ℝ} (hδ : 0 < δ) :
    prodIcc L (z (Fin.last n) - δ) (z (Fin.last n) + δ) ∈ 𝓝 z := by
  have h1 : ∀ᶠ x in 𝓝 z, (Fin.init x : Base n) ∈ L :=
    (analyticAt_init z).continuousAt.preimage_mem_nhds hL
  have h2 : ∀ᶠ x in 𝓝 z, x (Fin.last n) ∈ Icc (z (Fin.last n) - δ) (z (Fin.last n) + δ) :=
    (analyticAt_last z).continuousAt.preimage_mem_nhds (Icc_mem_nhds (by linarith) (by linarith))
  filter_upwards [h1, h2] with x hx1 hx2
  exact mem_prodIcc.2 ⟨hx1, hx2⟩

theorem norm_snoc_le {a : Base n} {y r : ℝ} (ha : ‖a‖ ≤ r) (hy : |y| ≤ r) :
    ‖(Fin.snoc a y : Base (n + 1))‖ ≤ r := by
  refine (pi_norm_le_iff_of_nonneg ((abs_nonneg y).trans hy)).2 fun i => ?_
  induction i using Fin.lastCases with
  | last => simpa only [Fin.snoc_last, Real.norm_eq_abs] using hy
  | cast i => simpa only [Fin.snoc_castSucc] using (norm_le_pi_norm a i).trans ha

theorem norm_init_le (x : Base (n + 1)) : ‖(Fin.init x : Base n)‖ ≤ ‖x‖ :=
  (pi_norm_le_iff_of_nonneg (norm_nonneg x)).2 fun i => norm_le_pi_norm x i.castSucc

theorem abs_last_le (x : Base (n + 1)) : |x (Fin.last n)| ≤ ‖x‖ :=
  norm_le_pi_norm x (Fin.last n)

theorem tendsto_init_snoc (w' : Base n) (y₀ : ℝ) :
    Tendsto (fun x : Base (n + 1) => (Fin.init x : Base n)) (𝓝 (Fin.snoc w' y₀)) (𝓝 w') := by
  have h := (analyticAt_init (Fin.snoc w' y₀ : Base (n + 1))).continuousAt.tendsto
  rwa [Fin.init_snoc] at h

end Transformer.Uniformization
