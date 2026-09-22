/-
# Attention's forward pass and Frank-Wolfe — uniqueness for the singular ODE

Formalization of Alcalde, Geshkovski, Ruiz-Balet — arXiv:2508.09628v1,
*Attention's forward pass and Frank-Wolfe*, §4, `thm: ode`, Part 2 of its proof.

`hardmaxSol` solves `eq: hardmax.ode` (`isHardmaxODESolution_hardmaxSol`); here
it is the only solution.  Where a solution `x` agrees with it at `t₀`, the
strict gaps of `inner_hardmaxSol_lt` persist for a while by continuity, so every
particle of `x` keeps moving towards a particle resting at its vertex.  On that
stretch both curves solve one equation, Lipschitz in the state, and Grönwall
keeps them together (`eventually_eq_hardmaxSol`); the set where they agree is
closed, so it is all of `[0, T]` (`eqOn_hardmaxSol`).

The source argues with explicit times instead: a solution moves at speed at
most `2𝖽(𝒦)`, so it stays in its cell for a time `δ / 2𝖽(𝒦)`, and the argument
is iterated.  The continuity argument here needs neither the bound nor `δ`.

Source: arXiv:2508.09628v1, `sec: proof.thm.ode`, Part 2.
-/

import Transformer.FrankWolfe.Section4_ODESolution
import Mathlib.Analysis.ODE.Gronwall
import Mathlib.Analysis.Calculus.Deriv.Prod

open scoped BigOperators Topology NNReal
open Real Set Filter

namespace Transformer
namespace FrankWolfe

variable {d n κ : ℕ} {B : ParamMatrix d} {X₀ : Idx n → EucSpace d} {v : Idx κ → EucSpace d}
  {σ : Idx n → Idx κ}

/-- Moving every particle towards a chosen one, `Z ↦ (Z_{b(i)} - Z_i)_i`, is
`2`-Lipschitz. -/
theorem lipschitzWith_sub_comp (b : Idx n → Idx n) :
    LipschitzWith 2 fun Z : Idx n → EucSpace d => fun i => Z (b i) - Z i := by
  refine LipschitzWith.of_dist_le_mul fun Z W => (dist_pi_le_iff (by positivity)).2 fun i => ?_
  calc dist (Z (b i) - Z i) (W (b i) - W i) ≤ dist (Z (b i)) (W (b i)) + dist (Z i) (W i) :=
        dist_sub_sub_le _ _ _ _
    _ ≤ dist Z W + dist Z W := add_le_add (dist_le_pi_dist _ _ _) (dist_le_pi_dist _ _ _)
    _ = ((2 : ℝ≥0) : ℝ) * dist Z W := by push_cast; ring

/-- **Uniqueness, the local step.**  Where a solution agrees with `hardmaxSol` at
some `t₀ ∈ [0, T)`, it keeps agreeing for a while.

The equation is not imposed at `t₀` itself (at `t₀ = 0` it is not imposed at
all), so the two curves are compared on `[a, t]` for `t₀ < a < t`, where both
solve one Lipschitz equation and Grönwall bounds their distance at `t` by the
one at `a`; the latter vanishes as `a → t₀`.

Source: arXiv:2508.09628v1, `sec: proof.thm.ode`, Part 2. -/
theorem eventually_eq_hardmaxSol (hgen : GenericPolytope B v X₀)
    (hσ : ∀ i, X₀ i ∈ cell B (configHull X₀) v (σ i)) {T : ℝ} {x : ℝ → Idx n → EucSpace d}
    (hx : IsHardmaxODESolution B T x) {t₀ : ℝ} (ht₀ : t₀ ∈ Ico 0 T)
    (hxt₀ : x t₀ = hardmaxSol X₀ (v ∘ σ) t₀) :
    ∀ᶠ t in 𝓝[>] t₀, x t = hardmaxSol X₀ (v ∘ σ) t := by
  set E := hardmaxSol X₀ (v ∘ σ)
  choose p hp hpσ using hgen.exists_particle hσ
  have hEp : ∀ k t, E t (p k) = v k := fun k t => hardmaxSol_of_vertex (hp k) (hpσ k) t
  have hIcc : Icc 0 T ∈ 𝓝[≥] t₀ := Icc_mem_nhdsGE_of_mem ht₀
  have hxt := (hx.1 t₀ (Ico_subset_Icc_self ht₀)).mono_of_mem_nhdsWithin hIcc
  have hxi := continuousWithinAt_pi.1 hxt
  -- the strict gaps of `x t₀ = E t₀` persist
  have hgap : ∀ᶠ t in 𝓝[≥] t₀, ∀ i j, E t₀ j ≠ v (σ i) →
      inner (𝕜 := ℝ) (B (x t i)) (x t j) < inner (𝕜 := ℝ) (B (x t i)) (x t (p (σ i))) := by
    refine eventually_all.2 fun i => eventually_all.2 fun j => ?_
    by_cases hj : E t₀ j = v (σ i)
    · exact Eventually.of_forall fun _ h => absurd hj h
    · have hBi := B.continuous.continuousAt.comp_continuousWithinAt (hxi i)
      refine (Tendsto.eventually_lt (hBi.inner (hxi j)) (hBi.inner (hxi (p (σ i)))) ?_).mono
        fun _ h _ => h
      show inner (𝕜 := ℝ) (B (x t₀ i)) (x t₀ j) < inner (𝕜 := ℝ) (B (x t₀ i)) (x t₀ (p (σ i)))
      rw [hxt₀, hEp]
      exact inner_hardmaxSol_lt hgen hσ ht₀.1 i (hardmaxSol_mem_configHull hgen ht₀.1 j) hj
  obtain ⟨u, hu, hsub⟩ := mem_nhdsGE_iff_exists_Ico_subset.1 (inter_mem hIcc hgap)
  obtain ⟨u', hu', hu'T⟩ : ∃ u' ∈ Ioo t₀ u, u' ≤ T :=
    have h : (t₀ + u) / 2 ∈ Ioo t₀ u := ⟨by linarith [mem_Ioi.1 hu], by linarith [mem_Ioi.1 hu]⟩
    ⟨_, h, (hsub (Ioo_subset_Ico_self h)).1.2⟩
  -- so on `(t₀, u')` every particle of `x` moves towards a particle resting at its vertex
  have hsel : ∀ t i, ∃ j, t ∈ Ioo t₀ u' → E t₀ j = v (σ i) ∧
      HasDerivAt (fun s => x s i) (x t j - x t i) t := by
    intro t i
    by_cases ht : t ∈ Ioo t₀ u'
    · have hgt := (hsub ⟨ht.1.le, ht.2.trans hu'.2⟩).2
      obtain ⟨_, ⟨j, rfl⟩, hmax, hder⟩ := hx.2 t ⟨ht₀.1.trans_lt ht.1, ht.2.trans_le hu'T⟩ i
      exact ⟨j, fun _ =>
        ⟨by_contra fun hj => (hgt i j hj).not_ge (hmax _ (mem_range_self _)), hder⟩⟩
    · exact ⟨i, fun h => absurd h ht⟩
  choose b hb using hsel
  -- there both curves solve `Ż = (Z_{b t i} - Z_i)_i`, which is `2`-Lipschitz in `Z`
  have hdist : ∀ a ∈ Ioo t₀ u', ∀ t ∈ Icc a u',
      dist (x t) (E t) ≤ dist (x a) (E a) * exp ((2 : ℝ≥0) * (t - a)) := fun a ha => by
    have hIoo : ∀ t ∈ Ico a u', t ∈ Ioo t₀ u' := fun t ht => ⟨ha.1.trans_le ht.1, ht.2⟩
    refine dist_le_of_trajectories_ODE (f := x) (g := E)
      (v := fun t (Z : Idx n → EucSpace d) i => Z (b t i) - Z i) (K := 2)
      (fun t => lipschitzWith_sub_comp (b t))
      (hx.1.mono (Icc_subset_Icc (ht₀.1.trans ha.1.le) hu'T)) (fun t ht => ?_)
      (continuous_hardmaxSol _ _).continuousOn (fun t ht => ?_) le_rfl
    · exact (hasDerivAt_pi.2 fun i => (hb t i (hIoo t ht)).2).hasDerivWithinAt
    · refine hasDerivWithinAt_pi.2 fun i => ?_
      show HasDerivWithinAt (fun s => E s i) (E t (b t i) - E t i) (Ici t) t
      rw [show E t (b t i) = v (σ i) from
        hardmaxSol_eq_vertex hgen hσ ht₀.1 (hb t i (hIoo t ht)).1 t]
      exact (hasDerivAt_hardmaxSol X₀ (v ∘ σ) i t).hasDerivWithinAt
  refine mem_nhdsGT_iff_exists_Ioo_subset.2 ⟨u', hu'.1, fun t ht => ?_⟩
  -- the bound at `t` from every `a ∈ (t₀, t)`, and `a → t₀`
  have hlim : Tendsto (fun a => dist (x a) (E a) * exp ((2 : ℝ≥0) * (t - a))) (𝓝[>] t₀)
      (𝓝 (dist (x t₀) (E t₀) * exp ((2 : ℝ≥0) * (t - t₀)))) :=
    (Filter.Tendsto.dist (hxt.mono Ioi_subset_Ici_self)
      (continuous_hardmaxSol _ _).continuousWithinAt).mul
      ((Continuous.tendsto (by fun_prop) t₀).mono_left nhdsWithin_le_nhds)
  rw [hxt₀, dist_self, zero_mul] at hlim
  refine dist_le_zero.1 (ge_of_tendsto hlim ?_)
  filter_upwards [Ioo_mem_nhdsGT ht.1] with a ha
  exact hdist a ⟨ha.1, ha.2.trans ht.2⟩ t ⟨ha.2.le, ht.2.le⟩

/-- **Uniqueness.**  A solution starting at `x^0` is `hardmaxSol` on all of
`[0, T]`.

Source: arXiv:2508.09628v1, `sec: proof.thm.ode`, Part 2. -/
theorem eqOn_hardmaxSol (hgen : GenericPolytope B v X₀)
    (hσ : ∀ i, X₀ i ∈ cell B (configHull X₀) v (σ i)) {T : ℝ} {x : ℝ → Idx n → EucSpace d}
    (hx : IsHardmaxODESolution B T x) (hx0 : x 0 = X₀) :
    EqOn x (hardmaxSol X₀ (v ∘ σ)) (Icc 0 T) := by
  have hcl : IsClosed ({t | x t = hardmaxSol X₀ (v ∘ σ) t} ∩ Icc 0 T) := by
    rw [inter_comm]
    exact (hx.1.prodMk (continuous_hardmaxSol X₀ (v ∘ σ)).continuousOn
      ).preimage_isClosed_of_isClosed isClosed_Icc isClosed_diagonal
  refine hcl.Icc_subset_of_forall_mem_nhdsWithin ?_ fun t ht =>
    eventually_eq_hardmaxSol hgen hσ hx ht.2 ht.1
  show x 0 = hardmaxSol X₀ (v ∘ σ) 0
  rw [hardmaxSol_zero, hx0]

/-- A single particle is a generic configuration, for every `B`: its hull is the
point, which is its only vertex and lies in its own cell.

Source: `GenericPolytope` (arXiv:2508.09628v1, §4, `thm: exp.fast.polytope`,
conditions 1 and 2) at `n = κ = 1`. -/
theorem genericPolytope_single (B : ParamMatrix d) (z : EucSpace d) :
    GenericPolytope B (fun _ : Idx 1 => z) (fun _ : Idx 1 => z) := by
  have hhull : configHull (fun _ : Idx 1 => z) = {z} := by
    rw [configHull, range_const, convexHull_singleton]
  have hcell : ∀ j : Idx 1, z ∈ cell B (configHull fun _ : Idx 1 => z) (fun _ => z) j := fun j =>
    ⟨by rw [hhull]; rfl, fun y hy => by
      rw [hhull] at hy
      obtain rfl : y = z := hy
      exact le_rfl⟩
  exact ⟨⟨fun a b _ => Subsingleton.elim a b, by rw [range_const, hhull, extremePoints_singleton]⟩,
    fun j => ⟨hcell j, fun i hij => absurd (Subsingleton.elim i j) hij⟩,
    fun i => ⟨0, hcell 0, fun j _ => Subsingleton.elim j 0⟩⟩

/-- The hypotheses of `eventually_eq_hardmaxSol` and `eqOn_hardmaxSol` are
satisfiable together: `B = I₁`, one particle at the origin
(`genericPolytope_single`), `σ = id`, the solution `hardmaxSol` itself on
`[0, 1]`, `t₀ = 0`. -/
example : ∃ (X₀ : Idx 1 → EucSpace 1) (σ : Idx 1 → Idx 1) (x : ℝ → Idx 1 → EucSpace 1),
    GenericPolytope (ContinuousLinearMap.id ℝ (EucSpace 1)) X₀ X₀ ∧
    (∀ i, X₀ i ∈ cell (ContinuousLinearMap.id ℝ (EucSpace 1)) (configHull X₀) X₀ (σ i)) ∧
    IsHardmaxODESolution (ContinuousLinearMap.id ℝ (EucSpace 1)) 1 x ∧ (0 : ℝ) ∈ Ico 0 1 ∧
    x 0 = hardmaxSol X₀ (X₀ ∘ σ) 0 ∧ x 0 = X₀ := by
  have hgen := genericPolytope_single (ContinuousLinearMap.id ℝ (EucSpace 1)) 0
  have hσ := fun i : Idx 1 => (hgen.ownCell (id i)).1
  exact ⟨_, id, _, hgen, hσ, isHardmaxODESolution_hardmaxSol hgen hσ 1, ⟨le_rfl, one_pos⟩, rfl,
    hardmaxSol_zero _ _⟩

end FrankWolfe
end Transformer
