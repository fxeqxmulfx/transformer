module

public import Transformer.Perspective.AppendixA_Barycenter
public import Transformer.Perspective.AppendixA_Flow
public import Transformer.Perspective.AppendixA_StayNear
public import Transformer.Perspective.UniformLipschitz

/-!
# Appendix A — the trajectories converging to a saddle start in a null set

Geshkovski, Letrouit, Polyanskiy, Rigollet — arXiv:2312.10794v5,
*A mathematical perspective on Transformers*, Appendix A, `l:nosaddleconv`.

A solution of `e:gradfl` converging to a non-trivial critical point `Z` falls
under one of three cases (`tendsto_critical_cases`). If `Z` has zero
barycentre, so had the start (`sum_eq_zero_of_tendsto`); if two particles sit
together opposite to the barycentre, they started together
(`coord_eq_of_tendsto`); otherwise `Z` is the saddle `signedTuple e {i₀}` of a
single particle `i₀` opposite to the others, `n ≥ 3`. The last case is the
source's argument (`measure_tendsto_signedTuple`): the countably many local
center-stable sets `stayNear e' i₀ r`, `e'` in a countable dense set, cover the
trajectory from an integer time `ℓ` on, so the start lies in their images under
the backward flow `Φ^{-ℓ}`, which are null since that flow is Lipschitz. So
the lemma itself, `no_saddle_convergence`, holds.
-/

@[expose] public section

open Set Filter MeasureTheory
open scoped Topology

namespace Transformer
namespace Perspective

variable {d n : ℕ}

/-- **The trajectories converging to a saddle with one particle apart start in a
null set** (`d ≥ 2`, `n ≥ 3`). With `r = 1/(60 n)` and `D` countable and dense
in the sphere, such a trajectory enters `stayNear e' i₀ r` at an integer time
`ℓ` for some `e' ∈ D` (`exists_nat_mem_stayNear`), so its start lies in the
image of the closure of `stayNear e' i₀ r`, a null set
(`measure_closure_stayNear`), under the backward flow `Φ^{-ℓ}`, a Lipschitz map
(`lipschitzWith_e0Flow_neg`); such images are null
(`measure_image_eq_zero_of_lipschitz`). -/
theorem measure_tendsto_signedTuple (hd : 2 ≤ d) (hn : 3 ≤ n) (P : Measure (SphereTuple d n))
    (hP : UniformTuple d n P) :
    P {X₀ : SphereTuple d n | ∃ X : ℝ → SphereTuple d n, X 0 = X₀ ∧ E0GradientAscent d n X ∧
      ∃ (e : SSphere d) (i₀ : Idx n), ∀ i : Idx n,
        Tendsto (fun t => (X t i : EucSpace d) - signedTuple e {i₀} i) atTop (𝓝 0)} = 0 := by
  obtain ⟨D, hDc, hDd⟩ := TopologicalSpace.exists_countable_dense (SSphere d)
  have hn0 : (0 : ℝ) < n := by exact_mod_cast (by omega : 0 < n)
  have hr0 : (0 : ℝ) < 1 / (60 * n) := by positivity
  have hr1 : 60 * (n : ℝ) * (1 / (60 * n)) ≤ 1 := le_of_eq (mul_one_div_cancel (by positivity))
  refine measure_mono_null (t := ⋃ e' ∈ D, ⋃ i₀ : Idx n, ⋃ ℓ : ℕ,
      (fun Y => e0Flow Y (-(ℓ : ℝ))) '' closure (stayNear e' i₀ (1 / (60 * n)))) ?_ ?_
  · rintro _ ⟨X, rfl, hX, e, i₀, hlim⟩
    obtain ⟨e', he'D, he'⟩ := hDd.exists_dist_lt e (half_pos hr0)
    rw [Subtype.dist_eq, dist_eq_norm] at he'
    obtain ⟨ℓ, hℓ⟩ := exists_nat_mem_stayNear hX e i₀ hlim e' he'
    simp only [mem_iUnion]
    exact ⟨e', he'D, i₀, ℓ, X ℓ, subset_closure hℓ, e0Flow_neg_eq hX ℓ⟩
  · rw [measure_biUnion_null_iff hDc]
    exact fun e' _ => measure_iUnion_null fun i₀ => measure_iUnion_null fun ℓ =>
      measure_image_eq_zero_of_lipschitz (by omega) P hP isClosed_closure
        (measure_closure_stayNear hd hn P hP e' i₀ hr1)
        (lipschitzWith_e0Flow_neg (by omega) ℓ (Nat.cast_nonneg ℓ))

/-- The dimension and size hypotheses of `measure_tendsto_signedTuple` hold for
`d = 2`, `n = 3`; the uniform law is quantified over. -/
example : 2 ≤ 2 ∧ 3 ≤ 3 := ⟨le_rfl, le_rfl⟩

/-- **The three ways to converge to a non-trivial critical point.**  A solution
of `e:gradfl` converging to a non-trivial critical point `Z` started with zero
barycentre, or with two particles together, or converges to a saddle
`signedTuple e {i₀}` with `n ≥ 3`.

`Z` with nonzero barycentre `s` is `signedTuple e S` for `S` the particles with
`⟨z_i, s⟩ < 0` (`eq_signedTuple_of_critical`): `S = ∅` is a consensus, two
particles of `S` started together (`coord_eq_of_tendsto`), and `S = {i₀}` has
`s = (n - 2) e ≠ 0`, so `n ≥ 3`. -/
theorem tendsto_critical_cases {X : ℝ → SphereTuple d n} (hX : E0GradientAscent d n X)
    {Z : SphereTuple d n} (hcrit : IsCriticalE0 d n Z) (hnt : NonTrivialTuple d n Z)
    (hlim : ∀ i : Idx n,
      Tendsto (fun t => (X t i : EucSpace d) - (Z i : EucSpace d)) atTop (𝓝 0)) :
    ∑ j : Idx n, (X 0 j : EucSpace d) = 0 ∨ (∃ i j : Idx n, i ≠ j ∧ X 0 i = X 0 j) ∨
      3 ≤ n ∧ ∃ (e : SSphere d) (i₀ : Idx n), ∀ i : Idx n,
        Tendsto (fun t => (X t i : EucSpace d) - signedTuple e {i₀} i) atTop (𝓝 0) := by
  by_cases hs : ∑ k : Idx n, (Z k : EucSpace d) = 0
  · exact Or.inl (sum_eq_zero_of_tendsto X hX Z hs hlim)
  obtain ⟨e, -, hZ⟩ := eq_signedTuple_of_critical Z hcrit hs
  obtain ⟨S, hZS, hSneg⟩ : ∃ S : Finset (Idx n), Z = signedTuple e S ∧
      ∀ i ∈ S, inner (𝕜 := ℝ) (Z i : EucSpace d) (∑ k : Idx n, (Z k : EucSpace d)) < 0 :=
    ⟨_, hZ, fun i hi => (Finset.mem_filter.mp hi).2⟩
  clear hZ
  subst hZS
  rcases lt_trichotomy S.card 1 with h0 | h1 | h2
  · obtain ⟨i, j, hij⟩ := hnt
    rw [Finset.card_eq_zero.mp (show S.card = 0 by omega)] at hij
    simp [coe_signedTuple] at hij
  · obtain ⟨i₀, rfl⟩ := Finset.card_eq_one.mp h1
    refine Or.inr (Or.inr ⟨?_, e, i₀, hlim⟩)
    obtain ⟨i, j, hij⟩ := hnt
    have hval : i.val ≠ j.val := Fin.val_ne_of_ne fun h => hij (by rw [h])
    have := i.isLt
    have := j.isLt
    by_contra hlt
    have hn2 : (n : ℝ) = 2 := by exact_mod_cast (by omega : n = 2)
    apply hs
    rw [sum_signedTuple, h1, hn2]
    norm_num
  · obtain ⟨i, hi, j, hj, hij⟩ := Finset.one_lt_card.mp h2
    refine Or.inr (Or.inl ⟨i, j, hij, coord_eq_of_tendsto X hX _ hlim ?_ (hSneg i hi)⟩)
    simp [coe_signedTuple, hi, hj]

/-- The hypotheses of `tendsto_critical_cases` hold for the constant trajectory at
the antipodal pair of the plane, which falls under the first case. -/
example : E0GradientAscent 2 2 (fun _ => antipodalPair 2 (basePoint 1)) ∧
    IsCriticalE0 2 2 (antipodalPair 2 (basePoint 1)) ∧
    NonTrivialTuple 2 2 (antipodalPair 2 (basePoint 1)) :=
  ⟨E0GradientAscent_const _ (antipodalPair_critical_nonTrivial 2 (basePoint 1)).1,
    antipodalPair_critical_nonTrivial 2 (basePoint 1)⟩

variable (d n) in
/-- **Lemma (l:nosaddleconv).** *No-saddle-convergence lemma.*

On a compact Riemannian manifold the set of initial conditions whose
gradient-ascent trajectory converges to a strict saddle of a smooth `f` has
zero volume (center-stable manifold theorem). It is stated here in the only
instance the survey uses it: `ℳ = (𝕊^{d-1})^n`, `f = 𝖤_0`, and — by
`yury_lemma` — the strict saddles are the non-trivial critical points. Volume
is the uniform measure `UniformTuple`.

The proof is the source's for the saddles `signedTuple e {i₀}`, one particle
opposite to the others (`measure_tendsto_signedTuple`): countably many local
center-stable sets `stayNear e' i₀ r` cover the trajectory from an integer time
`ℓ` on, each is null, its slices along `i₀` having at most one point
(`measure_closure_stayNear`), and the backward flow `Φ^{-ℓ}`, Lipschitz, keeps
it null. The other non-trivial critical points are reached only from the null
sets of zero barycentre and of two coincident particles
(`tendsto_critical_cases`).

**What the source says and what is changed here.**  The source states the
lemma for every compact Riemannian manifold; on `(𝕊⁰)^n`, of dimension `0`, it
holds vacuously, there being no strict saddles. The instance here puts the
non-trivial critical points in place of the strict saddles, which
`yury_lemma` justifies only for `d ≥ 2`. At `d = 1` every configuration is
critical and stationary, so the non-trivial ones, of mass `1 - 2^{1-n} > 0`
for `n ≥ 2`, converge to non-trivial critical points. The hypothesis
`2 ≤ d` is added, as in `yury_lemma`.

Source: arXiv:2312.10794v5, Appendix A, `l:nosaddleconv`. -/
theorem no_saddle_convergence (hd : 2 ≤ d) :
    ∀ P : Measure (SphereTuple d n), UniformTuple d n P →
      P { X₀ : SphereTuple d n |
          ∃ X : ℝ → SphereTuple d n, X 0 = X₀ ∧ E0GradientAscent d n X ∧
            ∃ Z : SphereTuple d n, IsCriticalE0 d n Z ∧ NonTrivialTuple d n Z ∧
              ∀ i : Idx n,
                Filter.Tendsto
                  (fun t : ℝ => ((X t i : EucSpace d) - (Z i : EucSpace d)))
                  Filter.atTop (nhds 0) } = 0 := by
  intro P hP
  rcases Nat.eq_zero_or_pos n with rfl | hn
  · refine measure_mono_null (fun X₀ hX₀ => ?_) measure_empty
    obtain ⟨-, -, -, -, -, ⟨i, -⟩, -⟩ := hX₀
    exact i.elim0
  refine measure_mono_null (fun X₀ hX₀ => ?_) (measure_union_null (measure_union_null
    (measure_sum_eq_zero hd hn P hP) (measure_exists_coords_eq hd P hP))
    (measure_iUnion_null fun h3 => measure_tendsto_signedTuple hd h3 P hP))
  obtain ⟨X, rfl, hX, Z, hcrit, hnt, hlim⟩ := hX₀
  rcases tendsto_critical_cases hX hcrit hnt hlim with h | h | ⟨h3, h⟩
  · exact Or.inl (Or.inl h)
  · exact Or.inl (Or.inr h)
  · exact Or.inr (Set.mem_iUnion.mpr ⟨h3, X, rfl, hX, h⟩)

/-- The hypothesis of `no_saddle_convergence` holds for the circle, `d = 2`; the
uniform law is quantified over. -/
example : 2 ≤ 2 := le_rfl

end Perspective
end Transformer
