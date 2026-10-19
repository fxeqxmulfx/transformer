module

public import Transformer.FrankWolfe.Section5_MetaTail
public import Transformer.FrankWolfe.Section5_BinomialExp

/-!
# Attention's forward pass and Frank-Wolfe — proved dynamic metastability

The corrected theorem is assembled from the actual attention transition
identity, the score gap near the vertices, the pathwise first-exit count,
and the dependent-step binomial bound. It uses no unproved result about
the preceding clustering phase. The zero-diameter case has a zero lower
bound in the stated formula and is included without division assumptions.
Source: arXiv:2508.09628v1, §5, `lem: metastab.1`.
-/

@[expose] public section

open scoped BigOperators Pointwise
open Real MeasureTheory

namespace Transformer
namespace FrankWolfe

/-- **Theorem (lem: metastab.1) — metastability, as repaired.**

In the setup of `lem: first.phase` — a polytope `𝒦` with vertices `v`, particles `X₀ ∈ 𝒦ⁿ`
attached to vertices by `σ`, each within `ρ` of its own vertex — let `ε > 0` and `γ ∈ (0,1)`. The
exit time

  `T₂ = inf{t : x_{ji}^t ∉ conv{x_{ji}^0}_j + B(0,ε) for some i, j}`

then satisfies, for `t > 1`,

  `ℙ[T₂ ≥ t] ≥ 1 - exp(M (1 + log(t n² / M) - β c₀ / 2))`,   `M = ⌈ε / (γ 𝖽(𝒦))⌉`,

where `𝖽(𝒦)` is the diameter. The right-hand side is close to `1` up to `t ∼ e^{β c₀ / 2}`.

**What the source says and what is changed here.**  The printed bound

  `ℙ[T₂ ≥ t] ≥ 1 - exp((1 + ε/γ) log(γt/ε) + (1 + ε/γ) log n - β c₀ ε / (2γ))`

is not what the proof gives, and the two differ in the exponent. The proof shows that a particle
cannot leave its group's `ε`-neighbourhood in fewer than `M ≈ ε/(γ𝖽(𝒦))` steps in which some
particle attends to another group, each such step having probability at most `n² e^{-λβ}`; the
printed exponent `β c₀ ε / (2γ)` drops the factor `1/𝖽(𝒦)` and puts `c₀` in place of `λ ≤ c₀`.
It is not invariant under rescaling the polytope, which the dynamics is (at inverse temperature
`β s²`), so it cannot be right as printed. Four repairs, each a missing hypothesis or a wrong
constant:

* the number of exits needed is `M`, not `ε/γ`, and the tail estimate is the binomial one,
  `ℙ[Bin(N,p) ≥ M] ≤ C(N,M) p^M ≤ (e N p / M)^M` — the printed `(N p / M)^M` lacks the `e`;
* `cl: group.cluster.bound` bounds `⟨x, v_m⟩ ≤ ⟨v_ℓ, v_m⟩ + ‖x - v_ℓ‖ ‖v_m‖`, so the norm `R` of
  the vertices enters, not `𝖽(𝒦)`; with `r = ρ + ε` the score of another group is at most
  `e^{-β (c₀ - 4Rr - r²)}` relative to the particle's own, and `λ = c₀ - 4Rr - r²` is positive only
  when `r` is small. The printed statement asks for that of `ε` alone (`ε_*`), while `Cτ` may be as
  large as `C c₀ / (2 max_j ‖v_i - v_j‖)`, so the smallness of `ρ + ε` is a hypothesis here:
  `4R(ρ + ε) + 2(ρ + ε)² ≤ c₀/2`, which gives `λ ≥ c₀/2`;
* `C`, `τ`, `ε_*` and `β_*` only ever served to make `ρ = Cτ` and `ε` small enough for that
  estimate, so they are replaced by it: `ρ` is the radius of the initial balls and no `β_*` is
  needed (`β ≥ 0`);
* `κ ≤ n` and `ε/γ ≥ 2𝖽(𝒦)` are not used by the argument and are dropped, which makes the
  statement stronger, not weaker. A separate `ρ > 0` input is also unnecessary: the initial
  ball membership already implies it when a particle exists, and the empty case is included.

`T₂` is not introduced as an `ℕ∞`-valued random variable. The event `{T₂ ≥ t}` is the event that
no particle has left its group's `ε`-neighbourhood at any step `s < t`, and that is what the
conclusion is stated about — the same set, without the extra definition. Relabelling `x_ℓ^0` as
`x_{ji}^0` is the fibre `σ⁻¹(i)`, so `conv{x_{ji}^0}_j` is `groupHull X₀ σ (σ i)`.

**Proof.** Count a bad step only while all particles are within `ρ + ε` of
their assigned vertices. Its probability conditional on any event of the
past is bounded by `n² exp(-β c₀/2)`, using the actual transition identity.
Before the first exit every bad step meets that condition. The pathwise
diameter bound therefore gives at least `M` counted steps on an exit path.
Intersections at prescribed times have probability at most `p^M`, and the
union over all `M`-element subsets of `floor(t)` gives the binomial bound.
The exponential series supplies the factorial estimate used to pass to
the stated exponent. Neither steps nor simultaneous particle updates are
assumed independent. When the diameter vanishes the displayed lower
bound equals zero; this case is proved directly from nonnegativity.

Source: arXiv:2508.09628v1, §5, `lem: metastab.1`, `eq: metastability.bound`. -/
theorem metastability {d κ n : ℕ} {K : Set (EucSpace d)} {v : Idx κ → EucSpace d}
    {c₀ R ρ ε γ β : ℝ} (hK : ConePolytope K v c₀) (hR : ∀ i, ‖v i‖ ≤ R)
    (hγ : γ ∈ Set.Ioo (0 : ℝ) 1) (hε : 0 < ε) (hβ : 0 ≤ β)
    (hsmall : 4 * R * (ρ + ε) + 2 * (ρ + ε) ^ 2 ≤ c₀ / 2)
    (X₀ : Idx n → EucSpace d) (σ : Idx n → Idx κ) (P : Measure (ℕ → Idx n → EucSpace d))
    (hX₀ : ∀ i, X₀ i ∈ K) (hρX : ∀ i, X₀ i ∈ Metric.ball (v (σ i)) ρ)
    (hP : IsSAProcess β γ X₀ P) (t : ℝ) (ht : 1 < t) :
    1 - Real.exp (⌈ε / (γ * Metric.diam K)⌉₊ *
        (1 + Real.log (t * n ^ 2 / ⌈ε / (γ * Metric.diam K)⌉₊) - β * (c₀ / 2)))
      ≤ (P {x : ℕ → Idx n → EucSpace d | ∀ s : ℕ, (s : ℝ) < t → ∀ i : Idx n,
          x s i ∈ groupHull X₀ σ (σ i) + Metric.ball (0 : EucSpace d) ε}).toReal := by
  let := hP.1
  by_cases hDzero : Metric.diam K = 0
  · simpa only [hDzero, mul_zero, div_zero, Nat.ceil_zero, Nat.cast_zero,
      zero_mul, Real.exp_zero, sub_self] using
      (ENNReal.toReal_nonneg (a := P {x : ℕ → Idx n → EucSpace d |
        ∀ s : ℕ, (s : ℝ) < t → ∀ i : Idx n,
          x s i ∈ groupHull X₀ σ (σ i) + Metric.ball (0 : EucSpace d) ε}))
  · have hDpos : 0 < Metric.diam K :=
      lt_of_le_of_ne Metric.diam_nonneg (fun h => hDzero h.symm)
    have hδ : 0 < γ * Metric.diam K := mul_pos hγ.1 hDpos
    have hM : 0 < ⌈ε / (γ * Metric.diam K)⌉₊ := Nat.ceil_pos.mpr (div_pos hε hδ)
    have hKcompact : IsCompact K := by
      rw [hK.hull]
      exact Set.Finite.isCompact_convexHull ℝ (Set.finite_range v)
    have hKv : Convex ℝ K := by
      rw [hK.hull]
      exact convex_convexHull ℝ _
    have hD : ∀ y ∈ K, ∀ y' ∈ K, ‖y - y'‖ ≤ Metric.diam K := by
      intro y hy y' hy'
      simpa only [dist_eq_norm] using Metric.dist_le_diam_of_mem hKcompact.isBounded hy hy'
    have hsurvival := hP.survival_real_horizon_bound hKcompact.isClosed hKv
      hγ.1.le hγ.2.le hδ hD hK.sep hR hsmall hβ hX₀ hρX t
    have htpos : 0 < t := zero_lt_one.trans ht
    have htail := choose_attention_tail_le_exp ⌊t⌋₊ ⌈ε / (γ * Metric.diam K)⌉₊ n hM
      htpos (Nat.floor_le htpos.le) (β * (c₀ / 2))
    calc 1 - Real.exp (⌈ε / (γ * Metric.diam K)⌉₊ *
          (1 + Real.log (t * n ^ 2 / ⌈ε / (γ * Metric.diam K)⌉₊) - β * (c₀ / 2)))
        ≤ 1 - (⌊t⌋₊.choose ⌈ε / (γ * Metric.diam K)⌉₊ : ℝ) *
            (n ^ 2 * Real.exp (-(β * (c₀ / 2)))) ^ ⌈ε / (γ * Metric.diam K)⌉₊ :=
          sub_le_sub_left htail 1
      _ ≤ _ := hsurvival

/-- The hypotheses of `metastability` are satisfiable, the process included: one vertex `e₀` in
`ℝ^1`, `𝒦 = {e₀}`, `c₀ = R = 1`, `γ = 1/2`, `ρ = ε = 1/20`, `β = 1`, one particle sitting at the
vertex, whose process is the point mass at the constant path (`isSAProcess_single`). -/
example :
    ConePolytope {(EuclideanSpace.single (0 : Fin 1) (1 : ℝ))}
      (fun _ : Idx 1 => (EuclideanSpace.single (0 : Fin 1) (1 : ℝ))) 1 ∧
    (∀ _ : Idx 1, ‖(EuclideanSpace.single (0 : Fin 1) (1 : ℝ))‖ ≤ 1) ∧
    (1 / 2 : ℝ) ∈ Set.Ioo (0 : ℝ) 1 ∧
    0 < (1 / 20 : ℝ) ∧ 0 ≤ (1 : ℝ) ∧
    4 * (1 : ℝ) * (1 / 20 + 1 / 20) + 2 * (1 / 20 + 1 / 20) ^ 2 ≤ 1 / 2 ∧
    (∀ i : Idx 1, (fun _ : Idx 1 => EuclideanSpace.single (0 : Fin 1) (1 : ℝ)) i ∈
      ({EuclideanSpace.single (0 : Fin 1) (1 : ℝ)} : Set (EucSpace 1))) ∧
    (∀ i : Idx 1, (fun _ : Idx 1 => EuclideanSpace.single (0 : Fin 1) (1 : ℝ)) i ∈
      Metric.ball ((fun _ : Idx 1 => EuclideanSpace.single (0 : Fin 1) (1 : ℝ)) i) (1 / 20)) ∧
    IsSAProcess 1 (1 / 2) (fun _ : Idx 1 => (EuclideanSpace.single (0 : Fin 1) (1 : ℝ)))
      (Measure.dirac fun _ => fun _ => (EuclideanSpace.single (0 : Fin 1) (1 : ℝ))) ∧
    1 < (2 : ℝ) := by
  refine ⟨⟨?_, ⟨?_, ?_⟩, fun i => ⟨0, by positivity, fun x hx y hy hxi hyi => ?_⟩,
    fun i j => rfl, one_pos, fun i j hij => absurd (Subsingleton.elim i j) hij⟩,
    fun _ => by simp, ⟨by norm_num, by norm_num⟩, by norm_num, by norm_num, by norm_num,
    by simp, fun i => Metric.mem_ball_self (by norm_num), isSAProcess_single _ _ _, by norm_num⟩
  · rw [Set.range_const, convexHull_singleton]
  · intro a b _
    exact Subsingleton.elim a b
  · rw [Set.range_const, extremePoints_singleton]
  · exact absurd hx hxi

end FrankWolfe
end Transformer
