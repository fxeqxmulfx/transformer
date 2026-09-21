/-
# Measure-to-measure interpolation — the ball lemmas in the source's order

`lem: two.balls` and `lem: tubular.mass.movement` of arXiv:2411.04551v3
(Appendix) choose the parameters, and with them the flow map `φ^t`, *before*
the initial measure: "there exist `𝐖, 𝐔, b` such that for any
`μ_0 ∈ 𝒫(𝕊^{d-1})` …".  In that order both are false as soon as the mass has
to move, in every dimension.

The obstruction is elementary.  `φ^T` is a bijection which is the identity off
`S` (the ball `𝓑_0`, or the union of the chain), so it maps `S` onto `S`: every
`x ∈ S` is `φ^T(y)` for some `y ∈ S`.  Take `x ∈ S` outside the target set `A`
and `μ_0 = δ_y`; then `μ(T) = δ_x` gives `A` no mass, while the claim asks for
`c · μ_0(S) = c > 0`.

The lemmas as stated in `BallTransport` choose `μ_0` first, as the proof does.
-/

import Transformer.Interpolation.BallTransport

open scoped ENNReal
open MeasureTheory

namespace Transformer
namespace Interpolation

variable (d : ℕ)

/-- A bijection which fixes every point outside `S` maps `S` onto `S`: every
point of `S` is the image of a point of `S`. -/
theorem exists_mem_eq_of_bijective_fix {X : Type*} {f : X → X}
    (hf : Function.Bijective f) {S : Set X} (hfix : ∀ x, x ∉ S → f x = x)
    {x : X} (hx : x ∈ S) : ∃ y ∈ S, f y = x := by
  obtain ⟨y, rfl⟩ := hf.2 x
  refine ⟨y, ?_, rfl⟩
  by_contra hy
  rw [hfix y hy] at hx
  exact hy hx

/-- The common core of `not_two_balls` and `not_tubular_mass_movement`: no
parameters and no transport map, the identity off `S`, move a fraction `c > 0`
of the mass of `S` into `A` for every initial measure, unless `S ⊆ A`. -/
theorem not_forall_mass_into (θ : TimeParams d) (φ : ℝ → SSphere d → SSphere d)
    (S A : Set (SSphere d)) (T c : ℝ) (hT : 0 ≤ T) (hc : 0 < c) (hSA : ¬ S ⊆ A)
    (hφ : IsTransportOutside d φ S T) :
    ¬ ∀ μ₀ : Perspective.ProbSphere d,
        (∃ μ : ℝ → Perspective.ProbSphere d, μ 0 = μ₀ ∧ cauchyPB d θ μ) ∧
        ∀ μ : ℝ → Perspective.ProbSphere d, μ 0 = μ₀ → cauchyPB d θ μ →
          (μ T : Measure (SSphere d)) = Measure.map (φ T) (μ₀ : Measure (SSphere d)) ∧
          ENNReal.ofReal c * (μ₀ : Measure (SSphere d)) S ≤ (μ T : Measure (SSphere d)) A := by
  intro h
  obtain ⟨x, hxS, hxA⟩ := Set.not_subset.mp hSA
  obtain ⟨-, hbij, hfix⟩ := hφ T ⟨hT, le_rfl⟩
  obtain ⟨y, hyS, rfl⟩ := exists_mem_eq_of_bijective_fix hbij hfix hxS
  obtain ⟨⟨μ, hμ0, hμ⟩, hall⟩ := h (Perspective.diracProb d y)
  obtain ⟨hmap, hbound⟩ := hall μ hμ0 hμ
  have h1 : ((Perspective.diracProb d y : Perspective.ProbSphere d) : Measure (SSphere d))
      = Measure.dirac y := rfl
  rw [hmap, h1, Measure.map_dirac, Measure.dirac_apply_of_mem hyS, Measure.dirac_apply,
    Set.indicator_of_notMem hxA, mul_one] at hbound
  exact absurd (ENNReal.ofReal_eq_zero.mp (le_antisymm hbound bot_le)) (not_le.mpr hc)

/-- **`lem: two.balls` in the source's order is false.**

Whenever `𝓑_0 ⊄ 𝓑_1` and `ε < 1`, no constant `(𝐖, 𝐔, b)` and flow map
`φ^t`, Lipschitz, invertible and the identity off `𝓑_0`, give
`μ(T, 𝓑_0 ∩ 𝓑_1) ≥ (1 - ε) μ_0(𝓑_0)` with `μ(T) = φ^T_# μ_0` for every
`μ_0`.  In any dimension; `two_balls` states the lemma with `μ_0` first.

Source: arXiv:2411.04551v3, Appendix, `lem: two.balls`. -/
theorem not_two_balls (z₀ z₁ : SSphere d) (R₀ R₁ : ℝ)
    (hsub : ¬ Metric.ball z₀ R₀ ⊆ Metric.ball z₁ R₁) (ε T : ℝ) (hε : ε < 1) (hT : 0 < T) :
    ¬ ∃ (W U : ParamMatrix d) (b : EucSpace d) (φ : ℝ → SSphere d → SSphere d),
        IsNeuralFlow d (fun _ => W) (fun _ => U) (fun _ => b) φ ∧
        IsTransportOutside d φ (Metric.ball z₀ R₀) T ∧
        ∀ μ₀ : Perspective.ProbSphere d,
          (∃ μ : ℝ → Perspective.ProbSphere d, μ 0 = μ₀ ∧ cauchyPB d (constNeural d W U b) μ) ∧
          ∀ μ : ℝ → Perspective.ProbSphere d, μ 0 = μ₀ → cauchyPB d (constNeural d W U b) μ →
            (μ T : Measure (SSphere d)) = Measure.map (φ T) (μ₀ : Measure (SSphere d)) ∧
            ENNReal.ofReal (1 - ε) * (μ₀ : Measure (SSphere d)) (Metric.ball z₀ R₀)
              ≤ (μ T : Measure (SSphere d)) (Metric.ball z₀ R₀ ∩ Metric.ball z₁ R₁) := by
  rintro ⟨W, U, b, φ, -, hφ, h⟩
  exact not_forall_mass_into d _ φ _ _ T (1 - ε) hT.le (by linarith)
    (fun hs => hsub fun x hx => (hs hx).2) hφ h

/-- **`lem: tubular.mass.movement` in the source's order is false.**

Whenever `⋃_k 𝓑_k ⊄ 𝓑_K` and `ε < 1`, no piecewise-constant `(𝐖, 𝐔, b)` and
flow map `φ^t`, Lipschitz, invertible and the identity off `⋃_k 𝓑_k`, give
`μ(T, 𝓑_K) ≥ (1 - ε)^K μ_0(⋃_k 𝓑_k)` with `μ(T) = φ^T_# μ_0` for every `μ_0`.
For `K ≥ 2` the chain hypotheses themselves force `⋃_k 𝓑_k ⊄ 𝓑_K`, `𝓑_0` being
nonempty and disjoint from `𝓑_K`.  `tubular_mass_movement` states the lemma
with `μ_0` first.

Source: arXiv:2411.04551v3, Appendix, `lem: tubular.mass.movement`. -/
theorem not_tubular_mass_movement (K : ℕ) (z : ℕ → SSphere d) (R : ℕ → ℝ)
    (hsub : ¬ (⋃ k ∈ Set.Iic K, Metric.ball (z k) (R k)) ⊆ Metric.ball (z K) (R K))
    (ε T : ℝ) (hε : ε < 1) (hT : 0 < T) :
    ¬ ∃ (W U : ℝ → ParamMatrix d) (b : ℝ → EucSpace d) (φ : ℝ → SSphere d → SSphere d),
        PiecewiseConstant d (neuralParams d W U b) T (K + 1) ∧
        IsNeuralFlow d W U b φ ∧
        IsTransportOutside d φ (⋃ k ∈ Set.Iic K, Metric.ball (z k) (R k)) T ∧
        ∀ μ₀ : Perspective.ProbSphere d,
          (∃ μ : ℝ → Perspective.ProbSphere d, μ 0 = μ₀ ∧ cauchyPB d (neuralParams d W U b) μ) ∧
          ∀ μ : ℝ → Perspective.ProbSphere d, μ 0 = μ₀ → cauchyPB d (neuralParams d W U b) μ →
            (μ T : Measure (SSphere d)) = Measure.map (φ T) (μ₀ : Measure (SSphere d)) ∧
            ENNReal.ofReal ((1 - ε) ^ K) *
                (μ₀ : Measure (SSphere d)) (⋃ k ∈ Set.Iic K, Metric.ball (z k) (R k))
              ≤ (μ T : Measure (SSphere d)) (Metric.ball (z K) (R K)) := by
  rintro ⟨W, U, b, φ, -, -, hφ, h⟩
  exact not_forall_mass_into d _ φ _ _ T ((1 - ε) ^ K) hT.le
    (pow_pos (by linarith) K) hsub hφ h

/-- A point of the sphere lies at distance `2` from its antipode. -/
theorem dist_antipode (x : SSphere d) : dist (antipode d x) x = 2 := by
  rw [Subtype.dist_eq, dist_eq_norm]
  change ‖-(x : EucSpace d) - x‖ = 2
  rw [← neg_add', norm_neg, ← two_smul ℝ, norm_smul, mem_sphere_zero_iff_norm.mp x.2]
  norm_num

/-- The hypotheses of `not_two_balls` are satisfiable inside the lemma's scope:
on the circle, `𝓑_0 = B(e, 3)` is the whole circle and meets `𝓑_1 = B(e, 1)`,
which misses the antipode `-e`. -/
example :
    (Metric.ball (basePoint 1) 3 ∩ Metric.ball (basePoint 1) 1 : Set (SSphere 2)).Nonempty ∧
      ¬ Metric.ball (basePoint 1) 3 ⊆ (Metric.ball (basePoint 1) 1 : Set (SSphere 2)) ∧
      (0 : ℝ) < 1 / 2 ∧ (1 / 2 : ℝ) < 1 ∧ (0 : ℝ) < 1 := by
  refine ⟨⟨basePoint 1, by simp, by simp⟩, fun h => ?_, by norm_num, by norm_num, one_pos⟩
  have hmem : antipode 2 (basePoint 1) ∈ Metric.ball (basePoint 1) 3 := by
    rw [Metric.mem_ball, dist_antipode]; norm_num
  have := h hmem
  rw [Metric.mem_ball, dist_antipode] at this
  norm_num at this

/-- The hypotheses of `not_tubular_mass_movement` are satisfiable inside the
lemma's scope: the chain `𝓑_0 = B(e, 3)`, `𝓑_1 = B(e, 1)` on the circle,
`K = 1`, whose union contains the antipode `-e ∉ 𝓑_1`. -/
example :
    (∀ k : ℕ, 1 ≤ k → k ≤ 1 →
        (Metric.ball (basePoint 1) ((fun k : ℕ => if k = 0 then (3 : ℝ) else 1) k)
          ∩ Metric.ball (basePoint 1) ((fun k : ℕ => if k = 0 then (3 : ℝ) else 1) (k - 1))
            : Set (SSphere 2)).Nonempty) ∧
      ¬ (⋃ k ∈ Set.Iic 1,
          Metric.ball (basePoint 1) ((fun k : ℕ => if k = 0 then (3 : ℝ) else 1) k))
        ⊆ (Metric.ball (basePoint 1) ((fun k : ℕ => if k = 0 then (3 : ℝ) else 1) 1)
            : Set (SSphere 2)) := by
  refine ⟨fun k hk hk1 => ⟨basePoint 1, ?_, ?_⟩, fun h => ?_⟩
  · obtain rfl : k = 1 := le_antisymm hk1 hk
    simp
  · obtain rfl : k = 1 := le_antisymm hk1 hk
    simp
  · have hmem : antipode 2 (basePoint 1) ∈ ⋃ k ∈ Set.Iic 1,
        Metric.ball (basePoint 1) ((fun k : ℕ => if k = 0 then (3 : ℝ) else 1) k) := by
      refine Set.mem_biUnion (x := 0) (by simp) ?_
      rw [Metric.mem_ball, dist_antipode]; norm_num
    have := h hmem
    rw [Metric.mem_ball, dist_antipode] at this
    norm_num at this

end Interpolation
end Transformer
