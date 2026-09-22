/-
# §6.1 — Exponential rate when `d ≥ n`

Geshkovski, Letrouit, Polyanskiy, Rigollet — arXiv:2312.10794v5,
*A mathematical perspective on Transformers*.

`Theorem thm: d.infty`, `eq: expconvtocons`, and the lemma it is a corollary
of, `lem: hemisphere.clustering` (*cone collapse*).  Both are asserted for
four dynamics at once: `SA`, `USA`, `eq: transformerSd.QKV` with arbitrary
constant `Q, K` and `V = I_d`, and "the natural analogue of `USA` with these
parameters", which the survey does not write out and which is `usaQKV` here.
-/

import Transformer.Perspective.Section1_IPS
import Transformer.Perspective.Section2_GradientFlow
import Transformer.Perspective.Section3_SmallBeta
import Transformer.Perspective.ConeFields

open scoped BigOperators
open Real MeasureTheory

namespace Transformer
namespace Perspective

variable (d n : ℕ)

/-- **The natural analogue of `USA` with parameters `Q, K`** (and `V = I_d`):

  `ẋ_i(t) = Proj_{x_i(t)} ( (1/n) Σ_j exp(β ⟨Q x_i, K x_j⟩) x_j )`,

`USA` with the inner product of the weights replaced by `⟨Q ·, K ·⟩`, exactly
as `eq: transformerSd.QKV` is `SA` with it replaced.

Source: arXiv:2312.10794v5, §6.1, `thm: d.infty` ("the natural analogue of
`USA` with these parameters"). -/
def usaQKV (β : ℝ) (Q K : ParamMatrix d) (X : ℝ → SphereTuple d n) : Prop :=
  ∀ t : ℝ, ∀ i : Idx n,
    HasDerivAt (fun s => (X s i : EucSpace d))
      (proj d ((X t i : EucSpace d))
        (((n : ℝ)⁻¹) •
          ∑ j : Idx n,
            Real.exp (β * inner (𝕜 := ℝ) (Q (X t i : EucSpace d)) (K (X t j : EucSpace d)))
            • ((X t j : EucSpace d)))) t

/-- `eq: expconvtocons` for the dynamics `dyn` from `X₀`: there are `x⋆` and
`C, λ > 0` with `‖x_i(t) - x⋆‖ ≤ C e^{-λ t}` for every solution through `X₀`,
every `i` and every `t ≥ 0`.

Source: arXiv:2312.10794v5, §6.1, `eq: expconvtocons`. -/
def ExpConvergent (dyn : (ℝ → SphereTuple d n) → Prop) (X₀ : SphereTuple d n) : Prop :=
  ∃ (x_star : SSphere d) (C lam : ℝ), 0 < C ∧ 0 < lam ∧
    ∀ X : ℝ → SphereTuple d n, X 0 = X₀ → dyn X →
      ∀ i : Idx n, ∀ t : ℝ, 0 ≤ t →
        ‖((X t i : EucSpace d)) - x_star‖ ≤ C * Real.exp (-(lam * t))

/-- **Lemma (lem: hemisphere.clustering).** *Cone collapse.*

Let `(x_i(0))_{i ∈ [n]} ∈ (𝕊^{d-1})^n` lie in an open hemisphere:
`⟨x_i(0), w⟩ > 0` for some `w ∈ 𝕊^{d-1}` and every `i`.  Then there are
`x⋆ ∈ 𝕊^{d-1}` and `C, λ > 0` with `‖x_i(t) - x⋆‖ ≤ C e^{-λ t}` for all `i`
and `t ≥ 0`, for the solution of `SA` and of `USA`, and — for arbitrary
`d × d` matrices `Q, K` — of `eq: transformerSd.QKV` with `V = I_d` and of
`usaQKV`.

**What the source says and what is changed here.**  The survey assumes
`β > 0`, and states the lemma for `n ≥ 1` and `d ≥ 2` ("which holds for any
`n ≥ 1` and `d ≥ 2`").  None of them is used, and all are dropped: the proof
needs of the weights `a_ij = c_i e^{β⟨Q x_i, K x_j⟩}` only that they lie in
some `[m, M]` with `m > 0`, which holds for every real `β` — on the sphere
they lie in `[e^{-2L}/n, e^{2L}/n]`, `L = |β| ‖Q‖ ‖K‖`
(`softmaxNorm_bounds`, `usaNorm_bounds`) — and `n = 0` is vacuous.  A
solution is one on all of `ℝ`, as `SA`, `USA`, `transformerODE` and `usaQKV`
are defined, where the survey's Cauchy problem is posed on `ℝ_{≥0}`; the
survey's "the unique solution" is `ExpConvergent`'s "every solution through
`X₀`", and uniqueness is how one `x⋆` serves them all
(`expLimit_of_attnField`).

The proof is not the survey's.  The survey first obtains `x⋆` qualitatively
(`eq: qual.conv`), then integrates `α̇ ≥ (1 - α)/(2 n e^{2β})` for
`α(t) = min_i ⟨x_i(t), x⋆⟩` (`e:diffineqalpha.step2`; its integration is
`exp_rate_of_diffineqalpha`).  Here the chart `y = x / ⟨x, w⟩` of the
hemisphere makes the flow a consensus dynamics
`ẏ_i = Σ_j b_ij (y_j - y_i)`, `b_ij ≥ m r₀`, whose width decays like
`e^{-2 m r₀ t}` (`Perspective.ConeWidth`); the `y_i` converge at that rate to
some `y⋆`, and `x⋆ = y⋆ / ‖y⋆‖` (`Perspective.ConeLimit`).
`Perspective.ConeFields` puts the four dynamics in that form.

Source: arXiv:2312.10794v5, §6.1, `lem: hemisphere.clustering`; restated as
`thm:cone-collapse` in arXiv:2512.01868v4, §4 (`SA` and `USA`, `n ≥ 1`,
`d ≥ 2`, `β > 0`). -/
theorem cone_collapse (β : ℝ) (X₀ : SphereTuple d n)
    (hX₀ : ∃ w : SSphere d, ∀ i : Idx n,
              0 < inner (𝕜 := ℝ) ((X₀ i : EucSpace d)) ((w : EucSpace d))) :
    ExpConvergent d n (Perspective.SA d n β) X₀ ∧
    ExpConvergent d n (Perspective.USA d n β) X₀ ∧
    ∀ Q K : ParamMatrix d,
      ExpConvergent d n
          (transformerODE d n β (fun _ => Q) (fun _ => K) (fun _ => ContinuousLinearMap.id ℝ _))
          X₀ ∧
        ExpConvergent d n (usaQKV d n β Q K) X₀ := by
  obtain ⟨w, hw⟩ := hX₀
  rcases Nat.eq_zero_or_pos n with rfl | hn
  · have h0 : ∀ dyn, ExpConvergent d 0 dyn X₀ := fun _ =>
      ⟨w, 1, 1, one_pos, one_pos, fun _ _ _ i => i.elim0⟩
    exact ⟨h0 _, h0 _, fun _ _ => ⟨h0 _, h0 _⟩⟩
  have hn' : (0 : ℝ) < n := Nat.cast_pos.mpr hn
  have hsoft : ∀ (Q K : ParamMatrix d) (dyn : (ℝ → SphereTuple d n) → Prop),
      (∀ X, dyn X → ∀ (t : ℝ) (i : Idx n), HasDerivAt (fun s => (X s i : EucSpace d))
        (attnField β Q K (softmaxNorm β Q K) (fun j => (X t j : EucSpace d)) i) t) →
      ExpConvergent d n dyn X₀ := fun Q K _ hdyn =>
    expLimit_of_attnField hdyn contDiff_softmaxNorm (div_pos (exp_pos _) hn')
      (fun _ hY i j => softmaxNorm_bounds hY i j) hn w hw
  have husa : ∀ (Q K : ParamMatrix d) (dyn : (ℝ → SphereTuple d n) → Prop),
      (∀ X, dyn X → ∀ (t : ℝ) (i : Idx n), HasDerivAt (fun s => (X s i : EucSpace d))
        (attnField β Q K (fun _ _ => (n : ℝ)⁻¹) (fun j => (X t j : EucSpace d)) i) t) →
      ExpConvergent d n dyn X₀ := fun Q K _ hdyn =>
    expLimit_of_attnField hdyn (fun _ => contDiff_const) (div_pos (exp_pos _) hn')
      (fun _ hY i j => usaNorm_bounds hY i j) hn w hw
  set I := ContinuousLinearMap.id ℝ (EucSpace d)
  refine ⟨hsoft I I _ fun X hX t i => (hX t i).congr_deriv ?_,
    husa I I _ fun X hX t i => (hX t i).congr_deriv ?_,
    fun Q K => ⟨hsoft Q K _ fun X hX t i => (hX t i).congr_deriv ?_,
      husa Q K _ fun X hX t i => (hX t i).congr_deriv ?_⟩⟩
  all_goals simp only [attnField, softmaxNorm, attnExp, partitionSA, partitionQKV, I,
    ContinuousLinearMap.id_apply]

/-- The hypothesis of `cone_collapse` is satisfiable: one particle at
`basePoint 1`, in the hemisphere around itself. -/
example : ∃ w : SSphere 2, ∀ i : Idx 1,
    0 < inner (𝕜 := ℝ) (((fun _ => basePoint 1 : SphereTuple 2 1) i : EucSpace 2))
      ((w : EucSpace 2)) := by
  refine ⟨basePoint 1, fun _ => ?_⟩
  rw [real_inner_self_eq_norm_mul_norm, mem_sphere_zero_iff_norm.mp (basePoint 1).2]
  norm_num

/-- **Theorem (thm: d.infty), eq: expconvtocons.** *Exponential clustering when
`d ≥ n`.*

For `n ≥ 1`, `β > 0` and `d ≥ n`, if the initial points
`(x_i(0))_{i ∈ [n]} ∈ (𝕊^{d-1})^n` are uniformly distributed, then almost
surely there exist `x⋆ ∈ 𝕊^{d-1}` and constants `C, λ > 0` such that

  `‖x_i(t) - x⋆‖ ≤ C e^{-λ t}`  for all `i` and `t ≥ 0`,

for the solution of `SA`, of `USA`, and — for arbitrary `d × d` matrices `Q`
and `K` — of `eq: transformerSd.QKV` with `V = I_d` and of its `USA` analogue
`usaQKV`.  `x⋆`, `C` and `λ` depend on the dynamics; the null set does not
depend on `Q, K`, which is how "almost surely … arbitrary `Q, K`" reads: the
proof needs of `X₀` only that it lie in an open hemisphere, and concludes by
`cone_collapse`.

*Almost surely* is part of the statement and not a turn of phrase: for `n = 2`
the antipodal pair admits no such rate for `SA` at any `β`
(`antipodalPair_not_exponential`), so the `∀ X₀` reading is false.  The
initial sequence is drawn from the uniform law `UniformTuple` of §4.

Not proved here.

Source: arXiv:2312.10794v5, §6.1, `thm: d.infty`, `eq: expconvtocons`. -/
theorem d_infty_exponential
    (hn : 1 ≤ n) (β : ℝ) (hβ : 0 < β) (hdn : n ≤ d) :
    ∀ P : Measure (SphereTuple d n), UniformTuple d n P →
      ∀ᵐ X₀ ∂P,
        ExpConvergent d n (Perspective.SA d n β) X₀ ∧
        ExpConvergent d n (Perspective.USA d n β) X₀ ∧
        ∀ Q K : ParamMatrix d,
          ExpConvergent d n
              (transformerODE d n β (fun _ => Q) (fun _ => K) (fun _ => ContinuousLinearMap.id ℝ _))
              X₀ ∧
            ExpConvergent d n (usaQKV d n β Q K) X₀ := by
  sorry

/-- The hypotheses of `d_infty_exponential` are satisfiable: `d = n = 1`,
`β = 1`. -/
example : 1 ≤ 1 ∧ (0 : ℝ) < 1 ∧ 1 ≤ 1 := ⟨le_rfl, one_pos, le_rfl⟩

end Perspective
end Transformer
