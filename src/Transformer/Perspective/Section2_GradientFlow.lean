/-
# §3.3–§3.4 — `USA` as a Wasserstein gradient flow, `SA` for a modified metric

Geshkovski, Letrouit, Polyanskiy, Rigollet — arXiv:2312.10794v5,
*A mathematical perspective on Transformers*.

Continues `Perspective.Section2_FlowMap`:

* `eq: logder`             — logarithmic-derivative form of `𝒳[μ]`,
* `USA`                    — the unnormalised `SA` model, and its vector field,
* `eq: pde.nosoftmaxZ`     — continuity equation for `USA`,
* `e:XmuE`, `eq: aggregation.eq`, `Lemma lem: dissipation`,
* §3.4 — `SA` is a gradient flow for a modified metric;
        `e:scalarproduct`, `eq: first.rewriting`.

The three PDEs of §3.3–§3.4 are `Perspective.auxCE` at three velocity fields;
what distinguishes them is exactly the identity `e:XmuE`, which says the first
two coincide.
-/

import Mathlib.Analysis.Calculus.Deriv.Add
import Mathlib.Analysis.Calculus.Deriv.Comp
import Mathlib.Analysis.Calculus.Deriv.Mul
import Mathlib.Analysis.SpecialFunctions.ExpDeriv
import Transformer.Perspective.Section2_FlowMap
import Transformer.Perspective.PartitionGradient

open scoped BigOperators
open Real MeasureTheory

namespace Transformer
namespace Perspective

variable (d n : ℕ)

/-! ### §3.3 — A Wasserstein gradient flow proxy (`USA`) -/

/-- **Equation (eq: logder).** Logarithmic-derivative form of `𝒳[μ]`:

  `𝒳[μ](x) = Proj_x ∇_x ( β⁻¹ log Z_{β,μ}(x) )`,

since `∇_x log Z_{β,μ}(x) = β Z_{β,μ}(x)⁻¹ ∫ exp(β ⟨x,y⟩) y dμ(y)`.

The differentiation under the integral sign is
`Perspective.gradient_log_partitionMu`.

Source: arXiv:2312.10794v5, §3.3, `eq: logder`. -/
theorem vectorField_eq_grad_log (β : ℝ) (hβ : 0 < β) (μ : ProbSphere d) :
    ∀ x : EucSpace d,
      vectorField d β μ x
        = proj d x (gradient (fun z => β⁻¹ * Real.log (partitionMu d β μ z)) x) := by
  intro x
  rw [gradient_log_partitionMu d β (ne_of_gt hβ) μ x, vectorField]

/-- The hypothesis of `vectorField_eq_grad_log` is satisfiable: `β = 1`. -/
example : (0 : ℝ) < 1 := one_pos

/-- **Equation (USA).** Unnormalised Self-Attention dynamics. -/
def USA (β : ℝ) (X : ℝ → SphereTuple d n) : Prop :=
  ∀ t : ℝ, ∀ i : Idx n,
    HasDerivAt (fun s => (X s i : EucSpace d))
      (proj d ((X t i : EucSpace d))
        (((n : ℝ)⁻¹) •
          ∑ j : Idx n,
            Real.exp (β * inner (𝕜 := ℝ)
                        ((X t i : EucSpace d))
                        ((X t j : EucSpace d)))
            • ((X t j : EucSpace d)))) t

/-- The mean-field vector field of `USA`: `eq: vfSd` with the partition
function dropped,

  `𝒳^{USA}[μ](x) = Proj_x ∫ exp(β ⟨x,y⟩) y dμ(y)`. -/
noncomputable def usaVectorField
    (β : ℝ) (μ : ProbSphere d) (x : EucSpace d) : EucSpace d :=
  proj d x
    (∫ y, Real.exp (β * inner (𝕜 := ℝ) x (y : EucSpace d)) • (y : EucSpace d)
      ∂(μ : Measure (SSphere d)))

/-- **Equation (eq: pde.nosoftmaxZ).** Continuity equation for `USA`: `eq: CE`
at the velocity field `𝒳^{USA}[μ(t)]`. -/
def usaContinuityEquation (β : ℝ) (μ : ℝ → ProbSphere d) : Prop :=
  auxCE d μ (fun t x => usaVectorField d β (μ t) x)

/-- **Lemma (e:XmuE).**  `𝒳^{USA}[μ] = Proj ∇ δ𝖤_β[μ]`: the first variation of
the interaction energy `𝖤_β` at `μ` is `x ↦ β⁻¹ Z_{β,μ}(x)`, and its ambient
gradient, projected onto the tangent space, is the `USA` vector field.

The differentiation under the integral sign is
`Perspective.gradient_partitionMu`.

Source: arXiv:2312.10794v5, §3.3, `e:XmuE`. -/
theorem usaVectorField_eq_grad_first_variation (β : ℝ) (hβ : 0 < β)
    (μ : ProbSphere d) :
    ∀ x : EucSpace d,
      usaVectorField d β μ x
        = proj d x (gradient (fun z => β⁻¹ * partitionMu d β μ z) x) := by
  intro x
  rw [gradient_partitionMu d β (ne_of_gt hβ) μ x, usaVectorField]

/-- The hypothesis of `usaVectorField_eq_grad_first_variation` is satisfiable:
`β = 1`. -/
example : (0 : ℝ) < 1 := one_pos

/-- **Equation (eq: aggregation.eq).** Aggregation form of the `USA`-PDE:
`eq: CE` driven by the gradient of the first variation of `𝖤_β`,

  `∂_t μ = -div( μ Proj ∇ (β⁻¹ Z_{β,μ}) )`.

By `e:XmuE` this is `eq: pde.nosoftmaxZ`; written this way it exhibits the
Wasserstein gradient-flow structure. -/
def aggregationEquation (β : ℝ) (μ : ℝ → ProbSphere d) : Prop :=
  auxCE d μ (fun t x =>
    proj d x (gradient (fun z => β⁻¹ * partitionMu d β (μ t) z) x))

/-- **Lemma (lem: dissipation).**  Along `USA` the interaction energy
dissipates at rate

  `d/dt 𝖤_β[μ(t)] = ∫ ‖𝒳^{USA}[μ(t)](x)‖² dμ(t,x)`

— the same identity as `eq: dissipation.softmax` with the partition-function
weight `Z_{β,μ}` removed, which is precisely what the normalisation costs.

*`β > 0` is a hypothesis*, the survey's standing assumption: at `β = 0` the
energy is Lean's junk value `(2·0)⁻¹ ∫∫ 1 = 0`, as for `dissipation_softmax`.

Not proved here: as for `dissipation_softmax`, differentiating the energy
along the flow is not formalized.

Source: arXiv:2312.10794v5, §3.3, `lem: dissipation`. -/
theorem usa_dissipation (β : ℝ) (hβ : 0 < β) (μ : ℝ → ProbSphere d)
    (hCE : usaContinuityEquation d β μ) :
    ∀ t : ℝ,
      HasDerivAt (fun s => interactionEnergy d β (μ s))
        (∫ x, ‖usaVectorField d β (μ t) (x : EucSpace d)‖ ^ 2
          ∂(μ t : Measure (SSphere d))) t := by
  sorry

/-- A Dirac mass is a stationary point of the `USA` vector field too: the only
point `δ_x` sees is `x`, so the integral is a multiple of `x` and `Proj_x`
kills it.  Dropping the partition function changes the length of the velocity,
never its direction. -/
theorem usaVectorField_diracProb_self (β : ℝ) (x : SSphere d) :
    usaVectorField d β (diracProb d x) (x : EucSpace d) = 0 := by
  have hx : ‖(x : EucSpace d)‖ = 1 := mem_sphere_zero_iff_norm.mp x.2
  have hμ : ((diracProb d x : ProbSphere d) : Measure (SSphere d))
      = Measure.dirac x := rfl
  rw [usaVectorField, hμ, integral_dirac]
  exact proj_smul_self hx _

/-- The constant curve at a Dirac mass solves `eq: pde.nosoftmaxZ`, so the
hypothesis of `usa_dissipation` is satisfiable. -/
theorem usaContinuityEquation_const_diracProb (β : ℝ) (x : SSphere d) :
    usaContinuityEquation d β (fun _ => diracProb d x) := by
  intro φ _ t
  have hμ : ((diracProb d x : ProbSphere d) : Measure (SSphere d))
      = Measure.dirac x := rfl
  have hrhs :
      (∫ y, inner (𝕜 := ℝ) (gradient φ (y : EucSpace d))
          (usaVectorField d β (diracProb d x) (y : EucSpace d))
        ∂((diracProb d x : ProbSphere d) : Measure (SSphere d))) = 0 := by
    rw [hμ, integral_dirac, usaVectorField_diracProb_self, inner_zero_right]
  rw [hrhs]
  exact hasDerivAt_const t _

/-- The hypotheses of `usa_dissipation` are satisfiable: `β = 1` and the
constant curve at a Dirac mass. -/
example : (0 : ℝ) < 1 ∧ usaContinuityEquation 1 1 (fun _ => diracProb 1 (basePoint 0)) :=
  ⟨one_pos, usaContinuityEquation_const_diracProb 1 1 (basePoint 0)⟩

/-! ### §3.4 — `SA` is a gradient flow for a modified metric -/

/-- The particle interaction energy
`𝖤_β(X) = (1/(2β)) Σ_i Σ_j exp(β ⟨V x_i, x_j⟩)`. -/
noncomputable def particleEnergy
    (β : ℝ) (V : ParamMatrix d) (X : SphereTuple d n) : ℝ :=
  (2 * β)⁻¹ *
    ∑ i : Idx n, ∑ j : Idx n,
      Real.exp (β * inner (𝕜 := ℝ) (V (X i)) ((X j) : EucSpace d))

/-- `Z_{β,i}(X) = Σ_j exp(β ⟨V x_i, x_j⟩)`. -/
noncomputable def particlePartition
    (β : ℝ) (V : ParamMatrix d) (X : SphereTuple d n) (i : Idx n) : ℝ :=
  ∑ j : Idx n, Real.exp (β * inner (𝕜 := ℝ) (V (X i)) ((X j) : EucSpace d))

/-- **Equation (e:scalarproduct).** Modified inner product on `T_X (𝕊^{d-1})^n`:

  `⟨(a_i), (b_i)⟩_X = Σ_i Z_{β,i}(X) ⟨a_i, b_i⟩`. -/
noncomputable def modifiedMetric
    (β : ℝ) (V : ParamMatrix d) (X : SphereTuple d n)
    (a b : Idx n → EucSpace d) : ℝ :=
  ∑ i : Idx n,
    particlePartition d n β V X i * inner (𝕜 := ℝ) (a i) (b i)

/-- **`SA` is a gradient flow for the modified metric.**

For `Q^⊤ K` self-adjoint and `V = Q^⊤ K`, the dynamics `eq: transformerSd.QKV`
is the gradient *ascent* flow of the particle energy `𝖤_β` with respect to
`e:scalarproduct`: for every curve `Y` through `X(t)` with velocity `b`,

  `d/ds 𝖤_β(Y(s))|_{s=0} = ⟨Ẋ(t), b⟩_{X(t)}`.

Taking `b = Ẋ(t)` recovers `Σ_i Z_{β,i} ‖ẋ_i‖² = d/dt 𝖤_β ≥ 0`, the particle
counterpart of `eq: dissipation.softmax`.

Self-adjointness is `hV`, `⟨V x, y⟩ = ⟨x, V y⟩`, spelled out rather than taken
through `ContinuousLinearMap.adjoint`, which would need the
`FiniteDimensional` infrastructure here.

The paper's standing assumption `V = Q^⊤K` is `hQK`, `⟨Q x, K y⟩ = ⟨V x, y⟩`,
and it is what makes the statement true: the attention weight the dynamics
uses has to be the exponent the energy differentiates.  Instantiating the
dynamics at `Q = K = V` instead — as this file did before the proof was
attempted — gives the weight `⟨V x_i, V x_j⟩ = ⟨V² x_i, x_j⟩` against the
value `V x_j`, and the identity is then false unless `V² = V`.

Source: arXiv:2312.10794v5, §3.4. -/
theorem sa_is_gradient_flow (β : ℝ) (Q K V : ParamMatrix d) (X : ℝ → SphereTuple d n)
    (hβ : 0 < β)
    (hV : ∀ x y : EucSpace d, inner (𝕜 := ℝ) (V x) y = inner (𝕜 := ℝ) x (V y))
    (hQK : ∀ x y : EucSpace d,
      inner (𝕜 := ℝ) (Q x) (K y) = inner (𝕜 := ℝ) (V x) y)
    (hX : Perspective.transformerODE d n β (fun _ => Q) (fun _ => K) (fun _ => V) X) :
    ∀ (t : ℝ) (Y : ℝ → SphereTuple d n) (b : Idx n → EucSpace d),
    Y 0 = X t →
    (∀ i : Idx n, HasDerivAt (fun s => (Y s i : EucSpace d)) (b i) 0) →
    HasDerivAt (fun s => particleEnergy d n β V (Y s))
      (modifiedMetric d n β V (X t)
        (fun i => deriv (fun s => (X s i : EucSpace d)) t) b) 0 := by
  intro t Y b hY0 hb
  have hβ0 : β ≠ 0 := ne_of_gt hβ
  -- `b` is tangent to the sphere: differentiate `⟨Y s i, Y s i⟩ = 1`.
  have htang : ∀ i : Idx n, inner (𝕜 := ℝ) ((X t i : EucSpace d)) (b i) = (0 : ℝ) := by
    intro i
    have hconst : (fun s : ℝ => inner (𝕜 := ℝ) ((Y s i : EucSpace d)) ((Y s i : EucSpace d)))
        = fun _ : ℝ => (1 : ℝ) := by
      funext s
      rw [real_inner_self_eq_norm_sq, mem_sphere_zero_iff_norm.mp (Y s i).2, one_pow]
    have h1 := HasDerivAt.inner (𝕜 := ℝ) (hb i) (hb i)
    rw [hconst] at h1
    have h2 := h1.unique (hasDerivAt_const (0 : ℝ) (1 : ℝ))
    rw [hY0, real_inner_comm ((X t i : EucSpace d)) (b i)] at h2
    linarith
  -- The weight is symmetric, because `V` is self-adjoint.
  have hsymm : ∀ i j : Idx n,
      Real.exp (β * inner (𝕜 := ℝ) (V ((X t i : EucSpace d))) ((X t j : EucSpace d)))
        = Real.exp (β * inner (𝕜 := ℝ) (V ((X t j : EucSpace d))) ((X t i : EucSpace d))) := by
    intro i j
    rw [hV, real_inner_comm]
  -- A symmetric weight makes the two terms of `d𝖤_β` the same sum.
  have pair : ∀ w g : Idx n → Idx n → ℝ, (∀ i j : Idx n, w i j = w j i) →
      (2 * β)⁻¹ * ∑ i : Idx n, ∑ j : Idx n, w i j * (β * (g j i + g i j))
        = ∑ i : Idx n, ∑ j : Idx n, w i j * g i j := by
    intro w g hw
    have h1 : ∑ i : Idx n, ∑ j : Idx n, w i j * g j i
        = ∑ i : Idx n, ∑ j : Idx n, w i j * g i j := by
      rw [Finset.sum_comm]
      exact Finset.sum_congr rfl fun i _ =>
        Finset.sum_congr rfl fun j _ => by rw [hw]
    have h2 : ∑ i : Idx n, ∑ j : Idx n, w i j * (β * (g j i + g i j))
        = β * (∑ i : Idx n, ∑ j : Idx n, w i j * g j i)
          + β * ∑ i : Idx n, ∑ j : Idx n, w i j * g i j := by
      rw [Finset.mul_sum, Finset.mul_sum, ← Finset.sum_add_distrib]
      refine Finset.sum_congr rfl fun i _ => ?_
      rw [Finset.mul_sum, Finset.mul_sum, ← Finset.sum_add_distrib]
      exact Finset.sum_congr rfl fun j _ => by ring
    rw [h2, h1]
    field_simp
    ring
  -- The dynamics, rewritten in the exponent the energy uses.
  have hQKw : ∀ i j : Idx n,
      Real.exp (β * inner (𝕜 := ℝ) (Q ((X t i : EucSpace d))) (K ((X t j : EucSpace d))))
        = Real.exp (β * inner (𝕜 := ℝ) (V ((X t i : EucSpace d))) ((X t j : EucSpace d))) := by
    intro i j
    rw [hQK]
  have hZ : ∀ i : Idx n,
      partitionQKV d n β (fun _ => Q) (fun _ => K) X t i
        = particlePartition d n β V (X t) i := by
    intro i
    simp only [partitionQKV, particlePartition]
    exact Finset.sum_congr rfl fun j _ => hQKw i j
  have hZpos : ∀ i : Idx n, (0 : ℝ) < particlePartition d n β V (X t) i := fun i =>
    Finset.sum_pos (fun j _ => Real.exp_pos _) ⟨i, Finset.mem_univ i⟩
  have hproj : ∀ (i : Idx n) (v : EucSpace d),
      inner (𝕜 := ℝ) (proj d ((X t i : EucSpace d)) v) (b i) = inner (𝕜 := ℝ) v (b i) := by
    intro i v
    rw [proj, inner_sub_left, real_inner_smul_left, htang i, mul_zero, sub_zero]
  -- The right-hand side: the partition function cancels, the projection is
  -- invisible against a tangent vector.
  have hmm : modifiedMetric d n β V (X t)
      (fun i => deriv (fun s => (X s i : EucSpace d)) t) b
      = ∑ i : Idx n, ∑ j : Idx n,
          Real.exp (β * inner (𝕜 := ℝ) (V ((X t i : EucSpace d))) ((X t j : EucSpace d)))
            * inner (𝕜 := ℝ) (V ((X t j : EucSpace d))) (b i) := by
    simp only [modifiedMetric]
    refine Finset.sum_congr rfl fun i _ => ?_
    rw [(hX t i).deriv]
    simp only [hZ, hQKw]
    rw [hproj, real_inner_smul_left, sum_inner, ← mul_assoc,
      mul_inv_cancel₀ (hZpos i).ne', one_mul]
    exact Finset.sum_congr rfl fun j _ => real_inner_smul_left _ _ _
  -- The left-hand side: differentiate the energy term by term.
  have hVb : ∀ i : Idx n,
      HasDerivAt (fun s => V ((Y s i : EucSpace d))) (V (b i)) 0 := by
    intro i
    simpa [Function.comp_def] using V.hasFDerivAt.comp_hasDerivAt 0 (hb i)
  have hterm : ∀ i j : Idx n,
      HasDerivAt
        (fun s => Real.exp (β * inner (𝕜 := ℝ) (V ((Y s i : EucSpace d))) ((Y s j : EucSpace d))))
        (Real.exp (β * inner (𝕜 := ℝ) (V ((X t i : EucSpace d))) ((X t j : EucSpace d))) *
          (β * (inner (𝕜 := ℝ) (V ((X t i : EucSpace d))) (b j)
                + inner (𝕜 := ℝ) (V ((X t j : EucSpace d))) (b i)))) 0 := by
    intro i j
    have hswap : inner (𝕜 := ℝ) (V (b i)) ((X t j : EucSpace d))
        = inner (𝕜 := ℝ) (V ((X t j : EucSpace d))) (b i) := by
      rw [hV, real_inner_comm]
    have hi : HasDerivAt
        (fun s => inner (𝕜 := ℝ) (V ((Y s i : EucSpace d))) ((Y s j : EucSpace d)))
        (inner (𝕜 := ℝ) (V ((Y 0 i : EucSpace d))) (b j)
          + inner (𝕜 := ℝ) (V (b i)) ((Y 0 j : EucSpace d))) 0 :=
      HasDerivAt.inner (𝕜 := ℝ) (hVb i) (hb j)
    have h := (hi.const_mul β).exp
    rw [hY0, hswap] at h
    exact h
  have sumDeriv : ∀ (A : Idx n → ℝ → ℝ) (A' : Idx n → ℝ),
      (∀ i : Idx n, HasDerivAt (A i) (A' i) 0) →
      HasDerivAt (fun s => ∑ i : Idx n, A i s) (∑ i : Idx n, A' i) 0 := by
    intro A A' h
    have he : (fun s : ℝ => ∑ i : Idx n, A i s) = ∑ i : Idx n, A i := by
      funext s; simp
    rw [he]
    exact HasDerivAt.sum fun i _ => h i
  have hgrid : HasDerivAt
      (fun s => ∑ i : Idx n, ∑ j : Idx n,
        Real.exp (β * inner (𝕜 := ℝ) (V ((Y s i : EucSpace d))) ((Y s j : EucSpace d))))
      (∑ i : Idx n, ∑ j : Idx n,
        Real.exp (β * inner (𝕜 := ℝ) (V ((X t i : EucSpace d))) ((X t j : EucSpace d))) *
          (β * (inner (𝕜 := ℝ) (V ((X t i : EucSpace d))) (b j)
                + inner (𝕜 := ℝ) (V ((X t j : EucSpace d))) (b i)))) 0 :=
    sumDeriv _ _ fun i => sumDeriv _ _ fun j => hterm i j
  have hvalue :
      (2 * β)⁻¹ * ∑ i : Idx n, ∑ j : Idx n,
        Real.exp (β * inner (𝕜 := ℝ) (V ((X t i : EucSpace d))) ((X t j : EucSpace d))) *
          (β * (inner (𝕜 := ℝ) (V ((X t i : EucSpace d))) (b j)
                + inner (𝕜 := ℝ) (V ((X t j : EucSpace d))) (b i)))
        = modifiedMetric d n β V (X t)
            (fun i => deriv (fun s => (X s i : EucSpace d)) t) b := by
    rw [hmm]
    exact pair
      (fun i j => Real.exp (β * inner (𝕜 := ℝ) (V ((X t i : EucSpace d))) ((X t j : EucSpace d))))
      (fun i j => inner (𝕜 := ℝ) (V ((X t j : EucSpace d))) (b i)) hsymm
  rw [← hvalue]
  simpa only [particleEnergy] using hgrid.const_mul ((2 * β)⁻¹)

/-- A single token sitting still solves `eq: transformerSd.QKV` with
`Q = K = V = I_d`: it attends only to itself, so the attention average is `x`
and `Proj_x x = 0`. -/
theorem transformerODE_const_one (β : ℝ) (p : SSphere d) :
    Perspective.transformerODE d 1 β
      (fun _ => ContinuousLinearMap.id ℝ (EucSpace d))
      (fun _ => ContinuousLinearMap.id ℝ (EucSpace d))
      (fun _ => ContinuousLinearMap.id ℝ (EucSpace d)) (fun _ _ => p) := by
  intro t _
  have hp : ‖(p : EucSpace d)‖ = 1 := mem_sphere_zero_iff_norm.mp p.2
  simp only [ContinuousLinearMap.coe_id', id_eq, Fin.sum_univ_one, smul_smul]
  rw [proj_smul_self hp]
  exact hasDerivAt_const t _

/-- The hypotheses of `sa_is_gradient_flow` are satisfiable, and by a genuine
solution: `β = 1`, `Q = K = V = I_d` — self-adjoint, and `⟨Q x, K y⟩ = ⟨V x, y⟩`
because all three are the identity — and the stationary one-token solution of
`transformerODE_const_one`. -/
example :
    (0 : ℝ) < 1 ∧
      (∀ x y : EucSpace 1,
        inner (𝕜 := ℝ) (ContinuousLinearMap.id ℝ (EucSpace 1) x) y
          = inner (𝕜 := ℝ) x (ContinuousLinearMap.id ℝ (EucSpace 1) y)) ∧
      (∀ x y : EucSpace 1,
        inner (𝕜 := ℝ) (ContinuousLinearMap.id ℝ (EucSpace 1) x)
            (ContinuousLinearMap.id ℝ (EucSpace 1) y)
          = inner (𝕜 := ℝ) (ContinuousLinearMap.id ℝ (EucSpace 1) x) y) ∧
      Perspective.transformerODE 1 1 1
        (fun _ => ContinuousLinearMap.id ℝ (EucSpace 1))
        (fun _ => ContinuousLinearMap.id ℝ (EucSpace 1))
        (fun _ => ContinuousLinearMap.id ℝ (EucSpace 1))
        (fun _ _ => basePoint 0) :=
  ⟨one_pos, fun _ _ => rfl, fun _ _ => rfl, transformerODE_const_one 1 1 (basePoint 0)⟩

/-- **Equation (eq: first.rewriting).** Rewriting of `eq: conteqSd` through
`eq: logder`: `eq: CE` at the velocity field `Proj_x ∇(β⁻¹ log Z_{β,μ(t)})`.
By `VectorFieldEqGradLog` it is `continuityEquation`; the difference with
`aggregationEquation` is exactly the logarithm, i.e. the normalisation. -/
def conteqFirstRewriting (β : ℝ) (μ : ℝ → ProbSphere d) : Prop :=
  auxCE d μ (fun t x =>
    proj d x (gradient (fun z => β⁻¹ * Real.log (partitionMu d β (μ t) z)) x))

end Perspective
end Transformer
