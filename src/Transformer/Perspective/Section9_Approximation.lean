/-
# §10 — Approximation, control, training

Geshkovski, Letrouit, Polyanskiy, Rigollet — arXiv:2312.10794v5,
*A mathematical perspective on Transformers*.

The original §10 is a brief survey of known results on universal
approximation.  Two of them are recorded here as `Prop`-valued definitions,
written out in full so that what is *not* proved is visible: the class of
approximating networks (`discreteLayer`, `vectorFieldQKV`), the sense of
approximation (uniform on a compact set; bounded-Lipschitz on measures) and
the quantifier order are all part of the statement.

Neither is a theorem: the proofs are Yun et al. and Furuya et al.
respectively, and neither is available here.
-/

import Transformer.Basic
import Transformer.Perspective.Section1_IPS
import Transformer.Perspective.Section2_FlowMap

open scoped BigOperators
open Real MeasureTheory

namespace Transformer
namespace Perspective

variable (d n : ℕ)

/-! ### Discrete-time Transformers -/

/-- Partition function of one discrete self-attention block:

  `Z_{β,i}(x) = Σ_j exp(β ⟨Q x_i, K x_j⟩)`. -/
noncomputable def discretePartition
    (β : ℝ) (Q K : ParamMatrix d) (x : Idx n → EucSpace d) (i : Idx n) : ℝ :=
  ∑ j : Idx n, Real.exp (β * inner (𝕜 := ℝ) (Q (x i)) (K (x j)))

/-- One discrete Transformer block in the residual form of `eq: resnet`:
self-attention followed by a one-hidden-layer feed-forward map,

  `x_i ↦ x_i + Z_{β,i}(x)⁻¹ Σ_j exp(β ⟨Q x_i, K x_j⟩) V x_j + w σ(a x_i + b)`.

This is the discrete-time counterpart of `fullTransformer` at one head, and
the class of maps the approximation result below composes. -/
noncomputable def discreteLayer
    (β : ℝ) (Q K V : ParamMatrix d) (σ : ℝ → ℝ)
    (w a : ParamMatrix d) (b : EucSpace d)
    (x : Idx n → EucSpace d) (i : Idx n) : EucSpace d :=
  x i
    + (discretePartition d n β Q K x i)⁻¹ •
        ∑ j : Idx n, Real.exp (β * inner (𝕜 := ℝ) (Q (x i)) (K (x j))) • V (x j)
    + w (EuclideanSpace.equiv _ ℝ |>.symm
          (fun k => σ ((EuclideanSpace.equiv _ ℝ (a (x i) + b)) k)))

/-- *Universal approximation (Yun et al.).*  Discrete-time Transformers with
translation parameters approximate an arbitrary continuous sequence-to-sequence
map uniformly on compacta, as the number of layers tends to `+∞`.

The network is presented as a recursion (`x (k+1) = discreteLayer … (x k)`)
rather than as a composition, so that the depth `L` and the per-layer
parameters `Q, K, V, w, a, b` are the existential witnesses and the initial
sequence ranges over the compact set.

A `Prop`-valued definition and not a theorem: the proof is Yun, Bhojanapalli,
Rawat, Reddi, Kumar, *Are Transformers universal approximators of
sequence-to-sequence functions?*, ICLR 2020.

Source: arXiv:2312.10794v5, §10. -/
def UniversalApproximationDiscrete (β : ℝ) : Prop :=
  ∀ f : (Idx n → EucSpace d) → (Idx n → EucSpace d), Continuous f →
  ∀ S : Set (Idx n → EucSpace d), IsCompact S →
  ∀ ε : ℝ, 0 < ε →
  ∃ (L : ℕ) (Q K V w a : ℕ → ParamMatrix d) (b : ℕ → EucSpace d) (σ : ℝ → ℝ),
    Continuous σ ∧
    ∀ x₀ ∈ S, ∀ x : ℕ → Idx n → EucSpace d,
      x 0 = x₀ →
      (∀ k : ℕ,
        x (k + 1) = discreteLayer d n β (Q k) (K k) (V k) σ (w k) (a k) (b k) (x k)) →
      ∀ i : Idx n, ‖x L i - f x₀ i‖ < ε

/-! ### Measure-to-measure flow maps -/

/-- The mean-field vector field of `eq: SA.QKV`, i.e. `eq: vfSd` with general
time-dependent `Q, K, V`:

  `𝒳_t[μ](x) = Proj_x ( (∫ exp(β ⟨Q_t x, K_t y⟩) dμ(y))⁻¹
                          ∫ exp(β ⟨Q_t x, K_t y⟩) V_t y dμ(y) )`.

`vectorField` is this one at `Q = K = V = I_d`. -/
noncomputable def vectorFieldQKV
    (β : ℝ) (Q K V : TimeParam d) (t : ℝ) (μ : ProbSphere d) (x : EucSpace d) :
    EucSpace d :=
  proj d x
    ((∫ y, Real.exp (β * inner (𝕜 := ℝ) (Q t x) (K t (y : EucSpace d)))
        ∂(μ : Measure (SSphere d)))⁻¹ •
      ∫ y, Real.exp (β * inner (𝕜 := ℝ) (Q t x) (K t (y : EucSpace d)))
            • V t (y : EucSpace d) ∂(μ : Measure (SSphere d)))

/-- *Measure-to-measure universal approximation* (Agrachev–Sarychev,
Furuya–de Hoop–Peyré et al.).  The time-`T` flow maps of the mean-field
Transformer dynamics approximate an arbitrary continuous self-map of
`𝒫(𝕊^{d-1})`.

Closeness of measures is the bounded-Lipschitz (Dudley) distance, spelled out
by testing against `1`-Lipschitz functions bounded by `1`; it metrizes the weak
topology, which is the one `Continuous Φ` refers to.  The flow map is presented
through `auxCE`: `m` is any solution of the continuity equation driven by
`vectorFieldQKV` with `m 0 = μ`, and it is `m T` that must be close to `Φ μ`.

A `Prop`-valued definition and not a theorem: neither the approximation result
nor well-posedness of the continuity equation is proved here.

Source: arXiv:2312.10794v5, §10. -/
def UniversalApproximationMeasure (β : ℝ) : Prop :=
  ∀ Φ : ProbSphere d → ProbSphere d, Continuous Φ →
  ∀ ε : ℝ, 0 < ε →
  ∃ (T : ℝ) (Q K V : TimeParam d), 0 < T ∧
    ∀ (μ : ProbSphere d) (m : ℝ → ProbSphere d),
      m 0 = μ →
      auxCE d m (fun t x => vectorFieldQKV d β Q K V t (m t) x) →
      ∀ φ : EucSpace d → ℝ, LipschitzWith 1 φ → (∀ x : EucSpace d, |φ x| ≤ 1) →
        |(∫ x, φ (x : EucSpace d) ∂(m T : Measure (SSphere d)))
            - ∫ x, φ (x : EucSpace d) ∂((Φ μ : ProbSphere d) : Measure (SSphere d))|
          < ε

end Perspective
end Transformer
