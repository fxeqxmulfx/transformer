/-
# §4 — A single cluster for small β

Geshkovski, Letrouit, Polyanskiy, Rigollet — arXiv:2312.10794v5,
*A mathematical perspective on Transformers*.

This file formalizes §4 of the survey:

* `e:Snonres0`              — the `β = 0` dynamics,
* `Theorem p:beta0`         — almost-sure consensus at `β = 0`,
* `Theorem th:beta_small`   — probability of clustering tends to `1` as `β → 0`,
* the auxiliary equations  `eq: harry.potter`, `e:Ps0n`, `e:approxsphere`,
                            `eq: youareawizardharry`,
* `Theorem thm: beta.tiny`  — clustering when `β ≤ C/n`.
-/

import Transformer.Basic
import Transformer.Perspective.Section1_IPS
import Transformer.Perspective.Section2_FlowMap
import Mathlib.Analysis.Calculus.MeanValue
import Mathlib.MeasureTheory.Constructions.Pi

open scoped BigOperators
open Real MeasureTheory

namespace Transformer
namespace Perspective

open Perspective

variable (d n : ℕ)

/-- **Equation (e:Snonres0).** The `β = 0` dynamics common to both `SA` and
`USA`:

  `ẋ_i(t) = Proj_{x_i(t)} ( (1/n) Σ_j x_j(t) )`. -/
def beta0Dynamics (X : ℝ → SphereTuple d n) : Prop :=
  ∀ t : ℝ, ∀ i : Idx n,
    HasDerivAt (fun s => (X s i : EucSpace d))
      (proj d ((X t i : EucSpace d))
        (((n : ℝ)⁻¹) • ∑ j : Idx n, ((X t j : EucSpace d)))) t

/-- The uniform law on `(𝕊^{d-1})^n`: the `n`-fold product of a rotation-invariant
Borel probability measure on the sphere.

That marginal is unique — a Borel probability measure on `𝕊^{d-1}` invariant
under every linear isometry of `ℝ^d` *is* `σ_d` — so `UniformTuple d n` holds
for exactly one `P`, and quantifying over all of them below is not a
strengthening of the paper's statement.  Writing it this way keeps the Haar
machinery out: invariance is stated through `Perspective.sphereMap`. -/
def UniformTuple (P : Measure (SphereTuple d n)) : Prop :=
  ∃ σ : Measure (SSphere d), IsProbabilityMeasure σ ∧
    (∀ U : EucSpace d ≃ₗᵢ[ℝ] EucSpace d, σ.map (sphereMap d U) = σ) ∧
    P = Measure.pi (fun _ : Idx n => σ)

/-- The set of initial sequences whose `β = 0` solution reaches consensus:
the `β = 0` counterpart of `clusteringSet`. -/
def consensusSet0 : Set (SphereTuple d n) :=
  { X₀ | ∃ x_star : SSphere d, ∀ X : ℝ → SphereTuple d n,
            X 0 = X₀ → beta0Dynamics d n X →
              ∀ i : Idx n,
                Filter.Tendsto (fun t : ℝ => ((X t i : EucSpace d) - x_star))
                  Filter.atTop (nhds 0) }

/-- **Theorem (p:beta0).** *Consensus at zero temperature.*

For `d, n ≥ 2`, and Lebesgue-almost any initial sequence
`(x_i(0))_{i ∈ [n]} ∈ (𝕊^{d-1})^n`, the unique solution to the Cauchy problem
for `e:Snonres0` satisfies

  `lim_{t → ∞} x_i(t) = x⋆`  for some `x⋆ ∈ 𝕊^{d-1}` and all `i`.

The almost-everywhere quantifier is not decoration, and the statement is false
without it: the antipodal pair `n = 2`, `x₂ = -x₁` is a stationary point of
`e:Snonres0` — the mean `(x₁ + x₂)/2` is `0`, so both velocities vanish — and
it never reaches consensus.  The exceptional set is null but non-empty, which
is why the conclusion is read against the uniform law `UniformTuple` and not
against every `X₀`.

Not proved here.

Source: arXiv:2312.10794v5, §4, `p:beta0`. -/
theorem beta0_consensus (hd : 2 ≤ d) (hn : 2 ≤ n) :
    ∀ P : Measure (SphereTuple d n), UniformTuple d n P →
      ∀ᵐ X₀ ∂P, X₀ ∈ consensusSet0 d n := by
  sorry

/-- The hypotheses of `beta0_consensus` are satisfiable: `d = n = 2`. -/
example : 2 ≤ 2 ∧ 2 ≤ 2 := ⟨le_rfl, le_rfl⟩

/-- The antipode `-x` of a point of `𝕊^{d-1}`. -/
def antipode (x : SSphere d) : SSphere d :=
  ⟨-(x : EucSpace d), by
    have hx : ‖(x : EucSpace d)‖ = 1 := mem_sphere_zero_iff_norm.mp x.2
    simp [hx]⟩

/-- The antipodal pair `(x, -x) ∈ (𝕊^{d-1})²`. -/
def antipodalPair (x : SSphere d) : SphereTuple d 2 :=
  fun i => if i = 0 then x else antipode d x

/-- **The exceptional set of `beta0_consensus` is not empty.**

The antipodal pair is a stationary point of `e:Snonres0`: its mean is `0`, so
`Proj_{x_i} 0 = 0` and the constant path is a solution.  The two particles stay
antipodal forever, so no `x⋆` can attract both, and `beta0_consensus` fails for
this one initial condition.  This is what forces the almost-everywhere
quantifier there — `∀ X₀` would be false.

Source: arXiv:2312.10794v5, §4, `p:beta0` (the "almost every" of the
statement). -/
theorem antipodalPair_not_mem_consensusSet0 (x : SSphere d) :
    antipodalPair d x ∉ consensusSet0 d 2 := by
  rintro ⟨x_star, hstar⟩
  have hsum : ∑ j : Idx 2, ((antipodalPair d x j : EucSpace d)) = 0 := by
    simp [antipodalPair, antipode, Fin.sum_univ_two]
  have hdyn : beta0Dynamics d 2 (fun _ => antipodalPair d x) := by
    intro t i
    simp only [hsum, smul_zero, proj, inner_zero_right, zero_smul, sub_self]
    exact hasDerivAt_const t _
  have h0 := tendsto_const_nhds_iff.mp (hstar (fun _ => antipodalPair d x) rfl hdyn 0)
  have h1 := tendsto_const_nhds_iff.mp (hstar (fun _ => antipodalPair d x) rfl hdyn 1)
  simp only [antipodalPair, antipode, sub_eq_zero] at h0 h1
  have hneg : (x : EucSpace d) = -(x : EucSpace d) := h0.trans h1.symm
  have h2 : (2 : ℝ) • (x : EucSpace d) = 0 := by
    rw [two_smul]
    nth_rewrite 2 [hneg]
    exact add_neg_cancel _
  have hx0 : (x : EucSpace d) = 0 := by
    rcases smul_eq_zero.mp h2 with h' | h'
    · norm_num at h'
    · exact h'
  have hx : ‖(x : EucSpace d)‖ = 1 := mem_sphere_zero_iff_norm.mp x.2
  rw [hx0] at hx
  norm_num at hx

/-- The subset `𝒮_β ⊂ (𝕊^{d-1})^n` of initial sequences for which the solution
to the Cauchy problem for `SA` (or `USA`) converges to a single cluster. -/
def clusteringSet
    (β : ℝ) : Set (SphereTuple d n) :=
  { X₀ | ∃ x_star : SSphere d, ∀ X : ℝ → SphereTuple d n,
            X 0 = X₀ → Perspective.SA d n β X →
              ∀ i : Idx n,
                Filter.Tendsto (fun t : ℝ => ((X t i : EucSpace d) - x_star))
                  Filter.atTop (nhds 0) }

/-- **An antipodal pair is a stationary point of `SA` at every `β`.**

At the first particle the attention weights are `e^{β}` on `x` and `e^{-β}` on
`-x`, at the second the other way round, so both weighted sums equal
`(e^{β} - e^{-β}) x` up to the sign carried by the particle itself: each is a
multiple of the particle, and `proj_smul_self` kills it.

Source: arXiv:2312.10794v5, §4, the exceptional set of `p:beta0`. -/
theorem SA_const_antipodalPair (β : ℝ) (x : SSphere d) :
    Perspective.SA d 2 β (fun _ => antipodalPair d x) := by
  have hx : ‖(x : EucSpace d)‖ = 1 := mem_sphere_zero_iff_norm.mp x.2
  have hnorm : ∀ i : Idx 2, ‖((antipodalPair d x i : SSphere d) : EucSpace d)‖ = 1 := by
    intro i; fin_cases i <;> simp [antipodalPair, antipode, hx]
  have hsum : ∀ i : Idx 2,
      ∑ j : Idx 2, Real.exp (β * inner (𝕜 := ℝ)
          ((antipodalPair d x i : SSphere d) : EucSpace d)
          ((antipodalPair d x j : SSphere d) : EucSpace d))
          • (((antipodalPair d x j : SSphere d) : EucSpace d))
        = (Real.exp β - Real.exp (-β))
            • (((antipodalPair d x i : SSphere d) : EucSpace d)) := by
    intro i
    fin_cases i <;>
      simp [antipodalPair, antipode, Fin.sum_univ_two, hx, inner_neg_right, mul_neg] <;>
      module
  intro t i
  refine (hasDerivAt_const t _).congr_deriv ?_
  rw [hsum i, smul_smul, proj_smul_self (hnorm i)]

/-- The antipodal pair is outside `𝒮_β` for every `β`: the constant path is a
solution of `SA` through it, and the two particles stay antipodal, so no `x⋆`
attracts both.  This is why `beta_tiny` and `beta_interval` are read almost
everywhere.

Source: arXiv:2312.10794v5, §4. -/
theorem antipodalPair_not_mem_clusteringSet (β : ℝ) (x : SSphere d) :
    antipodalPair d x ∉ clusteringSet d 2 β := by
  rintro ⟨x_star, hstar⟩
  have h0 := tendsto_const_nhds_iff.mp
    (hstar (fun _ => antipodalPair d x) rfl (SA_const_antipodalPair d β x) 0)
  have h1 := tendsto_const_nhds_iff.mp
    (hstar (fun _ => antipodalPair d x) rfl (SA_const_antipodalPair d β x) 1)
  simp only [antipodalPair, antipode, sub_eq_zero] at h0 h1
  have hneg : (x : EucSpace d) = -(x : EucSpace d) := h0.trans h1.symm
  have h2 : (2 : ℝ) • (x : EucSpace d) = 0 := by
    rw [two_smul]
    nth_rewrite 2 [hneg]
    exact add_neg_cancel _
  have hx0 : (x : EucSpace d) = 0 := by
    rcases smul_eq_zero.mp h2 with h' | h'
    · norm_num at h'
    · exact h'
  have hx : ‖(x : EucSpace d)‖ = 1 := mem_sphere_zero_iff_norm.mp x.2
  rw [hx0] at hx
  norm_num at hx

/-- **Theorem (th:beta_small).** *Clustering with high probability at small β.*

For fixed `d, n ≥ 2`, the probability (w.r.t. uniform initialization on
`(𝕊^{d-1})^n`) that the initial sequence belongs to `𝒮_β` tends to `1` as
`β → 0⁺`:

  `ℙ(𝒮_β) →[β → 0⁺] 1`.

Not proved here: the proof runs through `beta0_consensus`,
`sset0_probability_tendsto_one` and `distance_bound_at_time_m`.  Of the three
only the last is proved, and that from the Gronwall bound `e:approxsphere`
carried as a hypothesis.  Measurability of `clusteringSet` is part of what is being
asserted: `P (𝒮_β)` is the outer measure when the set is not measurable, so the
statement is the one the paper makes in either case.

Source: arXiv:2312.10794v5, §4, `th:beta_small`. -/
theorem clustering_probability_small_beta (hd : 2 ≤ d) (hn : 2 ≤ n) :
    ∀ P : Measure (SphereTuple d n), UniformTuple d n P →
      Filter.Tendsto (fun β : ℝ => (P (clusteringSet d n β)).toReal)
        (nhdsWithin 0 (Set.Ioi 0)) (nhds 1) := by
  sorry

/-- The hypotheses of `clustering_probability_small_beta` are satisfiable:
`d = n = 2`. -/
example : 2 ≤ 2 ∧ 2 ≤ 2 := ⟨le_rfl, le_rfl⟩

/-! ### Auxiliary objects used inside the proof of `th:beta_small`. -/

/-- A set of `n` points on `𝕊^{d-1}` is *α-clustered* if all pairwise inner
products exceed `α`. -/
def alphaClustered
    (α : ℝ) (X : SphereTuple d n) : Prop :=
  ∀ i j : Idx n, α < inner (𝕜 := ℝ) ((X i) : EucSpace d) ((X j) : EucSpace d)

/-- **Equation (eq: harry.potter).** *3/4-clustering at integer time `m`.*

The set of initial conditions `X₀` for which the solution to `e:Snonres0` is
`(3/4)`-clustered at time `m`:

  `⟨x_i^0(m), x_j^0(m)⟩ > 3/4`  for all `i, j ∈ [n]`. -/
def Sset0 (m : ℕ) : Set (SphereTuple d n) :=
  { X₀ | ∃ X : ℝ → SphereTuple d n,
            X 0 = X₀ ∧ beta0Dynamics d n X ∧
              alphaClustered d n (3/4 : ℝ) (X (m : ℝ)) }

/-- **Equation (e:Ps0n).** As `m → ∞`, `ℙ(𝒮_0^m) → 1`: almost every initial
sequence is `(3/4)`-clustered by the `β = 0` dynamics at a late enough integer
time, so the measure of `Sset0 d n m` tends to `1`.

Not proved here: it is the quantitative form of `beta0_consensus`, which is a
`sorry` here.

Source: arXiv:2312.10794v5, §4, `e:Ps0n`. -/
theorem sset0_probability_tendsto_one (hd : 2 ≤ d) (hn : 2 ≤ n) :
    ∀ P : Measure (SphereTuple d n), UniformTuple d n P →
      Filter.Tendsto (fun m : ℕ => (P (Sset0 d n m)).toReal)
        Filter.atTop (nhds 1) := by
  sorry

/-- The hypotheses of `sset0_probability_tendsto_one` are satisfiable:
`d = n = 2`. -/
example : 2 ≤ 2 ∧ 2 ≤ 2 := ⟨le_rfl, le_rfl⟩

/-- **Equation (e:approxsphere).** Gronwall bound:

  `‖x_i^β(t) - x_i^0(t)‖ ≤ O(β) e^{3t}`.

**What the source says and what is changed here.**  The form this development
gave the bound put `∃ C` *after* `β` and after the two solutions, which makes
it say nothing: for `β > 0` the constant `C = 2/β` already works, since two
points of `𝕊^{d-1}` are never further apart than `2 ≤ 2 e^{3t}`.  The survey's
`O(β)` is a constant depending on `d` and `n` alone, so the quantifier is
hoisted to the front here.  With it in front the bound has content at every
`β`, including `β = 0`, where it asserts that `e:Snonres0` has at most one
solution from a given initial tuple.

Not proved here.

Source: arXiv:2312.10794v5, §4, `e:approxsphere`. -/
theorem solutions_close_at_small_beta :
    ∃ C : ℝ, 0 < C ∧
      ∀ β : ℝ, 0 ≤ β →
        ∀ Xβ X0t : ℝ → SphereTuple d n,
          Xβ 0 = X0t 0 →
          Perspective.SA d n β Xβ → beta0Dynamics d n X0t →
          ∀ t : ℝ, 0 ≤ t → ∀ i : Idx n,
            ‖((Xβ t i : EucSpace d)) - ((X0t t i : EucSpace d))‖
              ≤ C * β * Real.exp (3 * t) := by
  sorry

/-- **A lone token does not move under `eq: SA`.**

At `n = 1` the attention average is the token itself — the single weight is
`e^{β}` and it is divided by the partition function `Z = e^{β}` — so the drift
is `Proj_x x = 0` and every solution is constant.  This is the converse of
`SA_const_consensus` in the one-particle case.

Source: arXiv:2312.10794v5, §2, `eq: SA`. -/
theorem const_of_SA_one (β : ℝ) (X : ℝ → SphereTuple d 1)
    (hX : Perspective.SA d 1 β X) (t : ℝ) (i : Idx 1) :
    (X t i : EucSpace d) = (X 0 i : EucSpace d) := by
  have hzero : ∀ s : ℝ, HasDerivAt (fun r => (X r i : EucSpace d)) 0 s := by
    intro s
    obtain rfl : i = 0 := Subsingleton.elim i 0
    have hx : ‖(X s (0 : Idx 1) : EucSpace d)‖ = 1 :=
      mem_sphere_zero_iff_norm.mp (X s 0).2
    have hxx : inner (𝕜 := ℝ) ((X s (0 : Idx 1) : EucSpace d))
        ((X s (0 : Idx 1) : EucSpace d)) = 1 := by
      rw [real_inner_self_eq_norm_mul_norm, hx]; ring
    have hZ : partitionSA d 1 β X s 0 = Real.exp β := by
      simp [partitionSA]
    have hsum : (∑ j : Idx 1,
        Real.exp (β * inner (𝕜 := ℝ) ((X s (0 : Idx 1) : EucSpace d))
            ((X s j : EucSpace d))) • ((X s j : EucSpace d)))
        = Real.exp β • ((X s (0 : Idx 1) : EucSpace d)) := by
      simp
    have hval : proj d ((X s (0 : Idx 1) : EucSpace d))
        ((partitionSA d 1 β X s 0)⁻¹ • ∑ j : Idx 1,
          Real.exp (β * inner (𝕜 := ℝ) ((X s (0 : Idx 1) : EucSpace d))
              ((X s j : EucSpace d))) • ((X s j : EucSpace d))) = 0 := by
      rw [hZ, hsum, smul_smul, inv_mul_cancel₀ (Real.exp_ne_zero β), one_smul,
        proj, hxx, one_smul, sub_self]
    exact (hX s 0).congr_deriv hval
  exact is_const_of_deriv_eq_zero (fun s => (hzero s).differentiableAt)
    (fun s => (hzero s).deriv) t 0

/-- The hypothesis of `const_of_SA_one` is satisfiable: the constant curve
solves `eq: SA` (`SA_const_consensus`). -/
example (β : ℝ) : Perspective.SA 1 1 β (fun _ _ => basePoint 0) :=
  SA_const_consensus 1 1 one_pos β (basePoint 0)

/-- **A lone token does not move under `e:Snonres0` either.**

At `n = 1` the mean of the tuple is the token itself, so the drift is again
`Proj_x x = 0`.

Source: arXiv:2312.10794v5, §4, `e:Snonres0`. -/
theorem const_of_beta0Dynamics_one (X : ℝ → SphereTuple d 1)
    (hX : beta0Dynamics d 1 X) (t : ℝ) (i : Idx 1) :
    (X t i : EucSpace d) = (X 0 i : EucSpace d) := by
  have hzero : ∀ s : ℝ, HasDerivAt (fun r => (X r i : EucSpace d)) 0 s := by
    intro s
    obtain rfl : i = 0 := Subsingleton.elim i 0
    have hx : ‖(X s (0 : Idx 1) : EucSpace d)‖ = 1 :=
      mem_sphere_zero_iff_norm.mp (X s 0).2
    have hxx : inner (𝕜 := ℝ) ((X s (0 : Idx 1) : EucSpace d))
        ((X s (0 : Idx 1) : EucSpace d)) = 1 := by
      rw [real_inner_self_eq_norm_mul_norm, hx]; ring
    have hval : proj d ((X s (0 : Idx 1) : EucSpace d))
        ((((1 : ℕ) : ℝ)⁻¹) • ∑ j : Idx 1, ((X s j : EucSpace d))) = 0 := by
      simp only [Nat.cast_one, inv_one, one_smul, Fin.sum_univ_one]
      rw [proj, hxx, one_smul, sub_self]
    exact (hX s 0).congr_deriv hval
  exact is_const_of_deriv_eq_zero (fun s => (hzero s).differentiableAt)
    (fun s => (hzero s).deriv) t 0

/-- The hypothesis of `const_of_beta0Dynamics_one` is satisfiable: the constant
curve solves `e:Snonres0`. -/
example : beta0Dynamics 1 1 (fun _ _ => basePoint 0) := by
  intro t i
  have hx : ‖((basePoint 0 : SSphere 1) : EucSpace 1)‖ = 1 :=
    mem_sphere_zero_iff_norm.mp (basePoint 0).2
  have hxx : inner (𝕜 := ℝ) ((basePoint 0 : EucSpace 1))
      ((basePoint 0 : EucSpace 1)) = 1 := by
    rw [real_inner_self_eq_norm_mul_norm, hx]; ring
  have hval : proj 1 ((basePoint 0 : EucSpace 1))
      ((((1 : ℕ) : ℝ)⁻¹) • ∑ _j : Idx 1, ((basePoint 0 : EucSpace 1))) = 0 := by
    simp only [Nat.cast_one, inv_one, one_smul, Fin.sum_univ_one]
    rw [proj, hxx, one_smul, sub_self]
  exact (hasDerivAt_const t _).congr_deriv hval.symm

/-- **Equation (eq: youareawizardharry).**  For any `m`, there exists `β_m > 0`
such that for `β ∈ [0, β_m]`,

  `‖x_i^β(m) - x_i^0(m)‖ ≤ 1/8`.

Proved here, from the Gronwall bound `e:approxsphere` — which is not proved —
carried as an explicit hypothesis: its constant `C` is the `C` below, and
`β_m = 1/(8 C e^{3m})` turns the bound at `t = m` into `1/8`.

Source: arXiv:2312.10794v5, §4, `eq: youareawizardharry`. -/
theorem distance_bound_at_time_m (m : ℕ) (C : ℝ) (hC : 0 < C)
    (hclose : ∀ β : ℝ, 0 ≤ β →
      ∀ Xβ X0t : ℝ → SphereTuple d n,
        Xβ 0 = X0t 0 →
        Perspective.SA d n β Xβ → beta0Dynamics d n X0t →
        ∀ t : ℝ, 0 ≤ t → ∀ i : Idx n,
          ‖((Xβ t i : EucSpace d)) - ((X0t t i : EucSpace d))‖
            ≤ C * β * Real.exp (3 * t)) :
    ∃ βm : ℝ, 0 < βm ∧
      ∀ β : ℝ, 0 ≤ β → β ≤ βm →
        ∀ Xβ X0t : ℝ → SphereTuple d n,
          Xβ 0 = X0t 0 →
          Perspective.SA d n β Xβ → beta0Dynamics d n X0t →
          ∀ i : Idx n,
            ‖((Xβ (m : ℝ) i : EucSpace d))
              - ((X0t (m : ℝ) i : EucSpace d))‖ ≤ (1/8 : ℝ) := by
  have hE : (0 : ℝ) < Real.exp (3 * m) := Real.exp_pos _
  refine ⟨1 / (8 * C * Real.exp (3 * m)), by positivity, ?_⟩
  intro β hβ0 hβm Xβ X0t hinit hSA h0 i
  have h := hclose β hβ0 Xβ X0t hinit hSA h0 (m : ℝ) (Nat.cast_nonneg m) i
  have hmono : C * β * Real.exp (3 * m)
      ≤ C * (1 / (8 * C * Real.exp (3 * m))) * Real.exp (3 * m) := by
    gcongr
  refine h.trans (hmono.trans (le_of_eq ?_))
  field_simp

/-- The hypotheses of `distance_bound_at_time_m` are satisfiable, and not
vacuously: at `n = 1` the lone token stands still under both dynamics
(`const_of_SA_one`, `const_of_beta0Dynamics_one`), so two solutions from the
same initial tuple coincide and `C = 1` witnesses the Gronwall bound. -/
example : (0 : ℝ) < 1 ∧
    ∀ β : ℝ, 0 ≤ β →
      ∀ Xβ X0t : ℝ → SphereTuple 1 1,
        Xβ 0 = X0t 0 →
        Perspective.SA 1 1 β Xβ → beta0Dynamics 1 1 X0t →
        ∀ t : ℝ, 0 ≤ t → ∀ i : Idx 1,
          ‖((Xβ t i : EucSpace 1)) - ((X0t t i : EucSpace 1))‖
            ≤ 1 * β * Real.exp (3 * t) := by
  refine ⟨one_pos, fun β hβ Xβ X0t hinit hSA h0 t _ i => ?_⟩
  rw [const_of_SA_one 1 β Xβ hSA t i, const_of_beta0Dynamics_one 1 X0t h0 t i,
    hinit, sub_self, norm_zero]
  exact mul_nonneg (mul_nonneg zero_le_one hβ) (Real.exp_pos _).le

/-- **Theorem (thm: beta.tiny).** *Cluster collapse when `β ≤ C/n`.*

Fix `d, n ≥ 2`.  There is a numerical constant `C > 0` such that whenever
`β ≤ C/n`:

For Lebesgue-almost any `(x_i(0))_{i ∈ [n]} ∈ (𝕊^{d-1})^n`, there exists
`x⋆ ∈ 𝕊^{d-1}` with `lim_{t→∞} x_i(t) = x⋆` for the unique solution of `SA`
(resp. `USA`) starting from `X₀` — that is, `𝒮_β` is co-null.

Moreover, when `d = 2` one can take `β ≤ 1` (`beta_tiny_circle`).

As in `beta0_consensus` the almost-everywhere quantifier is necessary: the
antipodal pair `n = 2`, `x₂ = -x₁` is stationary for `SA` at every `β`, since
the weighted mean `(e^{β} x₁ + e^{-β} x₂) / Z` is again a multiple of `x₁` and
its projection onto `T_{x₁} 𝕊^{d-1}` vanishes.

Not proved here.

Source: arXiv:2312.10794v5, §4, `thm: beta.tiny`. -/
theorem beta_tiny (hd : 2 ≤ d) (hn : 2 ≤ n) :
    ∃ C : ℝ, 0 < C ∧ ∀ β : ℝ, 0 ≤ β → β ≤ C / n →
      ∀ P : Measure (SphereTuple d n), UniformTuple d n P →
        ∀ᵐ X₀ ∂P, X₀ ∈ clusteringSet d n β := by
  sorry

/-- The hypotheses of `beta_tiny` are satisfiable: `d = n = 2`. -/
example : 2 ≤ 2 ∧ 2 ≤ 2 := ⟨le_rfl, le_rfl⟩

/-- *d = 2 improvement of `thm: beta.tiny` (Criscitiello-Boumal 2024).*

When `d = 2`, the constant in `beta_tiny` can be taken so that `β ≤ 1`.

Not proved here.

Source: arXiv:2312.10794v5, §4, remark after `thm: beta.tiny`. -/
theorem beta_tiny_circle (hn : 2 ≤ n) :
    ∀ β : ℝ, 0 ≤ β → β ≤ 1 →
      ∀ P : Measure (SphereTuple 2 n), UniformTuple 2 n P →
        ∀ᵐ X₀ ∂P, X₀ ∈ clusteringSet 2 n β := by
  sorry

/-- The hypothesis of `beta_tiny_circle` is satisfiable: `n = 2`. -/
example : 2 ≤ 2 := le_rfl

end Perspective
end Transformer
