module

public import Transformer.Grokking.FlatSpectrum.Training.SmallSteps

/-!
# A loss that is not convex: small steps still converge to a stationary point

Source: `review/flat_spectrum.md`, *Training with cross-entropy* (added), item XE13, the bullet
"Logits that are not linear in the product, such as a network between the product and the
logits". XE1–XE3 use that the logits are real-linear: then the loss is convex, and the
first-order inequality makes the field `G` its gradient (WD3). This file keeps the second
fact as a hypothesis and drops the first (item XE14).

Let `L` be real-analytic in the entries, and let `G` be its gradient along lines:
`d/dt L (X + t H) = ⟨G X, H⟩` at `t = 0`, for all `X` and `H`. Then `L` has the same
derivative along a curve `X + t H(t)` (`hasDerivAt_comp_add_smul_of_analytic`), so the
weight-decayed loss has the gradient `[G V + λ U; Gᴴ U + λ V]` on the factor space
(`gradient_flatLoss_of_analytic`) and the run of `decayStep` is gradient descent there
(`factors_iterate_gradientStep_of_analytic`). If `L` is also bounded below and `G` is
continuous, small steps converge from every start of a sublevel set, of any width and rank,
and the limit is stationary (`exists_pos_tendsto_decayRun_of_analytic`): this is the first
statement of WD9 without convexity. The cross-entropy of logits that are real-analytic in the
product is such a loss, being nonnegative.

What convexity adds in WD5 and WD9 is that the limit from a start of full row rank is a
decayed minimum. Without it that fails: `SquaredLogitRun` has a loss of this file with an
open set of starts whose runs converge to a factorization that is not a decayed minimum.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Training.Scope

open Matrix Filter Topology
open Transformer.Landscape.SaddleAvoidance (gradientStep)

variable {m n k : Type*} [Fintype m] [Fintype n] [Fintype k] {L : Matrix m n ℂ → ℝ}
  {G : Matrix m n ℂ → Matrix m n ℂ} {lam l₀ : ℝ}

/-- **A gradient along lines is the derivative along curves** for a loss that is real-analytic
in the entries: along a continuous curve `X + t H(t)` the loss has the derivative
`⟨G X, H(0)⟩` at `t = 0`. This is `hasDerivAt_comp_add_smul` without convexity. -/
theorem hasDerivAt_comp_add_smul_of_analytic
    (hgrad : ∀ X H : Matrix m n ℂ, HasDerivAt (fun t : ℝ => L (X + t • H)) (pairing (G X) H) 0)
    (hL : ∀ y : EuclideanSpace ℂ (m × n),
      AnalyticAt ℝ (fun y : EuclideanSpace ℂ (m × n) => L (of fun i j => y (i, j))) y)
    (X : Matrix m n ℂ) {H : ℝ → Matrix m n ℂ} (hH : Continuous H) :
    HasDerivAt (fun t : ℝ => L (X + t • H t)) (pairing (G X) (H 0)) 0 := by
  set vec : Matrix m n ℂ → EuclideanSpace ℂ (m × n) :=
    fun Y => WithLp.toLp 2 fun ij => Y ij.1 ij.2 with hvec
  have hd := (hL (vec X)).differentiableAt.hasFDerivAt
  have hcont : Continuous fun t : ℝ => vec (H t) :=
    (PiLp.continuous_toLp 2 _).comp (continuous_pi fun ij => hH.matrix_elem ij.1 ij.2)
  have hc : HasDerivAt (fun t : ℝ => vec X + t • vec (H t)) (vec (H 0)) 0 := by
    rw [hasDerivAt_iff_tendsto_slope_zero]
    refine ((hcont.tendsto 0).mono_left nhdsWithin_le_nhds).congr' ?_
    filter_upwards [self_mem_nhdsWithin] with t ht
    simp [smul_smul, inv_mul_cancel₀ (show t ≠ 0 from ht)]
  have hc0 : HasDerivAt (fun t : ℝ => vec X + t • vec (H 0)) (vec (H 0)) 0 := by
    simpa using ((hasDerivAt_id (0 : ℝ)).smul_const (vec (H 0))).const_add (vec X)
  have h1 := hd.comp_hasDerivAt_of_eq (0 : ℝ) hc (by simp)
  have h0 := hd.comp_hasDerivAt_of_eq (0 : ℝ) hc0 (by simp)
  have e : ∀ (t : ℝ) (Y : Matrix m n ℂ),
      (of fun i j => (vec X + t • vec Y) (i, j) : Matrix m n ℂ) = X + t • Y := fun t Y => by
    ext i j
    simp [hvec]
  have e1 : ((fun y : EuclideanSpace ℂ (m × n) => L (of fun i j => y (i, j))) ∘
      fun t : ℝ => vec X + t • vec (H t)) = fun t : ℝ => L (X + t • H t) := by
    funext t
    simp only [Function.comp_apply, e]
  have e0 : ((fun y : EuclideanSpace ℂ (m × n) => L (of fun i j => y (i, j))) ∘
      fun t : ℝ => vec X + t • vec (H 0)) = fun t : ℝ => L (X + t • H 0) := by
    funext t
    simp only [Function.comp_apply, e]
  rw [e1] at h1
  rw [e0] at h0
  rwa [h0.unique (hgrad X (H 0))] at h1

/-- The gradient of the weight-decayed loss on the factor space, for a loss that is
real-analytic in the entries with the gradient `G` along lines. -/
theorem gradient_flatLoss_of_analytic
    (hgrad : ∀ X H : Matrix m n ℂ, HasDerivAt (fun t : ℝ => L (X + t • H)) (pairing (G X) H) 0)
    (hL : ∀ y : EuclideanSpace ℂ (m × n),
      AnalyticAt ℝ (fun y : EuclideanSpace ℂ (m × n) => L (of fun i j => y (i, j))) y)
    (lam : ℝ) (x : EuclideanSpace ℂ ((m ⊕ n) × k)) :
    gradient (flatLoss L lam) x = flatten
      (G ((factors x).1 * (factors x).2ᴴ) * (factors x).2 + lam • (factors x).1,
        (G ((factors x).1 * (factors x).2ᴴ))ᴴ * (factors x).1 + lam • (factors x).2) :=
  (hasGradientAt_flatLoss_of_hasDerivAt hL lam x fun A B =>
    hasDerivAt_decayedLoss_of_hasDerivAt lam (factors x).1 A (factors x).2 B
      (hasDerivAt_comp_add_smul_of_analytic hgrad hL _
        (continuous_const.fun_add (continuous_id.fun_smul continuous_const)))).gradient

/-- The run of `decayStep` is gradient descent on `flatLoss`, read on the factor space, for a
loss that is real-analytic in the entries with the gradient `G` along lines. -/
theorem factors_iterate_gradientStep_of_analytic
    (hgrad : ∀ X H : Matrix m n ℂ, HasDerivAt (fun t : ℝ => L (X + t • H)) (pairing (G X) H) 0)
    (hL : ∀ y : EuclideanSpace ℂ (m × n),
      AnalyticAt ℝ (fun y : EuclideanSpace ℂ (m × n) => L (of fun i j => y (i, j))) y)
    (η lam : ℝ) (U₀ : Matrix m k ℂ) (V₀ : Matrix n k ℂ) (t : ℕ) :
    factors ((gradientStep (flatLoss L lam) η)^[t] (flatten (U₀, V₀))) =
      decayRun G η lam U₀ V₀ t := by
  induction t with
  | zero => exact factors_flatten _
  | succ t ih =>
    have hstep : ∀ x : EuclideanSpace ℂ ((m ⊕ n) × k),
        factors (gradientStep (flatLoss L lam) η x) = decayStep G η lam (factors x) := fun x => by
      rw [gradientStep, gradient_flatLoss_of_analytic hgrad hL, factors_sub, factors_smul,
        factors_flatten]
      rfl
    rw [Function.iterate_succ_apply', hstep, ih]
    exact (Function.iterate_succ_apply' _ _ _).symm

variable [DecidableEq m] [DecidableEq n]

/-- **Small steps converge without convexity** (`review/flat_spectrum.md`, item XE14): for a
loss that is real-analytic in the entries and bounded below, with a continuous gradient `G`
along lines, a weight decay `λ > 0` and a level `c`, some `η₀ > 0` has: for every `0 < η < η₀`
the run from every start with `F(U₀, V₀) ≤ c`, of any width and rank, converges, and its limit
is stationary, `M [U; V] = 0`. -/
theorem exists_pos_tendsto_decayRun_of_analytic
    (hgrad : ∀ X H : Matrix m n ℂ, HasDerivAt (fun t : ℝ => L (X + t • H)) (pairing (G X) H) 0)
    (hG : Continuous G)
    (hL : ∀ y : EuclideanSpace ℂ (m × n),
      AnalyticAt ℝ (fun y : EuclideanSpace ℂ (m × n) => L (of fun i j => y (i, j))) y)
    (hlam : 0 < lam) (hlow : ∀ X, l₀ ≤ L X) (c : ℝ) :
    ∃ η₀ > 0, ∀ η, 0 < η → η < η₀ → ∀ (U₀ : Matrix m k ℂ) (V₀ : Matrix n k ℂ),
      decayedLoss L lam U₀ V₀ ≤ c → ∃ (U : Matrix m k ℂ) (V : Matrix n k ℂ),
        Tendsto (decayRun G η lam U₀ V₀) atTop (𝓝 (U, V)) ∧
          lift lam (G (U * Vᴴ)) * stack U V = 0 := by
  obtain ⟨η₀, hη₀, h⟩ := exists_pos_forall_sublevel (k := k) hG hL hlam hlow c
  refine ⟨η₀, hη₀, fun η hη hlt U₀ V₀ hc => ?_⟩
  obtain ⟨-, -, hconvg, -⟩ := h η hη hlt
  obtain ⟨w, -, -, hlim⟩ := hconvg (flatten (U₀, V₀)) (by
    rw [flatLoss, factors_flatten]
    exact hc)
  have hrun : Tendsto (decayRun G η lam U₀ V₀) atTop (𝓝 (factors w)) :=
    ((continuous_factors.tendsto w).comp hlim).congr
      (factors_iterate_gradientStep_of_analytic hgrad hL η lam U₀ V₀)
  exact ⟨(factors w).1, (factors w).2, hrun,
    lift_mul_stack_eq_zero_of_tendsto_decayRun hG hη hrun⟩

/-! The witness: a convex loss is a loss of this file. For the squared loss of the `1 × 1`
kernel `1` the gradient along lines is `hasDerivAt_comp_add_smul`. `SquaredLogit` has a loss
that is not convex. -/

example := exists_pos_tendsto_decayRun_of_analytic (k := Fin 1)
  (fun X H => hasDerivAt_comp_add_smul
    (fitLoss_first_order (Set.univ : Set (Fin 1 × Fin 1)) 1) (continuous_mask_sub Set.univ 1) X
    (H := fun _ => H) continuous_const)
  (continuous_mask_sub Set.univ 1) (analyticAt_fitLoss Set.univ 1) one_pos
  (fitLoss_nonneg Set.univ 1) 1

example := hasDerivAt_comp_add_smul_of_analytic
  (fun X H => hasDerivAt_comp_add_smul
    (fitLoss_first_order (Set.univ : Set (Fin 1 × Fin 1)) 1) (continuous_mask_sub Set.univ 1) X
    (H := fun _ => H) continuous_const)
  (analyticAt_fitLoss Set.univ 1) 0 (H := fun _ => 1) continuous_const

example := gradient_flatLoss_of_analytic (k := Fin 1)
  (fun X H => hasDerivAt_comp_add_smul
    (fitLoss_first_order (Set.univ : Set (Fin 1 × Fin 1)) 1) (continuous_mask_sub Set.univ 1) X
    (H := fun _ => H) continuous_const)
  (analyticAt_fitLoss Set.univ 1) 1 0

example := factors_iterate_gradientStep_of_analytic
  (fun X H => hasDerivAt_comp_add_smul
    (fitLoss_first_order (Set.univ : Set (Fin 1 × Fin 1)) 1) (continuous_mask_sub Set.univ 1) X
    (H := fun _ => H) continuous_const)
  (analyticAt_fitLoss Set.univ 1) (1 / 10) 1 (1 : Matrix (Fin 1) (Fin 1) ℂ) 1 3

end Transformer.Grokking.FlatSpectrum.Training.Scope
