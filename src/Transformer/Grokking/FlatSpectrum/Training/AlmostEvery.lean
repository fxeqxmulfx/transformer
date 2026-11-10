module

public import Transformer.Grokking.FlatSpectrum.Training.ExpLoss
public import Transformer.Grokking.FlatSpectrum.Training.Width

/-!
# Almost every start: width `min(m, n)` suffices

Source: `review/flat_spectrum.md`, *Training reaches the optimal fit* (added), item WD10.
Full row rank of the start needs width `k ≥ m + n`. For almost every start the width
`k ≥ min(m, n)` suffices. Let `L` be real-analytic in the entries, convex with a continuous
gradient `G` and bounded below, and let `λ > 0`. By WD9 small steps from the sublevel set
`{F ≤ c}` converge, and the starts whose run converges to a strict saddle form a null set of
the factor space. So for almost every start in `{F ≤ c}` the limit is a critical point that is
not a strict saddle (`exists_pos_ae_tendsto_not_isStrictSaddle`), and by the theorems of
`Training.Width` it is

* a decayed minimum whenever its columns are dependent, at any width
  (`exists_pos_ae_tendsto_decayRun`);
* a decayed minimum at every width `k ≥ min(m, n)`
  (`exists_pos_ae_tendsto_decayRun_isDecayedMin`).

Hence decayed minima of width `min(m, n)` exist (`exists_isDecayedMin_of_card_le`), and for a
`p × p` kernel with equal singular values and no zero entry on a proper `Ω`, almost every run
of width `p` converges a fixed distance below `‖K‖_S1`
(`exists_pos_ae_tendsto_decayRun_nuclearNorm_add_le`, Theorem A). "Almost every" refers to the
Lebesgue measure `volume` of the factor space, the real inner product space of the entries of
`[U; V]`.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Training

open Matrix Filter Topology MeasureTheory
open Transformer.Landscape.SaddleAvoidance (gradientStep IsStrictSaddle)

variable {m n k : Type*} [Fintype m] [Fintype n] [Fintype k] [DecidableEq m] [DecidableEq n]
  {L : Matrix m n ℂ → ℝ} {G : Matrix m n ℂ → Matrix m n ℂ} {lam l₀ : ℝ}

/-- **Almost every run avoids the strict saddles** (`review/flat_spectrum.md`, item WD10): for
a loss that is real-analytic in the entries and bounded below, a continuous field `G`, a weight
decay `λ > 0` and a level `c`, some `η₀ > 0` has: for every `0 < η < η₀` and almost every start
`x` of the factor space with `F x ≤ c`, gradient descent from `x` converges to a critical point
that is not a strict saddle. -/
theorem exists_pos_ae_tendsto_not_isStrictSaddle (hG : Continuous G)
    (hL : ∀ y : EuclideanSpace ℂ (m × n),
      AnalyticAt ℝ (fun y : EuclideanSpace ℂ (m × n) => L (of fun i j => y (i, j))) y)
    (hlam : 0 < lam) (hlow : ∀ X, l₀ ≤ L X) (c : ℝ) :
    ∃ η₀ > 0, ∀ η, 0 < η → η < η₀ → ∀ᵐ x : EuclideanSpace ℂ ((m ⊕ n) × k),
      flatLoss L lam x ≤ c → ∃ w, gradient (flatLoss L lam) w = 0 ∧
        ¬ IsStrictSaddle (flatLoss L lam) w ∧
        Tendsto (fun t => (gradientStep (flatLoss L lam) η)^[t] x) atTop (𝓝 w) := by
  obtain ⟨η₀, hη₀, h⟩ := exists_pos_forall_sublevel (k := k) hG hL hlam hlow c
  refine ⟨η₀, hη₀, fun η hη hlt => ?_⟩
  obtain ⟨-, hnull, hconvg, -⟩ := h η hη hlt
  filter_upwards [measure_eq_zero_iff_ae_notMem.1 hnull] with x hx hc
  obtain ⟨w, -, hw, hlim⟩ := hconvg x hc
  exact ⟨w, hw, fun hs => hx ⟨hc, w, hs, hlim⟩, hlim⟩

/-- **Almost every run ends at a decayed minimum or with independent columns**
(`review/flat_spectrum.md`, item WD10): for a loss that is real-analytic in the entries,
satisfies the first-order inequality with a continuous `G` and is bounded below, a weight decay
`λ > 0` and a level `c`, some `η₀ > 0` has: for every `0 < η < η₀` and almost every start of
the factor space with `F ≤ c`, of any width, the run converges, its limit is stationary, and if
the columns of the limit `[U; V]` are dependent, the limit has the least weight-decayed loss
among the factorizations of every width. -/
theorem exists_pos_ae_tendsto_decayRun (hconv : ∀ X Y, L X + pairing (G X) (Y - X) ≤ L Y)
    (hG : Continuous G)
    (hL : ∀ y : EuclideanSpace ℂ (m × n),
      AnalyticAt ℝ (fun y : EuclideanSpace ℂ (m × n) => L (of fun i j => y (i, j))) y)
    (hlam : 0 < lam) (hlow : ∀ X, l₀ ≤ L X) (c : ℝ) :
    ∃ η₀ > 0, ∀ η, 0 < η → η < η₀ → ∀ᵐ x : EuclideanSpace ℂ ((m ⊕ n) × k),
      flatLoss L lam x ≤ c → ∃ (U : Matrix m k ℂ) (V : Matrix n k ℂ),
        Tendsto (decayRun G η lam (factors x).1 (factors x).2) atTop (𝓝 (U, V)) ∧
          lift lam (G (U * Vᴴ)) * stack U V = 0 ∧
          (¬ Function.Injective (stack U V).mulVec → IsDecayedMin L lam U V) := by
  obtain ⟨η₀, hη₀, h⟩ := exists_pos_ae_tendsto_not_isStrictSaddle (k := k) hG hL hlam hlow c
  refine ⟨η₀, hη₀, fun η hη hlt => ?_⟩
  filter_upwards [h η hη hlt] with x hx hc
  obtain ⟨w, hw, hns, hlim⟩ := hx hc
  refine ⟨(factors w).1, (factors w).2, ?_,
    lift_mul_stack_eq_zero_of_gradient_eq_zero hconv hG hL hw,
    isDecayedMin_of_not_isStrictSaddle hconv hG hL hw hns⟩
  refine ((continuous_factors.tendsto w).comp hlim).congr fun t => ?_
  have ht := factors_iterate_gradientStep hconv hG hL η lam (factors x).1 (factors x).2 t
  rwa [Prod.mk.eta, flatten_factors] at ht

/-- **At width `min(m, n)` almost every run ends at a decayed minimum**
(`review/flat_spectrum.md`, item WD10): for a loss that is real-analytic in the entries,
satisfies the first-order inequality with a continuous `G` and is bounded below, a weight decay
`λ > 0`, a level `c` and a width at least the number of rows of `U` or of `V`, some `η₀ > 0`
has: for every `0 < η < η₀` and almost every start of the factor space with `F ≤ c`, the run
converges and its limit has the least weight-decayed loss among the factorizations of every
width. -/
theorem exists_pos_ae_tendsto_decayRun_isDecayedMin
    (hconv : ∀ X Y, L X + pairing (G X) (Y - X) ≤ L Y) (hG : Continuous G)
    (hL : ∀ y : EuclideanSpace ℂ (m × n),
      AnalyticAt ℝ (fun y : EuclideanSpace ℂ (m × n) => L (of fun i j => y (i, j))) y)
    (hlam : 0 < lam) (hlow : ∀ X, l₀ ≤ L X) (c : ℝ)
    (hcard : Fintype.card m ≤ Fintype.card k ∨ Fintype.card n ≤ Fintype.card k) :
    ∃ η₀ > 0, ∀ η, 0 < η → η < η₀ → ∀ᵐ x : EuclideanSpace ℂ ((m ⊕ n) × k),
      flatLoss L lam x ≤ c → ∃ (U : Matrix m k ℂ) (V : Matrix n k ℂ),
        Tendsto (decayRun G η lam (factors x).1 (factors x).2) atTop (𝓝 (U, V)) ∧
          IsDecayedMin L lam U V := by
  obtain ⟨η₀, hη₀, h⟩ := exists_pos_ae_tendsto_not_isStrictSaddle (k := k) hG hL hlam hlow c
  refine ⟨η₀, hη₀, fun η hη hlt => ?_⟩
  filter_upwards [h η hη hlt] with x hx hc
  obtain ⟨w, hw, hns, hlim⟩ := hx hc
  refine ⟨(factors w).1, (factors w).2, ?_,
    isDecayedMin_of_not_isStrictSaddle_of_card_le hconv hG hL hlam hw hns hcard⟩
  refine ((continuous_factors.tendsto w).comp hlim).congr fun t => ?_
  have ht := factors_iterate_gradientStep hconv hG hL η lam (factors x).1 (factors x).2 t
  rwa [Prod.mk.eta, flatten_factors] at ht

/-- **Decayed minima of width `min(m, n)` exist** (`review/flat_spectrum.md`, item WD10): for
a loss that is real-analytic in the entries, satisfies the first-order inequality with a
continuous `G` and is bounded below, every weight decay `λ > 0` and every width at least the
number of rows of `U` or of `V` has a factorization of least weight-decayed loss among those of
every width. -/
theorem exists_isDecayedMin_of_card_le (hconv : ∀ X Y, L X + pairing (G X) (Y - X) ≤ L Y)
    (hG : Continuous G)
    (hL : ∀ y : EuclideanSpace ℂ (m × n),
      AnalyticAt ℝ (fun y : EuclideanSpace ℂ (m × n) => L (of fun i j => y (i, j))) y)
    (hlam : 0 < lam) (hlow : ∀ X, l₀ ≤ L X)
    (hcard : Fintype.card m ≤ Fintype.card k ∨ Fintype.card n ≤ Fintype.card k) :
    ∃ (U : Matrix m k ℂ) (V : Matrix n k ℂ), IsDecayedMin L lam U V := by
  have hcont : Continuous (flatLoss (k := k) L lam) :=
    continuous_iff_continuousAt.mpr fun x => (analyticAt_flatLoss hL lam x).continuousAt
  obtain ⟨η₀, hη₀, h⟩ := exists_pos_ae_tendsto_decayRun_isDecayedMin (k := k) hconv hG hL hlam
    hlow (flatLoss (k := k) L lam 0 + 1) hcard
  have hne : volume {x : EuclideanSpace ℂ ((m ⊕ n) × k) |
      flatLoss L lam x < flatLoss (k := k) L lam 0 + 1} ≠ 0 :=
    ((isOpen_lt hcont continuous_const).measure_pos volume ⟨0, by simp⟩).ne'
  obtain ⟨x, hx, hp⟩ := Measure.exists_mem_of_measure_ne_zero_of_ae hne
    (ae_restrict_of_ae (h (η₀ / 2) (half_pos hη₀) (half_lt_self hη₀)))
  obtain ⟨U, V, -, hmin⟩ := hp (le_of_lt hx)
  exact ⟨U, V, hmin⟩

/-- **Training at width `p` does not return a flat kernel** (`review/flat_spectrum.md`,
Theorem A with items WD7 and WD10): if the singular values of a square `K` are all equal, no
entry of `K` vanishes and `Ω` is proper, then some `δ > 0` has: for every `λ > 0`, every level
`c` and every width at least the size of `K`, some `η₀ > 0` has: for every `0 < η < η₀` and
almost every start of the factor space with `F ≤ c`, the run on the squared loss on `Ω`
converges, to factors with `‖U Vᴴ‖_S1 + δ ≤ ‖K‖_S1`. -/
theorem exists_pos_ae_tendsto_decayRun_nuclearNorm_add_le {K : Matrix n n ℂ} {s : ℝ}
    (h : ∀ i, singularValues K i = s) (hK : ∀ i j, K i j ≠ 0) {Ω : Set (n × n)}
    (hΩ : Ω ≠ Set.univ) (hcard : Fintype.card n ≤ Fintype.card k) :
    ∃ δ : ℝ, 0 < δ ∧ ∀ {lam : ℝ}, 0 < lam → ∀ c : ℝ, ∃ η₀ > 0, ∀ η, 0 < η → η < η₀ →
      ∀ᵐ x : EuclideanSpace ℂ ((n ⊕ n) × k), flatLoss (fitLoss Ω K) lam x ≤ c →
        ∃ U V : Matrix n k ℂ,
          Tendsto (decayRun (fun X => mask Ω (X - K)) η lam (factors x).1 (factors x).2) atTop
            (𝓝 (U, V)) ∧ nuclearNorm (U * Vᴴ) + δ ≤ nuclearNorm K := by
  have hmin := not_isMinOn_of_singularValues_eq h hK hΩ
  rw [isMinOn_iff] at hmin
  push Not at hmin
  obtain ⟨X', hX', hlt⟩ := hmin
  refine ⟨nuclearNorm K - nuclearNorm X', sub_pos.mpr hlt, fun {lam} hlam c => ?_⟩
  obtain ⟨η₀, hη₀, hae⟩ := exists_pos_ae_tendsto_decayRun_isDecayedMin (k := k)
    (fitLoss_first_order Ω K) (continuous_mask_sub Ω K) (analyticAt_fitLoss Ω K) hlam
    (fitLoss_nonneg Ω K) c (Or.inl hcard)
  refine ⟨η₀, hη₀, fun η hη hlt' => ?_⟩
  filter_upwards [hae η hη hlt'] with x hx hc
  obtain ⟨U, V, hlim, hdm⟩ := hx hc
  exact ⟨U, V, hlim, by linarith [hdm.nuclearNorm_le hlam hX']⟩

/-- The squared loss of the `1 × 1` kernel `1` at width one, `λ = 1`, below the level `1`. -/
example := exists_pos_ae_tendsto_not_isStrictSaddle (k := Fin 1)
  (continuous_mask_sub (Set.univ : Set (Fin 1 × Fin 1)) 1) (analyticAt_fitLoss Set.univ 1)
  one_pos (fitLoss_nonneg Set.univ 1) 1

/-- The exponential loss of `3 × 3` matrices at width one, below `min(m, n)`. -/
example := exists_pos_ae_tendsto_decayRun (m := Fin 3) (n := Fin 3) (k := Fin 1)
  expLoss_first_order continuous_expGrad analyticAt_expLoss one_pos expLoss_nonneg 10

/-- The exponential loss of `2 × 3` matrices at width `2 = min(m, n)`. -/
example := exists_pos_ae_tendsto_decayRun_isDecayedMin (m := Fin 2) (n := Fin 3) (k := Fin 2)
  expLoss_first_order continuous_expGrad analyticAt_expLoss one_pos expLoss_nonneg 10
  (Or.inl le_rfl)

example := exists_isDecayedMin_of_card_le (m := Fin 2) (n := Fin 3) (k := Fin 2)
  expLoss_first_order continuous_expGrad analyticAt_expLoss one_pos expLoss_nonneg
  (Or.inl le_rfl)

/-- With no cell seen, almost every run of width `3` on the `a³ + a b` kernel for `p = 3`
converges strictly below it. -/
example := exists_pos_ae_tendsto_decayRun_nuclearNorm_add_le (k := ZMod 3)
  (singularValues_cubicKernel 3)
  (fun a b => norm_ne_zero_iff.mp (by simp [cubicKernel, taskKernel]))
  (Set.empty_ne_univ (α := ZMod 3 × ZMod 3)) le_rfl

end Transformer.Grokking.FlatSpectrum.Training
