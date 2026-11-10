module

public import Transformer.Grokking.FlatSpectrum.Training.Sublevel

/-!
# Every convex real-analytic loss bounded below: small steps converge

Source: `review/flat_spectrum.md`, *Training reaches the optimal fit* (added), item WD9.
Let the loss `L` be real-analytic in the entries, satisfy the first-order inequality with a
continuous `G`, and be bounded below. For a weight decay `λ > 0` and a level `c` there is
`η₀ > 0` such that for every step `0 < η < η₀` the run of gradient descent with weight decay
from every start with `F(U₀, V₀) ≤ c`

* converges, whatever its width and rank, to a stationary point
  (`exists_pos_tendsto_decayRun`);
* converges to a decayed minimum when the start has full row rank
  (`exists_pos_tendsto_decayRun_isDecayedMin`, by WD5).

No bound on the gradient and no quadratic model is assumed: `{F ≤ c}` is compact, and
`exists_pos_forall_sublevel` supplies the step. So every such loss has a decayed minimum of
width `m + n` for every `λ > 0` (`exists_isDecayedMin`): `L + λ ‖·‖_S1` attains its minimum
over all matrices. The examples are the squared loss; `Training.ExpLoss` gives a loss whose
gradient is not Lipschitz.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Training

open Matrix Filter Topology

variable {m n k : Type*} [Fintype m] [Fintype n] [Fintype k] [DecidableEq m] [DecidableEq n]
  {L : Matrix m n ℂ → ℝ} {G : Matrix m n ℂ → Matrix m n ℂ} {lam l₀ : ℝ}

/-- **Small steps converge** (`review/flat_spectrum.md`, item WD9): for a loss that is
real-analytic in the entries, satisfies the first-order inequality with a continuous `G` and is
bounded below, a weight decay `λ > 0` and a level `c`, some `η₀ > 0` has: for every
`0 < η < η₀` the run from every start with `F(U₀, V₀) ≤ c`, of any width and rank, converges,
and its limit is stationary, `M [U; V] = 0`. -/
theorem exists_pos_tendsto_decayRun (hconv : ∀ X Y, L X + pairing (G X) (Y - X) ≤ L Y)
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
      (factors_iterate_gradientStep hconv hG hL η lam U₀ V₀)
  exact ⟨(factors w).1, (factors w).2, hrun,
    lift_mul_stack_eq_zero_of_tendsto_decayRun hG hη hrun⟩

/-- **Small steps converge to a decayed minimum** (`review/flat_spectrum.md`, item WD9):
under the hypotheses of `exists_pos_tendsto_decayRun`, some `η₀ > 0` has: for every
`0 < η < η₀` the run from every start of full row rank with `F(U₀, V₀) ≤ c` converges, and its
limit has the least weight-decayed loss among the factorizations of every width. -/
theorem exists_pos_tendsto_decayRun_isDecayedMin
    (hconv : ∀ X Y, L X + pairing (G X) (Y - X) ≤ L Y) (hG : Continuous G)
    (hL : ∀ y : EuclideanSpace ℂ (m × n),
      AnalyticAt ℝ (fun y : EuclideanSpace ℂ (m × n) => L (of fun i j => y (i, j))) y)
    (hlam : 0 < lam) (hlow : ∀ X, l₀ ≤ L X) (c : ℝ) :
    ∃ η₀ > 0, ∀ η, 0 < η → η < η₀ → ∀ (U₀ : Matrix m k ℂ) (V₀ : Matrix n k ℂ),
      decayedLoss L lam U₀ V₀ ≤ c → Function.Surjective (stack U₀ V₀).mulVec →
      ∃ (U : Matrix m k ℂ) (V : Matrix n k ℂ),
        Tendsto (decayRun G η lam U₀ V₀) atTop (𝓝 (U, V)) ∧ IsDecayedMin L lam U V := by
  obtain ⟨η₀, hη₀, h⟩ := exists_pos_forall_sublevel (k := k) hG hL hlam hlow c
  refine ⟨η₀, hη₀, fun η hη hlt U₀ V₀ hc hfull => ?_⟩
  obtain ⟨hmap, -, hconvg, hpd⟩ := h η hη hlt
  have hx₀ : flatLoss L lam (flatten (U₀, V₀)) ≤ c := by
    rw [flatLoss, factors_flatten]
    exact hc
  obtain ⟨w, -, -, hlim⟩ := hconvg _ hx₀
  have hiter := factors_iterate_gradientStep hconv hG hL η lam U₀ V₀
  have hrun : Tendsto (decayRun G η lam U₀ V₀) atTop (𝓝 (factors w)) :=
    ((continuous_factors.tendsto w).comp hlim).congr hiter
  refine ⟨(factors w).1, (factors w).2, hrun,
    isDecayedMin_of_tendsto_decayRun hconv hG hη hfull (fun t => ?_) hrun⟩
  have ht := hpd _ (hmap.iterate t hx₀)
  rwa [hiter t] at ht

/-- **Decayed minima exist** (`review/flat_spectrum.md`, item WD9): a loss that is
real-analytic in the entries, satisfies the first-order inequality with a continuous `G` and is
bounded below has, for every weight decay `λ > 0`, a factorization of width `m + n` with the
least weight-decayed loss among the factorizations of every width: the limit of a run from
`U₀ = [1 0]`, `V₀ = [0 1]` with a small step. -/
theorem exists_isDecayedMin (hconv : ∀ X Y, L X + pairing (G X) (Y - X) ≤ L Y)
    (hG : Continuous G)
    (hL : ∀ y : EuclideanSpace ℂ (m × n),
      AnalyticAt ℝ (fun y : EuclideanSpace ℂ (m × n) => L (of fun i j => y (i, j))) y)
    (hlam : 0 < lam) (hlow : ∀ X, l₀ ≤ L X) :
    ∃ (U : Matrix m (m ⊕ n) ℂ) (V : Matrix n (m ⊕ n) ℂ), IsDecayedMin L lam U V := by
  obtain ⟨η₀, hη₀, h⟩ := exists_pos_tendsto_decayRun_isDecayedMin (k := m ⊕ n) hconv hG hL hlam
    hlow (decayedLoss L lam (fromCols (1 : Matrix m m ℂ) (0 : Matrix m n ℂ))
      (fromCols (0 : Matrix n m ℂ) (1 : Matrix n n ℂ)))
  obtain ⟨U, V, -, hmin⟩ := h (η₀ / 2) (by positivity) (by linarith) _ _ le_rfl
    fun v => ⟨v, by rw [stack_fromCols_one, one_mulVec]⟩
  exact ⟨U, V, hmin⟩

/-! The witness: the squared loss of the `1 × 1` kernel `1` seen on its cell, `λ = 1`, below
the level `1`, at width one for the first theorem and at width two for the second. -/

example := exists_pos_tendsto_decayRun (k := Fin 1)
  (fitLoss_first_order (Set.univ : Set (Fin 1 × Fin 1)) 1) (continuous_mask_sub Set.univ 1)
  (analyticAt_fitLoss Set.univ 1) one_pos (fitLoss_nonneg Set.univ 1) 1

example := exists_pos_tendsto_decayRun_isDecayedMin (k := Fin 1 ⊕ Fin 1)
  (fitLoss_first_order (Set.univ : Set (Fin 1 × Fin 1)) 1) (continuous_mask_sub Set.univ 1)
  (analyticAt_fitLoss Set.univ 1) one_pos (fitLoss_nonneg Set.univ 1) 1

example := exists_isDecayedMin (fitLoss_first_order (Set.univ : Set (Fin 1 × Fin 1)) 1)
  (continuous_mask_sub Set.univ 1) (analyticAt_fitLoss Set.univ 1) one_pos
  (fitLoss_nonneg Set.univ 1)

end Transformer.Grokking.FlatSpectrum.Training
