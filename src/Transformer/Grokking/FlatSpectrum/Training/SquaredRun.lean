module

public import Transformer.Grokking.FlatSpectrum.Training.Convergent

/-!
# The run on the squared loss converges

Source: `review/flat_spectrum.md`, *Training reaches the optimal fit* (added), items WD7 and
WD8. The squared loss on `Ω` with `‖K‖ ≤ κ` meets the hypotheses of `exists_tendsto_decayRun`
with `γ = ρ/2 + κ`, `b = κ²/4` and `s = 1` (`Training.SquaredBounds`). So for `κ² ≤ λ ρ`,
`2 η (λ² + γ²) ≤ λ` and `η (λ + γ + 2ρ) ≤ 1` the run from the ball converges, for every width,
to a stationary point (`exists_tendsto_decayRun_fitLoss`), and from a start of full row rank to
a decayed minimum (`exists_tendsto_decayRun_fitLoss_isDecayedMin`). Hence trained fits exist for
every `λ > 0` (`exists_isDecayedMin_fitLoss`), and Theorem A holds for the run itself
(`exists_tendsto_decayRun_nuclearNorm_add_le`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Training

open Matrix Filter Topology

variable {m n k : Type*} [Fintype m] [Fintype n] [Fintype k] [DecidableEq m] [DecidableEq n]
  {Ω : Set (m × n)} {K : Matrix m n ℂ} {η lam κ ρ : ℝ}

/-- **The run on the squared loss converges** (`review/flat_spectrum.md`, item WD8): for the
squared loss on `Ω` with `‖K‖ ≤ κ`, weight decay `λ > 0`, a start of any width and rank in the
ball `‖U₀‖² + ‖V₀‖² ≤ ρ` with `κ² ≤ λ ρ`, and a step `0 < η` with `2 η (λ² + (ρ/2 + κ)²) ≤ λ`
and `η (λ + ρ/2 + κ + 2ρ) ≤ 1`, the run of gradient descent with weight decay converges, and
its limit is stationary. -/
theorem exists_tendsto_decayRun_fitLoss (hη : 0 < η) (hlam : 0 < lam) (hκ0 : 0 ≤ κ)
    (hκ : frobeniusSq K ≤ κ ^ 2) (hρ : κ ^ 2 ≤ lam * ρ)
    (hstep : 2 * η * (lam ^ 2 + (ρ / 2 + κ) ^ 2) ≤ lam)
    (hdesc : η * (lam + (ρ / 2 + κ) + 2 * ρ) ≤ 1) {U₀ : Matrix m k ℂ} {V₀ : Matrix n k ℂ}
    (hw : frobeniusSq U₀ + frobeniusSq V₀ ≤ ρ) :
    ∃ (U : Matrix m k ℂ) (V : Matrix n k ℂ),
      Tendsto (decayRun (fun X => mask Ω (X - K)) η lam U₀ V₀) atTop (𝓝 (U, V)) ∧
        lift lam (mask Ω (U * Vᴴ - K)) * stack U V = 0 := by
  have hρ0 : 0 ≤ ρ := (add_nonneg (frobeniusSq_nonneg U₀) (frobeniusSq_nonneg V₀)).trans hw
  exact exists_tendsto_decayRun (fitLoss_first_order Ω K) (continuous_mask_sub Ω K)
    (analyticAt_fitLoss Ω K) hη hlam (by linarith) zero_le_one
    (frobeniusSq_mask_sub_le hρ0 hκ0 hκ) (fun X _ => neg_le_pairing_mask_sub_of_le hκ X)
    (fun X Y _ _ => fitLoss_le_add Ω K X Y) (by linarith) hstep (by rw [mul_one]; exact hdesc)
    hw

/-- **The run on the squared loss converges, to a decayed minimum**
(`review/flat_spectrum.md`, item WD8): under the bounds of `exists_tendsto_decayRun_fitLoss`,
the run from a start of full row rank converges, and its limit has the least weight-decayed
loss among the factorizations of every width. Convergence is a conclusion: the loss is a
polynomial, so Łojasiewicz's theorem applies at a cluster point of the run. -/
theorem exists_tendsto_decayRun_fitLoss_isDecayedMin (hη : 0 < η) (hlam : 0 < lam)
    (hκ0 : 0 ≤ κ) (hκ : frobeniusSq K ≤ κ ^ 2) (hρ : κ ^ 2 ≤ lam * ρ)
    (hstep : 2 * η * (lam ^ 2 + (ρ / 2 + κ) ^ 2) ≤ lam)
    (hdesc : η * (lam + (ρ / 2 + κ) + 2 * ρ) ≤ 1) {U₀ : Matrix m k ℂ} {V₀ : Matrix n k ℂ}
    (hw : frobeniusSq U₀ + frobeniusSq V₀ ≤ ρ)
    (hfull : Function.Surjective (stack U₀ V₀).mulVec) :
    ∃ (U : Matrix m k ℂ) (V : Matrix n k ℂ),
      Tendsto (decayRun (fun X => mask Ω (X - K)) η lam U₀ V₀) atTop (𝓝 (U, V)) ∧
        IsDecayedMin (fitLoss Ω K) lam U V := by
  obtain ⟨U, V, hlim, -⟩ :=
    exists_tendsto_decayRun_fitLoss (Ω := Ω) hη hlam hκ0 hκ hρ hstep hdesc hw
  exact ⟨U, V, hlim,
    isDecayedMin_fitLoss_of_tendsto_decayRun hη hlam hκ0 hκ hρ hstep hw hfull hlim⟩

/-! The witness: the `1 × 1` kernel `1` seen on its cell, `λ = 1`, `η = 1/10`, `κ = 1`,
`ρ = 2`: `2 η (λ² + 2²) = 1` and `η (λ + 2 + 2ρ) = 7/10`. -/

/-- The width-one start `U₀ = V₀ = 1` has no full row rank; the run converges. -/
example := exists_tendsto_decayRun_fitLoss (Ω := (Set.univ : Set (Fin 1 × Fin 1))) (η := 1 / 10)
  (κ := 1) (ρ := 2) (by norm_num) one_pos zero_le_one stepWitness_kernel (by norm_num)
  (by norm_num) (by norm_num) (U₀ := (1 : Matrix (Fin 1) (Fin 1) ℂ)) (V₀ := 1)
  (by norm_num [frobeniusSq])

/-- From `U₀ = [1 0]`, `V₀ = [0 1]` the run converges to a decayed minimum. -/
example : ∃ U V : Matrix (Fin 1) (Fin 1 ⊕ Fin 1) ℂ,
    Tendsto (decayRun (fun X => mask (Set.univ : Set (Fin 1 × Fin 1)) (X - 1)) (1 / 10) 1
      (fromCols 1 0) (fromCols 0 1)) atTop (𝓝 (U, V)) ∧ IsDecayedMin (fitLoss Set.univ 1) 1 U V :=
  exists_tendsto_decayRun_fitLoss_isDecayedMin (κ := 1) (ρ := 2) (by norm_num) one_pos
    zero_le_one stepWitness_kernel (by norm_num) (by norm_num) (by norm_num) stepWitness_ball
    fun v => ⟨v, by rw [stack_fromCols_one, one_mulVec]⟩

/-- **Trained fits exist** (`review/flat_spectrum.md`, items WD7 and WD8): for every weight
decay `λ > 0` the squared loss on `Ω` has a factorization of width `m + n` with the least
weight-decayed loss among the factorizations of every width, the limit of the run from
`U₀ = [1 0]`, `V₀ = [0 1]` with a small enough step. -/
theorem exists_isDecayedMin_fitLoss (Ω : Set (m × n)) (K : Matrix m n ℂ) (hlam : 0 < lam) :
    ∃ (U : Matrix m (m ⊕ n) ℂ) (V : Matrix n (m ⊕ n) ℂ),
      IsDecayedMin (fitLoss Ω K) lam U V := by
  have hK := frobeniusSq_nonneg K
  have hκ0 : 0 ≤ frobeniusSq K + 1 := by linarith
  have hκ : frobeniusSq K ≤ (frobeniusSq K + 1) ^ 2 := by nlinarith
  have hc0 : 0 ≤ frobeniusSq (fromCols (1 : Matrix m m ℂ) (0 : Matrix m n ℂ)) +
      frobeniusSq (fromCols (0 : Matrix n m ℂ) (1 : Matrix n n ℂ)) :=
    add_nonneg (frobeniusSq_nonneg _) (frobeniusSq_nonneg _)
  obtain ⟨ρ, hρc, hρκ⟩ : ∃ ρ : ℝ,
      frobeniusSq (fromCols (1 : Matrix m m ℂ) (0 : Matrix m n ℂ)) +
        frobeniusSq (fromCols (0 : Matrix n m ℂ) (1 : Matrix n n ℂ)) ≤ ρ ∧
      (frobeniusSq K + 1) ^ 2 ≤ lam * ρ :=
    ⟨_, le_max_left _ ((frobeniusSq K + 1) ^ 2 / lam), by
      have h := le_max_right (frobeniusSq (fromCols (1 : Matrix m m ℂ) (0 : Matrix m n ℂ)) +
        frobeniusSq (fromCols (0 : Matrix n m ℂ) (1 : Matrix n n ℂ)))
        ((frobeniusSq K + 1) ^ 2 / lam)
      rw [div_le_iff₀ hlam] at h
      linarith⟩
  have hρ0 : 0 ≤ ρ := hc0.trans hρc
  have hA : 0 < 2 * (lam ^ 2 + (ρ / 2 + (frobeniusSq K + 1)) ^ 2) := by positivity
  have hB : 0 < lam + (ρ / 2 + (frobeniusSq K + 1)) + 2 * ρ := by linarith
  have hstep := min_le_left (lam / (2 * (lam ^ 2 + (ρ / 2 + (frobeniusSq K + 1)) ^ 2)))
    (1 / (lam + (ρ / 2 + (frobeniusSq K + 1)) + 2 * ρ))
  have hdesc := min_le_right (lam / (2 * (lam ^ 2 + (ρ / 2 + (frobeniusSq K + 1)) ^ 2)))
    (1 / (lam + (ρ / 2 + (frobeniusSq K + 1)) + 2 * ρ))
  rw [le_div_iff₀ hA] at hstep
  rw [le_div_iff₀ hB] at hdesc
  obtain ⟨U, V, -, h⟩ := exists_tendsto_decayRun_fitLoss_isDecayedMin (Ω := Ω)
    (lt_min (div_pos hlam hA) (one_div_pos.mpr hB)) hlam hκ0 hκ hρκ (by linarith) hdesc hρc
    fun v => ⟨v, by rw [stack_fromCols_one, one_mulVec]⟩
  exact ⟨U, V, h⟩

example := exists_isDecayedMin_fitLoss (Set.univ : Set (Fin 1 × Fin 1)) 1 one_pos

/-- **Training does not return a flat kernel** (`review/flat_spectrum.md`, Theorem A with items
WD7 and WD8): if the singular values of a square `K` are all equal, no entry of `K` vanishes and
`Ω` is proper, then some `δ > 0` has: for every `λ > 0`, every run on the squared loss on `Ω`
under the bounds of WD6 and WD8 from a start of full row rank converges, to factors with
`‖U Vᴴ‖_S1 + δ ≤ ‖K‖_S1`. -/
theorem exists_tendsto_decayRun_nuclearNorm_add_le {K : Matrix n n ℂ} {c : ℝ}
    (h : ∀ i, singularValues K i = c) (hK : ∀ i j, K i j ≠ 0) {Ω : Set (n × n)}
    (hΩ : Ω ≠ Set.univ) :
    ∃ δ : ℝ, 0 < δ ∧ ∀ {η lam κ ρ : ℝ}, 0 < η → 0 < lam → 0 ≤ κ → frobeniusSq K ≤ κ ^ 2 →
      κ ^ 2 ≤ lam * ρ → 2 * η * (lam ^ 2 + (ρ / 2 + κ) ^ 2) ≤ lam →
      η * (lam + (ρ / 2 + κ) + 2 * ρ) ≤ 1 → ∀ {U₀ V₀ : Matrix n k ℂ},
        frobeniusSq U₀ + frobeniusSq V₀ ≤ ρ → Function.Surjective (stack U₀ V₀).mulVec →
        ∃ U V : Matrix n k ℂ,
          Tendsto (decayRun (fun X => mask Ω (X - K)) η lam U₀ V₀) atTop (𝓝 (U, V)) ∧
            nuclearNorm (U * Vᴴ) + δ ≤ nuclearNorm K := by
  have hmin := not_isMinOn_of_singularValues_eq h hK hΩ
  rw [isMinOn_iff] at hmin
  push Not at hmin
  obtain ⟨X', hX', hlt⟩ := hmin
  refine ⟨nuclearNorm K - nuclearNorm X', sub_pos.mpr hlt, ?_⟩
  intro η lam κ ρ hη hlam hκ0 hκ hρ hstep hdesc U₀ V₀ hw hfull
  obtain ⟨U, V, hlim, hdm⟩ :=
    exists_tendsto_decayRun_fitLoss_isDecayedMin hη hlam hκ0 hκ hρ hstep hdesc hw hfull
  exact ⟨U, V, hlim, by linarith [hdm.nuclearNorm_le hlam hX']⟩

/-- With no cell seen, runs on the `a³ + a b` kernel for `p = 3` converge strictly below it. -/
example := exists_tendsto_decayRun_nuclearNorm_add_le (k := ZMod 3 ⊕ ZMod 3)
  (singularValues_cubicKernel 3)
  (fun a b => norm_ne_zero_iff.mp (by simp [cubicKernel, taskKernel]))
  (Set.empty_ne_univ (α := ZMod 3 × ZMod 3))

end Transformer.Grokking.FlatSpectrum.Training
