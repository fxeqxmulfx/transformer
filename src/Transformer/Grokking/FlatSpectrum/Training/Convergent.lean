module

public import Transformer.Landscape.SaddleAvoidance.Convergence
public import Transformer.Landscape.SaddleAvoidance.LocalInequality
public import Transformer.Grokking.FlatSpectrum.Training.Descent
public import Transformer.Grokking.FlatSpectrum.Training.FlatLoss
public import Transformer.Grokking.FlatSpectrum.Training.Recovery

/-!
# The run on the squared loss converges, to a decayed minimum

Source: `review/flat_spectrum.md`, *Training reaches the optimal fit* (added), item WD8.
For the squared loss with the step bounds of WD6 and WD8 the run stays in the ball
`‖U‖² + ‖V‖² ≤ ρ`, which is compact (`isCompact_setOf_frobeniusSq_add_le`), so it has a cluster
point. On the factor space the run is gradient descent on the analytic `flatLoss`
(`factors_iterate_gradientStep`), each step lowers the loss by `(η/2) ‖∇‖²`, and Łojasiewicz's
theorem (`Landscape.SaddleAvoidance.exists_isLojasiewiczAt_of_analyticAt`) gives the gradient
inequality at the cluster point. So the run converges
(`Landscape.SaddleAvoidance.tendsto_of_isLojasiewiczAt_of_le_sub`), and by WD5 and WD6 its
limit is a decayed minimum (`exists_tendsto_decayRun_isDecayedMin`). Convergence is proved,
not assumed.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Training

open Matrix Filter Topology
open Transformer.Landscape.SaddleAvoidance (gradientStep)

variable {m n k : Type*} [Fintype m] [Fintype n] [Fintype k]

omit [Fintype k] in
/-- An entry is at most the Frobenius norm. -/
lemma norm_apply_le_sqrt_frobeniusSq (A : Matrix m n ℂ) (i : m) (j : n) :
    ‖A i j‖ ≤ √(frobeniusSq A) := by
  refine Real.le_sqrt_of_sq_le ?_
  calc ‖A i j‖ ^ 2 ≤ ∑ j', ‖A i j'‖ ^ 2 :=
        Finset.single_le_sum (f := fun j' => ‖A i j'‖ ^ 2) (fun _ _ => sq_nonneg _)
          (Finset.mem_univ j)
    _ ≤ frobeniusSq A :=
        Finset.single_le_sum (f := fun i' => ∑ j', ‖A i' j'‖ ^ 2)
          (fun _ _ => Finset.sum_nonneg fun _ _ => sq_nonneg _) (Finset.mem_univ i)

/-- The ball `‖U‖² + ‖V‖² ≤ ρ` of factor pairs is compact. -/
lemma isCompact_setOf_frobeniusSq_add_le (ρ : ℝ) :
    IsCompact {p : Matrix m k ℂ × Matrix n k ℂ | frobeniusSq p.1 + frobeniusSq p.2 ≤ ρ} := by
  refine ((isCompact_setOf_norm_apply_le (m := m) (n := k) √ρ).prod
    (isCompact_setOf_norm_apply_le (m := n) (n := k) √ρ)).of_isClosed_subset
    (isClosed_le ((continuous_frobeniusSq.comp continuous_fst).add
      (continuous_frobeniusSq.comp continuous_snd)) continuous_const) fun p hp => ?_
  have h1 := frobeniusSq_nonneg p.1
  have h2 := frobeniusSq_nonneg p.2
  have hp' : frobeniusSq p.1 + frobeniusSq p.2 ≤ ρ := hp
  exact ⟨fun i j => (norm_apply_le_sqrt_frobeniusSq p.1 i j).trans
      (Real.sqrt_le_sqrt (by linarith)),
    fun i j => (norm_apply_le_sqrt_frobeniusSq p.2 i j).trans
      (Real.sqrt_le_sqrt (by linarith))⟩

variable {Ω : Set (m × n)} {K : Matrix m n ℂ} {η lam κ ρ : ℝ}

/-- The run of `decayStep` is gradient descent on `flatLoss`, read on the factor space. -/
theorem factors_iterate_gradientStep (Ω : Set (m × n)) (K : Matrix m n ℂ) (η lam : ℝ)
    (U₀ : Matrix m k ℂ) (V₀ : Matrix n k ℂ) (t : ℕ) :
    factors ((gradientStep (flatLoss Ω K lam) η)^[t] (flatten (U₀, V₀))) =
      decayRun (fun X => mask Ω (X - K)) η lam U₀ V₀ t := by
  induction t with
  | zero => exact factors_flatten _
  | succ t ih =>
    rw [Function.iterate_succ_apply', factors_gradientStep, ih]
    exact (Function.iterate_succ_apply' _ _ _).symm

variable [DecidableEq m] [DecidableEq n]

/-- **The run on the squared loss converges, to a decayed minimum**
(`review/flat_spectrum.md`, item WD8): for the squared loss on `Ω` with `‖K‖ ≤ κ`, weight decay
`λ > 0`, a start of full row rank in the ball `‖U₀‖² + ‖V₀‖² ≤ ρ` with `κ² ≤ λ ρ`, and a step
`0 < η` with `2 η (λ² + (ρ/2 + κ)²) ≤ λ` and `η (λ + ρ/2 + κ + 2ρ) ≤ 1`, the run of gradient
descent with weight decay converges, and its limit has the least weight-decayed loss among the
factorizations of every width. Convergence is a conclusion: the loss is a polynomial, so
Łojasiewicz's theorem applies at a cluster point of the run. -/
theorem exists_tendsto_decayRun_isDecayedMin (hη : 0 < η) (hlam : 0 < lam) (hκ0 : 0 ≤ κ)
    (hκ : frobeniusSq K ≤ κ ^ 2) (hρ : κ ^ 2 ≤ lam * ρ)
    (hstep : 2 * η * (lam ^ 2 + (ρ / 2 + κ) ^ 2) ≤ lam)
    (hdesc : η * (lam + (ρ / 2 + κ) + 2 * ρ) ≤ 1) {U₀ : Matrix m k ℂ} {V₀ : Matrix n k ℂ}
    (hw : frobeniusSq U₀ + frobeniusSq V₀ ≤ ρ)
    (hfull : Function.Surjective (stack U₀ V₀).mulVec) :
    ∃ (U : Matrix m k ℂ) (V : Matrix n k ℂ),
      Tendsto (decayRun (fun X => mask Ω (X - K)) η lam U₀ V₀) atTop (𝓝 (U, V)) ∧
        IsDecayedMin (fitLoss Ω K) lam U V := by
  have hmem : ∀ᶠ t in atTop, decayRun (fun X => mask Ω (X - K)) η lam U₀ V₀ t ∈
      {p : Matrix m k ℂ × Matrix n k ℂ | frobeniusSq p.1 + frobeniusSq p.2 ≤ ρ} :=
    Eventually.of_forall fun t => frobeniusSq_decayRun_le hη.le hlam hκ0 hκ hρ hstep hw t
  obtain ⟨p, -, hp⟩ := (isCompact_setOf_frobeniusSq_add_le ρ).exists_mapClusterPt
    (le_principal_iff.mpr (mem_map.mpr hmem))
  have hiter := factors_iterate_gradientStep Ω K η lam U₀ V₀
  have hflat : (fun t => (gradientStep (flatLoss Ω K lam) η)^[t] (flatten (U₀, V₀))) =
      flatten ∘ decayRun (fun X => mask Ω (X - K)) η lam U₀ V₀ :=
    funext fun t => by rw [Function.comp_apply, ← hiter t, flatten_factors]
  obtain ⟨a, ha0, ha1, hz⟩ := Landscape.SaddleAvoidance.exists_isLojasiewiczAt_of_analyticAt
    (analyticAt_flatLoss (k := k) Ω K lam (flatten p))
  have hlimE := Landscape.SaddleAvoidance.tendsto_of_isLojasiewiczAt_of_le_sub hη
    (κ := 1 / 2) (by norm_num) ha0 ha1 hz (analyticAt_flatLoss Ω K lam (flatten p)).continuousAt
    (x := flatten (U₀, V₀)) (fun t => by
      have h := decayedLoss_decayRun_succ_add_le (Ω := Ω) hη.le hlam hκ0 hκ hρ hstep hdesc hw t
      rw [norm_gradient_flatLoss_sq, flatLoss, flatLoss, hiter, hiter]
      linarith)
    (hflat ▸ hp.tendsto_comp (continuous_flatten.tendsto p))
  have hlim : Tendsto (decayRun (fun X => mask Ω (X - K)) η lam U₀ V₀) atTop (𝓝 p) := by
    have h := (continuous_factors.tendsto (flatten p)).comp hlimE
    rw [factors_flatten] at h
    exact h.congr hiter
  exact ⟨p.1, p.2, hlim,
    isDecayedMin_fitLoss_of_tendsto_decayRun hη hlam hκ0 hκ hρ hstep hw hfull hlim⟩

/-- The `1 × 1` kernel `1` seen on its cell, `λ = 1`, `η = 1/10`, from `U₀ = [1 0]`,
`V₀ = [0 1]`: the run converges to a decayed minimum; no limit is supplied. -/
example : ∃ U V : Matrix (Fin 1) (Fin 1 ⊕ Fin 1) ℂ,
    Tendsto (decayRun (fun X => mask (Set.univ : Set (Fin 1 × Fin 1)) (X - 1)) (1 / 10) 1
      (fromCols 1 0) (fromCols 0 1)) atTop (𝓝 (U, V)) ∧ IsDecayedMin (fitLoss Set.univ 1) 1 U V :=
  exists_tendsto_decayRun_isDecayedMin (κ := 1) (ρ := 2) (by norm_num) one_pos zero_le_one
    stepWitness_kernel (by norm_num) (by norm_num) (by norm_num) stepWitness_ball
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
  obtain ⟨U, V, -, h⟩ := exists_tendsto_decayRun_isDecayedMin (Ω := Ω)
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
    exists_tendsto_decayRun_isDecayedMin hη hlam hκ0 hκ hρ hstep hdesc hw hfull
  exact ⟨U, V, hlim, by linarith [hdm.nuclearNorm_le hlam hX']⟩

/-- With no cell seen, runs on the `a³ + a b` kernel for `p = 3` converge strictly below it. -/
example := exists_tendsto_decayRun_nuclearNorm_add_le (k := ZMod 3 ⊕ ZMod 3)
  (singularValues_cubicKernel 3)
  (fun a b => norm_ne_zero_iff.mp (by simp [cubicKernel, taskKernel]))
  (Set.empty_ne_univ (α := ZMod 3 × ZMod 3))

end Transformer.Grokking.FlatSpectrum.Training
