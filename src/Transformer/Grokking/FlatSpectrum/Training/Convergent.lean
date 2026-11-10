module

public import Transformer.Landscape.SaddleAvoidance.Convergence
public import Transformer.Landscape.SaddleAvoidance.LocalInequality
public import Transformer.Grokking.FlatSpectrum.Training.Descent
public import Transformer.Grokking.FlatSpectrum.Training.FlatLoss
public import Transformer.Grokking.FlatSpectrum.Training.Recovery

/-!
# The run converges, to a decayed minimum

Source: `review/flat_spectrum.md`, *Training reaches the optimal fit* (added), item WD8.
Let the loss `L` be real-analytic in the entries of its matrix, satisfy the first-order
inequality with a continuous `G`, and satisfy the three bounds of WD6 and WD8 at the matrices
with `‖X‖ ≤ ρ/2`. With the step bounds of WD6 and WD8 the run stays in the ball
`‖U‖² + ‖V‖² ≤ ρ`, which is compact (`isCompact_setOf_frobeniusSq_add_le`), so it has a cluster
point. On the factor space the run is gradient descent on the analytic `flatLoss`
(`factors_iterate_gradientStep`), each step lowers the loss by `(η/2) ‖∇‖²`, and Łojasiewicz's
theorem (`Landscape.SaddleAvoidance.exists_isLojasiewiczAt_of_analyticAt`) gives the gradient
inequality at the cluster point. So the run converges, from every start in the ball and for
every width, to a stationary point (`exists_tendsto_decayRun`, by
`Landscape.SaddleAvoidance.tendsto_of_isLojasiewiczAt_of_le_sub`). From a start of full row
rank the limit is a decayed minimum (`exists_tendsto_decayRun_isDecayedMin`, by WD5 and WD6).
Convergence is proved, not assumed. The examples are the squared loss.
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

variable {L : Matrix m n ℂ → ℝ} {G : Matrix m n ℂ → Matrix m n ℂ} {η lam b γ s ρ : ℝ}

/-- The run of `decayStep` is gradient descent on `flatLoss`, read on the factor space. -/
theorem factors_iterate_gradientStep (hconv : ∀ X Y, L X + pairing (G X) (Y - X) ≤ L Y)
    (hG : Continuous G)
    (hL : ∀ y : EuclideanSpace ℂ (m × n),
      AnalyticAt ℝ (fun y : EuclideanSpace ℂ (m × n) => L (of fun i j => y (i, j))) y)
    (η lam : ℝ) (U₀ : Matrix m k ℂ) (V₀ : Matrix n k ℂ) (t : ℕ) :
    factors ((gradientStep (flatLoss L lam) η)^[t] (flatten (U₀, V₀))) =
      decayRun G η lam U₀ V₀ t := by
  induction t with
  | zero => exact factors_flatten _
  | succ t ih =>
    rw [Function.iterate_succ_apply', factors_gradientStep hconv hG hL, ih]
    exact (Function.iterate_succ_apply' _ _ _).symm

variable [DecidableEq m] [DecidableEq n]

/-- **The run converges** (`review/flat_spectrum.md`, item WD8): let `L` be real-analytic in
the entries and satisfy the first-order inequality with a continuous `G`, with `‖G X‖ ≤ γ`,
`⟨G X, X⟩ ≥ −b` and `L Y ≤ L X + ⟨G X, Y − X⟩ + (s/2) ‖Y − X‖²` for `‖X‖, ‖Y‖ ≤ ρ/2`. For a
weight decay `λ > 0` with `4 b ≤ λ ρ` and a step `0 < η` with `2 η (λ² + γ²) ≤ λ` and
`η (λ + γ + 2 s ρ) ≤ 1`, the run from every start in the ball `‖U₀‖² + ‖V₀‖² ≤ ρ`, of any
width and any rank, converges, and its limit is stationary: `M [U; V] = 0`. -/
theorem exists_tendsto_decayRun (hconv : ∀ X Y, L X + pairing (G X) (Y - X) ≤ L Y)
    (hG : Continuous G)
    (hL : ∀ y : EuclideanSpace ℂ (m × n),
      AnalyticAt ℝ (fun y : EuclideanSpace ℂ (m × n) => L (of fun i j => y (i, j))) y)
    (hη : 0 < η) (hlam : 0 < lam) (hγ0 : 0 ≤ γ) (hs0 : 0 ≤ s)
    (hγ : ∀ X, frobeniusSq X ≤ (ρ / 2) ^ 2 → frobeniusSq (G X) ≤ γ ^ 2)
    (hb : ∀ X, frobeniusSq X ≤ (ρ / 2) ^ 2 → -b ≤ pairing (G X) X)
    (hs : ∀ X Y, frobeniusSq X ≤ (ρ / 2) ^ 2 → frobeniusSq Y ≤ (ρ / 2) ^ 2 →
      L Y ≤ L X + pairing (G X) (Y - X) + s / 2 * frobeniusSq (Y - X))
    (hρ : 4 * b ≤ lam * ρ) (hstep : 2 * η * (lam ^ 2 + γ ^ 2) ≤ lam)
    (hdesc : η * (lam + γ + 2 * s * ρ) ≤ 1) {U₀ : Matrix m k ℂ} {V₀ : Matrix n k ℂ}
    (hw : frobeniusSq U₀ + frobeniusSq V₀ ≤ ρ) :
    ∃ (U : Matrix m k ℂ) (V : Matrix n k ℂ),
      Tendsto (decayRun G η lam U₀ V₀) atTop (𝓝 (U, V)) ∧
        lift lam (G (U * Vᴴ)) * stack U V = 0 := by
  have hmem : ∀ᶠ t in atTop, decayRun G η lam U₀ V₀ t ∈
      {p : Matrix m k ℂ × Matrix n k ℂ | frobeniusSq p.1 + frobeniusSq p.2 ≤ ρ} :=
    Eventually.of_forall fun t => frobeniusSq_decayRun_le hη.le hlam hγ hb hρ hstep hw t
  obtain ⟨p, -, hp⟩ := (isCompact_setOf_frobeniusSq_add_le ρ).exists_mapClusterPt
    (le_principal_iff.mpr (mem_map.mpr hmem))
  have hiter := factors_iterate_gradientStep hconv hG hL η lam U₀ V₀
  have hflat : (fun t => (gradientStep (flatLoss L lam) η)^[t] (flatten (U₀, V₀))) =
      flatten ∘ decayRun G η lam U₀ V₀ :=
    funext fun t => by rw [Function.comp_apply, ← hiter t, flatten_factors]
  obtain ⟨a, ha0, ha1, hz⟩ := Landscape.SaddleAvoidance.exists_isLojasiewiczAt_of_analyticAt
    (analyticAt_flatLoss (k := k) hL lam (flatten p))
  have hlimE := Landscape.SaddleAvoidance.tendsto_of_isLojasiewiczAt_of_le_sub hη
    (κ := 1 / 2) (by norm_num) ha0 ha1 hz (analyticAt_flatLoss hL lam (flatten p)).continuousAt
    (x := flatten (U₀, V₀)) (fun t => by
      have h := decayedLoss_decayRun_succ_add_le (L := L) hη.le hlam hγ0 hs0 hγ hb hs hρ hstep
        hdesc hw t
      rw [norm_gradient_flatLoss_sq hconv hG hL, flatLoss, flatLoss, hiter, hiter]
      linarith)
    (hflat ▸ hp.tendsto_comp (continuous_flatten.tendsto p))
  have hlim : Tendsto (decayRun G η lam U₀ V₀) atTop (𝓝 p) := by
    have h := (continuous_factors.tendsto (flatten p)).comp hlimE
    rw [factors_flatten] at h
    exact h.congr hiter
  exact ⟨p.1, p.2, hlim, lift_mul_stack_eq_zero_of_tendsto_decayRun hG hη hlim⟩

/-- **The run converges, to a decayed minimum** (`review/flat_spectrum.md`, item WD8): under
the hypotheses of `exists_tendsto_decayRun`, the run from a start of full row rank converges,
and its limit has the least weight-decayed loss among the factorizations of every width.
Convergence is a conclusion: the loss is analytic, so Łojasiewicz's theorem applies at a
cluster point of the run. -/
theorem exists_tendsto_decayRun_isDecayedMin
    (hconv : ∀ X Y, L X + pairing (G X) (Y - X) ≤ L Y) (hG : Continuous G)
    (hL : ∀ y : EuclideanSpace ℂ (m × n),
      AnalyticAt ℝ (fun y : EuclideanSpace ℂ (m × n) => L (of fun i j => y (i, j))) y)
    (hη : 0 < η) (hlam : 0 < lam) (hγ0 : 0 ≤ γ) (hs0 : 0 ≤ s)
    (hγ : ∀ X, frobeniusSq X ≤ (ρ / 2) ^ 2 → frobeniusSq (G X) ≤ γ ^ 2)
    (hb : ∀ X, frobeniusSq X ≤ (ρ / 2) ^ 2 → -b ≤ pairing (G X) X)
    (hs : ∀ X Y, frobeniusSq X ≤ (ρ / 2) ^ 2 → frobeniusSq Y ≤ (ρ / 2) ^ 2 →
      L Y ≤ L X + pairing (G X) (Y - X) + s / 2 * frobeniusSq (Y - X))
    (hρ : 4 * b ≤ lam * ρ) (hstep : 2 * η * (lam ^ 2 + γ ^ 2) ≤ lam)
    (hdesc : η * (lam + γ + 2 * s * ρ) ≤ 1) {U₀ : Matrix m k ℂ} {V₀ : Matrix n k ℂ}
    (hw : frobeniusSq U₀ + frobeniusSq V₀ ≤ ρ)
    (hfull : Function.Surjective (stack U₀ V₀).mulVec) :
    ∃ (U : Matrix m k ℂ) (V : Matrix n k ℂ),
      Tendsto (decayRun G η lam U₀ V₀) atTop (𝓝 (U, V)) ∧ IsDecayedMin L lam U V := by
  obtain ⟨U, V, hlim, -⟩ :=
    exists_tendsto_decayRun hconv hG hL hη hlam hγ0 hs0 hγ hb hs hρ hstep hdesc hw
  exact ⟨U, V, hlim, isDecayedMin_of_tendsto_decayRun_of_ball hconv hG hη hlam hγ0 hγ hb hρ
    hstep hw hfull hlim⟩

/-! The witness: the squared loss of the `1 × 1` kernel `1` seen on its cell, `λ = 1`,
`η = 1/10`, `γ = 2`, `b = 1/4`, `s = 1`, `ρ = 2`. -/

/-- From the width-one start `U₀ = V₀ = 1`, which has no full row rank, the run converges to a
stationary point. -/
example := exists_tendsto_decayRun (fitLoss_first_order (Set.univ : Set (Fin 1 × Fin 1)) 1)
  (continuous_mask_sub Set.univ 1) (analyticAt_fitLoss Set.univ 1) (η := 1 / 10) (by norm_num)
  one_pos (by norm_num) zero_le_one stepWitness_bound (fun X _ => stepWitness_low X)
  (fun X Y _ _ => fitLoss_le_add Set.univ 1 X Y) (by norm_num) (by norm_num) (by norm_num)
  (U₀ := (1 : Matrix (Fin 1) (Fin 1) ℂ)) (V₀ := 1) (by norm_num [frobeniusSq])

/-- From `U₀ = [1 0]`, `V₀ = [0 1]` the run converges to a decayed minimum; no limit is
supplied. -/
example : ∃ U V : Matrix (Fin 1) (Fin 1 ⊕ Fin 1) ℂ,
    Tendsto (decayRun (fun X => mask (Set.univ : Set (Fin 1 × Fin 1)) (X - 1)) (1 / 10) 1
      (fromCols 1 0) (fromCols 0 1)) atTop (𝓝 (U, V)) ∧ IsDecayedMin (fitLoss Set.univ 1) 1 U V :=
  exists_tendsto_decayRun_isDecayedMin (fitLoss_first_order Set.univ 1)
    (continuous_mask_sub Set.univ 1) (analyticAt_fitLoss Set.univ 1) (by norm_num) one_pos
    (by norm_num) zero_le_one stepWitness_bound (fun X _ => stepWitness_low X)
    (fun X Y _ _ => fitLoss_le_add Set.univ 1 X Y) (by norm_num) (by norm_num) (by norm_num)
    stepWitness_ball fun v => ⟨v, by rw [stack_fromCols_one, one_mulVec]⟩

end Transformer.Grokking.FlatSpectrum.Training
