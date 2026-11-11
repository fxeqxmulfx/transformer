module

public import Transformer.Grokking.FlatSpectrum.Training.BoundedRun

/-!
# The run converges, to a decayed minimum

Source: `review/flat_spectrum.md`, *Training reaches the optimal fit* (added), item WD8.
Let the loss `L` be real-analytic in the entries of its matrix, satisfy the first-order
inequality with a continuous `G`, and satisfy the three bounds of WD6 and WD8 at the matrices
with `‖X‖ ≤ ρ/2`. With the step bounds of WD6 the run stays in the ball `‖U‖² + ‖V‖² ≤ ρ`,
and a run that stays in the ball converges to a stationary point under the step bound of WD8
(`exists_tendsto_decayRun_of_forall_le` of `BoundedRun`: compactness, the descent of the
analytic `flatLoss` and Łojasiewicz's theorem). So the run converges, from every start in
the ball and for every width, to a stationary point (`exists_tendsto_decayRun`). From a start
of full row rank the limit is a decayed minimum (`exists_tendsto_decayRun_isDecayedMin`, by
WD5 and WD6). Convergence is proved, not assumed. The examples are the squared loss.

The sign of `λ` enters only through WD6. For every real `λ`, a run from a start of full row
rank that stays in a ball where its step is admissible ends at a decayed minimum
(`exists_tendsto_decayRun_isDecayedMin_of_forall_le`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Training

open Matrix Filter Topology

variable {m n k : Type*} [Fintype m] [Fintype n] [Fintype k] [DecidableEq m] [DecidableEq n]
  {L : Matrix m n ℂ → ℝ} {G : Matrix m n ℂ → Matrix m n ℂ} {η lam b γ s ρ : ℝ}

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
        lift lam (G (U * Vᴴ)) * stack U V = 0 :=
  exists_tendsto_decayRun_of_forall_le hconv hG hL hη hγ0 hs0 hγ hs hdesc fun t =>
    frobeniusSq_decayRun_le hη.le hlam hγ hb hρ hstep hw t


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

/-- **A run that stays in a ball ends at a decayed minimum** (`review/flat_spectrum.md`, items
WD8 and XE17): for every weight decay `λ`, zero and negative ones included, under the bounds
of `exists_tendsto_decayRun_of_forall_le` and `η (|λ| + γ) < 1`, a run from a start of full
row rank that stays in the ball `‖U‖² + ‖V‖² ≤ ρ` converges, and its limit has the least
weight-decayed loss among the factorizations of every width. So where no such minimum exists
the run leaves the ball. -/
theorem exists_tendsto_decayRun_isDecayedMin_of_forall_le
    (hconv : ∀ X Y, L X + pairing (G X) (Y - X) ≤ L Y) (hG : Continuous G)
    (hL : ∀ y : EuclideanSpace ℂ (m × n),
      AnalyticAt ℝ (fun y : EuclideanSpace ℂ (m × n) => L (of fun i j => y (i, j))) y)
    (hη : 0 < η) (hγ0 : 0 ≤ γ) (hs0 : 0 ≤ s)
    (hγ : ∀ X, frobeniusSq X ≤ (ρ / 2) ^ 2 → frobeniusSq (G X) ≤ γ ^ 2)
    (hs : ∀ X Y, frobeniusSq X ≤ (ρ / 2) ^ 2 → frobeniusSq Y ≤ (ρ / 2) ^ 2 →
      L Y ≤ L X + pairing (G X) (Y - X) + s / 2 * frobeniusSq (Y - X))
    (hdesc : η * (lam + γ + 2 * s * ρ) ≤ 1) (hpd : η * (|lam| + γ) < 1)
    {U₀ : Matrix m k ℂ} {V₀ : Matrix n k ℂ}
    (hball : ∀ t, frobeniusSq (decayRun G η lam U₀ V₀ t).1 +
      frobeniusSq (decayRun G η lam U₀ V₀ t).2 ≤ ρ)
    (hfull : Function.Surjective (stack U₀ V₀).mulVec) :
    ∃ (U : Matrix m k ℂ) (V : Matrix n k ℂ),
      Tendsto (decayRun G η lam U₀ V₀) atTop (𝓝 (U, V)) ∧ IsDecayedMin L lam U V := by
  obtain ⟨U, V, hlim, -⟩ :=
    exists_tendsto_decayRun_of_forall_le hconv hG hL hη hγ0 hs0 hγ hs hdesc hball
  exact ⟨U, V, hlim, isDecayedMin_of_tendsto_decayRun hconv hG hη hfull (fun t =>
    one_sub_smul_lift_posDef hη.le hγ0 (hγ _ (frobeniusSq_mul_conjTranspose_le (hball t))) hpd)
    hlim⟩

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

/-- The same run, with the ball of WD6 supplied as a hypothesis on the run. -/
example := exists_tendsto_decayRun_isDecayedMin_of_forall_le
  (fitLoss_first_order (Set.univ : Set (Fin 1 × Fin 1)) 1) (continuous_mask_sub Set.univ 1)
  (analyticAt_fitLoss Set.univ 1) (η := 1 / 10) (lam := 1) (by norm_num) (by norm_num)
  zero_le_one stepWitness_bound (fun X Y _ _ => fitLoss_le_add Set.univ 1 X Y) (by norm_num)
  (by norm_num)
  (fun t => frobeniusSq_decayRun_le (by norm_num) one_pos stepWitness_bound
    (fun X _ => stepWitness_low X) (by norm_num) (by norm_num) stepWitness_ball t)
  fun v => ⟨v, by rw [stack_fromCols_one, one_mulVec]⟩

end Transformer.Grokking.FlatSpectrum.Training
