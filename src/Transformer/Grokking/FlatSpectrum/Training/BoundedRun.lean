module

public import Transformer.Landscape.SaddleAvoidance.Convergence
public import Transformer.Landscape.SaddleAvoidance.LocalInequality
public import Transformer.Grokking.FlatSpectrum.Training.Descent
public import Transformer.Grokking.FlatSpectrum.Training.FlatLoss
public import Transformer.Grokking.FlatSpectrum.Training.Recovery

/-!
# A run that stays in a ball converges

Source: `review/flat_spectrum.md`, *Training reaches the optimal fit* (added), item WD8, and
*Training with cross-entropy* (added), item XE17. Let the loss `L` be real-analytic in the
entries of its matrix, satisfy the first-order inequality with a continuous `G`, and satisfy
two bounds at the matrices with `‖X‖ ≤ ρ/2`: `‖G X‖ ≤ γ` and the quadratic upper model with
the constant `s`. The ball `‖U‖² + ‖V‖² ≤ ρ` is compact (`isCompact_setOf_frobeniusSq_add_le`),
so a run that stays in it has a cluster point. On the factor space the run is gradient descent
on the analytic `flatLoss` (`factors_iterate_gradientStep`), every step of a run that stays in
the ball lowers the loss by `(η/2) ‖∇‖²` when `η (λ + γ + 2 s ρ) ≤ 1`
(`decayedLoss_decayRun_succ_add_le_of_forall_le`), and Łojasiewicz's theorem
(`Landscape.SaddleAvoidance.exists_isLojasiewiczAt_of_analyticAt`) gives the gradient
inequality at the cluster point. So the run converges to a stationary point
(`exists_tendsto_decayRun_of_forall_le`, by
`Landscape.SaddleAvoidance.tendsto_of_isLojasiewiczAt_of_le_sub`).

The weight decay `λ` is any real number here, zero and negative ones included: `λ > 0` enters
WD8 only through WD6, which keeps the run in the ball. The examples are the squared loss.
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

variable {L : Matrix m n ℂ → ℝ} {G : Matrix m n ℂ → Matrix m n ℂ} {η lam γ s ρ : ℝ}

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

variable [DecidableEq m] [DecidableEq n] {U₀ : Matrix m k ℂ} {V₀ : Matrix n k ℂ}

/-- **Every step of a run that stays in a ball lowers the loss** (`review/flat_spectrum.md`,
item WD8), by at least `(η/2) ‖M_t W_t‖²`: for every weight decay `λ`, with `‖G X‖ ≤ γ` and
the quadratic upper model with the constant `s` for `‖X‖, ‖Y‖ ≤ ρ/2`, and
`η (λ + γ + 2 s ρ) ≤ 1`. -/
theorem decayedLoss_decayRun_succ_add_le_of_forall_le (hη : 0 ≤ η) (hγ0 : 0 ≤ γ) (hs0 : 0 ≤ s)
    (hγ : ∀ X, frobeniusSq X ≤ (ρ / 2) ^ 2 → frobeniusSq (G X) ≤ γ ^ 2)
    (hs : ∀ X Y, frobeniusSq X ≤ (ρ / 2) ^ 2 → frobeniusSq Y ≤ (ρ / 2) ^ 2 →
      L Y ≤ L X + pairing (G X) (Y - X) + s / 2 * frobeniusSq (Y - X))
    (hdesc : η * (lam + γ + 2 * s * ρ) ≤ 1)
    (hball : ∀ t, frobeniusSq (decayRun G η lam U₀ V₀ t).1 +
      frobeniusSq (decayRun G η lam U₀ V₀ t).2 ≤ ρ) (t : ℕ) :
    decayedLoss L lam (decayRun G η lam U₀ V₀ (t + 1)).1 (decayRun G η lam U₀ V₀ (t + 1)).2 +
      η / 2 * frobeniusSq (lift lam (G ((decayRun G η lam U₀ V₀ t).1 *
        (decayRun G η lam U₀ V₀ t).2ᴴ)) *
          stack (decayRun G η lam U₀ V₀ t).1 (decayRun G η lam U₀ V₀ t).2) ≤
      decayedLoss L lam (decayRun G η lam U₀ V₀ t).1 (decayRun G η lam U₀ V₀ t).2 := by
  have h1 := hball (t + 1)
  rw [decayRun, Function.iterate_succ_apply', ← decayRun] at h1 ⊢
  exact decayedLoss_decayStep_add_le hη hγ0 hs0
    (hγ _ (frobeniusSq_mul_conjTranspose_le (hball t))) hs (hball t) h1 hdesc

/-- **A run that stays in a ball converges** (`review/flat_spectrum.md`, items WD8 and XE17):
let `L` be real-analytic in the entries and satisfy the first-order inequality with a
continuous `G`, with `‖G X‖ ≤ γ` and `L Y ≤ L X + ⟨G X, Y − X⟩ + (s/2) ‖Y − X‖²` for
`‖X‖, ‖Y‖ ≤ ρ/2`. For every weight decay `λ` and a step `0 < η` with
`η (λ + γ + 2 s ρ) ≤ 1`, a run that stays in the ball `‖U‖² + ‖V‖² ≤ ρ` converges, and its
limit is stationary: `M [U; V] = 0`. -/
theorem exists_tendsto_decayRun_of_forall_le (hconv : ∀ X Y, L X + pairing (G X) (Y - X) ≤ L Y)
    (hG : Continuous G)
    (hL : ∀ y : EuclideanSpace ℂ (m × n),
      AnalyticAt ℝ (fun y : EuclideanSpace ℂ (m × n) => L (of fun i j => y (i, j))) y)
    (hη : 0 < η) (hγ0 : 0 ≤ γ) (hs0 : 0 ≤ s)
    (hγ : ∀ X, frobeniusSq X ≤ (ρ / 2) ^ 2 → frobeniusSq (G X) ≤ γ ^ 2)
    (hs : ∀ X Y, frobeniusSq X ≤ (ρ / 2) ^ 2 → frobeniusSq Y ≤ (ρ / 2) ^ 2 →
      L Y ≤ L X + pairing (G X) (Y - X) + s / 2 * frobeniusSq (Y - X))
    (hdesc : η * (lam + γ + 2 * s * ρ) ≤ 1)
    (hball : ∀ t, frobeniusSq (decayRun G η lam U₀ V₀ t).1 +
      frobeniusSq (decayRun G η lam U₀ V₀ t).2 ≤ ρ) :
    ∃ (U : Matrix m k ℂ) (V : Matrix n k ℂ),
      Tendsto (decayRun G η lam U₀ V₀) atTop (𝓝 (U, V)) ∧
        lift lam (G (U * Vᴴ)) * stack U V = 0 := by
  have hmem : ∀ᶠ t in atTop, decayRun G η lam U₀ V₀ t ∈
      {p : Matrix m k ℂ × Matrix n k ℂ | frobeniusSq p.1 + frobeniusSq p.2 ≤ ρ} :=
    Eventually.of_forall hball
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
      have h := decayedLoss_decayRun_succ_add_le_of_forall_le (L := L) hη.le hγ0 hs0 hγ hs
        hdesc hball t
      rw [norm_gradient_flatLoss_sq hconv hG hL, flatLoss, flatLoss, hiter, hiter]
      linarith)
    (hflat ▸ hp.tendsto_comp (continuous_flatten.tendsto p))
  have hlim : Tendsto (decayRun G η lam U₀ V₀) atTop (𝓝 p) := by
    have h := (continuous_factors.tendsto (flatten p)).comp hlimE
    rw [factors_flatten] at h
    exact h.congr hiter
  exact ⟨p.1, p.2, hlim, lift_mul_stack_eq_zero_of_tendsto_decayRun hG hη hlim⟩

/-! The witnesses, for the squared loss of the `1 × 1` kernel `1` with `γ = 2`, `s = 1`,
`ρ = 2` and `η = 1/10`: the run of weight decay `1` from `U₀ = [1 0]`, `V₀ = [0 1]`, which
stays in the ball by WD6; and the run without weight decay from the zero start, which stays
at it. -/

example := decayedLoss_decayRun_succ_add_le_of_forall_le
  (L := fitLoss (Set.univ : Set (Fin 1 × Fin 1)) 1) (η := 1 / 10) (lam := 1) (by norm_num)
  (by norm_num) zero_le_one stepWitness_bound (fun X Y _ _ => fitLoss_le_add Set.univ 1 X Y)
  (by norm_num)
  (fun t => frobeniusSq_decayRun_le (by norm_num) one_pos stepWitness_bound
    (fun X _ => stepWitness_low X) (by norm_num) (by norm_num) stepWitness_ball t) 3

example := exists_tendsto_decayRun_of_forall_le
  (fitLoss_first_order (Set.univ : Set (Fin 1 × Fin 1)) 1) (continuous_mask_sub Set.univ 1)
  (analyticAt_fitLoss Set.univ 1) (η := 1 / 10) (lam := 1) (by norm_num) (by norm_num)
  zero_le_one stepWitness_bound (fun X Y _ _ => fitLoss_le_add Set.univ 1 X Y) (by norm_num)
  (fun t => frobeniusSq_decayRun_le (by norm_num) one_pos stepWitness_bound
    (fun X _ => stepWitness_low X) (by norm_num) (by norm_num) stepWitness_ball t)

example := exists_tendsto_decayRun_of_forall_le
  (fitLoss_first_order (Set.univ : Set (Fin 1 × Fin 1)) 1) (continuous_mask_sub Set.univ 1)
  (analyticAt_fitLoss Set.univ 1) (η := 1 / 10) (lam := 0) (by norm_num) (by norm_num)
  zero_le_one stepWitness_bound (fun X Y _ _ => fitLoss_le_add Set.univ 1 X Y) (by norm_num)
  (U₀ := (0 : Matrix (Fin 1) (Fin 1) ℂ)) (V₀ := 0) fun t => by
    rw [decayRun, Function.iterate_fixed (by simp [decayStep])]
    simp [frobeniusSq]

end Transformer.Grokking.FlatSpectrum.Training
