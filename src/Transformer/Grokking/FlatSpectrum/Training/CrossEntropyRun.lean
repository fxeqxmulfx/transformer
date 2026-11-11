module

public import Mathlib.Analysis.Complex.ExponentialBounds
public import Transformer.Grokking.FlatSpectrum.Training.AlmostEvery
public import Transformer.Grokking.FlatSpectrum.Training.CrossEntropyBounds

/-!
# The run on the cross-entropy converges

Source: `review/flat_spectrum.md`, *Training with cross-entropy* (added), item XE3.
The cross-entropy `ceLoss A y` of `|S|` samples and `|C|` classes, with probes of norm at most
`a`, meets the hypotheses of items WD6 and WD8 at every matrix, with `γ = 2 a |S|`,
`b = |S| log |C|` and `s = |S| a²` (`Training.CrossEntropyBounds`). So for a weight decay
`λ > 0`, a radius `ρ` with `4 |S| log |C| ≤ λ ρ` and a step `η > 0` with

    2 η (λ² + (2 a |S|)²) ≤ λ    and    η (λ + 2 a |S| + 2 |S| a² ρ) ≤ 1,

the run of gradient descent with weight decay from the ball `‖U₀‖² + ‖V₀‖² ≤ ρ` converges, for
every width and rank, to a stationary point (`exists_tendsto_decayRun_ceLoss`), and from a
start of full row rank to a decayed minimum (`exists_tendsto_decayRun_ceLoss_isDecayedMin`).
Every start lies in such a ball. Convergence is a conclusion, by Łojasiewicz's theorem.

Items WD9 and WD10 apply as they stand: at a width `k ≥ min(m, n)` small steps from almost
every start of a sublevel set converge to a decayed minimum
(`exists_pos_ae_tendsto_decayRun_ceLoss_isDecayedMin`), and decayed minima of that width exist
for every `λ > 0` (`exists_isDecayedMin_ceLoss`). The witness is `signProbe` with `λ = 1`,
`ρ = 3` and `η = 1/10`.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Training

open Matrix Filter Topology MeasureTheory

variable {m n k S C : Type*} [Fintype m] [Fintype n] [Fintype k] [Fintype S] [Fintype C]
  [DecidableEq m] [DecidableEq n] {A : S → C → Matrix m n ℂ} {y : S → C} {η lam a ρ : ℝ}

/-- **The run on the cross-entropy converges** (`review/flat_spectrum.md`, item XE3): for
probes of norm at most `a`, a weight decay `λ > 0`, a start of any width and rank in the ball
`‖U₀‖² + ‖V₀‖² ≤ ρ` with `4 |S| log |C| ≤ λ ρ`, and a step `0 < η` with
`2 η (λ² + (2 a |S|)²) ≤ λ` and `η (λ + 2 a |S| + 2 |S| a² ρ) ≤ 1`, the run of gradient descent
with weight decay converges, and its limit is stationary. -/
theorem exists_tendsto_decayRun_ceLoss (hη : 0 < η) (hlam : 0 < lam) (ha0 : 0 ≤ a)
    (ha : ∀ s c, frobeniusSq (A s c) ≤ a ^ 2)
    (hρ : 4 * (Fintype.card S * Real.log (Fintype.card C)) ≤ lam * ρ)
    (hstep : 2 * η * (lam ^ 2 + (2 * a * Fintype.card S) ^ 2) ≤ lam)
    (hdesc : η * (lam + 2 * a * Fintype.card S + 2 * (Fintype.card S * a ^ 2) * ρ) ≤ 1)
    {U₀ : Matrix m k ℂ} {V₀ : Matrix n k ℂ} (hw : frobeniusSq U₀ + frobeniusSq V₀ ≤ ρ) :
    ∃ (U : Matrix m k ℂ) (V : Matrix n k ℂ),
      Tendsto (decayRun (ceGrad A y) η lam U₀ V₀) atTop (𝓝 (U, V)) ∧
        lift lam (ceGrad A y (U * Vᴴ)) * stack U V = 0 :=
  exists_tendsto_decayRun (ceLoss_first_order A y) (continuous_ceGrad A y)
    (analyticAt_ceLoss A y) hη hlam (by positivity) (by positivity)
    (fun X _ => frobeniusSq_ceGrad_le ha0 ha X) (fun X _ => neg_le_pairing_ceGrad A y X)
    (fun X Y _ _ => ceLoss_le_add ha0 ha X Y) hρ hstep hdesc hw

/-- **The run on the cross-entropy converges, to a decayed minimum**
(`review/flat_spectrum.md`, item XE3): under the bounds of `exists_tendsto_decayRun_ceLoss`,
the run from a start of full row rank converges, and its limit has the least weight-decayed
cross-entropy among the factorizations of every width. No convergence is assumed. -/
theorem exists_tendsto_decayRun_ceLoss_isDecayedMin (hη : 0 < η) (hlam : 0 < lam)
    (ha0 : 0 ≤ a) (ha : ∀ s c, frobeniusSq (A s c) ≤ a ^ 2)
    (hρ : 4 * (Fintype.card S * Real.log (Fintype.card C)) ≤ lam * ρ)
    (hstep : 2 * η * (lam ^ 2 + (2 * a * Fintype.card S) ^ 2) ≤ lam)
    (hdesc : η * (lam + 2 * a * Fintype.card S + 2 * (Fintype.card S * a ^ 2) * ρ) ≤ 1)
    {U₀ : Matrix m k ℂ} {V₀ : Matrix n k ℂ} (hw : frobeniusSq U₀ + frobeniusSq V₀ ≤ ρ)
    (hfull : Function.Surjective (stack U₀ V₀).mulVec) :
    ∃ (U : Matrix m k ℂ) (V : Matrix n k ℂ),
      Tendsto (decayRun (ceGrad A y) η lam U₀ V₀) atTop (𝓝 (U, V)) ∧
        IsDecayedMin (ceLoss A y) lam U V :=
  exists_tendsto_decayRun_isDecayedMin (ceLoss_first_order A y) (continuous_ceGrad A y)
    (analyticAt_ceLoss A y) hη hlam (by positivity) (by positivity)
    (fun X _ => frobeniusSq_ceGrad_le ha0 ha X) (fun X _ => neg_le_pairing_ceGrad A y X)
    (fun X Y _ _ => ceLoss_le_add ha0 ha X Y) hρ hstep hdesc hw hfull

variable (A y) in
/-- **At width `min(m, n)` almost every run on the cross-entropy ends at a decayed minimum**
(`review/flat_spectrum.md`, items WD10 and XE3): for a weight decay `λ > 0`, a level `c` and a
width at least the number of rows of `U` or of `V`, some `η₀ > 0` has: for every `0 < η < η₀`
and almost every start of the factor space with `F ≤ c`, the run converges and its limit has
the least weight-decayed cross-entropy among the factorizations of every width. -/
theorem exists_pos_ae_tendsto_decayRun_ceLoss_isDecayedMin (hlam : 0 < lam) (c : ℝ)
    (hcard : Fintype.card m ≤ Fintype.card k ∨ Fintype.card n ≤ Fintype.card k) :
    ∃ η₀ > 0, ∀ η, 0 < η → η < η₀ → ∀ᵐ x : EuclideanSpace ℂ ((m ⊕ n) × k),
      flatLoss (ceLoss A y) lam x ≤ c → ∃ (U : Matrix m k ℂ) (V : Matrix n k ℂ),
        Tendsto (decayRun (ceGrad A y) η lam (factors x).1 (factors x).2) atTop (𝓝 (U, V)) ∧
          IsDecayedMin (ceLoss A y) lam U V :=
  exists_pos_ae_tendsto_decayRun_isDecayedMin (ceLoss_first_order A y) (continuous_ceGrad A y)
    (analyticAt_ceLoss A y) hlam (ceLoss_nonneg A y) c hcard

variable (A y) in
/-- **The cross-entropy has decayed minima** (`review/flat_spectrum.md`, items WD10 and XE3):
for every weight decay `λ > 0` and every width at least the number of rows of `U` or of `V`,
some factorization has the least weight-decayed cross-entropy among those of every width. -/
theorem exists_isDecayedMin_ceLoss (hlam : 0 < lam)
    (hcard : Fintype.card m ≤ Fintype.card k ∨ Fintype.card n ≤ Fintype.card k) :
    ∃ (U : Matrix m k ℂ) (V : Matrix n k ℂ), IsDecayedMin (ceLoss A y) lam U V :=
  exists_isDecayedMin_of_card_le (ceLoss_first_order A y) (continuous_ceGrad A y)
    (analyticAt_ceLoss A y) hlam (ceLoss_nonneg A y) hcard

variable (A y) in
/-- Trained fits of the cross-entropy exist along every sequence of positive weight decays, at
a width at least the number of rows of `U` or of `V`. -/
lemma exists_seq_isDecayedMin_ceLoss {lams : ℕ → ℝ} (hpos : ∀ j, 0 < lams j)
    (hcard : Fintype.card m ≤ Fintype.card k ∨ Fintype.card n ≤ Fintype.card k) :
    ∃ (Us : ℕ → Matrix m k ℂ) (Vs : ℕ → Matrix n k ℂ),
      ∀ j, IsDecayedMin (ceLoss A y) (lams j) (Us j) (Vs j) := by
  choose Us Vs h using fun j => exists_isDecayedMin_ceLoss (k := k) A y (hpos j) hcard
  exact ⟨Us, Vs, h⟩

/-! The witness: one sample of a `1 × 1` product read with the code `±1`: `a = 1`, `γ = 2`,
`b = log 2 < 3/4`, `s = 1`. With `λ = 1`, `ρ = 3` and `η = 1/10`: `4 b ≤ λ ρ`,
`2 η (λ² + γ²) = 1` and `η (λ + γ + 2 s ρ) = 9/10`. -/

lemma signWitness_radius :
    4 * ((Fintype.card (Fin 1) : ℝ) * Real.log (Fintype.card Bool)) ≤ 1 * 3 := by
  simp only [Fintype.card_fin, Fintype.card_bool, Nat.cast_one, Nat.cast_ofNat, one_mul]
  linarith [Real.log_two_lt_d9]

/-- The width-one start `U₀ = V₀ = 1` has no full row rank; the run converges. -/
example := exists_tendsto_decayRun_ceLoss (y := fun _ => true) (η := 1 / 10) (by norm_num)
  one_pos zero_le_one frobeniusSq_signProbe_le signWitness_radius (by norm_num) (by norm_num)
  (U₀ := (1 : Matrix (Fin 1) (Fin 1) ℂ)) (V₀ := 1) (by norm_num [frobeniusSq])

/-- From `U₀ = [1 0]`, `V₀ = [0 1]` the run converges to a decayed minimum. -/
example : ∃ U V : Matrix (Fin 1) (Fin 1 ⊕ Fin 1) ℂ,
    Tendsto (decayRun (ceGrad signProbe fun _ => true) (1 / 10) 1 (fromCols 1 0) (fromCols 0 1))
      atTop (𝓝 (U, V)) ∧ IsDecayedMin (ceLoss signProbe fun _ => true) 1 U V :=
  exists_tendsto_decayRun_ceLoss_isDecayedMin (by norm_num) one_pos zero_le_one
    frobeniusSq_signProbe_le signWitness_radius (by norm_num) (by norm_num)
    (stepWitness_ball.trans (by norm_num)) fun v => ⟨v, by rw [stack_fromCols_one, one_mulVec]⟩

example := exists_pos_ae_tendsto_decayRun_ceLoss_isDecayedMin (k := Fin 1) signProbe
  (fun _ => true) one_pos 1 (Or.inl le_rfl)

example := exists_isDecayedMin_ceLoss (k := Fin 1) signProbe (fun _ => true) one_pos
  (Or.inl le_rfl)

example := exists_seq_isDecayedMin_ceLoss (k := Fin 1) signProbe (fun _ => true)
  (lams := fun j => 1 / ((j : ℝ) + 1)) (fun j => by positivity) (Or.inl le_rfl)

end Transformer.Grokking.FlatSpectrum.Training
