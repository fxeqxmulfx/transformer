module

public import Transformer.MHC.PeriRMS.Scores
public import Transformer.RoPE.ReRoPE.Section3_PhaseScore

/-!
# The phase score is bounded by `5/4 · ln N`

The score of a complex head (`Transformer.RoPE.ReRoPE.phaseScore`) is, for the query `q` and the
key `k` of real width `d` read as `d/2` complex coordinates,

    `(ln N / d) (Re Z + β (Im Z)² / d)`,   `Z = ∑_c conj(q_c) k_c`,   `β = a / √(1 + a²)`,

with `N` the number of visible keys and one learned `a` per layer and head. Its real part `Re Z`
is the ordinary dot product, so after the Q/K RMS of `Scores` and any phases the score is at most
`ln N` in absolute value when `β = 0`; the squared imaginary part enlarges that to

    `|score| ≤ (5/4) ln N`   (`causal_score_bound`)

for every `|β| ≤ 1`, in particular for the implemented `β` (`phaseCoefficient_abs_lt_one`). The
constant is `max_{x² + y² ≤ 1} |x| + y² = 5/4` at `|x| = 1/2` (`core_disk_bound`); it is a bound on
the score, not on its Jacobian.

* `hermitian_energy_bound`: Cauchy–Schwarz holds for the whole complex product, not only for its
  real part, so `|Z/d| ≤ 1` once both energies are at most `d` (`normalized_hermitian_disk`).
`PhaseScoreHead` reads the half-split head vectors of the model as such complex vectors and states
the invariances of the score.

Source: DiffGPT (`~/python_projects/gpt`, `fe1d528`), `research/main_model/proofs/ComplexPhaseScore.lean`
(`MainModelComplexPhaseScore`), with its definitions renamed. Exact real arithmetic: no
floating-point or convergence claim.
-/

@[expose] public section

namespace Transformer.MHC.PeriRMS

open scoped BigOperators ComplexConjugate
open Transformer Transformer.RoPE.ReRoPE Transformer.SequenceModel

variable {n : ℕ}

/-- The Hermitian product `Z = ∑_c conj(q_c) k_c` of two complex vectors. -/
noncomputable def hermitianProduct (q k : Fin n → ℂ) : ℂ := ∑ c, conj (q c) * k c

/-- The squared length `∑_c |q_c|²` of a complex vector. -/
noncomputable def pairEnergy (q : Fin n → ℂ) : ℝ := ∑ c, ‖q c‖ ^ 2

/-- The coefficient `β = a / √(1 + a²)` of the squared imaginary part, one learned `a` per head. -/
noncomputable def phaseCoefficient (a : ℝ) : ℝ := a / Real.sqrt (1 + a ^ 2)

/-- The core of the score, `Re z + β (Im z)²`, of the normalized product `z = Z / d`. -/
def phaseCore (β : ℝ) (z : ℂ) : ℝ := z.re + β * z.im ^ 2

theorem hermitianProduct_re (q k : Fin n → ℂ) :
    (hermitianProduct q k).re = ∑ c, ((q c).re * (k c).re + (q c).im * (k c).im) := by
  simp [hermitianProduct, Complex.mul_re]

theorem hermitianProduct_im (q k : Fin n → ℂ) :
    (hermitianProduct q k).im = ∑ c, ((q c).re * (k c).im - (q c).im * (k c).re) := by
  simp [hermitianProduct, Complex.mul_im, sub_eq_add_neg]

/-- The implemented coefficient is strictly inside `(-1, 1)`: the bound below needs only `≤ 1`. -/
theorem phaseCoefficient_abs_lt_one (a : ℝ) : |phaseCoefficient a| < 1 := by
  have hs := Real.sq_sqrt (by positivity : 0 ≤ 1 + a ^ 2)
  have hp : 0 < Real.sqrt (1 + a ^ 2) := Real.sqrt_pos.2 (by positivity)
  have ha : |a| < Real.sqrt (1 + a ^ 2) := by nlinarith [sq_abs a]
  simpa [phaseCoefficient, abs_div, abs_of_pos hp] using (div_lt_one hp).2 ha

/-- `a = 0` is the ordinary dot product: `β = 0`, the core is `Re z`. -/
theorem phaseCore_phaseCoefficient_zero (z : ℂ) : phaseCore (phaseCoefficient 0) z = z.re := by
  norm_num [phaseCore, phaseCoefficient]

/-- Cauchy–Schwarz for the complete complex product, not only its real part. -/
theorem hermitian_energy_bound (q k : Fin n → ℂ) :
    ‖hermitianProduct q k‖ ^ 2 ≤ pairEnergy q * pairEnergy k := by
  have ht : ‖hermitianProduct q k‖ ≤ ∑ c, ‖q c‖ * ‖k c‖ := by
    simpa only [hermitianProduct, norm_mul, Complex.norm_conj] using
      norm_sum_le Finset.univ (fun c => conj (q c) * k c)
  have hn : 0 ≤ ∑ c, ‖q c‖ * ‖k c‖ := by positivity
  calc
    _ ≤ (∑ c, ‖q c‖ * ‖k c‖) ^ 2 := by nlinarith [norm_nonneg (hermitianProduct q k)]
    _ ≤ pairEnergy q * pairEnergy k := Finset.sum_mul_sq_le_sq_mul_sq _ _ _

/-- Energies at most the real head width `d` put `Z / d` in the closed unit disk. Q/K RMS and
unit rotations supply the hypotheses; `d` is the real width, not the number of complex
coordinates. -/
theorem normalized_hermitian_disk (q k : Fin n → ℂ) (d : ℝ) (hd : 0 < d)
    (hq : pairEnergy q ≤ d) (hk : pairEnergy k ≤ d) :
    Complex.normSq (hermitianProduct q k / (d : ℂ)) ≤ 1 := by
  have he := hermitian_energy_bound q k
  have hn : 0 ≤ pairEnergy k := by unfold pairEnergy; positivity
  have hb : pairEnergy q * pairEnergy k ≤ d * d := mul_le_mul hq hk hn hd.le
  rw [Complex.normSq_eq_norm_sq, norm_div, Complex.norm_real, Real.norm_eq_abs,
    abs_of_pos hd, div_pow]
  apply (div_le_one (sq_pos_of_pos hd)).2
  nlinarith

/-- Two complex coordinates of energy `1 ≤ 2`: the hypotheses of the disk bound are satisfiable. -/
example : (0 : ℝ) < 2 ∧ pairEnergy (fun _ : Fin 1 => (1 : ℂ)) ≤ 2 ∧
    pairEnergy (fun _ : Fin 1 => Complex.I) ≤ 2 := by
  norm_num [pairEnergy]

/-- Sharp bound over the closed family `|β| ≤ 1`: `|Re z| + (Im z)² ≤ 5/4` on the unit disk. The
implemented coefficient has `|β| < 1` strictly. -/
theorem core_disk_bound (β : ℝ) (z : ℂ) (hb : |β| ≤ 1) (hz : Complex.normSq z ≤ 1) :
    |phaseCore β z| ≤ 5 / 4 := by
  have hz' : z.re ^ 2 + z.im ^ 2 ≤ 1 := by simpa [Complex.normSq_apply, pow_two] using hz
  have hi : 0 ≤ z.im ^ 2 := sq_nonneg _
  have hm : |β| * z.im ^ 2 ≤ z.im ^ 2 := by nlinarith
  have ha : |z.re| + z.im ^ 2 ≤ 5 / 4 := by
    nlinarith [sq_nonneg (|z.re| - 1 / 2), sq_abs z.re]
  calc
    |phaseCore β z| ≤ |z.re| + |β * z.im ^ 2| := abs_add_le _ _
    _ = |z.re| + |β| * z.im ^ 2 := by rw [abs_mul, abs_of_nonneg hi]
    _ ≤ 5 / 4 := by linarith

/-- `β = 1/2` and `z = i` are in the unit disk, with `|β| ≤ 1`. -/
example : |(1 / 2 : ℝ)| ≤ 1 ∧ Complex.normSq Complex.I ≤ 1 := by norm_num

/-- The score core of two complex vectors of energy at most `d` is at most `5/4`, for every
coefficient `|β| ≤ 1`. -/
theorem normalized_score_bound (β : ℝ) (hβ : |β| ≤ 1) (q k : Fin n → ℂ) (d : ℝ) (hd : 0 < d)
    (hq : pairEnergy q ≤ d) (hk : pairEnergy k ≤ d) :
    |phaseCore β (hermitianProduct q k / (d : ℂ))| ≤ 5 / 4 :=
  core_disk_bound _ _ hβ (normalized_hermitian_disk q k d hd hq hk)

/-- The same hypotheses with `β = 1/2`. -/
example : |phaseCore (1 / 2)
    (hermitianProduct (fun _ : Fin 1 => (1 : ℂ)) (fun _ : Fin 1 => Complex.I) / ((2 : ℝ) : ℂ))|
      ≤ 5 / 4 :=
  normalized_score_bound (1 / 2) (by norm_num) _ _ 2 (by norm_num) (by norm_num [pairEnergy])
    (by norm_num [pairEnergy])

/-- The unit-disk form is algebraically the gain times the ordinary real dot product plus the
squared-imaginary correction; the real width `d` stays in the denominators. -/
theorem registered_score_formula (N β d : ℝ) (z : ℂ) :
    Real.log N * phaseCore β (z / (d : ℂ)) =
      Real.log N / d * (z.re + β * (z.im / d) * z.im) := by
  by_cases hd : d = 0
  · simp [hd, phaseCore]
  · simp [phaseCore, div_eq_mul_inv, Complex.mul_re, Complex.mul_im]
    field_simp [hd]

/-- **The causal score is at most `(5/4) ln N`** for `N ≥ 1` visible keys, two complex vectors of
energy at most the real width `d`, and every coefficient `|β| ≤ 1`. -/
theorem causal_score_bound (N β d : ℝ) (hβ : |β| ≤ 1) (q k : Fin n → ℂ) (hN : 1 ≤ N)
    (hd : 0 < d) (hq : pairEnergy q ≤ d) (hk : pairEnergy k ≤ d) :
    |Real.log N / d * ((hermitianProduct q k).re +
      β * ((hermitianProduct q k).im / d) * (hermitianProduct q k).im)| ≤
      (5 / 4) * Real.log N := by
  rw [← registered_score_formula, abs_mul, abs_of_nonneg (Real.log_nonneg hN)]
  calc
    _ ≤ Real.log N * (5 / 4) := mul_le_mul_of_nonneg_left
      (normalized_score_bound β hβ q k d hd hq hk) (Real.log_nonneg hN)
    _ = _ := mul_comm _ _

/-- The implemented coefficient `β = a / √(1 + a²)` of the model, for every `a`. -/
theorem causal_score_bound_phaseCoefficient (N a d : ℝ) (q k : Fin n → ℂ) (hN : 1 ≤ N)
    (hd : 0 < d) (hq : pairEnergy q ≤ d) (hk : pairEnergy k ≤ d) :
    |Real.log N / d * ((hermitianProduct q k).re +
      phaseCoefficient a * ((hermitianProduct q k).im / d) * (hermitianProduct q k).im)| ≤
      (5 / 4) * Real.log N :=
  causal_score_bound N _ d (phaseCoefficient_abs_lt_one a).le q k hN hd hq hk

/-- Two keys visible, real width `2`, the vectors `1` and `i`, and `a = 1`. -/
example : |Real.log 2 / 2 * ((hermitianProduct (fun _ : Fin 1 => (1 : ℂ))
      (fun _ : Fin 1 => Complex.I)).re +
    phaseCoefficient 1 * ((hermitianProduct (fun _ : Fin 1 => (1 : ℂ))
      (fun _ : Fin 1 => Complex.I)).im / 2) *
      (hermitianProduct (fun _ : Fin 1 => (1 : ℂ)) (fun _ : Fin 1 => Complex.I)).im)| ≤
      (5 / 4) * Real.log 2 :=
  causal_score_bound_phaseCoefficient 2 1 2 _ _ (by norm_num) (by norm_num)
    (by norm_num [pairEnergy]) (by norm_num [pairEnergy])

end Transformer.MHC.PeriRMS
