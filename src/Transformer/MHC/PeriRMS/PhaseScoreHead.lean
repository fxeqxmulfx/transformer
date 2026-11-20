module

public import Transformer.MHC.PeriRMS.PhaseScore

/-!
# The phase score of the half-split heads

`PhaseScore` bounds the score of two complex vectors by `(5/4) ln N`; a head of the model is a
real vector `x : EucSpace d` in the half-split layout of `Transformer.GPTMini.RoPE`, whose complex
coordinate `c` is `x_c + i x_{c + d/2}` (`headPairs`). This module reads one as the other.

* `pairEnergy_headPairs_le`: the energy of the complex view is at most `‖x‖²`, so at most `d` after
  Q/K RMS, whatever rotations follow (`normalized_norm`).
* `headPhaseScore`: the score of two heads, the `phaseScore` of their pair layouts
  (`pairLayout`, row `0` the first halves, row `1` the second), whose rotary invariances
  `Transformer.RoPE.ReRoPE` proves; `headPhaseScore_eq` reads it as `g (Re Z + β (Im Z)² / d)`.
* `phaseScore_abs_le`, `headPhaseScore_abs_le`, `headPhaseScore_rotatedQK_abs_le`: the bound
  `(5/4) ln N` for the pair layout, for the half-split heads, and for heads after Q/K RMS and
  arbitrary rotary phases, with the gain `ln N / d`.
* `common_phase_hermitian`, `conjugate_core`, `nonlinear_witness`, `differential_remainder`: a
  common unit phase on `q` and `k` leaves `Z` unchanged, conjugating `Z` leaves the score
  unchanged, the score is not odd in the key for `β ≠ 0`, and its first variation in the complex
  input is exact. None of these is a Jacobian of the full model.

Source: DiffGPT (`~/python_projects/gpt`, `fe1d528`),
`research/main_model/proofs/ComplexPhaseScore.lean` and `research/main_model/README.txt`,
*Architecture*. Exact real arithmetic.
-/

@[expose] public section

namespace Transformer.MHC.PeriRMS

open Matrix
open scoped BigOperators ComplexConjugate
open Transformer Transformer.RoPE.ReRoPE Transformer.SequenceModel

variable {n : ℕ}

/-- A half-split head vector as complex coordinates: `x_c + i x_{c + d/2}`. -/
def headPairs (d : ℕ) (x : EucSpace d) (c : Fin (d / 2)) : ℂ :=
  ⟨x (GPTMini.ropeFst d c), x (GPTMini.ropeSnd d c)⟩

/-- The complex view drops at most an unpaired last coordinate: its energy is at most `‖x‖²`. -/
theorem pairEnergy_headPairs_le (d : ℕ) (x : EucSpace d) :
    pairEnergy (headPairs d x) ≤ ‖x‖ ^ 2 := by
  have h := GPTMini.sum_split d (fun i => x i ^ 2)
  have hnn : 0 ≤ ∑ j : Fin (d - 2 * (d / 2)), x ((GPTMini.ropeSplit d).symm (Sum.inr j)) ^ 2 :=
    Finset.sum_nonneg fun _ _ => sq_nonneg _
  have hE : pairEnergy (headPairs d x) = ∑ k : Fin (d / 2), x (GPTMini.ropeFst d k) ^ 2 +
      ∑ k : Fin (d / 2), x (GPTMini.ropeSnd d k) ^ 2 := by
    simp only [pairEnergy, headPairs, Complex.sq_norm, Complex.normSq_mk]
    rw [← Finset.sum_add_distrib]
    simp only [sq]
  rw [EuclideanSpace.norm_sq_eq]
  simp only [Real.norm_eq_abs, sq_abs]
  linarith

/-- The pair layout `Fin 2 × Fin (d / 2) → ℝ` of a half-split head vector, as `phaseScore` reads
it: row `0` the first halves, row `1` the second halves. -/
def pairLayout (d : ℕ) (x : EucSpace d) : Fin 2 × Fin (d / 2) → ℝ :=
  fun cl => splitPairs d x (cl.2, cl.1)

/-- The complex coordinates `toComplex` of the pair layout are `headPairs`. -/
theorem toComplex_pairLayout (d : ℕ) (x : EucSpace d) :
    toComplex (pairLayout d x) = headPairs d x := by
  funext c
  simp [toComplex, pairLayout, headPairs, splitPairs]

/-- The energy of the complex view of a pair layout is its squared length `x ⬝ᵥ x`. -/
theorem pairEnergy_toComplex {p : ℕ} (x : Fin 2 × Fin p → ℝ) :
    pairEnergy (toComplex x) = x ⬝ᵥ x := by
  simp only [pairEnergy, toComplex, Complex.sq_norm, Complex.normSq_mk, dotProduct,
    Fintype.sum_prod_type, Fin.sum_univ_two, Finset.sum_add_distrib]

/-- The squared length of the pair layout of a half-split head vector is at most `‖x‖²`: it drops
an unpaired last coordinate and no other. -/
theorem pairLayout_dotProduct_le (d : ℕ) (x : EucSpace d) :
    pairLayout d x ⬝ᵥ pairLayout d x ≤ ‖x‖ ^ 2 := by
  rw [← pairEnergy_toComplex, toComplex_pairLayout]
  exact pairEnergy_headPairs_le d x

/-- **The phase score of a pair layout is at most `(5/4) ln N`**: the gain `ln N / D` of `N ≥ 1`
visible keys, real width `D`, vectors of squared length at most `D`, and any `|β| ≤ 1`. -/
theorem phaseScore_abs_le {p : ℕ} (N β D : ℝ) (hN : 1 ≤ N) (hD : 0 < D) (hβ : |β| ≤ 1)
    (x y : Fin 2 × Fin p → ℝ) (hx : x ⬝ᵥ x ≤ D) (hy : y ⬝ᵥ y ≤ D) :
    |phaseScore (Real.log N / D) β D x y| ≤ 5 / 4 * Real.log N := by
  have h := causal_score_bound N β D hβ (toComplex x) (toComplex y) hN hD
    (by rwa [pairEnergy_toComplex]) (by rwa [pairEnergy_toComplex])
  rw [phaseScore_eq_re_im]
  have hZ : hermitianProduct (toComplex x) (toComplex y) =
      ∑ l, conj (toComplex x l) * toComplex y l := rfl
  rw [hZ] at h
  convert h using 3
  ring

/-- Two unit vectors of a one-pair head, `D = 2`, `β = 1/2`, two visible keys. -/
example : |phaseScore (Real.log 2 / 2) (1 / 2) 2
    (fun cl : Fin 2 × Fin 1 => if cl.1 = 0 then 1 else 0)
    (fun cl : Fin 2 × Fin 1 => if cl.1 = 0 then 0 else 1)| ≤ 5 / 4 * Real.log 2 :=
  phaseScore_abs_le 2 (1 / 2) 2 (by norm_num) (by norm_num) (by norm_num) _ _
    (by simp [dotProduct, Fintype.sum_prod_type])
    (by simp [dotProduct, Fintype.sum_prod_type])

/-- The score of two half-split heads of real width `d`, `g (Re Z + β (Im Z)² / d)` with
`Z = ∑_c conj(x_c) y_c` over the `d/2` complex coordinates. -/
noncomputable def headPhaseScore (d : ℕ) (g β : ℝ) (x y : EucSpace d) : ℝ :=
  phaseScore g β d (pairLayout d x) (pairLayout d y)

/-- The score of the heads is `g (Re Z + β (Im Z)² / d)` for their Hermitian product. -/
theorem headPhaseScore_eq (d : ℕ) (g β : ℝ) (x y : EucSpace d) :
    headPhaseScore d g β x y =
      g * ((hermitianProduct (headPairs d x) (headPairs d y)).re +
        β * (hermitianProduct (headPairs d x) (headPairs d y)).im ^ 2 / d) := by
  rw [headPhaseScore, phaseScore_eq_re_im, toComplex_pairLayout, toComplex_pairLayout]
  rfl

/-- **The phase score of half-split heads is at most `(5/4) ln N`** when both vectors have norm at
most `√d`, the norm Q/K RMS gives (`normalized_norm`), for any `|β| ≤ 1`. -/
theorem headPhaseScore_abs_le (d : ℕ) (hd : 0 < d) (N β : ℝ) (hN : 1 ≤ N) (hβ : |β| ≤ 1)
    (x y : EucSpace d) (hx : ‖x‖ ≤ Real.sqrt d) (hy : ‖y‖ ≤ Real.sqrt d) :
    |headPhaseScore d (Real.log N / d) β x y| ≤ 5 / 4 * Real.log N := by
  have hd' : (0 : ℝ) < d := by exact_mod_cast hd
  have hsq (v : EucSpace d) (hv : ‖v‖ ≤ Real.sqrt d) : ‖v‖ ^ 2 ≤ d :=
    (pow_le_pow_left₀ (norm_nonneg v) hv 2).trans_eq (Real.sq_sqrt hd'.le)
  have h := causal_score_bound N β d hβ (headPairs d x) (headPairs d y) hN hd'
    ((pairEnergy_headPairs_le d x).trans (hsq x hx))
    ((pairEnergy_headPairs_le d y).trans (hsq y hy))
  rw [headPhaseScore_eq]
  convert h using 3
  ring

/-- The heads `0` and `0` of width `2`: norm `0 ≤ √2`. -/
example : |headPhaseScore 2 (Real.log 2 / 2) (1 / 2) 0 0| ≤ 5 / 4 * Real.log 2 :=
  headPhaseScore_abs_le 2 (by norm_num) 2 (1 / 2) (by norm_num) (by norm_num) 0 0
    (by simp) (by simp)

/-- **After Q/K RMS and arbitrary rotary phases** the score of any two projected vectors is at most
`(5/4) ln N`: the angles `a`, `b` (LeRoPE at the positions of `q` and `k`, learned frequencies
included) are arbitrary, and so is routing before the projections. -/
theorem headPhaseScore_rotatedQK_abs_le {d : ℕ} (eps : ℝ) (heps : 0 < eps) (hd : 0 < d)
    (N β : ℝ) (hN : 1 ≤ N) (hβ : |β| ≤ 1) (a b : Fin (d / 2) → ℝ) (q k : EucSpace d) :
    |headPhaseScore d (Real.log N / d) β (rotatedQK eps a q) (rotatedQK eps b k)| ≤
      5 / 4 * Real.log N :=
  headPhaseScore_abs_le d hd N β hN hβ _ _ (normalized_norm eps heps hd a q)
    (normalized_norm eps heps hd b k)

/-- The model's head width `32`, `ε = 2⁻²³`, `64` visible keys, the implemented
`β = a / √(1 + a²)` for `a = 3`, and arbitrary projected `q`, `k`. -/
example (q k : EucSpace 32) :
    |headPhaseScore 32 (Real.log 64 / 32) (phaseCoefficient 3)
        (rotatedQK (1 / 8388608) (fun _ => 2) q) (rotatedQK (1 / 8388608) (fun _ => 3) k)| ≤
      5 / 4 * Real.log 64 :=
  headPhaseScore_rotatedQK_abs_le (1 / 8388608) (by norm_num) (by norm_num) 64 _
    (by norm_num) (phaseCoefficient_abs_lt_one 3).le _ _ q k

/-- A common unit phase leaves the Hermitian product unchanged. -/
theorem common_phase_hermitian (u : ℂ) (q k : Fin n → ℂ) (hu : ‖u‖ = 1) :
    hermitianProduct (fun c => u * q c) (fun c => u * k c) = hermitianProduct q k := by
  unfold hermitianProduct
  apply Finset.sum_congr rfl
  intro c _
  simp only [map_mul]
  calc
    conj u * conj (q c) * (u * k c) = (conj u * u) * (conj (q c) * k c) := by ring
    _ = conj (q c) * k c := by rw [Complex.conj_mul', hu]; norm_num

/-- The phase `i` is a unit. -/
example : ‖Complex.I‖ = 1 := by norm_num

/-- Conjugating the product leaves the real score unchanged: `(Im z)²` is even. -/
theorem conjugate_core (β : ℝ) (z : ℂ) : phaseCore β (conj z) = phaseCore β z := by
  simp [phaseCore]

/-- The coefficient of a reachable `a ≠ 0` is not odd in the key; every bilinear score is odd in
either input, so this witness excludes a hidden dot-product identity. -/
theorem nonlinear_witness :
    phaseCore (phaseCoefficient 1) Complex.I + phaseCore (phaseCoefficient 1) (-Complex.I) > 0 := by
  norm_num [phaseCore, phaseCoefficient]

/-- Exact polynomial remainder: the first variation of the core in the complex input. This is the
core alone, not a derivative of the full model. -/
theorem differential_remainder (β t : ℝ) (z dz : ℂ) :
    phaseCore β (z + (t : ℂ) * dz) = phaseCore β z +
      t * (dz.re + 2 * β * z.im * dz.im) + t ^ 2 * β * dz.im ^ 2 := by
  simp only [phaseCore, Complex.add_re, Complex.add_im, Complex.mul_re, Complex.mul_im,
    Complex.ofReal_re, Complex.ofReal_im, zero_mul, sub_zero, add_zero]
  ring

end Transformer.MHC.PeriRMS
