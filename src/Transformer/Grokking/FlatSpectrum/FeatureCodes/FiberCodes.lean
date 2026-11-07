module

public import Transformer.Grokking.FlatSpectrum.FeatureCodes.Margins

/-!
# Read out answers while ignoring differences inside an answer fiber

Source: `review/flat_spectrum.md`, *Answer equivalence*, QE1–QE3.
For any finite latent set and answer map `f`, transform each answer-fiber
indicator by an isometric complex matrix `F`. Its Hermitian readout on `F x`
is exactly the sum of real coordinates of `x` in that fiber. Thus columns
belonging to the same answer have identical scores even though the columns
are distinct. Errors balanced inside each fiber are invisible, with no bound
on their individual coordinates. A classification error budget can be placed
on fiber contrasts directly. No neural representation or learning is assumed.
-/

@[expose] public section
noncomputable section

namespace Transformer.Grokking.FlatSpectrum.FeatureCodes

open Matrix Finset

variable {S A : Type*} [Fintype S] [DecidableEq S] [DecidableEq A]

/-- The indicator of the latent inputs having a specified answer, QE1. -/
def fiberIndicator (f : S → A) (a : A) : S → ℂ := fun s => if f s = a then 1 else 0

/-- Transform the answer-fiber indicator by the chosen feature basis, QE1. -/
def fiberCode (F : Matrix S S ℂ) (f : S → A) (a : A) : S → ℂ :=
  F *ᵥ fiberIndicator f a

/-- A real linear Hermitian readout; code norms need not equal one, QE1. -/
def fiberReadout (F : Matrix S S ℂ) (f : S → A) (a : A) : (S → ℂ) →ₗ[ℝ] ℝ where
  toFun z := (star (fiberCode F f a) ⬝ᵥ z).re
  map_add' x y := by simp [dotProduct_add]
  map_smul' t x := by simp [dotProduct_smul]

/-- **The readout sees only the answer fiber**, QE1. Isometry removes the
feature basis, including a Fourier basis, from this exact calculation. -/
theorem fiber_readout_mulVec {F : Matrix S S ℂ} (hF : Fᴴ * F = 1)
    (f : S → A) (a : A) (x : S → ℂ) :
    fiberReadout F f a (F *ᵥ x) = ∑ s ∈ univ.filter (fun s => f s = a), (x s).re := by
  change (star (F *ᵥ fiberIndicator f a) ⬝ᵥ (F *ᵥ x)).re = _
  rw [star_mulVec, ← dotProduct_mulVec, mulVec_mulVec, hF, one_mulVec]
  simp [dotProduct, fiberIndicator, sum_filter, Complex.re_sum, apply_ite]

/-- A latent feature column has score one for its answer and zero for all
other answers, including answers outside the image of `f`, QE1. -/
theorem fiber_readout_column {F : Matrix S S ℂ} (hF : Fᴴ * F = 1)
    (f : S → A) (s : S) (a : A) :
    fiberReadout F f a (F *ᵥ Pi.single s 1) = if f s = a then 1 else 0 := by
  rw [fiber_readout_mulVec hF]
  simp [Pi.single_apply, apply_ite]

/-- **Same-answer features have exactly the same complete score vector**, QE2. -/
theorem fiber_readouts_eq_of_same_answer {F : Matrix S S ℂ} (hF : Fᴴ * F = 1)
    (f : S → A) {s t : S} (heq : f s = f t) :
    ∀ a, fiberReadout F f a (F *ᵥ Pi.single s 1) =
      fiberReadout F f a (F *ᵥ Pi.single t 1) := by
  intro a
  rw [fiber_readout_column hF, fiber_readout_column hF, heq]

/-- The columns nevertheless remain distinct under an isometry, QE2. -/
theorem feature_columns_ne {F : Matrix S S ℂ} (hF : Fᴴ * F = 1)
    {s t : S} (hne : s ≠ t) : F *ᵥ Pi.single s 1 ≠ F *ᵥ Pi.single t 1 := by
  intro heq
  have h := congrArg (fun x => Fᴴ *ᵥ x) heq
  simp only [mulVec_mulVec, hF, one_mulVec] at h
  have hs := congrFun h s
  simp [hne] at hs

/-- **Errors balanced inside fibers have no effect on any score**, QE2.
Individual latent or feature coordinates can have arbitrarily large errors. -/
theorem fiber_readouts_ignore_balanced_error {F : Matrix S S ℂ} (hF : Fᴴ * F = 1)
    (f : S → A) (x e : S → ℂ)
    (he : ∀ a, ∑ s ∈ univ.filter (fun s => f s = a), (e s).re = 0) :
    ∀ a, fiberReadout F f a (F *ᵥ (x + e)) = fiberReadout F f a (F *ᵥ x) := by
  intro a
  rw [fiber_readout_mulVec hF, fiber_readout_mulVec hF]
  simp only [Pi.add_apply, Complex.add_re, sum_add_distrib, he, add_zero]

/-- **Classify with a bound on the combined fiber contrast**, QE3. The bound
is taken after summing the error in each fiber; no coordinatewise recovery is
required. The exact unperturbed margin is one in this normalization. -/
theorem fiber_classifies_of_contrast_error {F : Matrix S S ℂ} (hF : Fᴴ * F = 1)
    (f : S → A) (s : S) (e : S → ℂ) {ε : ℝ}
    (he : ∀ a, a ≠ f s →
      |(∑ t ∈ univ.filter (fun t => f t = f s), (e t).re) -
        ∑ t ∈ univ.filter (fun t => f t = a), (e t).re| ≤ ε)
    (hε : ε < 1) : ∀ a, a ≠ f s →
    fiberReadout F f a (F *ᵥ (Pi.single s 1 + e)) <
      fiberReadout F f (f s) (F *ᵥ (Pi.single s 1 + e)) := by
  intro a ha
  rw [mulVec_add, map_add, map_add, fiber_readout_column hF, fiber_readout_column hF,
    fiber_readout_mulVec hF, fiber_readout_mulVec hF]
  simp only [ite_true, ite_eq_right (Ne.symm ha), zero_add]
  have h := (abs_le.mp (he a ha)).1
  linarith

example := fiber_readout_mulVec (F := (1 : Matrix (Fin 2) (Fin 2) ℂ)) (by simp)
  id 0 ![1, 2]
example := fiber_readout_column (F := (1 : Matrix (Fin 2) (Fin 2) ℂ)) (by simp) id 0 1
example := fiber_readouts_eq_of_same_answer
  (F := (1 : Matrix (Bool × Bool) (Bool × Bool) ℂ)) (by simp) Prod.fst
  (s := (true, false)) (t := (true, true)) rfl
example := feature_columns_ne (F := (1 : Matrix (Fin 2) (Fin 2) ℂ)) (by simp)
  (s := 0) (t := 1) (by decide)

example := fiber_readouts_ignore_balanced_error
  (F := (1 : Matrix (Bool × Bool) (Bool × Bool) ℂ)) (by simp) Prod.fst
  (Pi.single (true, false) 1) (fun s => if s.2 then (100 : ℂ) else -100)
  (by intro a; cases a <;> norm_num [sum_filter, Fintype.sum_prod_type, Fintype.sum_bool])

example := fiber_classifies_of_contrast_error
  (F := (1 : Matrix (Bool × Bool) (Bool × Bool) ℂ)) (by simp) Prod.fst
  (true, false) (fun s => if s.2 then (100 : ℂ) else -100) (ε := 0)
  (by intro a ha; cases a <;> norm_num [sum_filter, Fintype.sum_prod_type, Fintype.sum_bool] at *)
  (by norm_num)

end Transformer.Grokking.FlatSpectrum.FeatureCodes
