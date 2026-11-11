module

public import Transformer.Grokking.FlatSpectrum.Unimodular
public import Transformer.Grokking.FlatSpectrum.Training.CrossEntropyBounds

/-!
# The margin of a product and its cross-entropy

Source: `review/flat_spectrum.md`, *Training with cross-entropy* (added), item XE4.
The gap of the sample `s` against a class `c` is `z_{s,y_s}(X) − z_{s,c}(X)` (`gap`), linear
in the product, and the margin of `X` is the least gap over the samples and their wrong
classes (`margin`). The margin is attained (`exists_gap_eq_margin`), positively homogeneous
(`margin_smul`) and at most `2 a ‖X‖_S1` for probes of norm at most `a`
(`gap_le_mul_nuclearNorm`, from `‖X‖_S2 ≤ ‖X‖_S1`, `frobeniusSq_le_nuclearNorm_sq`). The
cross-entropy lies between two functions of the gaps:

* if every gap is at least `μ`, then `L X ≤ |S| log (1 + (|C| − 1) exp (−μ))`
  (`ceLoss_le_of_le_gap`), and so `L X ≤ |S| (|C| − 1) exp (−μ)` (`ceLoss_le_mul_exp`);
* every gap `g` has `log (1 + exp (−g)) ≤ L X` (`log_le_ceLoss`).

The witness is `signProbe` with the label `true`: its one gap is `2 Re X`.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Training

open Matrix
open Landscape.Neural (crossEntropy crossEntropy_nonneg)
open Geometry (oddsEnvelope crossEntropy_eq_log_odds crossEntropy_le_of_contrast_bounds)

variable {m n S C : Type*} [Fintype m] [Fintype n] [Fintype S] [Fintype C]

/-- The gap of the sample `s` against the class `c`: the logit of the label minus the logit of
`c`, item XE4. -/
noncomputable def gap (A : S → C → Matrix m n ℂ) (y : S → C) (X : Matrix m n ℂ) (s : S)
    (c : C) : ℝ :=
  pairing (A s (y s)) X - pairing (A s c) X

/-- The margin of `X`: the least gap over the samples and their wrong classes, item XE4. -/
noncomputable def margin (A : S → C → Matrix m n ℂ) (y : S → C) (X : Matrix m n ℂ) : ℝ :=
  ⨅ p : {p : S × C // p.2 ≠ y p.1}, gap A y X p.1.1 p.1.2

variable {A : S → C → Matrix m n ℂ} {y : S → C} {a μ : ℝ} {X : Matrix m n ℂ}

omit [Fintype S] [Fintype C] in
lemma gap_smul (t : ℝ) (X : Matrix m n ℂ) (s : S) (c : C) :
    gap A y (t • X) s c = t * gap A y X s c := by
  simp only [gap, pairing_smul_right]
  ring

lemma margin_le_gap (X : Matrix m n ℂ) {s : S} {c : C} (hc : c ≠ y s) :
    margin A y X ≤ gap A y X s c :=
  ciInf_le (Set.finite_range _).bddBelow (⟨(s, c), hc⟩ : {p : S × C // p.2 ≠ y p.1})

omit [Fintype S] [Fintype C] in
/-- Some sample has a wrong class when there are a sample and two classes. -/
lemma nonempty_wrong [Nonempty S] [Nontrivial C] (y : S → C) :
    Nonempty {p : S × C // p.2 ≠ y p.1} := by
  obtain ⟨s⟩ := ‹Nonempty S›
  obtain ⟨c, hc⟩ := exists_ne (y s)
  exact ⟨⟨(s, c), hc⟩⟩

omit [Fintype S] [Fintype C] in
lemma le_margin [Nonempty S] [Nontrivial C] (h : ∀ s c, c ≠ y s → μ ≤ gap A y X s c) :
    μ ≤ margin A y X :=
  have := nonempty_wrong y
  le_ciInf fun p => h _ _ p.2

variable (A y) in
/-- The margin is the gap of some sample against some wrong class. -/
lemma exists_gap_eq_margin [Nonempty S] [Nontrivial C] (X : Matrix m n ℂ) :
    ∃ s c, c ≠ y s ∧ gap A y X s c = margin A y X := by
  have := nonempty_wrong y
  obtain ⟨p, hp⟩ := exists_eq_ciInf_of_finite (f := fun p : {p : S × C // p.2 ≠ y p.1} =>
    gap A y X p.1.1 p.1.2)
  exact ⟨p.1.1, p.1.2, p.2, hp⟩

omit [Fintype S] [Fintype C] in
/-- The margin is positively homogeneous: `μ(t X) = t μ(X)` for `t ≥ 0`. -/
lemma margin_smul {t : ℝ} (ht : 0 ≤ t) (X : Matrix m n ℂ) :
    margin A y (t • X) = t * margin A y X := by
  simp only [margin, gap_smul]
  exact (Real.mul_iInf_of_nonneg ht _).symm

/-- `‖X‖_S2 ≤ ‖X‖_S1`: `Σ σ_i² ≤ (Σ σ_i)²`. -/
lemma frobeniusSq_le_nuclearNorm_sq [DecidableEq n] (X : Matrix m n ℂ) :
    frobeniusSq X ≤ nuclearNorm X ^ 2 := by
  rw [← sum_singularValues_sq, nuclearNorm]
  exact Finset.sum_sq_le_sq_sum_of_nonneg fun i _ => Real.sqrt_nonneg _

omit [Fintype S] [Fintype C] in
/-- A gap is at most `2 a ‖X‖_S1` for probes of norm at most `a`. -/
lemma gap_le_mul_nuclearNorm [DecidableEq n] (ha0 : 0 ≤ a)
    (ha : ∀ s c, frobeniusSq (A s c) ≤ a ^ 2) (X : Matrix m n ℂ) (s : S) (c : C) :
    gap A y X s c ≤ 2 * a * nuclearNorm X := by
  have hX : √(frobeniusSq X) ≤ nuclearNorm X :=
    (Real.sqrt_le_sqrt (frobeniusSq_le_nuclearNorm_sq X)).trans_eq
      (Real.sqrt_sq (nuclearNorm_nonneg X))
  have h1 := le_of_abs_le (abs_pairing_le_mul_sqrt ha0 (ha s (y s)) X)
  have h2 := neg_le_of_abs_le (abs_pairing_le_mul_sqrt ha0 (ha s c) X)
  have h3 := mul_le_mul_of_nonneg_left hX ha0
  unfold gap
  linarith

/-- One sample whose gaps are at least `μ` has cross-entropy at most
`log (1 + (|C| − 1) exp (−μ))`, and at most `(|C| − 1) exp (−μ)`. -/
lemma crossEntropy_le_of_le_gap {z : C → ℝ} {t : C} (h : ∀ c, c ≠ t → μ ≤ z t - z c) :
    crossEntropy z t ≤ Real.log (1 + (Fintype.card C - 1) * Real.exp (-μ)) ∧
      crossEntropy z t ≤ (Fintype.card C - 1) * Real.exp (-μ) := by
  classical
  have h1 := crossEntropy_le_of_contrast_bounds z (fun _ => -μ) t fun c hc => by
    linarith [h c hc]
  have e : oddsEnvelope (fun _ : C => -μ) t = (Fintype.card C - 1) * Real.exp (-μ) := by
    rw [oddsEnvelope, Finset.sum_const, Finset.card_erase_of_mem (Finset.mem_univ t),
      Finset.card_univ, nsmul_eq_mul, Nat.cast_pred (Fintype.card_pos_iff.mpr ⟨t⟩)]
  rw [e] at h1
  have h0 : 0 ≤ ((Fintype.card C : ℝ) - 1) * Real.exp (-μ) :=
    mul_nonneg (sub_nonneg.mpr (Nat.one_le_cast.mpr (Fintype.card_pos_iff.mpr ⟨t⟩)))
      (Real.exp_pos _).le
  exact ⟨h1, h1.trans (by linarith [Real.log_le_sub_one_of_pos (by linarith :
    (0 : ℝ) < 1 + (Fintype.card C - 1) * Real.exp (-μ))])⟩

/-- **A margin bounds the cross-entropy from above** (`review/flat_spectrum.md`, item XE4):
if every gap of `X` is at least `μ`, then `L X ≤ |S| log (1 + (|C| − 1) exp (−μ))`. -/
theorem ceLoss_le_of_le_gap (h : ∀ s c, c ≠ y s → μ ≤ gap A y X s c) :
    ceLoss A y X ≤ Fintype.card S * Real.log (1 + (Fintype.card C - 1) * Real.exp (-μ)) := by
  refine (Finset.sum_le_sum fun s _ => (crossEntropy_le_of_le_gap (h s)).1).trans_eq ?_
  rw [Finset.sum_const, Finset.card_univ, nsmul_eq_mul]

/-- The same bound without the logarithm: `L X ≤ |S| (|C| − 1) exp (−μ)`. -/
theorem ceLoss_le_mul_exp (h : ∀ s c, c ≠ y s → μ ≤ gap A y X s c) :
    ceLoss A y X ≤ Fintype.card S * (Fintype.card C - 1) * Real.exp (-μ) := by
  refine (Finset.sum_le_sum fun s _ => (crossEntropy_le_of_le_gap (h s)).2).trans_eq ?_
  rw [Finset.sum_const, Finset.card_univ, nsmul_eq_mul]
  ring

variable (A y) in
/-- **A gap bounds the cross-entropy from below** (`review/flat_spectrum.md`, item XE4):
`log (1 + exp (−g)) ≤ L X` for the gap `g` of every sample against every wrong class. -/
theorem log_le_ceLoss (X : Matrix m n ℂ) {s : S} {c : C} (hc : c ≠ y s) :
    Real.log (1 + Real.exp (-(gap A y X s c))) ≤ ceLoss A y X := by
  classical
  have h1 : Real.exp (-(gap A y X s c)) ≤
      oddsEnvelope (fun c' => pairing (A s c') X - pairing (A s (y s)) X) (y s) := by
    rw [gap, neg_sub]
    exact Finset.single_le_sum (f := fun c' => Real.exp (pairing (A s c') X -
      pairing (A s (y s)) X)) (fun _ _ => (Real.exp_pos _).le)
      (Finset.mem_erase.mpr ⟨hc, Finset.mem_univ c⟩)
  have h2 : Real.log (1 + Real.exp (-(gap A y X s c))) ≤
      crossEntropy (fun c' => pairing (A s c') X) (y s) := by
    rw [crossEntropy_eq_log_odds]
    exact Real.log_le_log (by positivity) (by linarith)
  exact h2.trans (Finset.single_le_sum (f := fun s => crossEntropy (fun c' =>
    pairing (A s c') X) (y s)) (fun _ _ => crossEntropy_nonneg _ _) (Finset.mem_univ s))

/-! The witness: one sample of a `1 × 1` product read with the code `±1` and labeled `true`.
At `X = 1` its gap is `2`. -/

lemma gap_signProbe_one (s : Fin 1) :
    gap signProbe (fun _ => true) (1 : Matrix (Fin 1) (Fin 1) ℂ) s false = 2 := by
  simp [gap, signProbe, pairing]
  norm_num

lemma le_gap_signProbe_one (s : Fin 1) (c : Bool) (hc : c ≠ true) :
    (2 : ℝ) ≤ gap signProbe (fun _ => true) (1 : Matrix (Fin 1) (Fin 1) ℂ) s c := by
  cases c with
  | false => exact (gap_signProbe_one s).ge
  | true => exact absurd rfl hc

example := le_margin (A := signProbe) (y := fun _ => true) le_gap_signProbe_one

example := margin_le_gap (A := signProbe) (y := fun _ => true) 1 (s := 0) (c := false) (by simp)

example := margin_smul (A := signProbe) (y := fun _ => true) zero_le_two 1

example := crossEntropy_le_of_le_gap (z := fun c : Bool => if c then (2 : ℝ) else 0) (t := true)
  (μ := 2) fun c hc => by cases c <;> simp_all

example := gap_le_mul_nuclearNorm (y := fun _ => true) zero_le_one frobeniusSq_signProbe_le 1
  0 false

example := ceLoss_le_of_le_gap (A := signProbe) (y := fun _ => true) le_gap_signProbe_one

example := ceLoss_le_mul_exp (A := signProbe) (y := fun _ => true) le_gap_signProbe_one

example := log_le_ceLoss signProbe (fun _ => true) 1 (s := 0) (c := false) (by simp)

end Transformer.Grokking.FlatSpectrum.Training
