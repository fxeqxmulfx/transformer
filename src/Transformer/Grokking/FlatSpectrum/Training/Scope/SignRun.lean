module

public import Transformer.Grokking.FlatSpectrum.Training.Scope.PaddedRun
public import Transformer.Grokking.FlatSpectrum.Training.Scope.SignGap
public import Transformer.Grokking.FlatSpectrum.Training.UnitBall

/-!
# The run without weight decay on the sign sample

Source: `review/flat_spectrum.md`, *Training with cross-entropy* (added), item XE19; the item
"Not covered" of XE17.

The sample of XE3 and XE17: a `1 × 1` product `x = U Vᴴ` of two rows of any width, read with
the code `±1` and labelled with the class of the probe `1`.

* Its gradient is `−w(Re x)` (`ceGrad_signProbe`), with the weight `signWeight`.
* A step without weight decay is `U' = U + η w V`, `V' = V + η w U` (`decayStep_signProbe`),
  so `U + V` is multiplied by `1 + η w` and `U − V` by `1 − η w`
  (`decayRun_signProbe_add`, `decayRun_signProbe_sub`).
* `Re x = (‖U + V‖² − ‖U − V‖²) / 4` (`pairing_one_mul_conjTranspose`) and
  `‖U‖² + ‖V‖² = (‖U + V‖² + ‖U − V‖²) / 2` (`frobeniusSq_add_frobeniusSq`), so the squared
  norms of the two modes follow the system `IsSignSystem`
  (`isSignSystem_decayRun_signProbe`), and the margin is their half difference
  (`margin_signProduct`).

So for every step `η > 0`, from every start with `U₀ + V₀ ≠ 0`, the two modes behave as in
`SignSystem` and `SignGap` (`tendsto_modes_decayRun_signProbe`), and the norm of the factors
tends to infinity (`tendsto_frobeniusSq_decayRun_signProbe_atTop`). `SignRunLimit` has the
loss, the margin and the direction.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Training.Scope

open Matrix Filter Topology

variable {k : Type*} [Fintype k] {η : ℝ} {U₀ V₀ : Matrix (Fin 1) k ℂ}

/-- **The gradient of the sign sample** (`review/flat_spectrum.md`, item XE19): `−w(Re x)`,
with the weight `w(r) = 2 / (1 + exp (2 r))`. -/
lemma ceGrad_signProbe (X : Matrix (Fin 1) (Fin 1) ℂ) :
    ceGrad signProbe (fun _ => true) X =
      (-signWeight (pairing 1 X)) • (1 : Matrix (Fin 1) (Fin 1) ℂ) := by
  have hz : pairing (-1 : Matrix (Fin 1) (Fin 1) ℂ) X = -pairing 1 X := by
    rw [show (-1 : Matrix (Fin 1) (Fin 1) ℂ) = (-1 : ℝ) • 1 by simp, pairing_smul_left]
    ring
  have hp := Geometry.sum_probability (fun c : Bool => pairing (signProbe 0 c) X) true
  have hf : Geometry.probability (fun c : Bool => pairing (signProbe 0 c) X) false =
      signWeight (pairing 1 X) / 2 := by
    simp only [Geometry.probability, Fintype.sum_bool, signProbe, ↓reduceIte,
      Bool.false_eq_true, hz, signWeight]
    rw [Real.exp_neg, two_mul, Real.exp_add]
    have he := Real.exp_pos (pairing 1 X)
    field_simp
    ring
  rw [Fintype.sum_bool] at hp
  rw [ceGrad, Fin.sum_univ_one, Fintype.sum_bool, eq_sub_of_add_eq hp, hf]
  simp only [signProbe, ↓reduceIte, Bool.false_eq_true]
  module

/-- **A step without weight decay on the sign sample** (`review/flat_spectrum.md`, item XE19):
`U' = U + η w V` and `V' = V + η w U`, with the weight `w` at the product. -/
lemma decayStep_signProbe (η : ℝ) (W : Matrix (Fin 1) k ℂ × Matrix (Fin 1) k ℂ) :
    decayStep (ceGrad signProbe fun _ => true) η 0 W =
      (W.1 + (η * signWeight (pairing 1 (W.1 * W.2ᴴ))) • W.2,
        W.2 + (η * signWeight (pairing 1 (W.1 * W.2ᴴ))) • W.1) := by
  simp only [decayStep, ceGrad_signProbe, zero_smul, add_zero, Matrix.smul_mul, Matrix.one_mul,
    conjTranspose_smul, conjTranspose_one, star_trivial]
  ext : 1 <;> module

/-- `Re tr (U Vᴴ) = (‖U + V‖² − ‖U − V‖²) / 4`. -/
lemma pairing_one_mul_conjTranspose {m : Type*} [Fintype m] [DecidableEq m]
    (U V : Matrix m k ℂ) :
    pairing (1 : Matrix m m ℂ) (U * Vᴴ) = (frobeniusSq (U + V) - frobeniusSq (U - V)) / 4 := by
  have h : pairing (1 : Matrix m m ℂ) (U * Vᴴ) = pairing U V := by
    rw [pairing_comm U V]
    simp only [pairing, conjTranspose_one, Matrix.one_mul]
    rw [Matrix.trace_mul_comm]
  rw [h, frobeniusSq_add, frobeniusSq_sub]
  ring

/-- `‖U‖² + ‖V‖² = (‖U + V‖² + ‖U − V‖²) / 2`. -/
lemma frobeniusSq_add_frobeniusSq {m : Type*} [Fintype m] (U V : Matrix m k ℂ) :
    frobeniusSq U + frobeniusSq V = (frobeniusSq (U + V) + frobeniusSq (U - V)) / 2 := by
  rw [frobeniusSq_add, frobeniusSq_sub]
  ring

local notation "R" => decayRun (ceGrad signProbe fun _ => true)

/-- The product `U_t V_tᴴ` of the run without weight decay on the sign sample
(`review/flat_spectrum.md`, item XE19). -/
noncomputable def signProduct (η : ℝ) (U₀ V₀ : Matrix (Fin 1) k ℂ) (t : ℕ) :
    Matrix (Fin 1) (Fin 1) ℂ :=
  (R η 0 U₀ V₀ t).1 * (R η 0 U₀ V₀ t).2ᴴ

/-- The sum of the factors is multiplied by `1 + η w` at every step. -/
lemma decayRun_signProbe_add (η : ℝ) (U₀ V₀ : Matrix (Fin 1) k ℂ) (t : ℕ) :
    (R η 0 U₀ V₀ (t + 1)).1 + (R η 0 U₀ V₀ (t + 1)).2 =
      (1 + η * signWeight (pairing 1 (signProduct η U₀ V₀ t))) •
        ((R η 0 U₀ V₀ t).1 + (R η 0 U₀ V₀ t).2) := by
  rw [decayRun_succ, decayStep_signProbe, signProduct]
  dsimp only
  module

/-- The difference of the factors is multiplied by `1 − η w` at every step. -/
lemma decayRun_signProbe_sub (η : ℝ) (U₀ V₀ : Matrix (Fin 1) k ℂ) (t : ℕ) :
    (R η 0 U₀ V₀ (t + 1)).1 - (R η 0 U₀ V₀ (t + 1)).2 =
      (1 - η * signWeight (pairing 1 (signProduct η U₀ V₀ t))) •
        ((R η 0 U₀ V₀ t).1 - (R η 0 U₀ V₀ t).2) := by
  rw [decayRun_succ, decayStep_signProbe, signProduct]
  dsimp only
  module

/-- **The squared norms of the two modes follow the system of `SignSystem`**
(`review/flat_spectrum.md`, item XE19). -/
lemma isSignSystem_decayRun_signProbe (η : ℝ) (U₀ V₀ : Matrix (Fin 1) k ℂ) :
    IsSignSystem η (fun t => frobeniusSq ((R η 0 U₀ V₀ t).1 + (R η 0 U₀ V₀ t).2))
      fun t => frobeniusSq ((R η 0 U₀ V₀ t).1 - (R η 0 U₀ V₀ t).2) :=
  ⟨fun t => by
    dsimp only
    rw [decayRun_signProbe_add, frobeniusSq_smul, signProduct, pairing_one_mul_conjTranspose],
  fun t => by
    dsimp only
    rw [decayRun_signProbe_sub, frobeniusSq_smul, signProduct, pairing_one_mul_conjTranspose]⟩

/-- The margin of the sign sample is `2 Re x`. -/
lemma margin_signProbe (X : Matrix (Fin 1) (Fin 1) ℂ) :
    margin signProbe (fun _ => true) X = 2 * pairing 1 X := by
  have hg : gap signProbe (fun _ => true) X 0 false = 2 * pairing 1 X := by
    rw [gap_signProbe]
    simp [pairing, Matrix.trace_fin_one]
  refine le_antisymm ((margin_le_gap X (s := 0) (c := false) (by simp)).trans_eq hg)
    (le_margin fun s c hc => ?_)
  cases c with
  | false => rw [Subsingleton.elim s 0, hg]
  | true => exact absurd rfl hc

/-- The margin of the product of the run is the half difference of the two modes. -/
lemma margin_signProduct (η : ℝ) (U₀ V₀ : Matrix (Fin 1) k ℂ) (t : ℕ) :
    margin signProbe (fun _ => true) (signProduct η U₀ V₀ t) =
      (frobeniusSq ((R η 0 U₀ V₀ t).1 + (R η 0 U₀ V₀ t).2) -
        frobeniusSq ((R η 0 U₀ V₀ t).1 - (R η 0 U₀ V₀ t).2)) / 2 := by
  rw [margin_signProbe, signProduct, pairing_one_mul_conjTranspose]
  ring

/-- **The two modes of the run** (`review/flat_spectrum.md`, item XE19, statements 1 to 3):
for every step `η > 0`, from a start with `U₀ + V₀ ≠ 0`, `‖U_t + V_t‖² → ∞`,
`‖U_t + V_t‖² − ‖U_t − V_t‖² → ∞`, and from some step on `‖U_t − V_t‖²` does not grow. -/
theorem tendsto_modes_decayRun_signProbe (hη : 0 < η) (h0 : U₀ + V₀ ≠ 0) :
    Tendsto (fun t => frobeniusSq ((R η 0 U₀ V₀ t).1 + (R η 0 U₀ V₀ t).2)) atTop atTop ∧
      Tendsto (fun t => frobeniusSq ((R η 0 U₀ V₀ t).1 + (R η 0 U₀ V₀ t).2) -
        frobeniusSq ((R η 0 U₀ V₀ t).1 - (R η 0 U₀ V₀ t).2)) atTop atTop ∧
      ∃ T, ∀ t, T ≤ t → frobeniusSq ((R η 0 U₀ V₀ t).1 - (R η 0 U₀ V₀ t).2) ≤
        frobeniusSq ((R η 0 U₀ V₀ T).1 - (R η 0 U₀ V₀ T).2) := by
  have sys := isSignSystem_decayRun_signProbe η U₀ V₀
  have ha0 : 0 < frobeniusSq (U₀ + V₀) :=
    lt_of_le_of_ne (frobeniusSq_nonneg _) fun h => h0 (eq_zero_of_frobeniusSq_eq_zero h.symm)
  have hb0 := frobeniusSq_nonneg (U₀ - V₀)
  exact ⟨sys.tendsto_fst_atTop hη ha0 hb0, sys.tendsto_sub_atTop hη ha0 hb0,
    sys.exists_forall_snd_le hη ha0 hb0⟩

/-- **Without weight decay the norm of the factors tends to infinity**
(`review/flat_spectrum.md`, item XE19, statement 1): on the sign sample, for every step
`η > 0` and every start with `U₀ + V₀ ≠ 0`, `‖U_t‖² + ‖V_t‖² → ∞`. -/
theorem tendsto_frobeniusSq_decayRun_signProbe_atTop (hη : 0 < η) (h0 : U₀ + V₀ ≠ 0) :
    Tendsto (fun t => frobeniusSq (R η 0 U₀ V₀ t).1 + frobeniusSq (R η 0 U₀ V₀ t).2)
      atTop atTop := by
  refine tendsto_atTop_mono (fun t => ?_)
    ((tendsto_modes_decayRun_signProbe hη h0).1.const_mul_atTop (by norm_num : (0 : ℝ) < 1 / 2))
  have h1 := frobeniusSq_add_frobeniusSq (R η 0 U₀ V₀ t).1 (R η 0 U₀ V₀ t).2
  have h2 := frobeniusSq_nonneg ((R η 0 U₀ V₀ t).1 - (R η 0 U₀ V₀ t).2)
  linarith

/-! The witness: the start `U₀ = [1 0]`, `V₀ = [0 1]` of the example of XE17, with the step
`1/10`. -/

/-- The start `U₀ = [1 0]`, `V₀ = [0 1]` has `U₀ + V₀ ≠ 0`. -/
lemma fromCols_one_zero_add_ne_zero :
    fromCols (1 : Matrix (Fin 1) (Fin 1) ℂ) (0 : Matrix (Fin 1) (Fin 1) ℂ) +
      fromCols (0 : Matrix (Fin 1) (Fin 1) ℂ) (1 : Matrix (Fin 1) (Fin 1) ℂ) ≠ 0 := fun h => by
  have := congrFun (congrFun h 0) (Sum.inl 0)
  simp at this

example := tendsto_modes_decayRun_signProbe (η := 1 / 10) (by norm_num)
  fromCols_one_zero_add_ne_zero

example := tendsto_frobeniusSq_decayRun_signProbe_atTop (η := 1 / 10) (by norm_num)
  fromCols_one_zero_add_ne_zero

end Transformer.Grokking.FlatSpectrum.Training.Scope
