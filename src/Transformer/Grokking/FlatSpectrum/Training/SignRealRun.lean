module

public import Transformer.Grokking.FlatSpectrum.Training.SignReal
public import Transformer.Grokking.FlatSpectrum.Training.CrossEntropyRun

/-!
# The code `±1`: from real factors the run returns a multiple of the flat kernel

Source: `review/flat_spectrum.md`, *Training with cross-entropy* (added), items XE3, XE11 and
XE13.
The gradient of the cross-entropy of the code `±1` has real entries
(`hasRealEntries_ceGrad_signCode`), so the run from factors with real entries has real entries,
and so does its limit. Let every cell be seen once and let the flat kernel `K`, of squared
singular value `c`, show the labels. Under the bounds of item XE3, for `0 < λ ≤ √c`,

    the run from real stacked factors of full row rank converges to a factorization of t(λ) K

(`exists_tendsto_decayRun_signCode_of_real`): its limit is a trained fit with real factors, and
the trained product with real entries is `t(λ) K`.

Trained fits with real factors exist for every `λ > 0` at the width `|m| + |n|`
(`exists_isDecayedMin_signCode_real`): the bounds of item XE3 can be met from the start
`U₀ = [1 0]`, `V₀ = [0 1]`. Trained fits with real factors, of every width, converge in
direction to `K / ‖K‖_S1` as `λ_j → 0⁺` (`tendsto_inv_smul_signCode_of_real_factors`,
`exists_seq_isDecayedMin_real_tendsto_signCode`).

The witness is the `1 × 1` kernel `1` with `λ = 1/2`, `η = 1/20` and `ρ = 6`.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Training

open Matrix Filter Topology

variable {m n k S : Type*} [Fintype m] [Fintype n] [Fintype k] [DecidableEq m] [DecidableEq n]
  [Fintype S] {cell : S → m × n} {y : S → Bool} {K : Matrix m n ℂ} {c₀ lam η ρ : ℝ}

/-- The gradient of the cross-entropy of the code `±1` has real entries. -/
lemma hasRealEntries_ceGrad_signCode (X : Matrix m n ℂ) :
    HasRealEntries (ceGrad (codeProbe signCode cell) y X) := by
  have hA : ∀ s c, HasRealEntries (codeProbe signCode cell s c) := fun s c i j => by
    rw [codeProbe, Matrix.single_apply]
    split_ifs <;> cases c <;> simp [signCode]
  exact .sum _ fun s _ => (HasRealEntries.sum _ fun c _ => (hA s c).smul _).sub (hA s _)

/-- **From real factors of full row rank the run converges to a factorization of `t(λ) K`**
(`review/flat_spectrum.md`, items XE3 and XE11): for the code `±1` with every cell seen once,
a flat kernel that shows the labels and `0 < λ ≤ √c`, under the bounds of item XE3 with
`a = 1`. No convergence is assumed. -/
theorem exists_tendsto_decayRun_signCode_of_real (hcell : Function.Bijective cell)
    (hK : ∀ s, K (cell s).1 (cell s).2 = signCode (y s)) (hc : 0 < c₀) (hflat : IsFlat K c₀)
    (hη : 0 < η) (hlam : 0 < lam) (hle : lam ≤ √c₀)
    (hρ : 4 * (Fintype.card S * Real.log (Fintype.card Bool)) ≤ lam * ρ)
    (hstep : 2 * η * (lam ^ 2 + (2 * Fintype.card S) ^ 2) ≤ lam)
    (hdesc : η * (lam + 2 * Fintype.card S + 2 * Fintype.card S * ρ) ≤ 1)
    {U₀ : Matrix m k ℂ} {V₀ : Matrix n k ℂ} (hw : frobeniusSq U₀ + frobeniusSq V₀ ≤ ρ)
    (hfull : Function.Surjective (stack U₀ V₀).mulVec) (hU₀ : HasRealEntries U₀)
    (hV₀ : HasRealEntries V₀) :
    ∃ (U : Matrix m k ℂ) (V : Matrix n k ℂ),
      Tendsto (decayRun (ceGrad (codeProbe signCode cell) y) η lam U₀ V₀) atTop (𝓝 (U, V)) ∧
        U * Vᴴ = signScale c₀ lam • K := by
  obtain ⟨U, V, hlim, hmin⟩ := exists_tendsto_decayRun_ceLoss_isDecayedMin
    (A := codeProbe signCode cell) (y := y) hη hlam zero_le_one
    (fun s c => by rw [frobeniusSq_codeProbe, one_pow]; cases c <;> simp [signCode]) hρ
    (by rw [mul_one]; exact hstep) (by rw [mul_one, one_pow, mul_one]; exact hdesc) hw hfull
  obtain ⟨hU, hV⟩ := hasRealEntries_of_tendsto_decayRun
    (fun X _ => hasRealEntries_ceGrad_signCode X) hU₀ hV₀ hlim
  exact ⟨U, V, hlim, (hmin.isNuclearMin hlam.le).eq_signScale_smul hcell hK hc hflat hlam hle
    (hU.mul hV.conjTranspose)⟩

variable (cell y) in
/-- **Trained fits with real factors exist** for every weight decay `λ > 0`, at the width
`|m| + |n|` (`review/flat_spectrum.md`, item XE11): the limit of the run of item XE3 from
`U₀ = [1 0]`, `V₀ = [0 1]`. -/
theorem exists_isDecayedMin_signCode_real (hlam : 0 < lam) :
    ∃ (U : Matrix m (m ⊕ n) ℂ) (V : Matrix n (m ⊕ n) ℂ),
      IsDecayedMin (ceLoss (codeProbe signCode cell) y) lam U V ∧ HasRealEntries U ∧
        HasRealEntries V := by
  have hρ0 : 0 ≤ max (4 * (Fintype.card S * Real.log (Fintype.card Bool)) / lam)
      (frobeniusSq (fromCols (1 : Matrix m m ℂ) (0 : Matrix m n ℂ))
        + frobeniusSq (fromCols (0 : Matrix n m ℂ) (1 : Matrix n n ℂ))) :=
    le_max_of_le_right (add_nonneg (frobeniusSq_nonneg _) (frobeniusSq_nonneg _))
  set ρ := max (4 * (Fintype.card S * Real.log (Fintype.card Bool)) / lam)
    (frobeniusSq (fromCols (1 : Matrix m m ℂ) (0 : Matrix m n ℂ))
      + frobeniusSq (fromCols (0 : Matrix n m ℂ) (1 : Matrix n n ℂ))) with hρdef
  have hE : 0 < 2 * (lam ^ 2 + (2 * 1 * (Fintype.card S : ℝ)) ^ 2) := by positivity
  have hD : 0 < lam + 2 * 1 * (Fintype.card S : ℝ) + 2 * (Fintype.card S * 1 ^ 2) * ρ := by
    positivity
  obtain ⟨U, V, hlim, hmin⟩ := exists_tendsto_decayRun_ceLoss_isDecayedMin
    (A := codeProbe signCode cell) (y := y) (ρ := ρ)
    (η := min (lam / (2 * (lam ^ 2 + (2 * 1 * (Fintype.card S : ℝ)) ^ 2)))
      (1 / (lam + 2 * 1 * (Fintype.card S : ℝ) + 2 * (Fintype.card S * 1 ^ 2) * ρ)))
    (lt_min (div_pos hlam hE) (one_div_pos.2 hD)) hlam zero_le_one
    (fun s c => by rw [frobeniusSq_codeProbe, one_pow]; cases c <;> simp [signCode])
    (by rw [← div_le_iff₀' hlam]; exact le_max_left _ _)
    ((mul_le_mul_of_nonneg_right (min_le_left _ _) hE.le).trans_eq'
      (by ring) |>.trans_eq (div_mul_cancel₀ _ hE.ne'))
    ((mul_le_mul_of_nonneg_right (min_le_right _ _) hD.le).trans_eq
      (div_mul_cancel₀ _ hD.ne'))
    (le_max_right _ _) fun v => ⟨v, by rw [stack_fromCols_one, one_mulVec]⟩
  obtain ⟨hU, hV⟩ := hasRealEntries_of_tendsto_decayRun
    (fun X _ => hasRealEntries_ceGrad_signCode X) (.fromCols .one .zero) (.fromCols .zero .one)
    hlim
  exact ⟨U, V, hmin, hU, hV⟩

/-- **Trained fits whose factors have real entries converge in direction to the flat
kernel** (`review/flat_spectrum.md`, items XE11 and XE13), at every width. -/
theorem tendsto_inv_smul_signCode_of_real_factors {lams : ℕ → ℝ}
    {Us : ℕ → Matrix m k ℂ} {Vs : ℕ → Matrix n k ℂ} (hcell : Function.Bijective cell)
    (hK : ∀ s, K (cell s).1 (cell s).2 = signCode (y s)) (hc : 0 < c₀) (hflat : IsFlat K c₀)
    (hpos : ∀ j, 0 < lams j) (hlam : Tendsto lams atTop (𝓝 0))
    (hmin : ∀ j, IsDecayedMin (ceLoss (codeProbe signCode cell) y) (lams j) (Us j) (Vs j))
    (hU : ∀ j, HasRealEntries (Us j)) (hV : ∀ j, HasRealEntries (Vs j)) :
    Tendsto (fun j => (nuclearNorm (Us j * (Vs j)ᴴ))⁻¹ • (Us j * (Vs j)ᴴ)) atTop
      (𝓝 ((nuclearNorm K)⁻¹ • K)) :=
  tendsto_inv_smul_signCode_of_real hcell hK hc hflat hpos hlam
    (fun j => (hmin j).isNuclearMin (hpos j).le) fun j => (hU j).mul (hV j).conjTranspose

/-- **Some trained fits with real factors exist along every `λ_j → 0⁺`, and all such fits
converge in direction to the flat kernel** (`review/flat_spectrum.md`, items XE11 and XE13),
for the code `±1` with every cell seen once. -/
theorem exists_seq_isDecayedMin_real_tendsto_signCode {lams : ℕ → ℝ}
    (hcell : Function.Bijective cell) (hK : ∀ s, K (cell s).1 (cell s).2 = signCode (y s))
    (hc : 0 < c₀) (hflat : IsFlat K c₀) (hpos : ∀ j, 0 < lams j)
    (hlam : Tendsto lams atTop (𝓝 0)) :
    ∃ (Us : ℕ → Matrix m (m ⊕ n) ℂ) (Vs : ℕ → Matrix n (m ⊕ n) ℂ),
      (∀ j, IsDecayedMin (ceLoss (codeProbe signCode cell) y) (lams j) (Us j) (Vs j)) ∧
      (∀ j, HasRealEntries (Us j)) ∧ (∀ j, HasRealEntries (Vs j)) ∧
      Tendsto (fun j => (nuclearNorm (Us j * (Vs j)ᴴ))⁻¹ • (Us j * (Vs j)ᴴ)) atTop
        (𝓝 ((nuclearNorm K)⁻¹ • K)) := by
  choose Us Vs hmin hU hV using fun j => exists_isDecayedMin_signCode_real cell y (hpos j)
  exact ⟨Us, Vs, hmin, hU, hV,
    tendsto_inv_smul_signCode_of_real_factors hcell hK hc hflat hpos hlam hmin hU hV⟩

/-! The witness: the `1 × 1` kernel `1` on its cell, `c = 1`. With `λ = 1/2`, `η = 1/20` and
`ρ = 6`: `4 log 2 ≤ 3 = λ ρ`, `2 η (λ² + 4) = 17/40` and `η (λ + 2 + 2 ρ) = 29/40`. -/

local notation "cell₀" => fun _ : Fin 1 => ((0 : Fin 1), (0 : Fin 1))

example := hasRealEntries_ceGrad_signCode (cell := cell₀) (y := fun _ => true) 1

/-- From `U₀ = [1 0]`, `V₀ = [0 1]` the run converges to a factorization of `t(1/2) • 1`. -/
example : ∃ U V : Matrix (Fin 1) (Fin 1 ⊕ Fin 1) ℂ,
    Tendsto (decayRun (ceGrad (codeProbe signCode cell₀) fun _ => true) (1 / 20) (1 / 2)
      (fromCols 1 0) (fromCols 0 1)) atTop (𝓝 (U, V)) ∧
      U * Vᴴ = signScale 1 (1 / 2) • (1 : Matrix (Fin 1) (Fin 1) ℂ) :=
  exists_tendsto_decayRun_signCode_of_real (ρ := 6)
    ⟨fun _ _ _ => Subsingleton.elim _ _, fun _ => ⟨0, Subsingleton.elim _ _⟩⟩
    (fun _ => by simp [signCode]) one_pos isFlat_one (by norm_num) (by norm_num)
    (by rw [Real.sqrt_one]; norm_num) (signWitness_radius.trans (by norm_num))
    (by simp only [Fintype.card_fin, Nat.cast_one]; norm_num)
    (by simp only [Fintype.card_fin, Nat.cast_one]; norm_num)
    (stepWitness_ball.trans (by norm_num))
    (fun v => ⟨v, by rw [stack_fromCols_one, one_mulVec]⟩) (.fromCols .one .zero)
    (.fromCols .zero .one)

example := exists_isDecayedMin_signCode_real cell₀ (fun _ => true) (lam := 1 / 2) (by norm_num)

/-- Along `λ_j = 1/(j + 1)` trained fits of the witness with real factors exist and converge
in direction to the kernel. -/
example := exists_seq_isDecayedMin_real_tendsto_signCode (cell := cell₀) (y := fun _ => true)
  (K := 1) (lams := fun j => 1 / ((j : ℝ) + 1))
  ⟨fun _ _ _ => Subsingleton.elim _ _, fun _ => ⟨0, Subsingleton.elim _ _⟩⟩
  (fun _ => by simp [signCode]) one_pos isFlat_one (fun j => by positivity)
  tendsto_one_div_add_atTop_nhds_zero_nat

end Transformer.Grokking.FlatSpectrum.Training
