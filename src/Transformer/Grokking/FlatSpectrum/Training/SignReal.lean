module

public import Transformer.Grokking.FlatSpectrum.Training.RealEntries
public import Transformer.Grokking.FlatSpectrum.Training.SignUnique
public import Transformer.Grokking.FlatSpectrum.Training.SignDirection

/-!
# The code `±1`: the trained product with real entries is a multiple of the flat kernel

Source: `review/flat_spectrum.md`, *Training with cross-entropy* (added), items XE11 and XE13.
Let every cell be seen once and let the flat kernel `K`, of squared singular value `c`, show
the labels of the code `±1`. For `0 < λ ≤ √c` put `t(λ) = ½ log (2 √c / λ − 1) ≥ 0`
(`signScale`, `signScale_nonneg`), so that `λ = 2 √c / (1 + e^{2 t(λ)})`
(`signDecay_signScale`). Then

* `t(λ) K` is a trained product for `λ` (`isNuclearMin_signScale_smul`);
* every trained product for `λ` has the real parts of `t(λ) K`
  (`IsNuclearMin.re_eq_signScale_mul`): trained products for one weight decay have the same
  real parts;
* so a trained product with real entries is `t(λ) K` (`IsNuclearMin.eq_signScale_smul`);
* hence trained products with real entries converge in direction to `K / ‖K‖_S1` as
  `λ_j → 0⁺` (`tendsto_inv_smul_signCode_of_real`).

The witness is `signKernel` with all four cells seen once.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Training

open Matrix Filter Topology

variable {m n S : Type*} [Fintype m] [Fintype n] [DecidableEq m] [DecidableEq n] [Fintype S]
  {cell : S → m × n} {y : S → Bool} {K X : Matrix m n ℂ} {c₀ lam : ℝ}

/-- The multiple `t(λ) = ½ log (2 √c / λ − 1)` of the flat kernel for the weight decay `λ`. -/
noncomputable def signScale (c₀ lam : ℝ) : ℝ := Real.log (2 * √c₀ / lam - 1) / 2

/-- `λ = 2 √c / (1 + e^{2 t(λ)})` for `0 < λ < 2 √c`. -/
lemma signDecay_signScale (hlam : 0 < lam) (hlt : lam < 2 * √c₀) :
    2 * √c₀ / (1 + Real.exp (2 * signScale c₀ lam)) = lam := by
  have hs : 0 < √c₀ := by linarith
  have h1 : 0 < 2 * √c₀ / lam - 1 := by
    rw [sub_pos, one_lt_div hlam]
    exact hlt
  rw [signScale, show 2 * (Real.log (2 * √c₀ / lam - 1) / 2) = Real.log (2 * √c₀ / lam - 1)
    by ring, Real.exp_log h1]
  field_simp
  ring

lemma signScale_nonneg (hlam : 0 < lam) (hle : lam ≤ √c₀) : 0 ≤ signScale c₀ lam :=
  div_nonneg (Real.log_nonneg (by
    rw [le_sub_iff_add_le, le_div_iff₀ hlam]
    linarith)) zero_le_two

lemma signScale_pos (hlam : 0 < lam) (hlt : lam < √c₀) : 0 < signScale c₀ lam :=
  div_pos (Real.log_pos (by
    rw [lt_sub_iff_add_lt, lt_div_iff₀ hlam]
    linarith)) two_pos

/-- **The multiple `t(λ) K` is a trained product for every weight decay `0 < λ ≤ √c`**
(`review/flat_spectrum.md`, item XE11). -/
theorem isNuclearMin_signScale_smul (hcell : Function.Bijective cell)
    (hK : ∀ s, K (cell s).1 (cell s).2 = signCode (y s)) (hc : 0 < c₀) (hflat : IsFlat K c₀)
    (hlam : 0 < lam) (hle : lam ≤ √c₀) :
    IsNuclearMin (ceLoss (codeProbe signCode cell) y) lam (signScale c₀ lam • K) := by
  have h := isNuclearMin_smul_signCode (y := y) hcell hK hc hflat (signScale_nonneg hlam hle)
  rwa [signDecay_signScale hlam (by linarith [Real.sqrt_pos.2 hc])] at h

/-- **Every trained product has the real parts of `t(λ) K`** (`review/flat_spectrum.md`, item
XE11), for the code `±1` with every cell seen once and `0 < λ ≤ √c`. -/
theorem IsNuclearMin.re_eq_signScale_mul (hcell : Function.Bijective cell)
    (hK : ∀ s, K (cell s).1 (cell s).2 = signCode (y s)) (hc : 0 < c₀) (hflat : IsFlat K c₀)
    (hlam : 0 < lam) (hle : lam ≤ √c₀)
    (hX : IsNuclearMin (ceLoss (codeProbe signCode cell) y) lam X) (i : m) (j : n) :
    (X i j).re = signScale c₀ lam * (K i j).re := by
  obtain ⟨s, hs⟩ := hcell.surjective (i, j)
  have h := hX.re_apply_eq_of_signCode hlam.le
    (isNuclearMin_signScale_smul hcell hK hc hflat hlam hle) s
  rwa [hs, Matrix.smul_apply, Complex.smul_re, smul_eq_mul] at h

/-- **A trained product with real entries is `t(λ) K`** (`review/flat_spectrum.md`, item
XE11). -/
theorem IsNuclearMin.eq_signScale_smul (hcell : Function.Bijective cell)
    (hK : ∀ s, K (cell s).1 (cell s).2 = signCode (y s)) (hc : 0 < c₀) (hflat : IsFlat K c₀)
    (hlam : 0 < lam) (hle : lam ≤ √c₀)
    (hX : IsNuclearMin (ceLoss (codeProbe signCode cell) y) lam X) (hreal : HasRealEntries X) :
    X = signScale c₀ lam • K := by
  ext i j
  have hKim : (K i j).im = 0 := by
    obtain ⟨s, hs⟩ := hcell.surjective (i, j)
    have h := hK s
    rw [hs] at h
    rw [h]
    cases y s <;> simp [signCode]
  refine Complex.ext ?_ ?_
  · rw [hX.re_eq_signScale_mul hcell hK hc hflat hlam hle, Matrix.smul_apply, Complex.smul_re,
      smul_eq_mul]
  · rw [hreal i j, Matrix.smul_apply, Complex.smul_im, hKim, smul_zero]

/-- **Trained products with real entries converge in direction to the flat kernel**
(`review/flat_spectrum.md`, items XE11 and XE13), for the code `±1` with every cell seen
once. -/
theorem tendsto_inv_smul_signCode_of_real {lams : ℕ → ℝ} {Xs : ℕ → Matrix m n ℂ}
    (hcell : Function.Bijective cell) (hK : ∀ s, K (cell s).1 (cell s).2 = signCode (y s))
    (hc : 0 < c₀) (hflat : IsFlat K c₀) (hpos : ∀ j, 0 < lams j)
    (hlam : Tendsto lams atTop (𝓝 0))
    (hmin : ∀ j, IsNuclearMin (ceLoss (codeProbe signCode cell) y) (lams j) (Xs j))
    (hreal : ∀ j, HasRealEntries (Xs j)) :
    Tendsto (fun j => (nuclearNorm (Xs j))⁻¹ • Xs j) atTop (𝓝 ((nuclearNorm K)⁻¹ • K)) := by
  refine tendsto_const_nhds.congr'
    ((hlam.eventually_lt_const (Real.sqrt_pos.2 hc)).mono fun j hj => ?_)
  beta_reduce
  rw [(hmin j).eq_signScale_smul hcell hK hc hflat (hpos j) hj.le (hreal j)]
  exact (inv_nuclearNorm_smul_smul (signScale_pos (hpos j) hj) K).symm

/-! The witness: `signKernel` with all four cells seen once, `c = 2`, at the weight decay
`1 ≤ √2`. -/

lemma one_le_sqrt_two : (1 : ℝ) ≤ √2 := Real.one_le_sqrt.2 one_le_two

example := signDecay_signScale (c₀ := 2) one_pos (by linarith [one_le_sqrt_two])

example := signScale_nonneg (c₀ := 2) one_pos one_le_sqrt_two

example := signScale_pos (c₀ := 2) (lam := 1 / 2) (by norm_num)
  (by linarith [one_le_sqrt_two])

example := isNuclearMin_signScale_smul Function.bijective_id signKernel_apply two_pos
  isFlat_signKernel one_pos one_le_sqrt_two

/-- The trained product `t signTwin` has the real parts of `t signKernel`. -/
example (i j : Fin 2) :=
  IsNuclearMin.re_eq_signScale_mul Function.bijective_id signKernel_apply two_pos
    isFlat_signKernel one_pos one_le_sqrt_two
    (isNuclearMin_smul_of_re_eq (K' := signTwin) (t := signScale 2 1) Function.bijective_id
      signKernel_apply two_pos isFlat_signKernel signTwin_re nuclearNorm_signTwin_le
      (signScale_nonneg one_pos one_le_sqrt_two) |>.of_le (by positivity) rfl le_rfl |>
      fun h => by rwa [signDecay_signScale one_pos (by linarith [one_le_sqrt_two])] at h) i j

lemma hasRealEntries_signKernel : HasRealEntries signKernel := fun i j => by
  fin_cases i <;> fin_cases j <;> simp [signKernel]

/-- The trained product `t(1) signKernel` has real entries. -/
example := (isNuclearMin_signScale_smul Function.bijective_id signKernel_apply two_pos
  isFlat_signKernel one_pos one_le_sqrt_two).eq_signScale_smul Function.bijective_id
  signKernel_apply two_pos isFlat_signKernel one_pos one_le_sqrt_two
  (hasRealEntries_signKernel.smul _)

/-- The trained products `t(λ_j) signKernel` for `λ_j = 1/(j + 1)` converge in direction to
the kernel. -/
example := tendsto_inv_smul_signCode_of_real (lams := fun j => 1 / ((j : ℝ) + 1))
  Function.bijective_id signKernel_apply two_pos isFlat_signKernel (fun j => by positivity)
  tendsto_one_div_add_atTop_nhds_zero_nat
  (fun j => isNuclearMin_signScale_smul Function.bijective_id signKernel_apply two_pos
    isFlat_signKernel (by positivity) (((div_le_one (by positivity)).2
      (by linarith [Nat.cast_nonneg (α := ℝ) j])).trans one_le_sqrt_two))
  fun j => hasRealEntries_signKernel.smul _

end Transformer.Grokking.FlatSpectrum.Training
