module

public import Transformer.SequenceModel.Attention.Spectral.Realization

/-!
# Embed recovered scores without changing their ordering

Source: `review/flat_spectrum.md`, *Unified spectral transformer certificate*,
J2. Every real class score is placed in its tied embedding coordinate. One
fixed positive scale bounds the whole finite table; final RMSNorm supplies
an additional positive row scale with a uniform lower bound.
-/

@[expose] public section
noncomputable section

namespace Transformer.SequenceModel.Attention.Spectral

open GPTMini Expressivity
open scoped BigOperators RealInnerProductSpace

/-- One finite positive scale, computed from all scores before any input is queried. -/
def tableMass {A V : Type*} [Fintype A] [Fintype V] (S : A → V → ℝ) : ℝ :=
  1 + ∑ x, ∑ v, |S x v|

/-- J2: the sentinel keeps the scaling valid even for a zero score table. -/
theorem tableMass_pos {A V : Type*} [Fintype A] [Fintype V] (S : A → V → ℝ) :
    0 < tableMass S := by unfold tableMass; positivity

/-- J2: one global scale bounds the absolute mass of every score row. -/
theorem row_mass_le {A V : Type*} [Fintype A] [Fintype V] (S : A → V → ℝ) (x : A) :
    ∑ v, |S x v| ≤ tableMass S := by
  have h : (∑ v, |S x v|) ≤ ∑ y, ∑ v, |S y v| :=
    Finset.single_le_sum (fun y _ => Finset.sum_nonneg (fun v _ => abs_nonneg (S y v)))
      (Finset.mem_univ x)
  unfold tableMass
  linarith

/-- The vector contains all real class scores in the original token bank. -/
def scoreVector (cfg : Config) (h1 : cfg.n_heads = 1)
    (e : (Fin cfg.vocab_size ⊕ Fin cfg.vocab_size) ↪ Fin cfg.head_dim)
    (s : Fin cfg.vocab_size → ℝ) : EucSpace cfg.d_model :=
  ∑ v, s v • merge cfg h1 (unit (e (.inl v)))

/-- J2: the actual tied-coordinate readout extracts the requested score. -/
theorem scoreVector_inner (cfg : Config) (h1 : cfg.n_heads = 1)
    (e : (Fin cfg.vocab_size ⊕ Fin cfg.vocab_size) ↪ Fin cfg.head_dim)
    (s : Fin cfg.vocab_size → ℝ) (v : Fin cfg.vocab_size) :
    ⟪scoreVector cfg h1 e s, merge cfg h1 (unit (e (.inl v)))⟫ = s v := by
  simp [scoreVector, sum_inner, real_inner_smul_left, LinearIsometry.inner_map_map,
    inner_unit, e.injective.eq_iff]

/-- J2: orthogonal unit token coordinates give a finite absolute-mass bound. -/
theorem scoreVector_norm (cfg : Config) (h1 : cfg.n_heads = 1)
    (e : (Fin cfg.vocab_size ⊕ Fin cfg.vocab_size) ↪ Fin cfg.head_dim)
    (s : Fin cfg.vocab_size → ℝ) : ‖scoreVector cfg h1 e s‖ ≤ ∑ v, |s v| := by
  simpa only [scoreVector, norm_smul, Real.norm_eq_abs, LinearIsometry.norm_map, norm_unit, mul_one]
    using norm_sum_le Finset.univ (fun v => s v • merge cfg h1 (unit (e (.inl v))))

/-- All requested scores, rescaled by the same number over the whole input table. -/
def normalizedScores (cfg : Config) (h1 : cfg.n_heads = 1)
    (e : (Fin cfg.vocab_size ⊕ Fin cfg.vocab_size) ↪ Fin cfg.head_dim)
    {A : Type*} [Fintype A] (S : A → Fin cfg.vocab_size → ℝ) (x : A) : EucSpace cfg.d_model :=
  (tableMass S)⁻¹ • scoreVector cfg h1 e (S x)

/-- J2: the realized residual target satisfies J1's actual geometric bound. -/
theorem normalizedScores_norm (cfg : Config) (h1 : cfg.n_heads = 1)
    (e : (Fin cfg.vocab_size ⊕ Fin cfg.vocab_size) ↪ Fin cfg.head_dim)
    {A : Type*} [Fintype A] (S : A → Fin cfg.vocab_size → ℝ) (x : A) :
    ‖normalizedScores cfg h1 e S x‖ ≤ 1 := by
  have hM := tableMass_pos S
  rw [normalizedScores, norm_smul, Real.norm_eq_abs, abs_of_pos (inv_pos.mpr hM)]
  calc
    _ ≤ (tableMass S)⁻¹ * tableMass S := mul_le_mul_of_nonneg_left
      ((scoreVector_norm cfg h1 e (S x)).trans (row_mass_le S x)) (by positivity)
    _ = 1 := inv_mul_cancel₀ hM.ne'

/-- J2: bounded residual targets give a uniform lower bound on final RMS gain. -/
theorem normScale_lower {d : ℕ} (hd : 0 < d) {eps : ℝ} (heps : 0 < eps)
    (v : EucSpace d) (hv : ‖v‖ ≤ 1) : normScale d eps 1 ≤ normScale d eps (‖v‖ ^ 2) := by
  have hn : ‖v‖ ^ 2 ≤ 1 := by nlinarith [norm_nonneg v]
  unfold normScale
  apply div_le_div_of_nonneg_left (Real.sqrt_nonneg _) (by positivity)
  exact Real.sqrt_le_sqrt (by linarith)

/-- A strictly positive uniform lower bound for the source's score multiplier. -/
def minimumGain (cfg : Config) (eps : ℝ) {A : Type*} [Fintype A]
    (S : A → Fin cfg.vocab_size → ℝ) : ℝ :=
  normScale cfg.d_model eps 1 * amplitude cfg eps / tableMass S

/-- J2: no separation or nonzero-score hypothesis is needed for positive scaling. -/
theorem minimumGain_pos (cfg : Config) (hw : 3 < cfg.d_model) {eps : ℝ} (heps : 0 < eps)
    {A : Type*} [Fintype A] (S : A → Fin cfg.vocab_size → ℝ) : 0 < minimumGain cfg eps S :=
  div_pos (mul_pos (normScale_pos _ cfg.d_model_pos eps heps 1 (by norm_num))
    (amplitude_pos cfg hw heps)) (tableMass_pos S)

private abbrev witnessCfg := Grid.sequenceConfig 3 (by decide) 1 0 1
private abbrev witnessBanks := banks 3 (by decide) 27 1

example := scoreVector_inner witnessCfg rfl witnessBanks (fun v => (v.val : ℝ) - 1) 2
example := scoreVector_norm witnessCfg rfl witnessBanks (fun v => (v.val : ℝ) - 1)
example := normalizedScores_norm witnessCfg rfl witnessBanks
  (fun x : Fin 2 => fun v => (x.val : ℝ) - v.val) 1
example := normScale_lower (d := 4) (by decide) (eps := 1 / 8388608) (by norm_num)
  (0 : EucSpace 4) (by simp)
example := minimumGain_pos witnessCfg (by decide) (eps := 1 / 8388608) (by norm_num)
  (fun x : Fin 2 => fun v => (x.val : ℝ) - v.val)

end Transformer.SequenceModel.Attention.Spectral
