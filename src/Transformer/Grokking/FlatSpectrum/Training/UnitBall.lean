module

public import Transformer.Grokking.FlatSpectrum.Training.MaxMarginLimit
public import Transformer.Grokking.FlatSpectrum.Training.Recovery

/-!
# The largest margin of the unit ball is attained

Source: `review/flat_spectrum.md`, *Training with cross-entropy* (added), item XE10.
The unit ball of the nuclear norm is compact: its matrices have entries of modulus at most one,
the nuclear norm is lower semicontinuous (`nuclearNorm_le_of_tendsto`, by its dual form), and
the gaps are continuous (`continuous_gap`). So a limit of products of the unit ball whose
margins tend to the largest margin `μ*` lies in the ball and has the margin `μ*`
(`margin_eq_maxMargin_of_tendsto`). Hence

* `μ*` is attained in the unit ball (`exists_margin_eq_maxMargin`);
* a product scaled by the inverse of its nuclear norm lies in the unit ball
  (`nuclearNorm_inv_smul_le`), with the margin `μ(X) / ‖X‖_S1` (`margin_inv_smul`), and every
  sequence of such normalized products has a convergent subsequence
  (`exists_subseq_tendsto_inv_smul`).

The witness is `signProbe`: its gap is `2 Re X` (`gap_signProbe`) and `μ* = 2`
(`maxMargin_signProbe`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Training

open Matrix Filter Topology

variable {m n S C : Type*} [Fintype m] [Fintype n] [DecidableEq n]
  {A : S → C → Matrix m n ℂ} {y : S → C}

omit [DecidableEq n] in
/-- A gap is continuous in the product: it is linear. -/
lemma continuous_gap (A : S → C → Matrix m n ℂ) (y : S → C) (s : S) (c : C) :
    Continuous fun X : Matrix m n ℂ => gap A y X s c :=
  (continuous_pairing continuous_const continuous_id).sub
    (continuous_pairing continuous_const continuous_id)

/-- The nuclear norm is lower semicontinuous: a limit of matrices with `‖Z_j‖_S1 ≤ b` has
`‖Z‖_S1 ≤ b`. -/
lemma nuclearNorm_le_of_tendsto {ι : Type*} {l : Filter ι} [l.NeBot] {Z : ι → Matrix m n ℂ}
    {Z₀ : Matrix m n ℂ} {b : ℝ} (hZ : Tendsto Z l (𝓝 Z₀))
    (hb : ∀ᶠ j in l, nuclearNorm (Z j) ≤ b) : nuclearNorm Z₀ ≤ b := by
  obtain ⟨W, hW, htr⟩ := exists_isContraction_trace_eq Z₀
  have h1 : Tendsto (fun j => pairing W (Z j)) l (𝓝 (pairing W Z₀)) :=
    ((continuous_pairing (f := fun _ : Matrix m n ℂ => W) (g := fun Y => Y) continuous_const
      continuous_id).tendsto Z₀).comp hZ
  have h2 : pairing W Z₀ = nuclearNorm Z₀ := by rw [pairing, htr, Complex.ofReal_re]
  rw [h2] at h1
  exact le_of_tendsto h1 (hb.mono fun j hj =>
    ((Complex.re_le_norm _).trans (norm_trace_le_nuclearNorm hW _)).trans hj)

/-- A product scaled by the inverse of its nuclear norm lies in the unit ball. -/
lemma nuclearNorm_inv_smul_le (X : Matrix m n ℂ) :
    nuclearNorm ((nuclearNorm X)⁻¹ • X) ≤ 1 := by
  rw [nuclearNorm_real_smul (inv_nonneg.mpr (nuclearNorm_nonneg X))]
  rcases (nuclearNorm_nonneg X).eq_or_lt with h | h
  · rw [← h, mul_zero]
    exact zero_le_one
  · exact (inv_mul_cancel₀ h.ne').le

/-- The margin of a normalized product is the margin for the norm: `μ(X) / ‖X‖_S1`. -/
lemma margin_inv_smul (X : Matrix m n ℂ) :
    margin A y ((nuclearNorm X)⁻¹ • X) = margin A y X / nuclearNorm X := by
  rw [margin_smul (inv_nonneg.mpr (nuclearNorm_nonneg X)), inv_mul_eq_div]

variable [DecidableEq m]

/-- The normalized products of every sequence have a convergent subsequence. -/
theorem exists_subseq_tendsto_inv_smul (X : ℕ → Matrix m n ℂ) :
    ∃ (Z : Matrix m n ℂ) (φ : ℕ → ℕ), StrictMono φ ∧
      Tendsto (fun j => (nuclearNorm (X (φ j)))⁻¹ • X (φ j)) atTop (𝓝 Z) := by
  have : FirstCountableTopology (Matrix m n ℂ) :=
    inferInstanceAs (FirstCountableTopology (m → n → ℂ))
  obtain ⟨Z, -, φ, hφ, hZ⟩ := (isCompact_setOf_norm_apply_le (m := m) (n := n)
    1).tendsto_subseq (x := fun j => (nuclearNorm (X j))⁻¹ • X j) fun j i l =>
      (norm_apply_le_nuclearNorm _ i l).trans (nuclearNorm_inv_smul_le _)
  exact ⟨Z, φ, hφ, hZ⟩

variable [Fintype S] [Fintype C]

omit [DecidableEq m] in
/-- **A limit in the unit ball of margins that tend to the largest has the largest margin**
(`review/flat_spectrum.md`, item XE10). -/
theorem margin_eq_maxMargin_of_tendsto [Nonempty S] [Nontrivial C] {Z : ℕ → Matrix m n ℂ}
    {Z₀ : Matrix m n ℂ} (hZ : Tendsto Z atTop (𝓝 Z₀)) (hball : ∀ j, nuclearNorm (Z j) ≤ 1)
    (hμ : Tendsto (fun j => margin A y (Z j)) atTop (𝓝 (maxMargin A y))) :
    nuclearNorm Z₀ ≤ 1 ∧ margin A y Z₀ = maxMargin A y := by
  have h1 := nuclearNorm_le_of_tendsto hZ (Eventually.of_forall hball)
  exact ⟨h1, le_antisymm (margin_le_maxMargin h1) (le_margin fun s c hc =>
    le_of_tendsto_of_tendsto' hμ (((continuous_gap A y s c).tendsto Z₀).comp hZ) fun j =>
      margin_le_gap (Z j) hc)⟩

/-- **The largest margin of the unit ball is attained** (`review/flat_spectrum.md`, item
XE10): some product with `‖Z‖_S1 ≤ 1` has `μ(Z) = μ*`. -/
theorem exists_margin_eq_maxMargin [Nonempty S] [Nontrivial C] (A : S → C → Matrix m n ℂ)
    (y : S → C) : ∃ Z : Matrix m n ℂ, nuclearNorm Z ≤ 1 ∧ margin A y Z = maxMargin A y := by
  have hne : (margin A y '' {X : Matrix m n ℂ | nuclearNorm X ≤ 1}).Nonempty :=
    ⟨_, ⟨0, nuclearNorm_zero.trans_le zero_le_one, rfl⟩⟩
  have hex : ∀ j : ℕ, ∃ Z : Matrix m n ℂ, nuclearNorm Z ≤ 1 ∧
      maxMargin A y - 1 / ((j : ℝ) + 1) < margin A y Z := fun j => by
    obtain ⟨_, ⟨Z, hZ, rfl⟩, hlt⟩ := exists_lt_of_lt_csSup hne
      (sub_lt_self (maxMargin A y) (by positivity : (0 : ℝ) < 1 / ((j : ℝ) + 1)))
    exact ⟨Z, hZ, hlt⟩
  choose Z hZ1 hZμ using hex
  have : FirstCountableTopology (Matrix m n ℂ) :=
    inferInstanceAs (FirstCountableTopology (m → n → ℂ))
  obtain ⟨Z₀, -, φ, hφ, hlim⟩ := (isCompact_setOf_norm_apply_le (m := m) (n := n)
    1).tendsto_subseq (x := Z) fun j i l => (norm_apply_le_nuclearNorm _ i l).trans (hZ1 j)
  have hμ : Tendsto (fun j => margin A y (Z j)) atTop (𝓝 (maxMargin A y)) := by
    refine tendsto_of_tendsto_of_tendsto_of_le_of_le ?_ tendsto_const_nhds
      (fun j => (hZμ j).le) fun j => margin_le_maxMargin (hZ1 j)
    simpa using tendsto_const_nhds.sub (tendsto_one_div_add_atTop_nhds_zero_nat (𝕜 := ℝ))
  exact ⟨Z₀, margin_eq_maxMargin_of_tendsto hlim (fun j => hZ1 (φ j))
    (hμ.comp hφ.tendsto_atTop)⟩

/-! The witness: one sample of a `1 × 1` product read with the code `±1`. Its gap is
`2 Re X`, so `μ* = 2`. -/

lemma gap_signProbe (Z : Matrix (Fin 1) (Fin 1) ℂ) :
    gap signProbe (fun _ => true) Z 0 false = 2 * (Z 0 0).re := by
  simp [gap, signProbe, pairing, Matrix.trace_fin_one]
  ring

lemma maxMargin_signProbe : maxMargin signProbe (fun _ => true) = 2 := by
  refine le_antisymm (csSup_le ⟨_, ⟨0, nuclearNorm_zero.trans_le zero_le_one, rfl⟩⟩ ?_)
    ((le_margin le_gap_signProbe_one).trans (margin_le_maxMargin nuclearNorm_one_fin_one.le))
  rintro _ ⟨X, hX, rfl⟩
  have h1 := margin_le_gap (A := signProbe) (y := fun _ => true) X (s := 0) (c := false)
    (by simp)
  have h2 := gap_le_mul_nuclearNorm (y := fun _ => true) zero_le_one frobeniusSq_signProbe_le
    X 0 false
  have hX' : nuclearNorm X ≤ 1 := hX
  linarith

example := continuous_gap signProbe (fun _ => true) 0 false

example := nuclearNorm_le_of_tendsto (l := atTop)
  (Z := fun _ : ℕ => (1 : Matrix (Fin 1) (Fin 1) ℂ)) tendsto_const_nhds
  (Eventually.of_forall fun _ => nuclearNorm_one_fin_one.le)

/-- The constant sequence `1` of the witness. -/
example := margin_eq_maxMargin_of_tendsto (A := signProbe) (y := fun _ => true)
  (Z := fun _ => 1) tendsto_const_nhds (fun _ => nuclearNorm_one_fin_one.le) (by
    rw [maxMargin_signProbe]
    exact tendsto_const_nhds.congr fun _ => le_antisymm (le_margin le_gap_signProbe_one)
      ((margin_le_maxMargin nuclearNorm_one_fin_one.le).trans_eq maxMargin_signProbe))

example := exists_margin_eq_maxMargin signProbe fun _ => true

end Transformer.Grokking.FlatSpectrum.Training
