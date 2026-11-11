module

public import Transformer.Grokking.FlatSpectrum.Training.Continuity

/-!
# Factors with real entries stay real along the run

Source: `review/flat_spectrum.md`, *Training with cross-entropy* (added), item XE11; the run is
that of item WD1.
A matrix has real entries when every entry has imaginary part `0` (`HasRealEntries`). Sums,
differences, real multiples, products, adjoints, blocks and limits of such matrices have real
entries.
So if the gradient field `G` maps matrices with real entries to such matrices, one step of
gradient descent with weight decay keeps factors with real entries
(`HasRealEntries.decayStep`), the whole run does (`hasRealEntries_decayRun`), and so does its
limit (`hasRealEntries_of_tendsto_decayRun`).

The witness is the run of the zero gradient field from the `1 × 1` factors `(0, 0)`.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Training

open Matrix Filter Topology

variable {m n k : Type*}

/-- A matrix with real entries. -/
def HasRealEntries (X : Matrix m n ℂ) : Prop := ∀ i j, (X i j).im = 0

namespace HasRealEntries

variable {X Y : Matrix m n ℂ}

lemma zero : HasRealEntries (0 : Matrix m n ℂ) := fun _ _ => rfl

lemma one [DecidableEq n] : HasRealEntries (1 : Matrix n n ℂ) := fun i j => by
  rw [Matrix.one_apply]
  split_ifs <;> simp

lemma add (hX : HasRealEntries X) (hY : HasRealEntries Y) : HasRealEntries (X + Y) :=
  fun i j => by rw [Matrix.add_apply, Complex.add_im, hX i j, hY i j, add_zero]

lemma sub (hX : HasRealEntries X) (hY : HasRealEntries Y) : HasRealEntries (X - Y) :=
  fun i j => by rw [Matrix.sub_apply, Complex.sub_im, hX i j, hY i j, sub_zero]

lemma smul (t : ℝ) (hX : HasRealEntries X) : HasRealEntries (t • X) :=
  fun i j => by rw [Matrix.smul_apply, Complex.smul_im, hX i j, smul_zero]

lemma conjTranspose (hX : HasRealEntries X) : HasRealEntries Xᴴ :=
  fun i j => by rw [Matrix.conjTranspose_apply, Complex.star_def, Complex.conj_im, hX j i,
    neg_zero]

lemma mul [Fintype k] {X : Matrix m k ℂ} {Y : Matrix k n ℂ} (hX : HasRealEntries X)
    (hY : HasRealEntries Y) : HasRealEntries (X * Y) := fun i j => by
  rw [Matrix.mul_apply, Complex.im_sum]
  exact Finset.sum_eq_zero fun l _ => by
    rw [Complex.mul_im, hX i l, hY l j, mul_zero, zero_mul, add_zero]

lemma sum {ι : Type*} (s : Finset ι) {f : ι → Matrix m n ℂ}
    (h : ∀ a ∈ s, HasRealEntries (f a)) : HasRealEntries (∑ a ∈ s, f a) := fun i j => by
  rw [Matrix.sum_apply, Complex.im_sum]
  exact Finset.sum_eq_zero fun a ha => h a ha i j

lemma fromCols {X : Matrix m n ℂ} {Y : Matrix m k ℂ} (hX : HasRealEntries X)
    (hY : HasRealEntries Y) : HasRealEntries (fromCols X Y) := fun i j => by
  cases j with
  | inl j => exact hX i j
  | inr j => exact hY i j

/-- A limit of matrices with real entries has real entries. -/
lemma of_tendsto {Xs : ℕ → Matrix m n ℂ} (h : Tendsto Xs atTop (𝓝 X))
    (hX : ∀ t, HasRealEntries (Xs t)) : HasRealEntries X := fun i j => by
  have h1 : Tendsto (fun t => (Xs t i j).im) atTop (𝓝 (X i j).im) :=
    ((Complex.continuous_im.comp (continuous_id.matrix_elem i j)).tendsto X).comp h
  exact tendsto_nhds_unique h1 (tendsto_const_nhds.congr fun t => (hX t i j).symm)

end HasRealEntries

variable [Fintype m] [Fintype n] [Fintype k] {G : Matrix m n ℂ → Matrix m n ℂ} {η lam : ℝ}
  {U₀ U : Matrix m k ℂ} {V₀ V : Matrix n k ℂ}

/-- **One step of gradient descent with weight decay keeps factors with real entries**, for a
gradient field that keeps real entries. -/
lemma HasRealEntries.decayStep (hG : ∀ X, HasRealEntries X → HasRealEntries (G X)) (η lam : ℝ)
    {UV : Matrix m k ℂ × Matrix n k ℂ} (hU : HasRealEntries UV.1) (hV : HasRealEntries UV.2) :
    HasRealEntries (decayStep G η lam UV).1 ∧ HasRealEntries (decayStep G η lam UV).2 := by
  have hX := hG _ (hU.mul hV.conjTranspose)
  exact ⟨hU.sub (((hX.mul hV).add (hU.smul lam)).smul η),
    hV.sub (((hX.conjTranspose.mul hU).add (hV.smul lam)).smul η)⟩

/-- **The run from factors with real entries has real entries.** -/
theorem hasRealEntries_decayRun (hG : ∀ X, HasRealEntries X → HasRealEntries (G X)) (η lam : ℝ)
    (hU₀ : HasRealEntries U₀) (hV₀ : HasRealEntries V₀) (t : ℕ) :
    HasRealEntries (decayRun G η lam U₀ V₀ t).1 ∧
      HasRealEntries (decayRun G η lam U₀ V₀ t).2 := by
  induction t with
  | zero => exact ⟨hU₀, hV₀⟩
  | succ t ih =>
    rw [decayRun, Function.iterate_succ_apply']
    exact HasRealEntries.decayStep hG η lam ih.1 ih.2

/-- **The limit of a run from factors with real entries has real entries.** -/
theorem hasRealEntries_of_tendsto_decayRun
    (hG : ∀ X, HasRealEntries X → HasRealEntries (G X)) (hU₀ : HasRealEntries U₀)
    (hV₀ : HasRealEntries V₀) (h : Tendsto (decayRun G η lam U₀ V₀) atTop (𝓝 (U, V))) :
    HasRealEntries U ∧ HasRealEntries V :=
  ⟨.of_tendsto h.fst_nhds fun t => (hasRealEntries_decayRun hG η lam hU₀ hV₀ t).1,
    .of_tendsto h.snd_nhds fun t => (hasRealEntries_decayRun hG η lam hU₀ hV₀ t).2⟩

/-! The witness: `1 × 1` matrices, and the run of the zero gradient field from `(0, 0)`, which
stays there. -/

example : HasRealEntries ((2 : ℝ) • (1 : Matrix (Fin 1) (Fin 1) ℂ) + 1 * 1ᴴ - 0) :=
  ((HasRealEntries.one.smul 2).add
    (HasRealEntries.one.mul HasRealEntries.one.conjTranspose)).sub .zero

example : HasRealEntries (fromCols (1 : Matrix (Fin 1) (Fin 1) ℂ) (0 : Matrix (Fin 1) (Fin 1) ℂ)) :=
  .fromCols .one .zero

example : HasRealEntries (∑ _a : Fin 2, (1 : Matrix (Fin 1) (Fin 1) ℂ)) :=
  .sum _ fun _ _ => .one

example := HasRealEntries.of_tendsto (tendsto_const_nhds (x := (1 : Matrix (Fin 1) (Fin 1) ℂ)))
  fun _ => .one

example := HasRealEntries.decayStep (G := fun _ : Matrix (Fin 1) (Fin 1) ℂ => 0)
  (fun _ _ => .zero) 1 1 (UV := (1, 1)) .one .one

example := hasRealEntries_decayRun (G := fun _ : Matrix (Fin 1) (Fin 1) ℂ => 0)
  (fun _ _ => .zero) 1 1 (U₀ := 1) (V₀ := 1) .one .one 3

example : HasRealEntries (0 : Matrix (Fin 1) (Fin 1) ℂ) ∧
    HasRealEntries (0 : Matrix (Fin 1) (Fin 1) ℂ) :=
  hasRealEntries_of_tendsto_decayRun (G := fun _ : Matrix (Fin 1) (Fin 1) ℂ => 0) (η := 1)
    (lam := 1) (fun _ _ => .zero) .zero .zero
    (tendsto_const_nhds.congr fun t => (Function.iterate_fixed (by simp [decayStep]) t).symm)

end Transformer.Grokking.FlatSpectrum.Training
