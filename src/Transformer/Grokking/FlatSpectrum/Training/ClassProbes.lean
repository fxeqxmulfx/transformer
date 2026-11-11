module

public import Transformer.Grokking.FlatSpectrum.Training.UnseenRows

/-!
# The probes of a table of logits

Source: `review/flat_spectrum.md`, *Training with cross-entropy* (added), item XE1, the second
case of probes.
The classes are the columns of the product. The sample `s` reads the row `row s`, and its
class `c` has the probe `e_{row s, c}` (`classProbe`), so that its logit is `Re X_{row s, c}`
(`pairing_classProbe`): the product is the table of the logits, and a gap is a difference of
two entries of a row (`gap_classProbe`). The probes have norm one (`frobeniusSq_classProbe`),
so `a = 1` in items XE2 and XE3, and they vanish off the rows that the samples read
(`classProbe_apply_of_ne`), so items XE7 and XE12 apply:

* under the bounds of item XE3 with `a = 1` the run from a start of full row rank converges
  to a trained fit whose table is zero on every row that no sample reads
  (`exists_tendsto_decayRun_classProbe`);
* every trained table is zero on such a row (`IsDecayedMin.row_eq_zero_of_classProbe`).

The witness is one sample of a `1 × 2` table with the label `true`.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Training

open Matrix Filter Topology

variable {m n k S : Type*} [Fintype m] [Fintype n] [Fintype k] [DecidableEq m] [DecidableEq n]

/-- The probes of a table of logits: the class `c` of the sample `s` has the probe
`e_{row s, c}`, item XE1. -/
def classProbe (row : S → m) (s : S) (c : n) : Matrix m n ℂ :=
  Matrix.single (row s) c 1

omit [Fintype k] in
/-- The logit of the class `c` at the sample `s` is `Re X_{row s, c}`. -/
lemma pairing_classProbe (row : S → m) (s : S) (c : n) (X : Matrix m n ℂ) :
    pairing (classProbe row s c) X = (X (row s) c).re := by
  rw [classProbe, pairing_single, map_one, one_mul]

omit [Fintype k] in
lemma frobeniusSq_classProbe (row : S → m) (s : S) (c : n) :
    frobeniusSq (classProbe row s c) = 1 := by
  rw [classProbe, frobeniusSq_single, norm_one, one_pow]

omit [Fintype m] [Fintype n] [Fintype k] in
/-- The probes vanish off the rows that the samples read. -/
lemma classProbe_apply_of_ne {row : S → m} {i : m} (h : ∀ s, row s ≠ i) (s : S) (c j : n) :
    classProbe row s c i j = 0 :=
  Matrix.single_apply_of_ne _ _ _ _ _ fun h' => h s h'.1

omit [Fintype k] in
/-- A gap of a table of logits is a difference of two entries of a row. -/
lemma gap_classProbe (row : S → m) (y : S → n) (X : Matrix m n ℂ) (s : S) (c : n) :
    gap (classProbe row) y X s c = (X (row s) (y s)).re - (X (row s) c).re := by
  rw [gap, pairing_classProbe, pairing_classProbe]

variable [Fintype S] {row : S → m} {y : S → n} {lam : ℝ} {U : Matrix m k ℂ}
  {V : Matrix n k ℂ}

/-- **A row that no sample reads is zero in every trained table of logits**
(`review/flat_spectrum.md`, items XE1 and XE12), for `λ > 0`. -/
theorem IsDecayedMin.row_eq_zero_of_classProbe (hlam : 0 < lam)
    (h : IsDecayedMin (ceLoss (classProbe row) y) lam U V) {i : m} (hi : ∀ s, row s ≠ i)
    (j : n) : (U * Vᴴ) i j = 0 :=
  h.row_eq_zero (Ω := {ij | ∃ s, row s = ij.1})
    (fun _ _ hXY => ceLoss_eq_of_seen (fun s c _ j' hn =>
      classProbe_apply_of_ne (fun s' hs' => hn (Exists.intro s' hs')) s c j') hXY) hlam
    (fun _ ⟨s, hs⟩ => hi s hs) j

/-- **The run on the cross-entropy of a table of logits converges**
(`review/flat_spectrum.md`, items XE1 and XE3 with `a = 1`): under the two bounds on the step
the run from a start of full row rank converges to a trained fit, whose table is zero on every
row that no sample reads. -/
theorem exists_tendsto_decayRun_classProbe {η ρ : ℝ} (hη : 0 < η) (hlam : 0 < lam)
    (hρ : 4 * (Fintype.card S * Real.log (Fintype.card n)) ≤ lam * ρ)
    (hstep : 2 * η * (lam ^ 2 + (2 * Fintype.card S) ^ 2) ≤ lam)
    (hdesc : η * (lam + 2 * Fintype.card S + 2 * Fintype.card S * ρ) ≤ 1)
    {U₀ : Matrix m k ℂ} {V₀ : Matrix n k ℂ} (hw : frobeniusSq U₀ + frobeniusSq V₀ ≤ ρ)
    (hfull : Function.Surjective (stack U₀ V₀).mulVec) :
    ∃ (U : Matrix m k ℂ) (V : Matrix n k ℂ),
      Tendsto (decayRun (ceGrad (classProbe row) y) η lam U₀ V₀) atTop (𝓝 (U, V)) ∧
        IsDecayedMin (ceLoss (classProbe row) y) lam U V ∧
          ∀ i, (∀ s, row s ≠ i) → ∀ j, (U * Vᴴ) i j = 0 := by
  obtain ⟨U, V, hlim, hmin⟩ := exists_tendsto_decayRun_ceLoss_isDecayedMin
    (A := classProbe row) (y := y) (a := 1) hη hlam zero_le_one
    (fun s c => (frobeniusSq_classProbe row s c).le.trans (one_pow 2).ge) hρ
    (by rwa [mul_one]) (by rwa [mul_one, one_pow, mul_one]) hw hfull
  exact ⟨U, V, hlim, hmin, fun i hi j => hmin.row_eq_zero_of_classProbe hlam hi j⟩

/-! The witness: one sample of a `1 × 2` table, with the columns `Bool` and the label `true`.
With `λ = 1`, `ρ = 3` and `η = 1/10` the bounds are those of the sample of item XE3. -/

example := pairing_classProbe (fun _ : Fin 1 => (0 : Fin 1)) 0 true 0

example := frobeniusSq_classProbe (n := Bool) (fun _ : Fin 1 => (0 : Fin 1)) 0 true

example := classProbe_apply_of_ne (row := fun _ : Fin 1 => (0 : Fin 2)) (i := 1)
  (fun _ => by simp) 0 true true

example := gap_classProbe (fun _ : Fin 1 => (0 : Fin 1)) (fun _ => true) 0 0 false

/-- A trained table with an unread row: the row is zero. -/
example : ∃ (U : Matrix (Fin 2) Bool ℂ) (V : Matrix Bool Bool ℂ), ∀ j, (U * Vᴴ) 1 j = 0 := by
  obtain ⟨U, V, h⟩ := exists_isDecayedMin_ceLoss (k := Bool)
    (classProbe fun _ : Fin 1 => (0 : Fin 2)) (fun _ => true) one_pos (Or.inr le_rfl)
  exact ⟨U, V, fun j => h.row_eq_zero_of_classProbe one_pos (fun _ => by simp) j⟩

/-- The run from `U₀ = [1 0]`, `V₀ = [0 1]` converges to a trained table. -/
example := exists_tendsto_decayRun_classProbe (row := fun _ : Fin 1 => (0 : Fin 1))
  (y := fun _ => true) (η := 1 / 10) (by norm_num) one_pos signWitness_radius (by norm_num)
  (by norm_num)
  (U₀ := fromCols (1 : Matrix (Fin 1) (Fin 1) ℂ) (0 : Matrix (Fin 1) Bool ℂ))
  (V₀ := fromCols (0 : Matrix Bool (Fin 1) ℂ) (1 : Matrix Bool Bool ℂ)) (by
    rw [← frobeniusSq_stack, stack_fromCols_one]
    norm_num [frobeniusSq, Matrix.one_apply])
  fun v => ⟨v, by rw [stack_fromCols_one, one_mulVec]⟩

end Transformer.Grokking.FlatSpectrum.Training
