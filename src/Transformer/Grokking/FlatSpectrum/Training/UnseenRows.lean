module

public import Transformer.Grokking.FlatSpectrum.UnseenRow
public import Transformer.Grokking.FlatSpectrum.Training.SeenMargin

/-!
# A trained product is zero on an unseen row or column

Source: `review/flat_spectrum.md`, *Training with cross-entropy* (added), item XE12.
For `λ > 0` and a loss that reads only the seen cells `Ω`, the product of a trained fit is a
minimum of M from `Ω` for its own values (item XE7), and such a minimum is zero on every row
and on every column without a cell of `Ω` (item CW8). So

* a trained product is zero on an unseen row (`IsDecayedMin.row_eq_zero`) and on an unseen
  column (`IsDecayedMin.col_eq_zero`);
* for the cross-entropy of an answer code it is zero at every cell whose row or whose column
  no sample sees (`IsDecayedMin.apply_eq_zero_of_codeProbe`), so every logit of a held-out
  sample at such a cell is zero (`IsDecayedMin.pairing_codeProbe_eq_zero`) and the held-out
  cross-entropy is that of chance, `|S'| log |C|` (`IsDecayedMin.ceLoss_heldOut_eq`);
* under the bounds of item XE3 the run converges, and the product of its limit is zero on the
  unseen rows and columns (`exists_tendsto_decayRun_ceLoss_eq_zero`).

The witness is a `2 × 1` product with the code `±1` whose sample sees the cell `(0, 0)`: the
row `1` is unseen.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Training

open Matrix Filter Topology

variable {m n k S C S' : Type*} [Fintype m] [Fintype n] [Fintype k] [DecidableEq m]
  [DecidableEq n] {Ω : Set (m × n)} {L : Matrix m n ℂ → ℝ} {lam : ℝ} {U : Matrix m k ℂ}
  {V : Matrix n k ℂ}

/-- **A trained product is zero on an unseen row** (`review/flat_spectrum.md`, item XE12): if
the loss reads only the cells of `Ω`, `λ > 0` and no cell of the row `i` lies in `Ω`. -/
theorem IsDecayedMin.row_eq_zero
    (hL : ∀ X Y : Matrix m n ℂ, (∀ ij ∈ Ω, X ij.1 ij.2 = Y ij.1 ij.2) → L X = L Y)
    (hlam : 0 < lam) (h : IsDecayedMin L lam U V) {i : m} (hi : ∀ j, (i, j) ∉ Ω) (j : n) :
    (U * Vᴴ) i j = 0 :=
  row_eq_zero_of_isMinOn hi (fun _ _ => rfl) (h.isMinOn_nuclearNorm hL hlam) j

/-- **A trained product is zero on an unseen column** (`review/flat_spectrum.md`, item
XE12). -/
theorem IsDecayedMin.col_eq_zero
    (hL : ∀ X Y : Matrix m n ℂ, (∀ ij ∈ Ω, X ij.1 ij.2 = Y ij.1 ij.2) → L X = L Y)
    (hlam : 0 < lam) (h : IsDecayedMin L lam U V) {j : n} (hj : ∀ i, (i, j) ∉ Ω) (i : m) :
    (U * Vᴴ) i j = 0 :=
  col_eq_zero_of_isMinOn hj (fun _ _ => rfl) (h.isMinOn_nuclearNorm hL hlam) i

variable [Fintype S] [Fintype C] {φ : C → ℂ} {cell : S → m × n} {y : S → C}

/-- **Under the cross-entropy of an answer code a trained product is zero at a cell whose row
or whose column no sample sees** (`review/flat_spectrum.md`, item XE12). -/
theorem IsDecayedMin.apply_eq_zero_of_codeProbe (hlam : 0 < lam)
    (h : IsDecayedMin (ceLoss (codeProbe φ cell) y) lam U V) {i : m} {j : n}
    (hij : (∀ s, (cell s).1 ≠ i) ∨ ∀ s, (cell s).2 ≠ j) : (U * Vᴴ) i j = 0 := by
  have hL : ∀ X Y : Matrix m n ℂ, (∀ ij ∈ Set.range cell, X ij.1 ij.2 = Y ij.1 ij.2) →
      ceLoss (codeProbe φ cell) y X = ceLoss (codeProbe φ cell) y Y := fun X Y hXY =>
    ceLoss_eq_of_seen (fun s c _ _ hn => codeProbe_apply_of_notMem hn s c) hXY
  rcases hij with hi | hj
  · exact h.row_eq_zero hL hlam (fun _ ⟨s, hs⟩ => hi s (congrArg Prod.fst hs)) j
  · exact h.col_eq_zero hL hlam (fun _ ⟨s, hs⟩ => hj s (congrArg Prod.snd hs)) i

variable [Fintype S'] {cell' : S' → m × n}

omit [Fintype S'] in
/-- Every logit of a held-out sample at such a cell is zero: the classes tie. -/
theorem IsDecayedMin.pairing_codeProbe_eq_zero (hlam : 0 < lam)
    (h : IsDecayedMin (ceLoss (codeProbe φ cell) y) lam U V) {s' : S'}
    (hs' : (∀ s, (cell s).1 ≠ (cell' s').1) ∨ ∀ s, (cell s).2 ≠ (cell' s').2) (c : C) :
    pairing (codeProbe φ cell' s' c) (U * Vᴴ) = 0 := by
  rw [pairing_codeProbe, h.apply_eq_zero_of_codeProbe hlam hs', mul_zero, Complex.zero_re]

/-- **The held-out cross-entropy on unseen rows and columns is that of chance**
(`review/flat_spectrum.md`, item XE12): `|S'| log |C|`, whatever the held-out labels. -/
theorem IsDecayedMin.ceLoss_heldOut_eq (hlam : 0 < lam)
    (h : IsDecayedMin (ceLoss (codeProbe φ cell) y) lam U V)
    (hs' : ∀ s', (∀ s, (cell s).1 ≠ (cell' s').1) ∨ ∀ s, (cell s).2 ≠ (cell' s').2)
    (y' : S' → C) :
    ceLoss (codeProbe φ cell') y' (U * Vᴴ) = Fintype.card S' * Real.log (Fintype.card C) := by
  rw [← ceLoss_zero (A := codeProbe φ cell') (y := y')]
  refine ceLoss_eq_of_seen (Ω := Set.range cell')
    (fun s c _ _ hn => codeProbe_apply_of_notMem hn s c) ?_
  rintro _ ⟨s', rfl⟩
  exact h.apply_eq_zero_of_codeProbe hlam (hs' s')

/-- **The run on the cross-entropy ends at a product that is zero on the unseen rows and
columns** (`review/flat_spectrum.md`, items XE3 and XE12): for probes of norm at most `a` that
vanish off `Ω`, under the bounds of item XE3 the run from a start of full row rank converges,
and the product of its limit is zero on every row and every column without a cell of `Ω`. -/
theorem exists_tendsto_decayRun_ceLoss_eq_zero {A : S → C → Matrix m n ℂ} {η a ρ : ℝ}
    (hη : 0 < η) (hlam : 0 < lam) (ha0 : 0 ≤ a) (ha : ∀ s c, frobeniusSq (A s c) ≤ a ^ 2)
    (hρ : 4 * (Fintype.card S * Real.log (Fintype.card C)) ≤ lam * ρ)
    (hstep : 2 * η * (lam ^ 2 + (2 * a * Fintype.card S) ^ 2) ≤ lam)
    (hdesc : η * (lam + 2 * a * Fintype.card S + 2 * (Fintype.card S * a ^ 2) * ρ) ≤ 1)
    {U₀ : Matrix m k ℂ} {V₀ : Matrix n k ℂ} (hw : frobeniusSq U₀ + frobeniusSq V₀ ≤ ρ)
    (hfull : Function.Surjective (stack U₀ V₀).mulVec)
    (hA : ∀ s c i j, (i, j) ∉ Ω → A s c i j = 0) :
    ∃ (U : Matrix m k ℂ) (V : Matrix n k ℂ),
      Tendsto (decayRun (ceGrad A y) η lam U₀ V₀) atTop (𝓝 (U, V)) ∧
        ∀ i j, ((∀ j', (i, j') ∉ Ω) ∨ ∀ i', (i', j) ∉ Ω) → (U * Vᴴ) i j = 0 := by
  obtain ⟨U, V, hlim, hmin⟩ :=
    exists_tendsto_decayRun_ceLoss_isDecayedMin hη hlam ha0 ha hρ hstep hdesc hw hfull
  have hL : ∀ X Y : Matrix m n ℂ, (∀ ij ∈ Ω, X ij.1 ij.2 = Y ij.1 ij.2) →
      ceLoss A y X = ceLoss A y Y := fun X Y h => ceLoss_eq_of_seen hA h
  exact ⟨U, V, hlim, fun i j hij => hij.elim (fun hi => hmin.row_eq_zero hL hlam hi j)
    fun hj => hmin.col_eq_zero hL hlam hj i⟩

/-! The witness: a `2 × 1` product with the code `±1`; the sample sees the cell `(0, 0)` and
the row `1` is unseen. -/

local notation "cell₁" => fun _ : Fin 1 => ((0 : Fin 2), (0 : Fin 1))

local notation "cell₂" => fun _ : Fin 1 => ((1 : Fin 2), (0 : Fin 1))

/-- A trained fit exists, and its product is zero on the unseen row, where the held-out
sample at `(1, 0)` has zero logits and the loss `log 2`. -/
example : ∃ (U : Matrix (Fin 2) (Fin 1) ℂ) (V : Matrix (Fin 1) (Fin 1) ℂ),
    (U * Vᴴ) 1 0 = 0 ∧ pairing (codeProbe signCode cell₂ 0 true) (U * Vᴴ) = 0 ∧
      ceLoss (codeProbe signCode cell₂) (fun _ => true) (U * Vᴴ)
        = (Fintype.card (Fin 1) : ℝ) * Real.log (Fintype.card Bool) := by
  obtain ⟨U, V, h⟩ := exists_isDecayedMin_ceLoss (k := Fin 1) (codeProbe signCode cell₁)
    (fun _ => true) one_pos (Or.inr le_rfl)
  exact ⟨U, V, h.row_eq_zero (Ω := Set.range cell₁)
    (fun X Y hXY => ceLoss_eq_of_seen (fun s c _ _ hn => codeProbe_apply_of_notMem hn s c) hXY)
    one_pos (by simp) 0,
    h.pairing_codeProbe_eq_zero (cell' := cell₂) (s' := 0) one_pos (Or.inl fun _ => by simp) true,
    h.ceLoss_heldOut_eq one_pos (fun _ => Or.inl fun _ => by simp) _⟩

example {U : Matrix (Fin 2) (Fin 1) ℂ} {V : Matrix (Fin 1) (Fin 1) ℂ}
    (h : IsDecayedMin (ceLoss (codeProbe signCode cell₁) fun _ => true) 1 U V) :
    (U * Vᴴ) 1 0 = 0 :=
  h.apply_eq_zero_of_codeProbe one_pos (Or.inl fun _ => by simp)

/-- With no cell seen, the product of the limit of the run on the probes `0` is zero: every
column is unseen. -/
example := exists_tendsto_decayRun_ceLoss_eq_zero (y := fun _ : Fin 1 => true) (η := 1 / 10)
  (a := 0) (by norm_num) one_pos le_rfl (A := 0) (fun _ _ => by simp [frobeniusSq])
  signWitness_radius (by norm_num) (by norm_num) (stepWitness_ball.trans (by norm_num))
  (fun v => ⟨v, by rw [stack_fromCols_one, one_mulVec]⟩) (Ω := ∅) (fun _ _ _ _ _ => rfl)

end Transformer.Grokking.FlatSpectrum.Training
