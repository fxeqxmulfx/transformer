module

public import Transformer.Grokking.FlatSpectrum.CoupledMoments.DecodedRecovery.Certificate

/-!
# Every fitting completion has the certified decoded answers

Source: `review/flat_spectrum.md`, CM18. `accepted_decoding` is the soundness of the
certificate: if it is accepted, then every pair of phases that fits the training samples, the
unknown target's among them, decodes at each reached row and reached column to the label that
the certificate proposes. The latent states themselves are fixed only up to the sign and the
shift of `pinned_rigid`, which the decoder does not see. Correctness is per cell: nothing is
said about a row or a column that the certificate does not reach.

No hypothesis mentions a label outside the training samples, a sign, or a latent state of a
cell: the phases `r`, `s` are arbitrary among those that fit.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.CoupledMoments.DecodedRecovery

open Schatten

variable {q : ℕ} {C n S : Type*} {d : ZMod q → C} {cell : S → n × n} {labels : S → C}

/-- **Every reached row and column is fixed up to one sign and one shift** (CM18). -/
theorem pinned_rigid (hq : Odd q) (hd : PairedDecoder d) {c : Certificate n q}
    (hc : Accepts d cell labels c) {r s : n → ZMod q} (hfit : Fits d cell labels r s) :
    ∃ ε g : ZMod q, (ε = 1 ∨ ε = -1) ∧ (∀ i, Pinned cell c (.inl i) → r i = ε * c.row i + g) ∧
      ∀ j, Pinned cell c (.inr j) → s j = ε * c.col j - g := by
  obtain ⟨ε, g, hε, htop, hbot, hanch⟩ := anchor_rigid hq hd hc hfit
  have hεε : ε * ε = 1 := by rcases hε with rfl | rfl <;> simp
  have main (v : n ⊕ n) (hv : Pinned cell c v) :
      Sum.elim (fun i => r i = ε * c.row i + g) (fun j => s j = ε * c.col j - g) v := by
    induction hv with
    | top => exact htop
    | bot => exact hbot
    | anchored ht hb => exact hanch _ ht hb
    | @row i j j' hj hj' hne ho ho' ihj ihj' =>
      have hi : Pinned cell c (.inl i) := .row hj hj' hne ho ho'
      have ej : s j = ε * c.col j - g := ihj
      have ej' : s j' = ε * c.col j' - g := ihj'
      have h1 : d (ε * (r i - g) + c.col j) = d (c.row i + c.col j) := by
        rw [← hd.mul_sign hε (ε * (r i - g) + c.col j), hc.agree hfit hi hj ho, ej]
        congr 1
        linear_combination (r i - g) * hεε
      have h2 : d (ε * (r i - g) + c.col j') = d (c.row i + c.col j') := by
        rw [← hd.mul_sign hε (ε * (r i - g) + c.col j'), hc.agree hfit hi hj' ho', ej']
        congr 1
        linear_combination (r i - g) * hεε
      have hx := step_unique hq hd hne h1 h2
      change r i = ε * c.row i + g
      linear_combination ε * hx - (r i - g) * hεε
    | @col j i i' hi hi' hne ho ho' ihi ihi' =>
      have hj : Pinned cell c (.inr j) := .col hi hi' hne ho ho'
      have ei : r i = ε * c.row i + g := ihi
      have ei' : r i' = ε * c.row i' + g := ihi'
      have h1 : d (ε * (s j + g) + c.row i) = d (c.col j + c.row i) := by
        rw [← hd.mul_sign hε (ε * (s j + g) + c.row i), add_comm (c.col j),
          hc.agree hfit hi hj ho, ei]
        congr 1
        linear_combination (s j + g) * hεε
      have h2 : d (ε * (s j + g) + c.row i') = d (c.col j + c.row i') := by
        rw [← hd.mul_sign hε (ε * (s j + g) + c.row i'), add_comm (c.col j),
          hc.agree hfit hi' hj ho', ei']
        congr 1
        linear_combination (s j + g) * hεε
      have hx := step_unique hq hd hne h1 h2
      change s j = ε * c.col j - g
      linear_combination ε * hx - (s j + g) * hεε
  exact ⟨ε, g, hε, fun i hi => main _ hi, fun j hj => main _ hj⟩

/-- **Soundness of the certificate** (CM18): at a reached row and a reached column, every pair
of phases that fits the training samples decodes to the proposed label. Applied to the phases
of the unknown target, this is its label at the cell, observed or not. -/
theorem accepted_decoding (hq : Odd q) (hd : PairedDecoder d) {c : Certificate n q}
    (hc : Accepts d cell labels c) {r s : n → ZMod q} (hfit : Fits d cell labels r s) {i j : n}
    (hi : Pinned cell c (.inl i)) (hj : Pinned cell c (.inr j)) :
    d (r i + s j) = c.decoded d i j := by
  obtain ⟨ε, g, hε, hr, hs⟩ := pinned_rigid hq hd hc hfit
  rw [hr i hi, hs j hj, Certificate.decoded, ← hd.mul_sign hε (c.row i + c.col j)]
  congr 1
  ring

/-- Two completions of the training samples have the same label wherever an accepted
certificate reaches: decoded answers are identified although latent states are not. -/
theorem fits_agree (hq : Odd q) (hd : PairedDecoder d) {c : Certificate n q}
    (hc : Accepts d cell labels c) {r s r' s' : n → ZMod q} (hfit : Fits d cell labels r s)
    (hfit' : Fits d cell labels r' s') {i j : n} (hi : Pinned cell c (.inl i))
    (hj : Pinned cell c (.inr j)) : d (r i + s j) = d (r' i + s' j) :=
  (accepted_decoding hq hd hc hfit hi hj).trans (accepted_decoding hq hd hc hfit' hi hj).symm

/-! The folded table with a withheld cell meets every hypothesis. The second fit below is the
reflected and shifted one: its latent states differ, its decoded labels do not. -/

example := pinned_rigid (by decide) (pairedDecoder_fold 2) heldCert_accepts held_fits
example : foldDecoder 2 (foldedRow 2 + foldedColumn 1) = heldCert.decoded (foldDecoder 2) 2 1 :=
  accepted_decoding (by decide) (pairedDecoder_fold 2) heldCert_accepts held_fits held_row
    held_col

/-- The reflection and a shift of the phases of CM16 fit the same samples. -/
lemma held_fits_reflected : Fits (foldDecoder 2) heldCells heldLabels
    (fun i => -foldedRow i + 1) (fun j => -foldedColumn j - 1) := by
  intro t
  rw [← held_fits t, ← foldDecoder_even 2 (foldedRow _ + foldedColumn _)]
  congr 1
  ring

example := fits_agree (by decide) (pairedDecoder_fold 2) heldCert_accepts held_fits
  held_fits_reflected held_row held_col
example : (fun i => -foldedRow i + 1) 2 + (fun j => -foldedColumn j - 1) 1 ≠
    foldedRow 2 + foldedColumn 1 := by decide

end Transformer.Grokking.FlatSpectrum.CoupledMoments.DecodedRecovery
