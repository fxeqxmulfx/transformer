module

public import Transformer.Grokking.Protocol.TwistedSplits
public import Transformer.Grokking.Protocol.LabelWitness

/-!
# Labelled splits with a symmetry: the twin splits

Source: arXiv:2201.02177v1 (Power et al.), Figure 1 and section 1, "distinct abstract
symbols"; `experiments/grok_division/PROTOCOL.md`, *The split*;
`review/grok_division_convergence.md`, section 2.

`LabelCoverage` shows that a true form of the target admits labelled splits only. To be
labelled is not enough. Take the residues `2` and `95 = −2` (`twins`). The equations
`2 / 1 = 2` and `95 / 1 = 95` (`twinTwo`, `twinMinus`) are read as the same row of tokens up to
the exchange of the two tokens (`twinSwap`), targets included. Call a split a twin split
(`twin`) when its training equations show the twin residues in those two equations only. The
exchange of the tokens, with the exchange of the two equations in the list (`twinOrder`),
carries a twin split to a twin split (`carries_twin`, `twinOrder_mul_mem_twin`), and it keeps
a twin split labelled (`twinOrder_mul_mem_labelled`). `LabelNotSufficient` draws the
consequence: the run of a twin split cannot tell which of the two tokens is `2`.
-/

@[expose] public section

namespace Transformer.Grokking.Protocol

/-- The twin residues: `2` and `95 = −2`. -/
def twins : Finset (Fin 97) := {2, 95}

/-- The renaming that exchanges the tokens of the twin residues. -/
def twinSwap : Equiv.Perm (Fin 239) := Equiv.swap (residueToken 2) (residueToken 95)

/-- `2 / 1 = 2`. -/
def twinTwo : Equation := (2, 0)

/-- `95 / 1 = 95`. -/
def twinMinus : Equation := (95, 0)

/-- The reordering of the list of equations that exchanges `2 / 1 = 2` and `95 / 1 = 95`. -/
def twinOrder : Equiv.Perm (Fin (97 * 96)) :=
  Equiv.swap (finProdFinEquiv twinTwo) (finProdFinEquiv twinMinus)

/-- An equation that shows no twin residue: not as numerator, denominator or quotient. -/
def AvoidsTwins (x : Equation) : Prop :=
  numerator x ∉ twins ∧ denominator x ∉ twins ∧ x.1 ∉ twins

instance : DecidablePred AvoidsTwins := fun x => by
  unfold AvoidsTwins
  infer_instance

/-- An equation that a twin split may train on: it shows no twin residue, or it is
`2 / 1 = 2` or `95 / 1 = 95`. -/
def TwinAllowed (x : Equation) : Prop := AvoidsTwins x ∨ x = twinTwo ∨ x = twinMinus

instance : DecidablePred TwinAllowed := fun x => by
  unfold TwinAllowed
  infer_instance

/-- **The twin splits**: the training equations show the twin residues in `2 / 1 = 2` and
`95 / 1 = 95` only. -/
def twin : Set (Equiv.Perm (Fin (97 * 96))) := {π | ∀ i : Fin 4656, TwinAllowed (trainEq π i)}

/-- The reordered split holds the exchanged equation in every slot. -/
lemma trainEq_twinOrder_mul (π : Equiv.Perm (Fin (97 * 96))) (i : Fin 4656) :
    trainEq (twinOrder * π) i = Equiv.swap twinTwo twinMinus (trainEq π i) := by
  have h := Equiv.congr_fun (Equiv.symm_trans_swap_trans twinTwo twinMinus finProdFinEquiv)
    (π (Fin.castLE (by norm_num : 4656 ≤ 97 * 96) i))
  simp only [Equiv.trans_apply] at h
  simp only [trainEq, trainAt, equations, Equiv.Perm.mul_apply, twinOrder, ← h,
    Equiv.symm_apply_apply]

lemma twinSwap_residueToken {r : Fin 97} (hr : r ∉ twins) :
    twinSwap (residueToken r) = residueToken r := by
  refine Equiv.swap_apply_of_ne_of_ne (fun h => hr ?_) (fun h => hr ?_)
  · rw [residueToken_injective h]
    decide
  · rw [residueToken_injective h]
    decide

/-- The supervision of an equation that shows no twin residue is left alone. -/
lemma fixed_supervised_twin {x : Equation} (hx : AvoidsTwins x) :
    (supervised x).Fixed twinSwap := by
  obtain ⟨ha, hb, hc⟩ := hx
  have h0 : twinSwap eosToken = eosToken := by decide
  have h1 : twinSwap eqToken = eqToken := by decide
  have h2 : twinSwap divToken = divToken := by decide
  refine ⟨?_, fun pc hpc => ?_⟩
  · change ∀ t : Fin 6, twinSwap (read x t) = read x t
    intro t
    fin_cases t <;>
      simp [read, h0, h1, h2, twinSwap_residueToken ha, twinSwap_residueToken hb,
        twinSwap_residueToken hc]
  · simp only [supervised, List.mem_cons, List.mem_nil_iff, or_false] at hpc
    rcases hpc with rfl | rfl
    · exact twinSwap_residueToken hc
    · exact h0

/-- An equation that shows no twin residue is neither of the two equations. -/
lemma swap_of_avoidsTwins {x : Equation} (hx : AvoidsTwins x) :
    Equiv.swap twinTwo twinMinus x = x :=
  Equiv.swap_apply_of_ne_of_ne (fun h => hx.2.2 (by rw [h]; decide))
    (fun h => hx.2.2 (by rw [h]; decide))

/-- The supervision of `95 / 1 = 95`, renamed, is that of `2 / 1 = 2`. -/
lemma rename_supervised_twinMinus :
    (supervised twinMinus).rename twinSwap = supervised twinTwo := by
  have ht : twinSwap ∘ read twinMinus = read twinTwo := by
    funext t
    fin_cases t <;> decide
  have hy : ([((4 : Fin 6), residueToken twinMinus.1), ((5 : Fin 6), eosToken)].map
      fun pc => (pc.1, twinSwap pc.2)) =
      [((4 : Fin 6), residueToken twinTwo.1), ((5 : Fin 6), eosToken)] := by decide
  simp only [Example.rename, supervised, ht, hy]

/-- The supervision of `2 / 1 = 2`, renamed, is that of `95 / 1 = 95`. -/
lemma rename_supervised_twinTwo :
    (supervised twinTwo).rename twinSwap = supervised twinMinus := by
  have ht : twinSwap ∘ read twinTwo = read twinMinus := by
    funext t
    fin_cases t <;> decide
  have hy : ([((4 : Fin 6), residueToken twinTwo.1), ((5 : Fin 6), eosToken)].map
      fun pc => (pc.1, twinSwap pc.2)) =
      [((4 : Fin 6), residueToken twinMinus.1), ((5 : Fin 6), eosToken)] := by decide
  simp only [Example.rename, supervised, ht, hy]

variable {π : Equiv.Perm (Fin (97 * 96))}

/-- **The exchange of the twin tokens and of the two equations carries every twin split.** -/
theorem carries_twin (hπ : π ∈ twin) : Carries twinSwap twinOrder π := fun i => by
  rw [trainEq_twinOrder_mul]
  rcases hπ i with h | h | h
  · rw [swap_of_avoidsTwins h]
    exact Example.rename_of_fixed (S := paperSizes) twinSwap (fixed_supervised_twin h)
  · rw [h, Equiv.swap_apply_left]
    exact rename_supervised_twinMinus
  · rw [h, Equiv.swap_apply_right]
    exact rename_supervised_twinTwo

/-- **The reordering keeps a twin split a twin split.** -/
theorem twinOrder_mul_mem_twin (hπ : π ∈ twin) : twinOrder * π ∈ twin := fun i => by
  rw [trainEq_twinOrder_mul]
  rcases hπ i with h | h | h
  · rw [swap_of_avoidsTwins h]
    exact Or.inl h
  · rw [h, Equiv.swap_apply_left]
    exact Or.inr (Or.inr rfl)
  · rw [h, Equiv.swap_apply_right]
    exact Or.inr (Or.inl rfl)

/-- **The reordering keeps a labelled twin split labelled.** -/
theorem twinOrder_mul_mem_labelled (hπ : π ∈ twin) (hl : π ∈ labelled) :
    twinOrder * π ∈ labelled := fun r => by
  obtain ⟨i, hi⟩ := hl r
  obtain ⟨i₂, h₂⟩ := hl 2
  obtain ⟨i₉, h₉⟩ := hl 95
  have hmin : trainEq π i₉ = twinMinus := by
    rcases hπ i₉ with h | h | h
    · exact absurd (h₉ ▸ h.2.2) (by decide)
    · exact absurd (h ▸ h₉ : twinTwo.1 = 95) (by decide)
    · exact h
  have htwo : trainEq π i₂ = twinTwo := by
    rcases hπ i₂ with h | h | h
    · exact absurd (h₂ ▸ h.2.2) (by decide)
    · exact h
    · exact absurd (h ▸ h₂ : twinMinus.1 = 2) (by decide)
  rcases hπ i with h | h | h
  · exact ⟨i, by rw [trainEq_twinOrder_mul, swap_of_avoidsTwins h, hi]⟩
  · refine ⟨i₉, ?_⟩
    rw [trainEq_twinOrder_mul, hmin, Equiv.swap_apply_right, ← hi, h]
  · refine ⟨i₂, ?_⟩
    rw [trainEq_twinOrder_mul, htwo, Equiv.swap_apply_left, ← hi, h]

/-! The witnesses: the residue `3` is no twin, and `6 / 4 = 50` shows no twin residue.
`TwinWitness` has a twin split that is labelled, the witness of the hypotheses on a split. -/

example : twinSwap (residueToken 3) = residueToken 3 := twinSwap_residueToken (by decide)

example : (supervised (50, 3)).Fixed twinSwap := fixed_supervised_twin (by decide)

example : Equiv.swap twinTwo twinMinus (50, 3) = (50, 3) := swap_of_avoidsTwins (by decide)

end Transformer.Grokking.Protocol
