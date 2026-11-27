module

public import Transformer.Grokking.FiniteRecovery.ArgandSamples
public import Transformer.Grokking.FiniteRecovery.ArgandCausality

/-! Exact capacity and observation comparisons for the last-token family,
and an active-layer lower bound for tasks that depend on earlier inputs. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.Argand

open Finset Matrix Transformer.Grokking.FlatSpectrum Transformer.PhasorV2

variable {A : Type*} {T H d layers repeats : ℕ}

/-- A strict classifier with a competing label needs a nonempty stream. -/
theorem Model.width_pos_of_realizes [Nontrivial A] (model : Model A H d layers)
    {target : Word A (T + 1) → A} (hrealizes : model.realizes repeats target) :
    0 < H * d := by
  by_contra! hn
  have hw : H * d = 0 := by omega
  have hz (a : A) : model.embedding a = 0 := by
    ext i
    have hi := i.isLt
    omega
  obtain ⟨a⟩ := (inferInstance : Nonempty A)
  let word : Word A (T + 1) := fun _ => a
  obtain ⟨b, hb⟩ := exists_ne (target word)
  simpa [Model.logits, Model.readout, hz] using hrealizes word b hb

/-- A task that distinguishes two prefixes with the same final token
requires both a layer and a pass in the actual Argand forward pass. -/
theorem Model.active_pass_of_prefix_dependency (model : Model A H d layers)
    {target : Word A (T + 1) → A} (hrealizes : model.realizes repeats target)
    {u v : Word A (T + 1)} (hlast : u (Fin.last T) = v (Fin.last T))
    (hdiff : target u ≠ target v) : 0 < layers ∧ 0 < repeats := by
  have contradict (h : ∀ a, model.logits repeats u (Fin.last T) a =
      model.logits repeats v (Fin.last T) a) : False := by
    have hu := hrealizes u (target v) hdiff.symm
    have hv := hrealizes v (target u) hdiff
    rw [h, h] at hu
    exact (lt_asymm hu hv)
  constructor
  · by_contra! hn
    have hz : layers = 0 := by omega
    subst layers
    exact contradict fun a => by
      simp only [Model.logits, Model.hidden, Model.hiddenWith, List.ofFn_zero, List.foldl_nil]
      change model.readout (argandRMS model.eps
        (((@id (Word (CSpace (H * d)) (T + 1)))^[repeats]) _ (Fin.last T))) a =
        model.readout (argandRMS model.eps
        (((@id (Word (CSpace (H * d)) (T + 1)))^[repeats]) _ (Fin.last T))) a
      simp [hlast]
  · by_contra! hn
    have hz : repeats = 0 := by omega
    subst repeats
    exact contradict (model.logits_no_passes_eq_of_token_eq hlast)

/-- A balanced word cut for any binary alphabet order. -/
def binaryLastTokenCut {n s : ℕ} (e : Fin (s + 1) ≃ Word Bool (n + 1)) :
    Fin (s + 1) × Fin (s + 1) ≃ Word Bool ((n + 1) + n + 1) :=
  (Equiv.prodCongr e e).trans (appendCut Bool (n + 1) (n + 1))

private theorem binaryLastTokenCut_last {n s : ℕ} (e : Fin (s + 1) ≃ Word Bool (n + 1))
    (u v : Fin (s + 1)) :
    binaryLastTokenCut e (u, v) (Fin.last ((n + 1) + n)) = e v (Fin.last n) := by
  have hl : finSumFinEquiv (Sum.inr (Fin.last n) : Fin (n + 1) ⊕ Fin (n + 1)) =
      Fin.last ((n + 1) + n) := by simp [finSumFinEquiv_apply_right]
  change Sum.elim (e u) (e v) (finSumFinEquiv.symm (Fin.last ((n + 1) + n))) = _
  rw [← hl, Equiv.symm_apply_apply]
  rfl

/-- On the same binary last-token family for which two observations
suffice, unconstrained nuclear completion requires a full staircase
on the balanced cut. This compares two methods on identical targets. -/
theorem isLeast_binaryLastToken_nuclear_observations {n s : ℕ}
    (e : Fin (s + 1) ≃ Word Bool (n + 1)) (mode : Option Bool) :
    let cut := binaryLastTokenCut e
    let target := lastTokenTask (T := (n + 1) + n) mode
    let phase : Bool → ℂ := fun b => if b then 1 else -1
    IsLeast {k : ℕ | ∃ inputs : Finset (Word Bool ((n + 1) + n + 1)), inputs.card = k ∧
      NuclearRecovers (observedCells cut inputs) (targetKernel cut phase target)} (2 * s + 1) := by
  dsimp only
  let phase : Bool → ℂ := fun b => if b then 1 else -1
  let right := fun v => phase (mode.getD (e v (Fin.last n)))
  have heq : targetKernel (binaryLastTokenCut e) phase (lastTokenTask mode) =
      vecMulVec (fun _ => 1) right := by
    ext u v
    simp [targetKernel, vecMulVec_apply, lastTokenTask, binaryLastTokenCut_last, right]
  rw [heq]
  apply isLeast_rankOne_nuclear_observations
  intro u v
  simp only [vecMulVec_apply, one_mul, right, phase]
  split <;> simp

private noncomputable def enumerateBinaryWords (n : ℕ) :
    Fin (2 ^ (n + 1) - 1 + 1) ≃ Word Bool (n + 1) :=
  (Fintype.equivFinOfCardEq (by
    rw [card_word, Fintype.card_bool, Nat.sub_add_cancel (Nat.one_le_pow _ _ (by decide))])).symm

example (mode : Option Bool) := isLeast_binaryLastToken_nuclear_observations (enumerateBinaryWords 3) mode
example (mode : Option Bool) := isLeast_binaryLastToken_nuclear_observations (enumerateBinaryWords 7) mode
example (mode : Option Bool) := isLeast_binaryLastToken_nuclear_observations (enumerateBinaryWords 15) mode

example (model : Model Bool 1 1 layers) {target : Word Bool 8 → Bool}
    (h : model.realizes repeats target)
    (hfirst : target (fun _ => false) ≠ target (fun j => decide (j = 0))) :
    0 < layers ∧ 0 < repeats :=
  model.active_pass_of_prefix_dependency h (by decide) hfirst

end Transformer.Grokking.FiniteRecovery.Argand
