module

public import Transformer.Grokking.FiniteRecovery.Argand
public import Transformer.Grokking.FiniteRecovery.Observations

/-! Identifiability of token functions represented by the full Argand
forward pass at fixed dimensions. The example gives an exact count for
the class consisting of the two constants and last-token copying. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.Argand

open Transformer.PhasorV2
open scoped InnerProductSpace

/-- A token-valued target is realized when its logit is strictly largest
at every input, with the classifier read at the final position. -/
def Model.realizes {A : Type*} {T H d layers : ℕ} (model : Model A H d layers)
    (repeats : ℕ) (target : Word A (T + 1) → A) : Prop :=
  ∀ word a, a ≠ target word →
    model.logits repeats word (Fin.last T) a < model.logits repeats word (Fin.last T) (target word)

/-- Functions representable at the indicated widths, depth and repeat count. -/
def realizableFamily (A : Type*) (T H d layers repeats : ℕ) : Set (Word A (T + 1) → A) :=
  {f | ∃ model : Model A H d layers, 0 < model.eps ∧ model.realizes repeats f}

/-- If an Argand configuration represents the flat class, any fixed mask
identifying the entire Argand class must meet the PT4 observation bound. -/
theorem observations_le_of_flat_subset {p left right T H d layers repeats : ℕ} [Fact p.Prime]
    (cut : Word (ZMod p) left × Word (ZMod p) right ≃ Word (ZMod p) (T + 1))
    (hsubset : unknownCouplingFamily cut ⊆ realizableFamily (ZMod p) T H d layers repeats)
    (inputs : Finset (Word (ZMod p) (T + 1))) {learner : Learner (ZMod p) (T + 1)}
    (hlearn : Recovers inputs (realizableFamily (ZMod p) T H d layers repeats) learner) :
    p ^ left + p ^ right + left * right - 1 ≤ inputs.card :=
  (isLeast_unknownCoupling_observations cut).2
    ⟨inputs, rfl, learner, fun {_} hf => hlearn (hsubset hf)⟩

/-- The two constants and the identity are selected by the embeddings of a
one-coordinate Argand. All residual branches in this example have zero
weights, so the result holds at every repeat count. -/
def lastTokenTask {T : ℕ} (mode : Option Bool) (word : Word Bool (T + 1)) : Bool :=
  mode.getD (word (Fin.last T))

def lastTokenFamily (T : ℕ) : Set (Word Bool (T + 1) → Bool) :=
  Set.range lastTokenTask

private def inactiveLayer : Layer 1 1 :=
  ⟨⟨0, 0, 0, 0⟩, ⟨0, 0, 0⟩, 1, ⟨0, 0⟩, ⟨0, 0⟩⟩

private theorem inactiveLayer_forward {P : Type*} [Fintype P] (eps scale : ℝ)
    (geometry : AttentionGeometry P 1) (x : P → CSpace 1) :
    inactiveLayer.forwardWith eps scale geometry x = x := by
  funext i
  simp [inactiveLayer, Layer.forwardWith, sublayer, attentionWith, argandRMS]

private def scalarVector (v : ℝ) : CSpace 1 :=
  WithLp.toLp 2 fun _ => (v : ℂ)

private theorem argandRMS_scalar (v : ℝ) :
    argandRMS 1 (scalarVector v) = scalarVector (v / (|v| + 1)) := by
  have hnorm : ‖scalarVector v‖ = |v| := by
    rw [PiLp.norm_eq_of_L2, Fin.sum_univ_one, Real.sqrt_sq_eq_abs]
    simp [scalarVector, Real.norm_eq_abs]
  rw [argandRMS, hnorm]
  ext i
  rw [PiLp.smul_apply]
  change (↑(Real.sqrt (↑(1 : ℕ) : ℝ) / (|v| + 1)) : ℂ) • (v : ℂ) = ↑(v / (|v| + 1))
  rw [smul_eq_mul, Nat.cast_one, Real.sqrt_one, ← Complex.ofReal_mul]
  congr 1
  simp [div_eq_mul_inv, mul_comm]

private def scalarModel (E : Bool → ℝ) (layers : ℕ) : Model Bool 1 1 layers :=
  ⟨scalarVector ∘ E, fun _ => inactiveLayer, 0, 1⟩

private theorem scalarModel_logits {T : ℕ} (E : Bool → ℝ) (layers repeats : ℕ)
    (word : Word Bool T) (i : Fin T) (token : Bool) :
    let x := E (word i)
    let y := x / (|x| + 1)
    (scalarModel E layers).logits repeats word i token = y / (|y| + 1) * E token / 2 := by
  have hpass : (fun z : Word (CSpace 1) T =>
      (List.ofFn fun _ : Fin layers => inactiveLayer).foldl
        (fun x layer =>
          layer.forwardWith 1 (1 / Real.sqrt (2 * (layers : ℝ) * repeats)) (causalGeometry 0) x)
        z) = id := by
    funext z
    have h (xs : List (Layer 1 1)) (hx : ∀ a ∈ xs, a = inactiveLayer) :
        xs.foldl (fun x layer =>
          layer.forwardWith 1 (1 / Real.sqrt (2 * (layers : ℝ) * repeats)) (causalGeometry 0) x)
          z = z := by
      induction xs with
      | nil => rfl
      | cons a as ih =>
        rw [List.foldl_cons, hx a (by simp), inactiveLayer_forward]
        exact ih fun b hb => hx b (List.mem_cons_of_mem a hb)
    exact h _ (by simp)
  have hhidden : (scalarModel E layers).hidden repeats word i =
      scalarVector ((E (word i) / (|E (word i)| + 1)) /
        (|E (word i) / (|E (word i)| + 1)| + 1)) := by
    simp only [Model.hidden, Model.hiddenWith, scalarModel, hpass, Function.iterate_id, id_eq,
      Function.comp_apply, argandRMS_scalar]
  rw [Model.logits, hhidden]
  simp only [Model.readout, scalarModel, Function.comp_apply, PiLp.inner_apply, Fin.sum_univ_one, scalarVector,
    RCLike.inner_apply, Complex.mul_re, Complex.conj_re, Complex.conj_im,
    Complex.ofReal_re, Complex.ofReal_im, zero_mul, sub_zero]

/-- This class is contained in the full Argand class at one complex embedding
coordinate and four FFN channels, with any number of shared layer passes. -/
theorem lastTokenFamily_subset_realizable (T layers repeats : ℕ) :
    lastTokenFamily T ⊆ realizableFamily Bool T 1 1 layers repeats := by
  rintro _ ⟨mode, rfl⟩
  let E (token : Bool) : ℝ :=
    match mode with
    | none => if token then 1 else -1
    | some a => if token = a then 2 else 1
  refine ⟨scalarModel E layers, by simp [scalarModel], fun word a hne => ?_⟩
  rw [scalarModel_logits, scalarModel_logits]
  cases mode with
  | none =>
    cases hword : word (Fin.last T) <;> cases a <;>
      norm_num [lastTokenTask, hword] at hne <;> norm_num [E, lastTokenTask, hword]
  | some b =>
    cases b <;> cases hword : word (Fin.last T) <;> cases a <;>
      norm_num [lastTokenTask, hword] at hne <;> norm_num [E, lastTokenTask, hword]

private def lastTokenMask (T : ℕ) : Finset (Word Bool (T + 1)) :=
  {fun _ => false, fun _ => true}

private theorem two_le_card_of_recovers_lastTokenFamily {T : ℕ}
    (inputs : Finset (Word Bool (T + 1))) {learner : Learner Bool (T + 1)}
    (hlearn : Recovers inputs (lastTokenFamily T) learner) : 2 ≤ inputs.card := by
  by_contra! h
  obtain ⟨word, hsingle⟩ := Finset.card_le_one_iff_subset_singleton.mp (by omega : inputs.card ≤ 1)
  let b := word (Fin.last T)
  have hagree : observe inputs (lastTokenTask none) = observe inputs (lastTokenTask (some b)) := by
    apply observe_eq_iff_eqOn.mpr
    intro w hw
    rw [show w = word from Finset.mem_singleton.mp (hsingle hw)]
    rfl
  have heq := (exists_learner_iff_injOn inputs _).mp ⟨learner, hlearn⟩
    (Set.mem_range_self none) (Set.mem_range_self (some b)) hagree
  have hne : (!b) ≠ b := by cases b <;> decide
  apply hne
  exact congrFun heq (fun _ => !b)

/-- Two fixed observations are necessary and sufficient for this realized
class. One word with each possible final token attains the bound. -/
theorem isLeast_lastToken_observations (T : ℕ) :
    IsLeast {n : ℕ | ∃ inputs : Finset (Word Bool (T + 1)), inputs.card = n ∧
      ∃ learner, Recovers inputs (lastTokenFamily T) learner} 2 := by
  refine ⟨⟨lastTokenMask T, ?_, (exists_learner_iff_injOn _ _).mpr ?_⟩, ?_⟩
  · apply Finset.card_pair
    intro heq
    exact Bool.false_ne_true (congrFun heq (Fin.last T))
  · rintro _ ⟨mode, rfl⟩ _ ⟨mode', rfl⟩ hobs
    have heq := observe_eq_iff_eqOn.mp hobs
    funext word
    have hmem : (fun _ : Fin (T + 1) => word (Fin.last T)) ∈ lastTokenMask T := by
      cases word (Fin.last T) <;> simp [lastTokenMask]
    have hs := heq hmem
    exact hs
  · rintro _ ⟨inputs, rfl, learner, hlearn⟩
    exact two_le_card_of_recovers_lastTokenFamily inputs hlearn

/-- In particular, one observation cannot identify every function represented
by Argand at this width, regardless of the number of layers and repeats. -/
theorem two_le_observations_of_argand_recovery {T layers repeats : ℕ}
    (inputs : Finset (Word Bool (T + 1))) {learner : Learner Bool (T + 1)}
    (hlearn : Recovers inputs (realizableFamily Bool T 1 1 layers repeats) learner) :
    2 ≤ inputs.card :=
  two_le_card_of_recovers_lastTokenFamily inputs fun {_} hf =>
    hlearn (lastTokenFamily_subset_realizable T layers repeats hf)

example : lastTokenFamily 7 ⊆ realizableFamily Bool 7 1 1 1 1 :=
  lastTokenFamily_subset_realizable 7 1 1

example := isLeast_lastToken_observations 7
example := isLeast_lastToken_observations 15
example := isLeast_lastToken_observations 31

example (word : Word Bool 8) (learner : Learner Bool 8) :
    ¬ Recovers {word} (realizableFamily Bool 7 1 1 2 3) learner := by
  intro h
  have hcard := two_le_observations_of_argand_recovery {word} h
  rw [Finset.card_singleton] at hcard
  omega

end Transformer.Grokking.FiniteRecovery.Argand
