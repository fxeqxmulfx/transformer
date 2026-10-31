module

public import Transformer.CRASP.Normalized.Source

/-!
# Compile finite-context source inputs without adding input tokens

Extension of arXiv:2506.16055v3, Appendix F, initial letter/residue formulas;
`review/flat_spectrum.md`, *Depth, width and precision*, R4.
On words of length at most T, residues modulo T+1 name each positive position
exactly. Tags exist only in the compiler's hidden states. The input alphabet
remains the original token vocabulary, and all source phases remain fixed.
-/

@[expose] public section

noncomputable section

namespace Transformer.CRASP.Normalized

open GPTMini SequenceModel.Attention

section Readout

variable {σ : Type*} [DecidableEq σ]

open scoped Classical in
/-- Source final RMSNorm and tied raw embedding scores, with strict correctness
as in the capacity theorems. Ties are rejected. -/
def tiedReadout (cfg : Config) (E : Fin cfg.vocab_size → EucSpace cfg.d_model)
    (eps : ℝ) {T p s : ℕ} (target : Fin cfg.vocab_size) : State cfg T p s → Bool
  | none => false
  | some q => decide (∀ v, v ≠ target →
      inner (𝕜 := ℝ) (rmsNormEps eps (vector q.2)) (E v) <
        inner (𝕜 := ℝ) (rmsNormEps eps (vector q.2)) (E target))

/-- End-to-end output simulation for the normalized rounded source stack.
The input-prefix hypothesis is discharged by `sat_initialFormula` for actual
embedded, rounded tokens; it makes no assumption about the learned output. -/
theorem source_output_correct (cfg : Config) (B : ℕ → BlockParams cfg)
    (E : Fin cfg.vocab_size → EucSpace cfg.d_model) (eps xeps : ℝ)
    {T : ℕ} (p s : ℕ) (angle : ℕ → Fin T → Fin (cfg.head_dim / 2) → ℝ)
    (initial : State cfg T p s → FormP σ) (q : ℕ → State cfg T p s)
    (w : List σ) (n : ℕ) {i : ℕ} (hi : 0 < i)
    (hinit : ∀ j ∈ Finset.Icc 1 i, ∀ a, (initial a).sat w j = decide (q j = a))
    (target : Fin cfg.vocab_size) :
    let L := fun l => sourceLayer cfg (B l) eps xeps p s (angle l)
    (outputFormula (tiedReadout cfg E eps target) (formula L none initial n)).sat w i =
      tiedReadout cfg E eps target (run L none q n i) := by
  apply sat_outputFormula
  intro r
  exact source_stack_sat cfg B eps xeps p s angle initial q w n hi hinit r

end Readout

/-- The source embedding RMSNorm, rounded once to the stored-state format. -/
def embedToken (cfg : Config) (E : Fin cfg.vocab_size → EucSpace cfg.d_model)
    (eps : ℝ) {T : ℕ} (p s : ℕ) (i : Fin T) (a : Fin cfg.vocab_size) :
    State cfg T p s := some (i, fun c => Fx.round p s (rmsNormEps eps (E a) c))

open scoped Classical in
/-- A hidden state's finite position and embedded token are both depth-zero
predicates on the original token word, within the specified context length. -/
def inputFormula (cfg : Config) (E : Fin cfg.vocab_size → EucSpace cfg.d_model)
    (eps : ℝ) {T : ℕ} (p s : ℕ) (r : State cfg T p s) : FormP (Fin cfg.vocab_size) :=
  FormP.any (Finset.univ.toList.map fun z : Fin T × Fin cfg.vocab_size =>
    .and (.mod (T + 1) (z.1.val + 1)) (.and (.sym z.2)
      (FormP.truth (decide (embedToken cfg E eps p s z.1 z.2 = r)))))

/-- Input normalization, fixed-context position tags and rounding require no
counting level; arbitrary real embeddings are permitted. -/
theorem inputFormula_mem (cfg : Config) (E : Fin cfg.vocab_size → EucSpace cfg.d_model)
    (eps : ℝ) {T : ℕ} (p s : ℕ) (r : State cfg T p s) :
    inputFormula cfg E eps p s r ∈ TLClMod (Fin cfg.vocab_size) 0 := by
  apply FormP.any_mem
  rintro φ hφ
  obtain ⟨z, _, rfl⟩ := List.mem_map.mp hφ
  exact FormP.and_mem ⟨rfl, le_rfl⟩
    (FormP.and_mem ⟨rfl, le_rfl⟩ (FormP.truth_mem _ _))

/-- Source tokens use zero-based positions. Logic reads the first token at 1;
the separate zero position is an excluded sentinel, not an extra source token. -/
def input (cfg : Config) (E : Fin cfg.vocab_size → EucSpace cfg.d_model)
    (eps : ℝ) (T p s : ℕ) (w : List (Fin cfg.vocab_size)) (i : ℕ) : State cfg T p s :=
  if h : 0 < i ∧ i ≤ w.length ∧ i ≤ T then
    embedToken cfg E eps p s ⟨i - 1, by omega⟩ (w[i - 1]'(by omega))
  else none

/-- The bounded residue predicate identifies the exact source position and
actual normalized embedding on the unchanged token alphabet. -/
theorem sat_inputFormula (cfg : Config) (E : Fin cfg.vocab_size → EucSpace cfg.d_model)
    (eps : ℝ) {T : ℕ} (p s : ℕ) (w : List (Fin cfg.vocab_size)) {i : ℕ}
    (hi : 0 < i) (hw : i ≤ w.length) (hT : i ≤ T) (r : State cfg T p s) :
    (inputFormula cfg E eps p s r).sat w i = decide (input cfg E eps T p s w i = r) := by
  classical
  have hidx : i - 1 < w.length := by omega
  let tag : Fin T := ⟨i - 1, by omega⟩
  have hin : input cfg E eps T p s w i = embedToken cfg E eps p s tag w[i - 1] := by
    rw [input, dite_eq_left ⟨hi, hw, hT⟩]
  rw [Bool.eq_iff_iff, decide_eq_true_iff, inputFormula, FormP.sat_any, hin]
  constructor
  · rintro ⟨φ, hφ, hs⟩
    obtain ⟨z, _, rfl⟩ := List.mem_map.mp hφ
    have hz : z.1.val + 1 < T + 1 := by have := z.1.isLt; omega
    simp only [FormP.sat, Bool.and_eq_true, FormP.sat_truth, decide_eq_true_eq,
      List.getElem?_eq_getElem hidx, Option.some.injEq,
      Nat.mod_eq_of_lt (show i < T + 1 by omega), Nat.mod_eq_of_lt hz] at hs
    have ht : tag = z.1 := by apply Fin.ext; dsimp [tag]; omega
    rw [ht, hs.2.1]
    exact hs.2.2
  · intro h
    refine ⟨_, List.mem_map_of_mem (a := (tag, w[i - 1]))
      (Finset.mem_toList.mpr (Finset.mem_univ _)), ?_⟩
    have htag : tag.val + 1 = i := by dsimp [tag]; omega
    simp only [FormP.sat, FormP.sat_truth, htag, h,
      List.getElem?_eq_getElem hidx, decide_true, Bool.and_self]

/-- The complete rounded source classifier compiles to counting depth n,
including source embeddings, all n blocks, final RMSNorm and tied readout. -/
theorem source_classifier_mem (cfg : Config) (B : ℕ → BlockParams cfg)
    (E : Fin cfg.vocab_size → EucSpace cfg.d_model) (eps xeps : ℝ)
    {T : ℕ} (p s : ℕ) (angle : ℕ → Fin T → Fin (cfg.head_dim / 2) → ℝ)
    (n : ℕ) (target : Fin cfg.vocab_size) :
    outputFormula (tiedReadout cfg E eps target)
      (formula (fun l => sourceLayer cfg (B l) eps xeps p s (angle l)) none
        (inputFormula cfg E eps p s) n) ∈ TLClMod (Fin cfg.vocab_size) n := by
  apply outputFormula_mem
  intro r
  simpa only [Nat.zero_add] using
    source_stack_mem cfg B eps xeps p s angle _ (inputFormula_mem cfg E eps p s) n r

/-- End-to-end equivalence on every original token word within the context
bound. No annotated input alphabet or assumed activation predicates remain. -/
theorem source_classifier_correct (cfg : Config) (B : ℕ → BlockParams cfg)
    (E : Fin cfg.vocab_size → EucSpace cfg.d_model) (eps xeps : ℝ)
    {T : ℕ} (p s : ℕ) (angle : ℕ → Fin T → Fin (cfg.head_dim / 2) → ℝ)
    (w : List (Fin cfg.vocab_size)) (hw : w.length ≤ T) (n : ℕ) {i : ℕ}
    (hi : 0 < i) (hile : i ≤ w.length) (target : Fin cfg.vocab_size) :
    let L := fun l => sourceLayer cfg (B l) eps xeps p s (angle l)
    (outputFormula (tiedReadout cfg E eps target)
      (formula L none (inputFormula cfg E eps p s) n)).sat w i =
        tiedReadout cfg E eps target (run L none (input cfg E eps T p s w) n i) := by
  apply source_output_correct cfg B E eps xeps p s angle _ _ w n hi
  intro j hj r
  have hjb := Finset.mem_Icc.mp hj
  exact sat_inputFormula cfg E eps p s w (by omega) (hjb.2.trans hile)
    ((hjb.2.trans hile).trans hw) r

/-- Bounded-prefix assumptions have a nonempty, multi-token source witness. -/
example : ([0, 1, 2] : List (Fin witnessConfig.vocab_size)).length ≤ 8 ∧
    0 < 3 ∧ 3 ≤ ([0, 1, 2] : List (Fin witnessConfig.vocab_size)).length ∧
    3 ≤ 8 := by decide

end Transformer.CRASP.Normalized
