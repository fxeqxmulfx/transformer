import Mathlib.Data.Nat.Find
import Mathlib.Data.List.OfFn
import Mathlib.Tactic

/-!
# Description complexity and sample-level memorization

arXiv:2505.24832v3, Section 2.2, Definitions 2 and 3. A decoder may fail
to describe a string. The paper's minimum is then undefined, represented
by `none`, rather than by a fabricated zero. Subtraction is signed.
-/

namespace Transformer.Memorization

/-- Section 2.2, Definition 2: programs, data, and encoded models are bits. -/
abbrev BitString := List Bool

/-- Section 2.2, Definition 2: evaluation with a list of side-information
strings; `none` means that the program does not produce an output. -/
abbrev Decoder := BitString → List BitString → Option BitString

/-- Section 2.2, Definition 2: a description of exactly `k` bits. -/
def HasDescription (decode : Decoder) (x : BitString) (side : List BitString)
    (k : ℕ) : Prop := ∃ p, p.length = k ∧ decode p side = some x

/-- Section 2.2, Definition 2: the actual minimum program length, when a
program exists. No computability of this minimization is asserted. -/
noncomputable def descriptionComplexity (decode : Decoder) (x : BitString)
    (side : List BitString) : Option ℕ := by
  classical
  exact if h : ∃ k, HasDescription decode x side k then some (Nat.find h) else none

/-- Section 2.2, Definition 2: every decoding program bounds the minimum. -/
theorem descriptionComplexity_le_program (decode : Decoder) (x p : BitString)
    (side : List BitString) (hp : decode p side = some x) :
    ∃ k, descriptionComplexity decode x side = some k ∧ k ≤ p.length := by
  classical
  have hx : ∃ k, HasDescription decode x side k := ⟨p.length, p, rfl, hp⟩
  exact ⟨Nat.find hx, by simp [descriptionComplexity, hx],
    Nat.find_min' hx (show HasDescription decode x side p.length from ⟨p, rfl, hp⟩)⟩

/-- Section 2.2: the empty program really describes the empty string
under the literal decoder; the decoding hypothesis is satisfiable. -/
example : (fun p : BitString => fun _ : List BitString => some p) [] [] = some [] := rfl

/-- Section 2.2, Definition 2: a finite complexity is attained by a real
program, not merely a numerical lower bound. -/
theorem descriptionComplexity_attained (decode : Decoder) (x : BitString)
    (side : List BitString) (k : ℕ) (hk : descriptionComplexity decode x side = some k) :
    ∃ p, decode p side = some x ∧ p.length = k := by
  classical
  unfold descriptionComplexity at hk
  split_ifs at hk with hx
  simp only [Option.some.injEq] at hk
  obtain ⟨p, hp, hd⟩ := Nat.find_spec hx
  exact ⟨p, hd, hp.trans hk⟩

/-- Section 2.2: an attained complexity of zero is possible. -/
example : descriptionComplexity (fun p _ => some p) [] [] = some 0 := by
  simp [descriptionComplexity, HasDescription, Nat.find_eq_zero]

/-- Section 2.2, Definition 2: a supplied minimal description determines
the complexity exactly. -/
theorem descriptionComplexity_eq_minimal (decode : Decoder) (x p : BitString)
    (side : List BitString) (hp : decode p side = some x)
    (hmin : ∀ q, decode q side = some x → p.length ≤ q.length) :
    descriptionComplexity decode x side = some p.length := by
  obtain ⟨k, hk, hle⟩ := descriptionComplexity_le_program decode x p side hp
  obtain ⟨q, hq, hlen⟩ := descriptionComplexity_attained decode x side k hk
  have hge : p.length ≤ k := by simpa [hlen] using hmin q hq
  simpa [le_antisymm hle hge] using hk

/-- Section 2.2: the decoding and minimality hypotheses hold for an empty
literal program. -/
example : (fun p : BitString => fun _ : List BitString => some p) [] [] = some [] ∧
    ∀ q : BitString, some q = some [] → ([] : BitString).length ≤ q.length := by
  exact ⟨rfl, fun _ _ => Nat.zero_le _⟩

/-- Section 2.2, Definition 2: an empty set of descriptions has no minimum. -/
theorem descriptionComplexity_none (decode : Decoder) (x : BitString)
    (side : List BitString) (h : ∀ p, decode p side ≠ some x) :
    descriptionComplexity decode x side = none := by
  classical
  have hn : ¬∃ k, HasDescription decode x side k := by
    rintro ⟨k, p, hp, hd⟩
    exact h p hd
  simp [descriptionComplexity, hn]

/-- Section 2.2: an always-failing interpreter witnesses the hypothesis. -/
example : ∀ p : BitString, (none : Option BitString) ≠ some p := by simp

/-- Section 2.2, Definition 3, corrected argument order: information
about `x` in the trained model is `K(x)-K(x | trained)`. The source first
writes `Iᵏ(trained,x)` and later `memᵏ(x,trained)`; these are not generally
equal for an arbitrary decoder. This definition follows the data-first
order used by Section 2.1 and the compression estimator in Section 2.3. -/
noncomputable def kolmogorovMem (decode : Decoder) (x trained : BitString) : Option ℤ := do
  let total ← descriptionComplexity decode x []
  let conditional ← descriptionComplexity decode x [trained]
  pure ((total : ℤ) - conditional)

/-- Section 2.2, Definition 3: unintended sample-level memorization. -/
noncomputable def kolmogorovUnintended (decode : Decoder)
    (x reference trained : BitString) : Option ℤ := do
  let baseline ← descriptionComplexity decode x [reference]
  let joint ← descriptionComplexity decode x [reference, trained]
  pure ((baseline : ℤ) - joint)

/-- Section 2.2, Definition 3: signed intended memorization. -/
noncomputable def kolmogorovGeneralization (decode : Decoder)
    (x reference trained : BitString) : Option ℤ := do
  let total ← kolmogorovMem decode x trained
  let unintended ← kolmogorovUnintended decode x reference trained
  pure (total - unintended)

end Transformer.Memorization
