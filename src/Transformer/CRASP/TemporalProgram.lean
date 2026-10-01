/-
# A uniform-attention program for a temporal formula

arXiv:2506.16055v3, Appendix B.2, `thm:TLCl_to_rtfr`.
Boolean subformulas are stored in parallel; separate scratch coordinates
average each comparison's signed integer source contributions.
-/

import Transformer.CRASP.TermAffine
import Transformer.CRASP.FixedSign
import Transformer.CRASP.AttentionKernel

namespace Transformer.CRASP.TemporalProgram

universe u
variable {σ : Type u}

/-- The finite set of formula nodes in the program (Appendix B.2). -/
abbrev Node (φ : Form σ) := {ψ : Form σ // ψ ∈ φ.subformulas}

/-- The node collection is finite even when the token alphabet is infinite (B.2). -/
instance (φ : Form σ) : Finite (Node φ) := by
  classical
  exact Finite.of_injective
    (fun ψ : Node φ => (⟨ψ.val, List.mem_toFinset.mpr ψ.property⟩ :
      {ψ : Form σ // ψ ∈ φ.subformulas.toFinset}))
    (by
      intro ψ χ h
      apply Subtype.ext
      exact congrArg (fun x : {ψ : Form σ // ψ ∈ φ.subformulas.toFinset} => x.val) h)

/-- An enumeration of the program's formula nodes (Appendix B.2). -/
noncomputable instance (φ : Form σ) : Fintype (Node φ) := Fintype.ofFinite _

/-- BOS flag, Boolean memory, and separate comparison scratch coordinates (B.2). -/
abbrev Tag (φ : Form σ) := Option (Node φ ⊕ Node φ)

/-- The number of coordinates of the compiled model (Appendix B.2). -/
noncomputable def dimension (φ : Form σ) : ℕ := Fintype.card (Tag φ)

/-- Coordinate names converted to the transformer's `Fin d` indexing (B.2). -/
noncomputable def coordinate (φ : Form σ) : Tag φ ≃ Fin (dimension φ) :=
  Fintype.equivFin (Tag φ)

/-- The precision chosen from a bound on every value projection (Appendix B.2). -/
abbrev State (φ : Form σ) := Tag φ → Fx (φ.capacity + 2) 0

/-- Named features converted to model coordinates (Appendix B.2). -/
noncomputable def encode (φ : Form σ) (H : State φ) :
    Fin (dimension φ) → Fx (φ.capacity + 2) 0 := fun c => H ((coordinate φ).symm c)

/-- Model coordinates read by their feature names (Appendix B.2). -/
noncomputable def decode (φ : Form σ) (h : Fin (dimension φ) → Fx (φ.capacity + 2) 0) :
    State φ := fun c => h (coordinate φ c)

/-- Encoding preserves every named feature (Appendix B.2). -/
@[simp] theorem decode_encode (φ : Form σ) (H : State φ) : decode φ (encode φ H) = H := by
  funext c
  simp [decode, encode]

open scoped Classical in
/-- Reading a Boolean memory slot, with false outside the program (B.2). -/
noncomputable def read (φ : Form σ) (H : State φ) (ψ : Form σ) : Bool :=
  if h : ψ ∈ φ.subformulas then decide ((H (some (.inl ⟨ψ, h⟩))).m = 1) else false

/-- The BOS flag is a Boolean memory feature (Appendix B.2). -/
def readBos (φ : Form σ) (H : State φ) : Bool := decide ((H none).m = 1)

open scoped Classical in
/-- Reading the negative sign of a comparison scratch coordinate (B.2). -/
noncomputable def readComparison (φ : Form σ) (H : State φ) (ψ : Form σ) : Bool :=
  if h : ψ ∈ φ.subformulas then decide ((H (some (.inr ⟨ψ, h⟩))).val < 0) else false

/-- Boolean evaluation using symbol memory and the freshly computed comparisons.
Source: arXiv:2506.16055v3, Appendix B.2, feed-forward construction. -/
noncomputable def evaluate (φ : Form σ) (H : State φ) : Form σ → Bool
  | .sym a => read φ H (.sym a)
  | ψ@(.lt _ _) => readComparison φ H ψ
  | .neg ψ => !(evaluate φ H ψ)
  | .and ψ χ => evaluate φ H ψ && evaluate φ H χ
  | .pnp _ => false

/-- The signed source contribution for one comparison (Appendix B.2). -/
noncomputable def source (φ : Form σ) (H : State φ) (ψ : Node φ) : ℤ :=
  match ψ.val with
  | .lt t u =>
      if readBos φ H then (t.constant : ℤ) - u.constant
      else (t.countBodies.countP (read φ H) : ℤ) - u.countBodies.countP (read φ H)
  | _ => 0

/-- The uniform value projection, with only scratch slots nonzero (B.2). -/
noncomputable def values (φ : Form σ) (H : State φ) : State φ
  | none | some (.inl _) => 0
  | some (.inr ψ) => Fx.round (φ.capacity + 2) 0 (source φ H ψ : ℝ)

variable [DecidableEq σ]

/-- Initial Boolean features use the one-token input or the BOS empty word (B.2). -/
noncomputable def embedding (φ : Form σ) (a : Option σ) : State φ
  | none => Fx.ofBool φ.capacity a.isNone
  | some (.inl ψ) =>
      let w := a.elim [] fun a => [a]
      Fx.ofBool φ.capacity (ψ.val.sat w w.length)
  | some (.inr _) => 0

/-- The feed-forward step stores the new Boolean values and clears scratch (B.2). -/
noncomputable def feedForward (φ : Form σ) (H : State φ) : State φ
  | none => Fx.ofBool φ.capacity (readBos φ H)
  | some (.inl ψ) => Fx.ofBool φ.capacity (evaluate φ H ψ.val)
  | some (.inr _) => 0

/-- The actual rounded transformer implementing the parallel program.
Source: arXiv:2506.16055v3, Appendix B.2, `thm:TLCl_to_rtfr`. -/
noncomputable def model (φ : Form σ) (k : ℕ) :
    RTfr (Option σ) (φ.capacity + 2) 0 (dimension φ) k where
  E a := encode φ (embedding φ a)
  WQ _ _ _ := 0
  WK _ _ _ := 0
  WV _ h := encode φ (values φ (decode φ h))
  ff _ h := encode φ (feedForward φ (decode φ h))
  Wout h := Fx.ofBool φ.capacity (read φ (decode φ h) φ)

end Transformer.CRASP.TemporalProgram
