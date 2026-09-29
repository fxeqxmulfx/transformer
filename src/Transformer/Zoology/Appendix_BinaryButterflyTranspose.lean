/-
# Transposing binary butterfly factors

Arora et al., arXiv:2312.04927v1, Appendix `def: kaleidoscope`.
Transposition retains the main diagonal and reads each off-diagonal
coefficient at the partner coordinate. The partner is an involution on
the full row-major layout, so the adjoint identity is exact.
-/

import Transformer.Zoology.Appendix_BinaryButterflyProgram

open scoped BigOperators

namespace Transformer.Zoology

/-- Row-major scalar coordinates of a power-of-two sequence layout.
Source: Appendix `prop: butterfly-hyena`, row-major n × d input. -/
abbrev ButterflyGrid (p k : ℕ) := Fin (butterflyWidth p) × Fin (butterflyWidth k)

/-- The row or feature partner paired by one block factor.
Source: Appendix `def: butterfly`, two-half block pairing. -/
def BinaryButterflyStage.partner {p k : ℕ} (s : BinaryButterflyStage p k)
    (c : ButterflyGrid p k) : ButterflyGrid p k :=
  match s.axis with
  | .inl t => (butterflyToggleIndex p t.val c.1, c.2)
  | .inr t => (c.1, butterflyToggleIndex k t.val c.2)

/-- The full-layout partner is an involution.
Source: Appendix `def: butterfly`, paired off-diagonal entries. -/
theorem BinaryButterflyStage.partner_involutive {p k : ℕ} (s : BinaryButterflyStage p k) :
    Function.Involutive s.partner := by
  intro c
  cases ha : s.axis with
  | inl t =>
      simp only [BinaryButterflyStage.partner, ha,
        butterflyToggleIndex_involutive p t.val c.1, Prod.eta]
  | inr t =>
      simp only [BinaryButterflyStage.partner, ha,
        butterflyToggleIndex_involutive k t.val c.2, Prod.eta]

/-- Factor action in scalar-coordinate form.
Source: Appendix `eq: butterfly-split`, paired coordinate contribution. -/
theorem BinaryButterflyStage.apply_partner {p k : ℕ} (s : BinaryButterflyStage p k)
    (u : RealSequence (butterflyWidth p) (butterflyWidth k)) (c : ButterflyGrid p k) :
    s.apply u c.1 c.2 = s.main c.1 c.2 * u c.1 c.2 +
      s.off c.1 c.2 * u (s.partner c).1 (s.partner c).2 := by
  cases ha : s.axis <;> simp [BinaryButterflyStage.apply, BinaryButterflyStage.partner, ha]

/-- The transposed factor exchanges the paired off-diagonal coefficients.
Source: Appendix `def: kaleidoscope`, real transpose in BB*. -/
def BinaryButterflyStage.transpose {p k : ℕ} (s : BinaryButterflyStage p k) :
    BinaryButterflyStage p k := {
  axis := s.axis
  main := s.main
  off := fun i q => s.off (s.partner (i, q)).1 (s.partner (i, q)).2
}

/-- Transposition retains the same partner permutation.
Source: Appendix `def: kaleidoscope`, transpose of a block factor. -/
theorem BinaryButterflyStage.transpose_partner {p k : ℕ} (s : BinaryButterflyStage p k) :
    s.transpose.partner = s.partner := rfl

/-- Ordinary Euclidean inner product on the row-major scalar entries.
Source: Appendix `def: kaleidoscope`, real matrix transpose. -/
def butterflyGridDot {p k : ℕ}
    (u v : RealSequence (butterflyWidth p) (butterflyWidth k)) : ℝ :=
  ∑ c : ButterflyGrid p k, u c.1 c.2 * v c.1 c.2

/-- The compiled transpose really is the adjoint of the original factor.
Source: Appendix `def: kaleidoscope`, meaning of BB*; no symmetry of the
factor's coefficients is assumed. -/
theorem BinaryButterflyStage.transpose_adjoint {p k : ℕ} (s : BinaryButterflyStage p k)
    (u v : RealSequence (butterflyWidth p) (butterflyWidth k)) :
    butterflyGridDot (s.apply u) v = butterflyGridDot u (s.transpose.apply v) := by
  let e : ButterflyGrid p k ≃ ButterflyGrid p k :=
    Function.Involutive.toPerm s.partner s.partner_involutive
  have hoff := e.sum_comp (fun c =>
    u c.1 c.2 * (s.off (s.partner c).1 (s.partner c).2 *
      v (s.partner c).1 (s.partner c).2))
  dsimp [e] at hoff
  have hpair : ∀ c, s.partner (s.partner c) = c := s.partner_involutive
  simp only [hpair] at hoff
  unfold butterflyGridDot
  simp only [BinaryButterflyStage.apply_partner,
    BinaryButterflyStage.transpose_partner]
  simp only [BinaryButterflyStage.transpose, Prod.eta, add_mul, mul_add,
    Finset.sum_add_distrib]
  congr 1
  · apply Finset.sum_congr rfl
    intro c _
    ring
  · rw [← hoff]
    apply Finset.sum_congr rfl
    intro c _
    ring

end Transformer.Zoology
