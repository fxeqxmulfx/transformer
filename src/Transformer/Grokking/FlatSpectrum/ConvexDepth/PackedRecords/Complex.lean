module

public import Transformer.Grokking.FlatSpectrum.ConvexDepth.PackedRecords.Observations

/-!
# Complete complex readouts and indexed CE from packed observations

Source: `review/flat_spectrum.md`, CM68. Integer packing changes only stored
observations. Every joint coefficient and every real/imaginary head coordinate
is retained. The statements concern actual complete products, without replacing
them by an arithmetic target or a task-specific decoder.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexDepth.PackedRecords

open scoped BigOperators
open LatentPrograms (Parameters features head)

variable {J X I : Type*} {s c : ℕ} [Fintype J]

/-- Reading the complete vector of precomputed states gives the same full
complex products as reevaluating each program, CM68. No unitary or target
hypothesis is required for this exact input-representation identity. -/
theorem logits_states (U : Matrix (Fin s) (Fin s) ℂ) (program : J → X → Fin s)
    (q : Parameters J (Fin s) (Fin c)) (x : X) :
    headLogits (features U coordinate) (head U 1 q) (fun j => program j x) =
      headLogits (features U program) (head U 1 q) x := by
  funext d
  unfold headLogits
  apply Finset.sum_congr rfl
  intro k hk
  rfl

/-- Each decoded packed record preserves every actual raw-class logit,
CM68, for an arbitrary complete joint tensor and arbitrary complex source. -/
theorem logits_roundtrip (hs : s ≤ UInt16.size) (hc : c ≤ UInt16.size)
    (U : Matrix (Fin s) (Fin s) ℂ) (program : J → X → Fin s)
    (q : Parameters J (Fin s) (Fin c)) (r : X × Fin c) :
    headLogits (features U coordinate) (head U 1 q)
      (decode (encode (states program r)) (encode_valid hs hc (states program r))).1 =
      headLogits (features U program) (head U 1 q) r.1 := by
  rw [decode_encode hs hc]
  exact logits_states U program q r.1

/-- Every indexed batch has exactly the same ordinary CE and denominator,
CM68. The IDs can carry repeated equal observations and any actual short tail.
No floating-point reassociation or optimizer convergence is asserted here. -/
theorem batchLoss_roundtrip (hs : s ≤ UInt16.size) (hc : c ≤ UInt16.size)
    (U : Matrix (Fin s) (Fin s) ℂ) (program : J → X → Fin s)
    (rows : I → X × Fin c) (batch : Finset I) (q : Parameters J (Fin s) (Fin c)) :
    BoundedLatent.batchLoss U coordinate
      (fun i => (decode (encode (states program (rows i)))
        (encode_valid hs hc (states program (rows i)))).1)
      (fun i => (decode (encode (states program (rows i)))
        (encode_valid hs hc (states program (rows i)))).2) batch q =
      BoundedLatent.batchLoss U program (fun i => (rows i).1) (fun i => (rows i).2)
        batch q := by
  unfold BoundedLatent.batchLoss
  congr 1
  apply Finset.sum_congr rfl
  intro i hi
  dsimp only
  rw [logits_roundtrip hs hc, decode_encode hs hc]
  rfl

end Transformer.Grokking.FlatSpectrum.ConvexDepth.PackedRecords
