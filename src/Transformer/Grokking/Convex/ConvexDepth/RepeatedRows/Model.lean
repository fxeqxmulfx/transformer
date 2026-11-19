module

public import Transformer.Grokking.Convex.ConvexDepth.RepeatedRows.Basic
public import Transformer.Grokking.Convex.ConvexDepth.ObservedRows

/-!
# A lossless representation of every unknown decoder row

Source: `review/convex/moments.md`, CM70. Encode every unknown row separately,
retaining shared masses and all explicit exceptions. Decoding recovers the
entire CM65 parameter space and the entire original CM64 feasible domain.
The nonlinear choice of storage layout is not an optimization parameterization.
-/

@[expose] public section

namespace Transformer.Grokking.Convex.ConvexDepth.RepeatedRows

open ObservedRows
open LatentPrograms (Parameters retained)

variable {J S C : Type*} [Fintype C]

/-- Stored masses and every unknown row, CM70. An arbitrary nonuniform row
keeps every value unequal to its baseline in explicit storage. -/
abbrev Stored (labels : J → S → Option C) :=
  (J → ℝ) × ((j : J) → Unknown labels j → Row C ℝ)

/-- Store a complete reduced decoder using arbitrary per-row baselines, CM70. -/
noncomputable def pack (labels : J → S → Option C)
    (baseline : (j : J) → Unknown labels j → ℝ) (r : Reduced labels) : Stored labels :=
  (r.1, fun j s => encode (baseline j s) (r.2 j s))

/-- Recover all logical unknown coordinates and every shared mass, CM70. -/
noncomputable def unpack (labels : J → S → Option C) (r : Stored labels) : Reduced labels :=
  (r.1, fun j s => read (r.2 j s))

/-- Packing is lossless on the entire reduced parameter space, CM70,
without a feasibility, uniformity or positive-mass premise. -/
theorem unpack_pack (labels : J → S → Option C)
    (baseline : (j : J) → Unknown labels j → ℝ) (r : Reduced labels) :
    unpack labels (pack labels baseline r) = r := by
  refine Prod.ext ?_ ?_
  · rfl
  · funext j s
    exact read_encode (baseline j s) (r.2 j s)

/-- Distinct full reduced tensors retain distinct encodings, CM70. -/
theorem pack_injective (labels : J → S → Option C)
    (baseline : (j : J) → Unknown labels j → ℝ) : Function.Injective (pack labels baseline) := by
  intro a b h
  have he := congrArg (unpack labels) h
  rw [unpack_pack, unpack_pack] at he
  exact he

variable [DecidableEq C]

/-- Every reconstructed original coordinate is preserved, CM70. Known rows
retain their exact anchors and unknown rows retain all original raw classes. -/
theorem expand_unpack_pack (labels : J → S → Option C)
    (baseline : (j : J) → Unknown labels j → ℝ) (r : Reduced labels) :
    expand labels (unpack labels (pack labels baseline r)) = expand labels r := by
  have he := unpack_pack labels baseline r
  exact congrArg (expand labels) he

variable [Fintype J] [Fintype S]

/-- The original full-coordinate distance to an arbitrary proposal is
unchanged, CM70. Anchored-copy weights and all unknown terms remain present;
the proposal need not lie in the feasible domain. -/
theorem projection_distance (labels : J → S → Option C)
    (baseline : (j : J) → Unknown labels j → ℝ) (r : Reduced labels)
    (proposal : Parameters J S C) :
    fullSquaredDistance (expand labels (unpack labels (pack labels baseline r))) proposal =
      fullSquaredDistance (expand labels r) proposal := by
  have he := expand_unpack_pack labels baseline r
  exact congrArg (fun q => fullSquaredDistance q proposal) he

variable {X : Type*} [DecidableEq S]

omit [Fintype S] in
/-- Decoded stored feasible points cover exactly the original complete
CM64 domain, CM70. Compatible candidates and nonuniform completions survive.
Storage tags introduce no new mathematical constraint or identifying premise. -/
theorem full_domain_coverage (program : J → X → S) (order : List J)
    (samples : List (X × C)) (δ R : ℝ)
    (baseline : (j : J) → Unknown (dictionary program samples) j → ℝ) :
    (fun r => expand (dictionary program samples) (unpack (dictionary program samples) r)) ''
      ((unpack (dictionary program samples)) ⁻¹'
        domain (dictionary program samples) (retained program order samples) δ R) =
      BoundedLatent.domain program order samples δ R := by
  ext q
  constructor
  · rintro ⟨r, hr, rfl⟩
    exact expand_mem program order samples hr
  · intro hq
    let r := restrict (dictionary program samples) q
    refine ⟨pack (dictionary program samples) baseline r, ?_, ?_⟩
    · change unpack (dictionary program samples) (pack _ baseline r) ∈
        domain (dictionary program samples) (retained program order samples) δ R
      rw [unpack_pack]
      exact restrict_mem program order samples hq
    · change expand (dictionary program samples)
        (unpack (dictionary program samples) (pack _ baseline r)) = q
      rw [unpack_pack]
      exact expand_restrict program order samples hq

example := unpack_pack witnessLabels (fun _ _ => 0) (witnessPoint 4 false)
example := pack_injective witnessLabels (fun _ _ => 0)
example := expand_unpack_pack witnessLabels (fun _ _ => 0) (witnessPoint 4 true)
example := projection_distance witnessLabels (fun _ _ => 0) (witnessPoint 4 false)
  (BoundedLatent.witnessPoint 0 true)
example := full_domain_coverage LatentPrograms.witnessProgram [false, true]
  LatentPrograms.witnessSamples 1 8 (fun _ _ => 0)

end Transformer.Grokking.Convex.ConvexDepth.RepeatedRows
