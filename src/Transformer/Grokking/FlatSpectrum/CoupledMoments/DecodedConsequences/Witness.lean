module

public import Transformer.Grokking.FlatSpectrum.CoupledMoments.DecodedConsequences.Discovery

/-!
# Decoded identification with a genuinely noncyclic joint module

Source: `review/flat_spectrum.md`, CM29. Fifteen cells of a four-state table
are observed, and every class appears. The lower-right answer follows from
two actual row additions. All decoded cells are certified although the gauged
observed joint module has a nonzero direction fixing a unit class coordinate.
This proves that CM28 cyclicity is not necessary for decoded identification.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.CoupledMoments.DecodedConsequences

open CompositeDiscovery

/-- All categorical cells except the lower-right one, CM29. -/
def fourCells : Fin 15 → Fin 4 × Fin 4 :=
  ![(0, 0), (1, 0), (2, 0), (3, 0), (0, 1), (1, 1), (2, 1), (3, 1),
    (0, 2), (1, 2), (2, 2), (3, 2), (0, 3), (1, 3), (2, 3)]

/-- The fifteen actual observed class names, CM29. -/
def fourLabels : Fin 15 → ZMod 4 := ![0, 2, 0, 2, 1, 3, 1, 3, 2, 0, 2, 0, 3, 1, 3]

/-- Distinct categorical row names share phases, CM29. -/
def fourRows : Fin 4 → ZMod 4 := ![0, 2, 0, 2]

/-- The four column phases of the unknown target witness. -/
def fourColumns : Fin 4 → ZMod 4 := ![0, 1, 2, 3]

/-- A nonconstant target used only to state correctness, not in `choose`. -/
def fourTarget (i j : Fin 4) : ZMod 4 := fourRows i + fourColumns j

/-- Every class is actually seen, while one genuine answer is held out. -/
theorem four_observed : (∀ c : ZMod 4, ∃ t, fourLabels t = c) ∧
    (∀ t, fourCells t ≠ (3, 3)) ∧
    (∀ t, fourTarget (fourCells t).1 (fourCells t).2 = fourLabels t) := by
  decide

/-- Two additions combine the (1,3), (3,0) and (1,0) observed rows. -/
def fourProgram : List (RowOp (Fin 15) (ZMod 4)) := [.add 13 3 1, .add 13 1 (-1)]

/-- Replay, coefficient by coefficient, certifies the unobserved query, CM29. -/
theorem four_query : ∀ v,
    reduceRows fourProgram (observedMatrix fourCells fourLabels) 13 v = queryRow 3 3 (1 : ZMod 4) v := by
  decide

/-- For observed queries reuse their actual rows; the single unobserved query
uses the displayed finite program and an actually observed class name. -/
def fourBank (i j : Fin 4) : List (Candidate (Fin 15) (ZMod 4) (ZMod 4)) :=
  if i = 3 ∧ j = 3 then [⟨fourLabels 13, fourProgram, 13⟩]
  else
    let t : Fin 15 := ⟨(i.val + 4 * j.val) % 15, Nat.mod_lt _ (by decide)⟩
    [⟨fourLabels t, [], t⟩]

/-- Every query has an actually checked coefficient certificate, CM29. No
decoded-target or code-correctness premise enters this finite check. -/
theorem four_coverage : ∀ i j, ∃ p ∈ fourBank i j, Passes fourCells fourLabels i j p := by
  decide

/-- The finite observed-only certifier recovers the whole nonconstant table,
including its missing answer, without cyclicity, CM29. -/
theorem four_recovers : ∀ i j,
    choose fourCells fourLabels i j (fourBank i j) = some (fourTarget i j) :=
  choose_recovers id (fun _ _ h => h) fourRows fourColumns fourTarget
    (fun _ _ => rfl) four_observed.2.2 fourBank four_coverage

/-- The actual normalized fitted joint vector. -/
def fourVector : Fin 4 ⊕ (Fin 4 ⊕ ZMod 4) → ZMod 4 := pack fourRows fourColumns id

/-- A second, nonzero joint direction with class-one coordinate zero. -/
def fourTorsion : Fin 4 ⊕ (Fin 4 ⊕ ZMod 4) → ZMod 4 :=
  pack fourRows ![0, 0, 2, 2] (fun c => if c.val < 2 then 0 else 2)

/-- Both displayed vectors satisfy every actual observation and the two
shift gauges. The second vector is not a multiple of the first. -/
theorem four_members :
    fourVector ∈ jointKernel (ZMod 4) fourCells fourLabels 0 0 ∧
    fourTorsion ∈ jointKernel (ZMod 4) fourCells fourLabels 0 0 := by
  constructor <;> refine ⟨?_, ?_, ?_⟩ <;> decide

/-- The actual observed joint module is noncyclic although every decoded
answer is identified, CM29. All four classes were observed. -/
theorem four_not_cyclic : ¬ Cyclic (jointKernel (ZMod 4) fourCells fourLabels 0 0) := by
  intro hc
  obtain ⟨a, ha⟩ := scalar_of_cyclic hc four_members.1 four_members.2
    ⟨.inr (.inr 1), by decide⟩
  have hc1 := congrFun ha (.inr (.inr 1))
  have hzero : a = 0 := by
    change (0 : ZMod 4) = a * 1 at hc1
    simpa only [mul_one] using hc1.symm
  have hr1 := congrFun ha (.inl 1)
  have hbad : (2 : ZMod 4) = 0 := by
    simpa [hzero, fourTorsion, fourRows, pack] using hr1
  exact (by decide : (2 : ZMod 4) ≠ 0) hbad

/-- A genuine held-out answer coexists with noncyclic observed constraints. -/
theorem four_strict_extension :
    ¬ Cyclic (jointKernel (ZMod 4) fourCells fourLabels 0 0) ∧
    (∀ t, fourCells t ≠ (3, 3)) ∧
    choose fourCells fourLabels 3 3 (fourBank 3 3) = some 1 := by
  refine ⟨four_not_cyclic, four_observed.2.1, ?_⟩
  exact four_recovers 3 3

example : fourTarget 3 3 = 1 :=
  decoded_of_program id (fun _ _ h => h) fourRows fourColumns fourTarget
    (fun _ _ => rfl) four_observed.2.2 fourProgram 13 3 3 1 four_query

example : ∃ p ∈ fourBank 3 3, p.label = (1 : ZMod 4) ∧ Passes fourCells fourLabels 3 3 p :=
  choose_certificate (fourBank 3 3) four_strict_extension.2.2

example : ∃ c, choose fourCells fourLabels 3 3 (fourBank 3 3) = some c :=
  choose_exists (fourBank 3 3) (four_coverage 3 3)

example : fourTarget 3 3 = 1 :=
  choose_sound id (fun _ _ h => h) fourRows fourColumns fourTarget
    (fun _ _ => rfl) four_observed.2.2 3 3 (fourBank 3 3) four_strict_extension.2.2

example : ∀ i j, choose fourCells fourLabels i j (fourBank i j) = some (fourTarget i j) :=
  choose_recovers id (fun _ _ h => h) fourRows fourColumns fourTarget
    (fun _ _ => rfl) four_observed.2.2 fourBank four_coverage

end Transformer.Grokking.FlatSpectrum.CoupledMoments.DecodedConsequences
