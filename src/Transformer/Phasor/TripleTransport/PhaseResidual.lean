module

public import Transformer.Phasor.TripleTransport
public import Transformer.Phasor.InputPhaseRead

/-!
# A learned phase residual in triple transport

Source: the “Learned phase residual” experiment in
`experiments/phasor_value_transport/README.md`. The triple gate and its
four values share one contextual backbone. An additional learned projection
reads the same causal input prefix and joins its output before the tied
classifier. The statements apply to every dictionary size and sequence length.
-/

@[expose] public section
namespace Transformer.Phasor.TripleTransport

variable {D m : ℕ}

/-- In the experiment's “Learned phase residual” section, this is the sum of a
single triple-transport mixture and the phase projection of the same inputs.
The four value reads and their gating state are computed from the dense prefix. -/
noncomputable def phaseRead (gate : EucSpace D → Fin 4 → ℝ)
    (values : List (EucSpace D) → EucSpace D × (Fin 4 → EucSpace D)) (P : PhasePrefixRead D m)
    (eps : ℝ) (x : ℕ → EucSpace D) (i : ℕ) : EucSpace D :=
  let v := values (List.ofFn fun j : Fin (i+1) => x j)
  mix gate v.1 v.2 + P.angleResidual (P.inputAngles eps x (i+1))

/-- The zero-projection control in “Learned phase residual” recovers the original
triple mixture for any increment geometry, gates and retrieved values. -/
theorem phaseRead_zero (gate : EucSpace D → Fin 4 → ℝ)
    (values : List (EucSpace D) → EucSpace D × (Fin 4 → EucSpace D)) (P : PhasePrefixRead D m)
    (eps : ℝ) (x : ℕ → EucSpace D) (i : ℕ) :
    phaseRead gate values {P with sineOut := 0, cosineOut := 0} eps x i =
      let v := values (List.ofFn fun j : Fin (i+1) => x j)
      mix gate v.1 v.2 := by
  simp [phaseRead, PhasePrefixRead.angleResidual]

/-- The identical-prefix check in “Learned phase residual” holds for arbitrary
dense inputs to both the phase projection and the four gated values. -/
theorem phaseRead_congr (gate : EucSpace D → Fin 4 → ℝ)
    (values : List (EucSpace D) → EucSpace D × (Fin 4 → EucSpace D)) (P : PhasePrefixRead D m)
    (eps : ℝ) (x y : ℕ → EucSpace D) (i : ℕ) (h : ∀ j ≤ i, x j = y j) :
    phaseRead gate values P eps x i = phaseRead gate values P eps y i := by
  have samePrefix : (fun j : Fin (i+1) => x j) = (fun j : Fin (i+1) => y j) := by
    funext j
    exact h j (Nat.le_of_lt_succ j.isLt)
  have angles : P.inputAngles eps x (i+1) = P.inputAngles eps y (i+1) := by
    funext k
    unfold PhasePrefixRead.inputAngles prefixPhase
    apply Finset.sum_congr rfl
    intro j hj
    change P.increments (GPTMini.rmsNormEps eps (x j)) k = _
    rw [h j (Nat.le_of_lt_succ (Finset.mem_range.mp hj))]
  simp only [phaseRead, samePrefix, angles]

example (gate : EucSpace 64 → Fin 4 → ℝ)
    (values : List (EucSpace 64) → EucSpace 64 × (Fin 4 → EucSpace 64))
    (P : PhasePrefixRead 64 32) (x y : ℕ → EucSpace 64) :
    phaseRead gate values P 1 x 5 =
      phaseRead gate values P 1 (fun j => if j ≤ 5 then x j else y j) 5 :=
  phaseRead_congr gate values P 1 _ _ _ (fun j hj => by simp [hj])

/-- The matched removals in “Learned phase residual” partition the two periodic
projections at any causal prefix. The repeated mixture on each side accounts
for the same dense backbone in both ablated forwards. -/
theorem phaseRead_projection_partition (gate : EucSpace D → Fin 4 → ℝ)
    (values : List (EucSpace D) → EucSpace D × (Fin 4 → EucSpace D)) (P : PhasePrefixRead D m)
    (eps : ℝ) (x : ℕ → EucSpace D) (i : ℕ) :
    phaseRead gate values P eps x i +
        phaseRead gate values {P with sineOut := 0, cosineOut := 0} eps x i =
      phaseRead gate values {P with cosineOut := 0} eps x i +
        phaseRead gate values {P with sineOut := 0} eps x i := by
  simp [phaseRead, PhasePrefixRead.angleResidual, PhasePrefixRead.inputAngles]
  abel

end Transformer.Phasor.TripleTransport
