module

public import Transformer.Grokking.FiniteRecovery.EqualMomentSymmetry
public import Transformer.Grokking.FiniteRecovery.TransitionFirstNoise
public import Transformer.Grokking.FiniteRecovery.Observations

/-! Three observed cells determine a rank-one phase table. A stock six-layer
Argand can represent it, yet its ordinary gradient training cannot distinguish
the missing answer token from an equally initialized unused vocabulary row. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.Argand.HiddenCorner

open Set MeasureTheory Filter Matrix FlatSpectrum
open scoped NNReal

abbrev Cell := Fin 2 × Fin 2

/-- The two rows of labels are (1,0) and (2,1). Token 2 is hidden. -/
def label (x : Cell) : Fin 64 := ⟨x.1.val+(1-x.2.val),by omega⟩

/-- Give each cell its own input token, disjoint from the answer tokens. -/
def prompt (x : Cell) : Word (Fin 64) 1 := fun _ => ⟨8+2*x.1.val+x.2.val,by omega⟩

def observedCell : Fin 3 → Cell := ![(0,0),(0,1),(1,1)]
def observation (j : Fin 3) : LabeledPosition (Fin 64) :=
  ⟨0,prompt (observedCell j),0,label (observedCell j)⟩

/-- The actual training cells are precisely the three-cell staircase. -/
theorem observed_staircase : range observedCell = staircase 1 := by
  ext ⟨i,j⟩
  simp only [staircase,Set.mem_range,Set.mem_union]
  fin_cases i <;> fin_cases j <;> decide

/-- The character of the label table factors into two unit-modulus vectors,
so the flat-spectrum nuclear norm has a unique minimizing completion. -/
theorem phase_recovered : NuclearRecovers (range observedCell)
    (fun i j => (ZMod.stdAddChar ((label (i,j)).val : ZMod 64) : ℂ)) := by
  let u (i : Fin 2) := (ZMod.stdAddChar (i.val : ZMod 64) : ℂ)
  let v (j : Fin 2) := (ZMod.stdAddChar (1 - (j.val : ZMod 64)) : ℂ)
  have he : (fun i j => (ZMod.stdAddChar ((label (i,j)).val : ZMod 64) : ℂ)) = vecMulVec u v := by
    funext i j
    have hexp : ((label (i,j)).val : ZMod 64) = (i.val : ZMod 64) + (1 - (j.val : ZMod 64)) := by
      fin_cases i <;> fin_cases j <;> decide
    simp [hexp,vecMulVec_apply,u,v,AddChar.map_add_eq_mul]
  rw [observed_staircase,he]
  exact forall_isMinOn_vecMulVec_staircase (fun i j => by simp [vecMulVec_apply,u,v])

/-- A single set of weights in the experimental 4 × 16, six-layer
architecture gives all four correct labels at strict margins. -/
theorem strictly_representable : ∃ w : Weights (Fin 64) 4 16 6, StrictOn
    (fun x => (Model.fromWeights w).logits 1 (prompt x) 0) label := by
  let next (k : Fin 64) : Fin 64 := if k = 9 then 0 else if k = 10 then 2 else 1
  obtain ⟨w,hw⟩ := sixLayer_transition_realization (T := 0) next
  refine ⟨w,fun x a ha => ?_⟩
  have he : transitionTask next (prompt x) = label x := by
    obtain ⟨i,j⟩ := x
    fin_cases i <;> fin_cases j <;> rfl
  simpa only [he,Fin.last_zero] using hw (prompt x) a (fun h => ha (h.trans he))

local notation "Draw" => (ComplexParameter (Fin 64) 4 16 6 → ℂ)

/-- At each fixed step, and also for strict correctness on the entire
tail, the probability is at most one half for stock exchangeable Gaussian
initialization. Any sequence and multiplicities of training observations
are allowed. The finite-time first-hit event may visit both orderings. -/
theorem hidden_success_le_half (stream : ℕ → List (Fin 3)) (rate : ℕ → ℝ) (decay : ℝ) :
    let μ := independentGaussianDraws (fun _ : ComplexParameter (Fin 64) 4 16 6 => (1/50 : ℝ≥0)^2)
    let score (z : Draw) (t : ℕ) := equalMomentScore 1 decay rate (fun k => (stream k).map observation)
      (prompt (1,0)) 0 z t
    let wins z t := ∀ c : Fin 64, c ≠ label (1,0) → score z t c < score z t (label (1,0))
    (∀ t, μ.real {z | wins z t} ≤ 1/2) ∧ μ.real {z | ∀ᶠ t in atTop, wins z t} ≤ 1/2 := by
  intro μ score wins
  let e := Equiv.swap (2 : Fin 64) 3
  have hi (j : Fin 3) : relabelPosition e (observation j) = observation j := by
    fin_cases j <;> rfl
  have hb (t : ℕ) : ((stream t).map observation).map (relabelPosition e) = (stream t).map observation := by
    rw [List.map_map,show relabelPosition e ∘ observation = observation from funext hi]
  obtain ⟨fixed,tail⟩ := equalMoment_swap_bound (H := 4) (d := 16) (L := 6) e 2 3
    (by decide) (by decide) 1 decay rate _ hb (prompt (1,0)) (by rfl) 0 ((1/50 : ℝ≥0)^2)
  have hsub (z : Draw) (t : ℕ) (h : wins z t) : score z t 3 < score z t 2 :=
    h 3 (by decide)
  constructor
  · intro t
    apply le_trans _ (fixed t)
    exact measureReal_mono fun z hz => hsub z t hz
  · apply le_trans _ tail
    exact measureReal_mono (fun z (hz : ∀ᶠ t in atTop, wins z t) =>
      hz.mono fun t ht => hsub z t ht)

example (stream : ℕ → List (Fin 3)) :=
  hidden_success_le_half stream (fun t => 3*wsdRate 20000 t) (1/10)

end Transformer.Grokking.FiniteRecovery.Argand.HiddenCorner
