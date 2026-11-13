module

public import Transformer.Grokking.FlatSpectrum.CoupledMoments.CompositeDiscovery.Certificate
public import Transformer.Grokking.FlatSpectrum.CoupledMoments.CompositeDiscovery.Witness

/-!
# The precise scope of a certified categorical strip view

Source: `review/flat_spectrum.md`, CM28, fixed categorical views. A native
CM25 view may remove constant rows or columns and map the remaining names to
an interior. Its certificate identifies that interior under the cyclic model.
Lifting identifies a full target represented by the same view, with the stated
constant strips. It does not prove that arbitrary unknown strip answers are
constant. Different proposed views therefore carry different model hypotheses.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.CoupledMoments.CompositeDiscovery

open Schatten

variable {n m C S : Type*}

/-- Lift an interior table through categorical maps and constant strips,
CM28. `none` denotes a strip; `some` is an interior name, not its numeric value. -/
def viewTable (row column : n → Option m) (constant : C) (y : m → m → C) : n → n → C :=
  fun i j => match row i, column j with
    | some a, some b => y a b
    | _, _ => constant

/-- The observed-only interior learner followed by the declared categorical
view, CM28. No unknown full target or its missing answers are arguments. -/
noncomputable def discoverView (q : ℕ) (cell : S → m × m) (labels : S → C)
    (i₀ : m) (c₀ : C) (row column : n → Option m) (constant : C) : Option (n → n → C) :=
  (discover q cell labels i₀ c₀).map (viewTable row column constant)

/-- Under unknown interior realizability and its observable joint condition,
the full target represented by this fixed view is recovered, CM28. Constant
strip structure belongs to the target model; the interior proof does not infer it. -/
theorem discoverView_eq_some {q : ℕ} {cell : S → m × m} {labels : S → C}
    (e : C ≃ ZMod q) (r s : m → ZMod q)
    (hreal : ∀ t, e (labels t) = r (cell t).1 + s (cell t).2) (i₀ : m) (c₀ : C)
    (hK : Cyclic (jointKernel (ZMod q) cell labels i₀ c₀))
    (row column : n → Option m) (constant : C) :
    discoverView q cell labels i₀ c₀ row column constant =
      some (viewTable row column constant (phaseLabels e r s)) := by
  unfold discoverView
  rw [discover_eq_some e r s hreal i₀ c₀ hK]
  rfl

/-- Every compatible interior completion gives the same lifted table within
this fixed categorical view, CM28, even if its row/column maps repeat names. -/
theorem view_completion_eq_target {q : ℕ} {cell : S → m × m} {labels : S → C}
    (e : C ≃ ZMod q) (r s : m → ZMod q)
    (hreal : ∀ t, e (labels t) = r (cell t).1 + s (cell t).2) (i₀ : m) (c₀ : C)
    (hK : Cyclic (jointKernel (ZMod q) cell labels i₀ c₀))
    {y : m → m → C} (hy : Completes q cell labels y)
    (row column : n → Option m) (constant : C) :
    viewTable row column constant y = viewTable row column constant (phaseLabels e r s) := by
  have he := (completes_iff e r s hreal i₀ c₀ hK y).mp hy
  exact congrArg (viewTable row column constant) he

/-- A six-state witness with an additional categorical strip name. The
interior remains the nonconstant repeated-phase witness with held-out cells. -/
theorem six_view_recovers :
    discoverView 6 sixCells sixLabels 0 1 (id : Option (Fin 7) → Option (Fin 7)) id 3 =
      some (viewTable id id 3 (phaseLabels sixCode sixPhase sixPhase)) :=
  discoverView_eq_some sixCode sixPhase sixPhase six_real 0 1 six_cyclic id id 3

example := discoverView_eq_some sixCode sixPhase sixPhase six_real 0 1 six_cyclic
  (id : Option (Fin 7) → Option (Fin 7)) id 3

example := view_completion_eq_target sixCode sixPhase sixPhase six_real 0 1 six_cyclic
  (y := phaseLabels sixCode sixPhase sixPhase) ⟨sixCode, sixPhase, sixPhase, six_real, rfl⟩
  (id : Option (Fin 7) → Option (Fin 7)) id 3

/-- A concrete strip answer and a nonconstant held-out interior answer.
The strip value follows from the stated view model, not from interior samples. -/
theorem six_view_answers :
    (discoverView 6 sixCells sixLabels 0 1 (id : Option (Fin 7) → Option (Fin 7)) id 3).map
      (fun y => (y none (some 2), y (some 2) (some 3))) = some (3, 5) := by
  rw [six_view_recovers]
  decide

end Transformer.Grokking.FlatSpectrum.CoupledMoments.CompositeDiscovery
