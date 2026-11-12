module

public import Transformer.Grokking.FlatSpectrum.ConvexCausal.Features
public import Transformer.Grokking.FlatSpectrum.ConvexChannel.Basic
public import Transformer.Grokking.FlatSpectrum.ConvexChannel.Representation
public import Transformer.Grokking.FlatSpectrum.CoupledMoments.Schatten.Basic

/-!
# Compatibility with the earlier channel and spectral coordinates

Source: `review/convex_causal.md`, CCa9. Every earlier feasible channel or
coupled-moment table is a conditional probability table in the new interface.
The calibrated loss stays convex on any convex restriction, including the
existing fixed operator/nuclear budgets. Universal capacity uses the larger
prefix domain, since tight spectral budgets can exclude valid task tables.
This does not claim unchanged optimizer values after changing the readout.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexCausal

variable {X C S : Type*} [Fintype C] [Nonempty C] [Fintype S]

/-- Fixed convex structural or spectral restrictions retain calibrated convex CE (CCa9). -/
theorem loss_on_domain {D : Set (X → C → ℝ)} (hD : Convex ℝ D)
    (hrows : ∀ Q ∈ D, Rows Q) {ε : ℝ} (hε : 0 < ε) (hε1 : ε ≤ 1)
    (input : S → X) (label : S → C) : ConvexOn ℝ D (loss ε input label) :=
  (loss_convex hε hε1 input label).subset hrows hD

omit [Nonempty C] in
/-- Old channel outputs are valid joint probabilities before the repaired readout (CCa9). -/
theorem channel_rows {Q K V : Type*} [Fintype K] [Nonempty V]
    {W : ConvexChannel.Channel Q K V C} (hW : ConvexChannel.Feasible W) :
    Rows (fun x : Q × (K → V) => ConvexChannel.logits W x.1 x.2) := by
  constructor
  · intro x c
    exact ConvexChannel.logits_nonneg hW x.1 x.2 c
  · intro x
    exact ConvexChannel.sum_logits hW x.1 x.2

omit [Nonempty C] in
/-- Flattening the pair coordinates retains all coupled-moment probabilities (CCa9). -/
theorem spectral_rows {m n F : Type*} [Fintype m] [Fintype n] [DecidableEq n]
    {φ : F → C → ℂ} {R B : F → ℝ} {P : CoupledMoments.Table m n C}
    (hP : CoupledMoments.Schatten.BudgetFeasible φ R B P) :
    Rows (fun x : m × n => P x.1 x.2) := by
  constructor
  · intro x c
    exact hP.1.1.1 x.1 x.2 c
  · intro x
    exact hP.1.1.2 x.1 x.2

/-- The calibrated loss is convex on the original full spectral-budget domain (CCa9). -/
theorem spectral_loss_convex {m n F : Type*} [Fintype m] [Fintype n] [DecidableEq n]
    (φ : F → C → ℂ) (R B : F → ℝ) {ε : ℝ} (hε : 0 < ε) (hε1 : ε ≤ 1)
    (input : S → m × n) (label : S → C) :
    ConvexOn ℝ {P | CoupledMoments.Schatten.BudgetFeasible φ R B P}
      (fun P => loss ε input label (fun x : m × n => P x.1 x.2)) := by
  refine ⟨CoupledMoments.Schatten.budget_convex φ R B, ?_⟩
  intro P hP Q hQ a b ha hb hab
  exact (loss_convex hε hε1 input label).2 (spectral_rows hP) (spectral_rows hQ) ha hb hab

example := loss_on_domain (X := Bool) (C := Bool) rows_convex (fun Q hQ => hQ)
  (ε := 1 / 4) (by norm_num) (by norm_num) (id : Bool → Bool) Bool.not

example := channel_rows (ConvexChannel.feasible_generator (fun (_ : Bool) (v : Bool) => !v))

example := spectral_rows (CoupledMoments.Schatten.budget_uniform (m := Fin 1) (n := Fin 1)
  (φ := fun (_ : Unit) (c : Bool) => if c then (1 : ℂ) else -1)
  (R := fun _ => 1) (B := fun _ => 1) (by intro f; simp)
  (by intro f; norm_num) (by intro f; norm_num))

example := spectral_loss_convex (m := Fin 1) (n := Fin 1)
  (fun (_ : Unit) (c : Bool) => if c then (1 : ℂ) else -1)
  (fun _ => 1) (fun _ => 1) (ε := 1 / 4) (by norm_num) (by norm_num)
  (fun _ : Unit => (0, 0)) (fun _ => true)

end Transformer.Grokking.FlatSpectrum.ConvexCausal
