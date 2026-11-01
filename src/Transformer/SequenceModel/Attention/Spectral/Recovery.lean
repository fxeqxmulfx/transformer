module

public import Transformer.Grokking.FlatSpectrum.FeatureCodes.PoweredDivision

/-!
# Recover the score-producing powered-division bank

Source: `review/flat_spectrum.md`, *Unified spectral transformer certificate*,
J3. Connected observation patterns with a perfect matching recover each
character kernel under the stated ideal nuclear-norm objective. The recovered
bank itself, rather than a supplied answer table, yields the score margin.
The global minimum hypothesis is not attributed to Schedule-Free AdamW.
-/

@[expose] public section
noncomputable section

namespace Transformer.SequenceModel.Attention.Spectral

open Matrix Grokking.FlatSpectrum
open scoped BigOperators

variable {p : ℕ} [Fact p.Prime] {R : Type*} [Fintype R]

/-- Agreement with observed character features, with multiplicity in the bank retained. -/
def Fits (χ : R → MulChar (ZMod p) ℂ) (k : ℕ) (Ω : Set ((ZMod p)ˣ × (ZMod p)ˣ))
    (X : R → Matrix (ZMod p)ˣ (ZMod p)ˣ ℂ) : Prop :=
  ∀ r, ∀ ij ∈ Ω, X r ij.1 ij.2 = FeatureCodes.powerKernel (χ r) k ij.1 ij.2

/-- The ideal global optimization premise, on the actual observed feature constraints. -/
def Minimizes (χ : R → MulChar (ZMod p) ℂ) (k : ℕ) (Ω : Set ((ZMod p)ˣ × (ZMod p)ˣ))
    (X : R → Matrix (ZMod p)ˣ (ZMod p)ˣ ℂ) : Prop :=
  IsMinOn (fun Y => ∑ r, nuclearNorm (Y r)) {Y | Fits χ k Ω Y} X

/-- J3: all recovered coordinates equal their character computations on
every unit pair, including unobserved pairs. -/
theorem recovered_bank (χ : R → MulChar (ZMod p) ℂ) (k : ℕ)
    (Ω : Set ((ZMod p)ˣ × (ZMod p)ˣ)) (hconn : (patternGraph Ω).Connected)
    (hmatch : ∃ σ : Equiv.Perm (ZMod p)ˣ, ∀ a, (a, σ a) ∈ Ω)
    (X : R → Matrix (ZMod p)ˣ (ZMod p)ˣ ℂ) (hfit : Fits χ k Ω X) (hmin : Minimizes χ k Ω X) :
    X = fun r => FeatureCodes.powerKernel (χ r) k := by
  exact (FeatureCodes.bank_minimum_iff
    (fun r Y hY => FeatureCodes.powerKernel_minimum_iff (χ r) k hconn hmatch Y hY) hfit).mp hmin

/-- J3: recovered feature readouts have a uniform positive classification
margin whenever the positively weighted character bank separates answers. -/
theorem recovered_unit_margin (χ : R → MulChar (ZMod p) ℂ) (k : ℕ)
    {w : R → ℝ} (hw : ∀ r, 0 ≤ w r)
    (hsep : ∀ a b : (ZMod p)ˣ, a ≠ b → ∃ r, 0 < w r ∧ χ r a ≠ χ r b)
    (Ω : Set ((ZMod p)ˣ × (ZMod p)ˣ)) (hconn : (patternGraph Ω).Connected)
    (hmatch : ∃ σ : Equiv.Perm (ZMod p)ˣ, ∀ a, (a, σ a) ∈ Ω)
    (X : R → Matrix (ZMod p)ˣ (ZMod p)ˣ ℂ) (hfit : Fits χ k Ω X) (hmin : Minimizes χ k Ω X) :
    ∃ m > 0, ∀ a b v : (ZMod p)ˣ, v ≠ (a / b) ^ k →
      m ≤ FeatureCodes.score w (FeatureCodes.unitCode χ) (fun r => X r a b) ((a / b) ^ k) -
        FeatureCodes.score w (FeatureCodes.unitCode χ) (fun r => X r a b) v := by
  have hX := recovered_bank χ k Ω hconn hmatch X hfit hmin
  subst X
  obtain ⟨m, hm, hg⟩ := FeatureCodes.exists_positive_margin hw (FeatureCodes.unitCode χ)
    (FeatureCodes.unitCode_unit χ) hsep
  exact ⟨m, hm, fun a b v hv => hg ((a / b) ^ k) v hv.symm⟩

/-- J3: true character kernels satisfy the stated ideal optimization
premises on every admissible observation pattern. -/
theorem true_bank_minimizes (χ : R → MulChar (ZMod p) ℂ) (k : ℕ)
    (Ω : Set ((ZMod p)ˣ × (ZMod p)ˣ)) (hconn : (patternGraph Ω).Connected)
    (hmatch : ∃ σ : Equiv.Perm (ZMod p)ˣ, ∀ a, (a, σ a) ∈ Ω) :
    Minimizes χ k Ω (fun r => FeatureCodes.powerKernel (χ r) k) := by
  exact (FeatureCodes.bank_minimum_iff
    (fun r Y hY => FeatureCodes.powerKernel_minimum_iff (χ r) k hconn hmatch Y hY)
    (fun _ _ _ => rfl)).mpr rfl

/-- Full observation is one concrete admissible pattern for every prime. -/
theorem full_pattern_connected :
    (patternGraph (Set.univ : Set ((ZMod p)ˣ × (ZMod p)ˣ))).Connected :=
  (rigid_iff_connected _).mp (fun _ _ h i j => h (i, j) (Set.mem_univ _))

example := recovered_bank (p := 3) (fun _ : Unit => 1) 2 Set.univ full_pattern_connected
  ⟨Equiv.refl _, fun _ => Set.mem_univ _⟩ (fun _ => FeatureCodes.powerKernel 1 2)
  (fun _ _ _ => rfl) (true_bank_minimizes _ _ _ full_pattern_connected ⟨Equiv.refl _, fun _ => Set.mem_univ _⟩)
example := true_bank_minimizes (p := 3) (fun _ : Unit => 1) 2 Set.univ full_pattern_connected
  ⟨Equiv.refl _, fun _ => Set.mem_univ _⟩

end Transformer.SequenceModel.Attention.Spectral
