module

public import Transformer.Grokking.FlatSpectrum.FeatureCodes.Recovery
public import Transformer.Grokking.FlatSpectrum.ReadoutChoice
public import Transformer.Grokking.FlatSpectrum.ConnectedRecovery

/-!
# Powered division through a jointly separating character bank

Source: `review/flat_spectrum.md`, *Joint feature recovery and classification*, item 6.
For every prime and every natural exponent, each complex character kernel of `(a/b)^k`
on nonzero operands has rank one. A connected observation graph containing a perfect
matching recovers every component under the ideal nuclear-norm learner. Joint answer
separation then gives classification. A faithful character exists, and its positive
classification margin can be chosen independently of k. The domain is explicitly the
unit group: zero numerators and actual neural weights are not covered by these theorems.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.FeatureCodes

open Matrix
open scoped BigOperators

variable {p : ℕ} [Fact p.Prime] {R : Type*} [Fintype R]

/-- The answer code formed by a bank of multiplicative characters, source item 6. -/
def unitCode (χ : R → MulChar (ZMod p) ℂ) (a : (ZMod p)ˣ) (r : R) : ℂ := χ r a

omit [Fintype R] in
/-- Every coordinate is on the unit circle, including a nonfaithful character. -/
theorem unitCode_unit (χ : R → MulChar (ZMod p) ℂ) (a : (ZMod p)ˣ) (r : R) :
    Complex.normSq (unitCode χ a r) = 1 := by
  rw [unitCode, Complex.normSq_eq_norm_sq, norm_mulChar_eq_one _ a.ne_zero]
  norm_num

/-- A scalar character readout of powered division on nonzero operands, source item 6. -/
def powerKernel (χ : MulChar (ZMod p) ℂ) (k : ℕ) : Matrix (ZMod p)ˣ (ZMod p)ˣ ℂ :=
  fun a b => χ ((a / b) ^ k : (ZMod p)ˣ)

/-- The **explicit separable computation** of every powered-division feature, source
item 6. Natural exponent zero is included without a special case. -/
theorem powerKernel_eq_outer (χ : MulChar (ZMod p) ℂ) (k : ℕ) :
    powerKernel χ k = vecMulVec (fun a : (ZMod p)ˣ => χ a ^ k)
      (fun b : (ZMod p)ˣ => star (χ b) ^ k) := by
  ext a b
  change χ (↑((a / b) ^ k)) = χ a ^ k * star (χ b) ^ k
  rw [Units.val_pow_eq_pow_val, Units.val_div_eq_div_val, map_pow, div_eq_mul_inv,
    map_mul, ← MulChar.inv_apply', ← MulChar.star_apply', mul_pow]

/-- Every character readout has **exactly rank one** for every natural exponent, source
item 6. The lower bound uses its nonzero entry at the pair of group identities. -/
theorem rank_powerKernel (χ : MulChar (ZMod p) ℂ) (k : ℕ) : (powerKernel χ k).rank = 1 := by
  refine le_antisymm (powerKernel_eq_outer χ k ▸ rank_vecMulVec_le _ _) ?_
  rw [Nat.one_le_iff_ne_zero, Ne, rank_eq_finrank_span_cols, Submodule.finrank_eq_zero,
    Submodule.span_eq_bot]
  intro h
  have := congrFun (h _ ⟨1, rfl⟩) 1
  simp [powerKernel, col] at this

/-- Jointly separating characters classify powered division by multiplying their operand
features. Source item 6: no single character is required to be faithful. -/
theorem power_scores_correct {w : R → ℝ} (hw : ∀ r, 0 ≤ w r)
    (χ : R → MulChar (ZMod p) ℂ)
    (hsep : ∀ a b : (ZMod p)ˣ, a ≠ b → ∃ r, 0 < w r ∧ χ r a ≠ χ r b)
    (k : ℕ) (a b c : (ZMod p)ˣ) (hc : c ≠ (a / b) ^ k) :
    score w (unitCode χ) (fun r => χ r a ^ k * star (χ r b) ^ k) c <
      score w (unitCode χ) (fun r => χ r a ^ k * star (χ r b) ^ k) ((a / b) ^ k) := by
  have hfeature : (fun r => χ r a ^ k * star (χ r b) ^ k) = unitCode χ ((a / b) ^ k) := by
    funext r
    exact (congrFun (congrFun (powerKernel_eq_outer (χ r) k) a) b).symm
  rw [hfeature]
  exact (codeScore_lt_self_iff hw (unitCode χ) (unitCode_unit χ) _ _).mpr
    (hsep _ _ hc.symm)

/-- **Ideal recovery at every exponent**, source item 6. The same connected mask and
perfect matching work for all characters and powers, independently of code separation. -/
theorem powerKernel_minimum_iff (χ : MulChar (ZMod p) ℂ) (k : ℕ)
    {Ω : Set ((ZMod p)ˣ × (ZMod p)ˣ)} (hconn : (patternGraph Ω).Connected)
    (hmatch : ∃ σ : Equiv.Perm (ZMod p)ˣ, ∀ a, (a, σ a) ∈ Ω)
    (X : Matrix (ZMod p)ˣ (ZMod p)ˣ ℂ)
    (hfit : ∀ ij ∈ Ω, X ij.1 ij.2 = powerKernel χ k ij.1 ij.2) :
    IsMinOn nuclearNorm {Y | ∀ ij ∈ Ω, Y ij.1 ij.2 = powerKernel χ k ij.1 ij.2} X ↔
      X = powerKernel χ k := by
  have hn (a : (ZMod p)ˣ) : ‖χ a‖ = 1 := norm_mulChar_eq_one χ a.ne_zero
  have hrec := (forall_isMinOn_vecMulVec_iff_connected
    (a := fun a : (ZMod p)ˣ => χ a ^ k) (b := fun b : (ZMod p)ˣ => star (χ b) ^ k)
    (fun a => by rw [norm_pow, hn, one_pow])
    (fun b => by rw [norm_pow, norm_star, hn, one_pow])).mpr ⟨hconn, hmatch⟩
  rw [powerKernel_eq_outer] at hfit ⊢
  exact hrec X hfit

/-- **Powered-division classification after joint recovery**, source item 6. This combines
the graph criterion, the sum-of-nuclear-norms objective, and joint code separation. -/
theorem power_classification_of_minimum {w : R → ℝ} (hw : ∀ r, 0 ≤ w r)
    (χ : R → MulChar (ZMod p) ℂ)
    (hsep : ∀ a b : (ZMod p)ˣ, a ≠ b → ∃ r, 0 < w r ∧ χ r a ≠ χ r b)
    (k : ℕ) {Ω : Set ((ZMod p)ˣ × (ZMod p)ˣ)}
    (hconn : (patternGraph Ω).Connected)
    (hmatch : ∃ σ : Equiv.Perm (ZMod p)ˣ, ∀ a, (a, σ a) ∈ Ω)
    (X : R → Matrix (ZMod p)ˣ (ZMod p)ˣ ℂ)
    (hfit : ∀ r, ∀ ij ∈ Ω, X r ij.1 ij.2 = powerKernel (χ r) k ij.1 ij.2)
    (hmin : IsMinOn (fun Y => ∑ r, nuclearNorm (Y r))
      {Y : R → Matrix (ZMod p)ˣ (ZMod p)ˣ ℂ |
        ∀ r, ∀ ij ∈ Ω, Y r ij.1 ij.2 = powerKernel (χ r) k ij.1 ij.2} X) :
    ∀ a b c, c ≠ (a / b) ^ k →
      score w (unitCode χ) (fun r => X r a b) c <
        score w (unitCode χ) (fun r => X r a b) ((a / b) ^ k) := by
  apply classification_of_bank_minimum hw (unitCode χ) (unitCode_unit χ) hsep
    (fun a b => (a / b) ^ k) (X := X) (Ω := Ω) _ hfit hmin
  intro r Y hY
  exact powerKernel_minimum_iff (χ r) k hconn hmatch Y hY

/-- **A faithful route with a uniform positive margin exists** for every prime, source
item 6. The same character and margin work for all k. The conclusion uses the explicit
operand computation, rather than assuming that the target code has already been produced. -/
theorem exists_power_division_margin : ∃ χ : MulChar (ZMod p) ℂ, ∃ δ > 0,
    ∀ (k : ℕ) (a b c : (ZMod p)ˣ), c ≠ (a / b) ^ k →
      δ ≤ score (fun _ : Unit => 1) (unitCode (fun _ => χ))
          (fun _ => χ a ^ k * star (χ b) ^ k) ((a / b) ^ k) -
        score (fun _ : Unit => 1) (unitCode (fun _ => χ))
          (fun _ => χ a ^ k * star (χ b) ^ k) c := by
  obtain ⟨χ, hinj⟩ := exists_injective_mulChar (p := p)
  have hsep (a b : (ZMod p)ˣ) (hab : a ≠ b) :
      ∃ r : Unit, 0 < (1 : ℝ) ∧ unitCode (fun _ : Unit => χ) a r ≠
        unitCode (fun _ : Unit => χ) b r :=
    ⟨(), zero_lt_one, fun h => hab (Units.ext (hinj h))⟩
  obtain ⟨δ, hδ, hgap⟩ := exists_positive_margin (fun _ : Unit => zero_le_one)
    (unitCode (fun _ => χ)) (unitCode_unit _) hsep
  refine ⟨χ, δ, hδ, ?_⟩
  intro k a b c hc
  have hfeature : (fun _ : Unit => χ a ^ k * star (χ b) ^ k) =
      unitCode (fun _ => χ) ((a / b) ^ k) := by
    funext r
    exact (congrFun (congrFun (powerKernel_eq_outer χ k) a) b).symm
  rw [hfeature]
  exact hgap _ _ hc.symm

section

local instance : Fact (Nat.Prime 7) := ⟨by decide⟩
local instance : Fact (Nat.Prime 97) := ⟨by decide⟩

example := unitCode_unit (p := 3) (fun _ : Bool => 1) 1 false
example := powerKernel_eq_outer (p := 3) 1 0
example := rank_powerKernel (p := 7) 1 2
example := exists_power_division_margin (p := 2)
example := exists_power_division_margin (p := 97)

example : ∃ χ : MulChar (ZMod 7) ℂ, ∀ (a b c : (ZMod 7)ˣ), c ≠ (a / b) ^ 3 →
    score (fun _ : Unit => 1) (unitCode (fun _ => χ))
        (fun _ => χ a ^ 3 * star (χ b) ^ 3) c <
      score (fun _ : Unit => 1) (unitCode (fun _ => χ))
        (fun _ => χ a ^ 3 * star (χ b) ^ 3) ((a / b) ^ 3) := by
  obtain ⟨χ, hχ⟩ := exists_injective_mulChar (p := 7)
  exact ⟨χ, fun a b c hc => power_scores_correct (fun _ : Unit => zero_le_one)
    (fun _ => χ) (fun a b hab => ⟨(), zero_lt_one,
      fun h => hab (Units.ext (hχ h))⟩) 3 a b c hc⟩

example := powerKernel_minimum_iff (p := 3) 1 2 (Ω := Set.univ)
  ((rigid_iff_connected _).mp (fun _ _ h i j => h (i, j) (Set.mem_univ _)))
  ⟨Equiv.refl _, fun _ => Set.mem_univ _⟩ (powerKernel 1 2) (fun _ _ => rfl)

example : ∃ χ : MulChar (ZMod 3) ℂ, ∀ a b c : (ZMod 3)ˣ, c ≠ (a / b) ^ 2 →
    score (fun _ : Unit => 1) (unitCode (fun _ => χ))
        (fun _ => powerKernel χ 2 a b) c <
      score (fun _ : Unit => 1) (unitCode (fun _ => χ))
        (fun _ => powerKernel χ 2 a b) ((a / b) ^ 2) := by
  obtain ⟨χ, hχ⟩ := exists_injective_mulChar (p := 3)
  have hconn := (rigid_iff_connected (Set.univ : Set ((ZMod 3)ˣ × (ZMod 3)ˣ))).mp
    (fun _ _ h i j => h (i, j) (Set.mem_univ _))
  have hmatch : ∃ σ : Equiv.Perm (ZMod 3)ˣ, ∀ a, (a, σ a) ∈ Set.univ :=
    ⟨Equiv.refl _, fun _ => Set.mem_univ _⟩
  refine ⟨χ, power_classification_of_minimum (fun _ : Unit => zero_le_one)
    (fun _ => χ) (fun a b hab => ⟨(), zero_lt_one, fun h => hab (Units.ext (hχ h))⟩)
    2 hconn hmatch (fun _ => powerKernel χ 2) (fun _ _ _ => rfl) ?_⟩
  exact (bank_minimum_iff (fun _ Y hY => powerKernel_minimum_iff χ 2 hconn hmatch Y hY)
    (fun _ _ _ => rfl)).mpr rfl

end

end Transformer.Grokking.FlatSpectrum.FeatureCodes
