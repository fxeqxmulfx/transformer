module

public import Transformer.Grokking.FlatSpectrum.ConvexDepth.ClassGeometry

/-!
# Joint program masses and scaled coordinate decoders

Source: `review/flat_spectrum.md`, CM57. Five fixed input features may carry a
quantity, its indicator, two structural roles and a supplied copy coordinate.
The first five learned coordinates are already scaled decoder coefficients;
the sixth is the program mass. No two learned factors are multiplied. Training
anchors and the explicit identity-copy prior are homogeneous linear equations.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexDepth.CoordinatePrograms

open scoped BigOperators

variable {P A : Type*} [Fintype P]

/-- The unscaled linear decoder used only in the semantic realizability prior,
CM57. It is not multiplied by a separately learned mass in the model. -/
def dot (β h : Fin 5 → ℝ) : ℝ := ∑ j, β j * h j

/-- Actual scaled coordinate center, linear in the learned parameters, CM57. -/
def center (θ : Fin 6 → ℝ) (h : Fin 5 → ℝ) : ℝ :=
  ∑ j : Fin 5, θ j.castSucc * h j

/-- Five complete Fourier column costs and the quadratic-class offset cost,
CM57. These account for all sixteen realified coefficients per program/class. -/
def costs (j : Fin 6) : ℝ := if j = 5 then 2 else 4

/-- Every parameter has strictly positive complete expansion cost, CM57. -/
theorem costs_pos (j : Fin 6) : 0 < costs j := by
  unfold costs
  split <;> norm_num

/-- Local complete coefficient charge, including the program mass, CM57. -/
noncomputable def charge (θ : Fin 6 → ℝ) : ℝ := ∑ j, costs j * |θ j|

/-- Convex-combination control of the full local charge, CM57. -/
theorem charge_combination (θ η : Fin 6 → ℝ) (a b : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) :
    charge (fun j => a * θ j + b * η j) ≤ a * charge θ + b * charge η := by
  unfold charge
  rw [Finset.mul_sum, Finset.mul_sum, ← Finset.sum_add_distrib]
  apply Finset.sum_le_sum
  intro j hj
  have h := mul_le_mul_of_nonneg_left (abs_add_le (a * θ j) (b * η j))
    (le_of_lt (costs_pos j))
  simpa [abs_mul, abs_of_nonneg ha, abs_of_nonneg hb, mul_add,
    mul_left_comm] using h

/-- All coefficients vanish when the perspective budget has zero mass,
CM57. This prevents a nominally inactive program from contributing free logits. -/
theorem zero_mass (θ : Fin 6 → ℝ) (B : ℝ)
    (hB : charge θ ≤ B * θ 5) (hg : θ 5 = 0) (j : Fin 6) : θ j = 0 := by
  have hterm : costs j * |θ j| ≤ charge θ :=
    Finset.single_le_sum (fun k _ => mul_nonneg (le_of_lt (costs_pos k)) (abs_nonneg _))
      (Finset.mem_univ j)
  rw [hg, mul_zero] at hB
  have hz : |θ j| = 0 := by nlinarith [abs_nonneg (θ j), costs_pos j]
  exact abs_eq_zero.mp hz

/-- Actual score before the common input-only normalizer, CM57. -/
def score {C : Type*} (e : C → ℝ) (θ : Fin 6 → ℝ) (h : Fin 5 → ℝ) (c : C) : ℝ :=
  2 * e c * center θ h - θ 5 * (e c) ^ 2

/-- Fixed observations, a declared identity-copy prior, perspective budgets,
and a global complete budget. No heldout target appears in this domain, CM57. -/
def domain (rows : P → A → Fin 5 → ℝ) (values : P → A → ℝ)
    (B : P → ℝ) (R δ : ℝ) : Set (P × Fin 6 → ℝ) :=
  {θ | θ ∈ ParameterSharing.weightedDomain (fun p => costs p.2) R ∧
    (∀ p, 0 ≤ θ (p, 5)) ∧ δ ≤ ∑ p, θ (p, 5) ∧
    (∀ p, charge (fun j => θ (p, j)) ≤ B p * θ (p, 5)) ∧
    (∀ p, θ (p, 4) = θ (p, 5)) ∧
    ∀ p a, center (fun j => θ (p, j)) (rows p a) = θ (p, 5) * values p a}

/-- All learned coordinates enter the center affinely, CM57. -/
theorem center_combination (θ η : Fin 6 → ℝ) (h : Fin 5 → ℝ) (a b : ℝ) :
    center (fun j => a * θ j + b * η j) h = a * center θ h + b * center η h := by
  simp only [center, Fin.sum_univ_succ]
  ring

set_option maxHeartbeats 800000 in
/-- Joint masses, scaled decoders and all observed constraints form a convex
domain. The finite input-program family and supplied copy geometry are fixed,
CM57; discrete program discovery is not claimed convex. -/
theorem domain_convex (rows : P → A → Fin 5 → ℝ) (values : P → A → ℝ)
    (B : P → ℝ) (R δ : ℝ) : Convex ℝ (domain rows values B R δ) := by
  intro θ hθ η hη a b ha hb hab
  refine ⟨ParameterSharing.weightedDomain_convex (fun p : P × Fin 6 => costs p.2)
    (fun p => le_of_lt (costs_pos p.2)) R
    hθ.1 hη.1 ha hb hab, ?_, ?_, ?_, ?_, ?_⟩
  · intro p
    exact add_nonneg (mul_nonneg ha (hθ.2.1 p)) (mul_nonneg hb (hη.2.1 p))
  · change δ ≤ ∑ p, (a * θ (p, 5) + b * η (p, 5))
    rw [Finset.sum_add_distrib, ← Finset.mul_sum, ← Finset.mul_sum]
    have h := add_le_add (mul_le_mul_of_nonneg_left hθ.2.2.1 ha)
      (mul_le_mul_of_nonneg_left hη.2.2.1 hb)
    rwa [← add_mul, hab, one_mul] at h
  · intro p
    change charge (fun j => a * θ (p, j) + b * η (p, j)) ≤
      B p * (a * θ (p, 5) + b * η (p, 5))
    calc
      _ ≤ a * charge (fun j => θ (p, j)) + b * charge (fun j => η (p, j)) :=
        charge_combination _ _ a b ha hb
      _ ≤ a * (B p * θ (p, 5)) + b * (B p * η (p, 5)) := add_le_add
        (mul_le_mul_of_nonneg_left (hθ.2.2.2.1 p) ha)
        (mul_le_mul_of_nonneg_left (hη.2.2.2.1 p) hb)
      _ = _ := by ring
  · intro p
    change a * θ (p, 4) + b * η (p, 4) = a * θ (p, 5) + b * η (p, 5)
    rw [hθ.2.2.2.2.1 p, hη.2.2.2.2.1 p]
  · intro p r
    change center (fun j => a * θ (p, j) + b * η (p, j)) (rows p r) =
      (a * θ (p, 5) + b * η (p, 5)) * values p r
    rw [center_combination, hθ.2.2.2.2.2 p r, hη.2.2.2.2.2 p r]
    ring

/-- Explicit copy-prior row, not a heldout target observation, CM57. -/
def copyRow (j : Fin 5) : ℝ := if j = 4 then 1 else 0

/-- Observations augmented with the separately declared copy prior, CM57. -/
def augmentedRows (rows : A → Fin 5 → ℝ) : Option A → Fin 5 → ℝ
  | none => copyRow
  | some a => rows a

/-- The copy coefficient is explicitly known to be one, CM57. -/
def augmentedValues (values : A → ℝ) : Option A → ℝ
  | none => 1
  | some a => values a

/-- A fixed linear decoder distributes over the observable span certificate,
CM57. The coefficients may be signed. -/
theorem dot_combination [Fintype A] (β : Fin 5 → ℝ) (rows : A → Fin 5 → ℝ) (weights : A → ℝ) :
    dot β (fun j => ∑ a, weights a * rows a j) = ∑ a, weights a * dot β (rows a) := by
  unfold dot
  simp only [Finset.mul_sum]
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro a ha
  apply Finset.sum_congr rfl
  intro j hj
  ring

/-- Training anchors and the explicit copy constraint imply all augmented
anchor equations, CM57. -/
theorem augmented_anchors (θ : Fin 6 → ℝ) (rows : A → Fin 5 → ℝ) (values : A → ℝ)
    (hcopy : θ 4 = θ 5) (hobs : ∀ a, center θ (rows a) = θ 5 * values a)
    (a : Option A) : center θ (augmentedRows rows a) = θ 5 * augmentedValues values a := by
  cases a with
  | none => simpa [center, augmentedRows, augmentedValues, copyRow] using hcopy
  | some a => exact hobs a

/-- Unknown decoder realizability extends to the supplied copy row precisely
when its copy coefficient is one, CM57. -/
theorem augmented_realizable (β : Fin 5 → ℝ) (rows : A → Fin 5 → ℝ) (values : A → ℝ)
    (hcopy : β 4 = 1) (hobs : ∀ a, values a = dot β (rows a)) (a : Option A) :
    augmentedValues values a = dot β (augmentedRows rows a) := by
  cases a with
  | none => simpa [dot, augmentedRows, augmentedValues, copyRow] using hcopy.symm
  | some a => exact hobs a

end Transformer.Grokking.FlatSpectrum.ConvexDepth.CoordinatePrograms
