/-
# Nonlinear recovery cannot convexify every convex loss

Ergen, Neyshabur, Mehta, arXiv:2211.11052v1, §2–§3.1, equation
`eq:attention_only_obj` permits an arbitrary convex loss.  Affine losses
with any real slope belong to that class.  If a fixed regularizer plus every
such loss of a decoded prediction is convex, then the decoder preserves
convex combinations.  The exact original prediction-budget epigraph proved
nonconvex in the preceding module therefore cannot be represented this way,
even with a nonlinear decoder and a changed regularizer.
-/

import Transformer.Convexifying.Section3_AtomicEpigraph

namespace Transformer.Convexifying

/-- The attainable projected prediction-budget pairs of a parameter model
with feasible set `C`, scalar decoder `decode`, and regularizer `penalty`.
Source: arXiv:2211.11052v1, `eq:attention_only_obj`, abstract candidate
reformulation. -/
def decodedEpigraph {E : Type*} [AddCommGroup E] [Module ℝ E] (C : Set E)
    (decode penalty : E → ℝ) : Set (ℝ × ℝ) :=
  {db | ∃ z ∈ C, decode z = db.1 ∧ penalty z ≤ db.2}

/-- Convexity of every training objective with a scalar affine loss forces
the decoded prediction to preserve convex combinations on the feasible set.
The regularizer is fixed while the slope varies over all real numbers.
Source: `eq:attention_only_obj`, arbitrary convex-loss hypothesis. -/
theorem decoder_jensen_of_all_affine_losses {E : Type*} [AddCommGroup E] [Module ℝ E]
    (C : Set E) (decode penalty : E → ℝ)
    (hall : ∀ M : ℝ, ConvexOn ℝ C (fun z => penalty z + M * decode z))
    {x y : E} (hx : x ∈ C) (hy : y ∈ C)
    {a b : ℝ} (ha : 0 ≤ a) (hb : 0 ≤ b) (hab : a + b = 1) :
    decode (a • x + b • y) = a * decode x + b * decode y := by
  let z := a • x + b • y
  let A := decode z - (a * decode x + b * decode y)
  let B := a * penalty x + b * penalty y - penalty z
  have hbound (M : ℝ) : M * A ≤ B := by
    have h := (hall M).2 hx hy ha hb hab
    change penalty (a • x + b • y) + M * decode (a • x + b • y) ≤
      a * (penalty x + M * decode x) +
        b * (penalty y + M * decode y) at h
    dsimp [A, B, z]
    nlinarith
  by_contra hne
  have hA : A ≠ 0 := sub_ne_zero.mpr hne
  have h := hbound ((B + 1) / A)
  have hcancel : (B + 1) / A * A = B + 1 := by
    field_simp [hA]
  rw [hcancel] at h
  linarith

/-- The prediction-budget epigraph of any model whose objective is convex
for every affine loss is convex, even if its decoder is initially given as
an arbitrary nonlinear function.  Source: `eq:attention_only_obj`, abstract
candidate reformulation. -/
theorem decodedEpigraph_convex_of_all_affine_losses {E : Type*} [AddCommGroup E] [Module ℝ E]
    (C : Set E) (decode penalty : E → ℝ)
    (hall : ∀ M : ℝ, ConvexOn ℝ C (fun z => penalty z + M * decode z)) :
    Convex ℝ (decodedEpigraph C decode penalty) := by
  apply convex_iff_forall_pos.mpr
  intro x hx y hy a b ha hb hab
  change (∃ z ∈ C, decode z = x.1 ∧ penalty z ≤ x.2) at hx
  change (∃ z ∈ C, decode z = y.1 ∧ penalty z ≤ y.2) at hy
  obtain ⟨u, hu, huout, hucost⟩ := hx
  obtain ⟨v, hv, hvout, hvcost⟩ := hy
  have hconv : Convex ℝ C := (hall 0).1
  have hz : a • u + b • v ∈ C :=
    hconv hu hv (le_of_lt ha) (le_of_lt hb) hab
  refine ⟨a • u + b • v, hz, ?_, ?_⟩
  · rw [decoder_jensen_of_all_affine_losses C decode penalty hall
      hu hv (le_of_lt ha) (le_of_lt hb) hab, huout, hvout]
    simp
  · have hpen := (hall 0).2 hu hv (le_of_lt ha) (le_of_lt hb) hab
    have h₁ : 0 ≤ a * (x.2 - penalty u) :=
      mul_nonneg (le_of_lt ha) (sub_nonneg.mpr hucost)
    have h₂ : 0 ≤ b * (y.2 - penalty v) :=
      mul_nonneg (le_of_lt hb) (sub_nonneg.mpr hvcost)
    change penalty (a • u + b • v) ≤ a * x.2 + b * y.2
    have hpen' : penalty (a • u + b • v) ≤
        a * penalty u + b * penalty v := by simpa using hpen
    linarith

/-- **No universal convex training reformulation with nonlinear recovery.**
Let a convex parameter set, any decoder, and any fixed regularizer be given.
If `penalty + M * decode` is convex for every affine loss slope `M`, their
prediction-budget epigraph cannot equal that of the original attention
model with trainable scores and four-matrix weight decay.  Thus nonlinear
recovery or changing the parameter penalty does not repair an exact
all-convex-loss reformulation under this epigraph notion of equivalence.
Source: arXiv:2211.11052v1, §2–§3.1, `eq:attention_only_obj`. -/
theorem no_universal_convex_training_lift {E : Type*} [AddCommGroup E] [Module ℝ E]
    (C : Set E) (decode penalty : E → ℝ)
    (hall : ∀ M : ℝ, ConvexOn ℝ C (fun z => penalty z + M * decode z)) :
    decodedEpigraph C decode penalty ≠ originalMixtureEpigraph := by
  intro hEq
  have hconv := decodedEpigraph_convex_of_all_affine_losses C decode penalty hall
  rw [hEq] at hconv
  exact originalMixtureEpigraph_not_convex hconv

/-- The universal affine-loss convexity hypothesis is satisfiable for a
linear decoder and zero regularizer on a nonempty convex parameter space.
Source: arXiv:2211.11052v1, `eq:attention_only_obj`. -/
example : ∀ M : ℝ, ConvexOn ℝ (Set.univ : Set (Fin 1 → ℝ))
    (fun z => (0 : ℝ) + M * z 0) := by
  intro M
  constructor
  · exact convex_univ
  · intro x _ y _ a b _ _ _
    change (0 : ℝ) + M * (a * x 0 + b * y 0) ≤
      a * (0 + M * x 0) + b * (0 + M * y 0)
    nlinarith

end Transformer.Convexifying
