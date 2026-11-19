module

public import Transformer.Grokking.FlatSpectrum.ConvexDepth.ParameterSharing

/-!
# An identifying prior for previously unseen output classes

Source: `review/convex/moments.md`, CM56. Class coordinates are supplied structure,
not recovered from flatness. Two distinct observed feature values identify an
unknown affine decoder in those coordinates. Slope and intercept are scaled
with a shared margin, so the three actual score parameters enter linearly.
The fitted function reads only two observed pairs and the supplied coordinates.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexDepth.ClassGeometry

variable {C X : Type*}

/-- Actual quadratic class scores, affine in all three head parameters, CM56. -/
def score (e : C → ℝ) (θ : Fin 3 → ℝ) (x : ℝ) (c : C) : ℝ :=
  2 * e c * (θ 0 * x + θ 1) - θ 2 * (e c) ^ 2

/-- Fit from two observed pairs only, with a declared positive margin, CM56.
The identification theorems require distinct feature values. -/
noncomputable def fitted (e : C → ℝ) (x₀ x₁ : ℝ) (c₀ c₁ : C) (g : ℝ) : Fin 3 → ℝ :=
  ![g * ((e c₁ - e c₀) / (x₁ - x₀)),
    g * (e c₀ - ((e c₁ - e c₀) / (x₁ - x₀)) * x₀), g]

/-- The observed anchors are satisfied by the explicit two-pair fit, CM56. -/
theorem fitted_anchors (e : C → ℝ) (x₀ x₁ : ℝ) (c₀ c₁ : C) (g : ℝ)
    (hx : x₀ ≠ x₁) :
    let θ := fitted e x₀ x₁ c₀ c₁ g
    θ 0 * x₀ + θ 1 = θ 2 * e c₀ ∧ θ 0 * x₁ + θ 1 = θ 2 * e c₁ := by
  have hd : x₁ - x₀ ≠ 0 := sub_ne_zero.mpr hx.symm
  constructor
  · dsimp [fitted]
    ring
  · dsimp [fitted]
    field_simp [hd]
    ring

/-- Anchor equations identify the scaled slope and intercept under affine
realizability in the supplied coordinates, CM56. No query label is an input. -/
theorem anchors_identify (e : C → ℝ) (θ : Fin 3 → ℝ) (x₀ x₁ a b : ℝ) (c₀ c₁ : C)
    (hx : x₀ ≠ x₁) (hr₀ : e c₀ = a * x₀ + b) (hr₁ : e c₁ = a * x₁ + b)
    (h₀ : θ 0 * x₀ + θ 1 = θ 2 * e c₀)
    (h₁ : θ 0 * x₁ + θ 1 = θ 2 * e c₁) :
    θ 0 = θ 2 * a ∧ θ 1 = θ 2 * b := by
  rw [hr₀] at h₀
  rw [hr₁] at h₁
  have hp : (θ 0 - θ 2 * a) * (x₁ - x₀) = 0 := by
    linear_combination h₁ - h₀
  have ha : θ 0 = θ 2 * a := by
    rcases mul_eq_zero.mp hp with h | h
    · linarith
    · exact (sub_ne_zero.mpr hx.symm h).elim
  constructor
  · exact ha
  · rw [ha] at h₀
    nlinarith [h₀]

/-- Exact score difference after identifying the center, CM56. This algebraic
lemma alone does not identify a center at an unseen query. -/
theorem score_difference (e : C → ℝ) (θ : Fin 3 → ℝ) (x : ℝ) (y c : C)
    (h : θ 0 * x + θ 1 = θ 2 * e y) :
    score e θ x y - score e θ x c = θ 2 * (e y - e c) ^ 2 := by
  unfold score
  rw [h]
  ring

/-- Injective supplied coordinates and positive margin turn the identified
center into a strict raw-class winner, CM56. Every output class is compared. -/
theorem strict_score (e : C → ℝ) (he : Function.Injective e) (θ : Fin 3 → ℝ)
    (x : ℝ) (y c : C) (hg : 0 < θ 2) (hc : c ≠ y)
    (h : θ 0 * x + θ 1 = θ 2 * e y) : score e θ x c < score e θ x y := by
  have hn : e y - e c ≠ 0 := sub_ne_zero.mpr (fun h => hc (he h).symm)
  have hp := mul_pos hg (sq_pos_of_ne_zero hn)
  rw [← score_difference e θ x y c h] at hp
  linarith

/-- Training-only two-anchor recovery at every input, including unseen raw
classes, CM56. The unknown target is affine in the supplied class geometry;
neither a correct query score nor optimizer convergence is a hypothesis. -/
theorem fitted_prediction (e : C → ℝ) (he : Function.Injective e)
    (f : X → ℝ) (target : X → C) (a b : ℝ)
    (realizable : ∀ x, e (target x) = a * f x + b)
    (x₀ x₁ : X) (hx : f x₀ ≠ f x₁) (g : ℝ) (hg : 0 < g)
    (x : X) (c : C) (hc : c ≠ target x) :
    score e (fitted e (f x₀) (f x₁) (target x₀) (target x₁) g) (f x) c <
    score e (fitted e (f x₀) (f x₁) (target x₀) (target x₁) g) (f x) (target x) := by
  let θ := fitted e (f x₀) (f x₁) (target x₀) (target x₁) g
  obtain ⟨h₀, h₁⟩ := fitted_anchors e (f x₀) (f x₁) (target x₀) (target x₁) g hx
  obtain ⟨ha, hb⟩ := anchors_identify e θ (f x₀) (f x₁) a b (target x₀) (target x₁)
    hx (realizable x₀) (realizable x₁) h₀ h₁
  have hcenter : θ 0 * f x + θ 1 = θ 2 * e (target x) := by
    rw [ha, hb, realizable x]
    ring
  exact strict_score e he θ (f x) (target x) c (by simpa [θ, fitted] using hg) hc hcenter

/-- Sufficient complete column costs when all supplied coordinates lie in
the unit interval in absolute value, CM56. -/
def costs : Fin 3 → ℝ := ![4, 4, 2]

/-- All three sufficient expansion costs are nonnegative, CM56. -/
theorem costs_nonneg (p : Fin 3) : 0 ≤ costs p := by
  fin_cases p <;> norm_num [costs]

/-- A complete budget, margin lower bound, and two observed linear anchor
equations. No unobserved target appears in this feasible domain, CM56. -/
def domain (e : C → ℝ) (x₀ x₁ : ℝ) (c₀ c₁ : C) (R δ : ℝ) : Set (Fin 3 → ℝ) :=
  {θ | θ ∈ ParameterSharing.weightedDomain costs R ∧ δ ≤ θ 2 ∧
    θ 0 * x₀ + θ 1 = θ 2 * e c₀ ∧ θ 0 * x₁ + θ 1 = θ 2 * e c₁}

/-- Training-only anchor constraints preserve convexity, CM56. -/
theorem domain_convex (e : C → ℝ) (x₀ x₁ : ℝ) (c₀ c₁ : C) (R δ : ℝ) :
    Convex ℝ (domain e x₀ x₁ c₀ c₁ R δ) := by
  intro θ hθ η hη a b ha hb hab
  refine ⟨ParameterSharing.weightedDomain_convex costs costs_nonneg R
    hθ.1 hη.1 ha hb hab, ?_, ?_, ?_⟩
  · change δ ≤ a * θ 2 + b * η 2
    have h := add_le_add (mul_le_mul_of_nonneg_left hθ.2.1 ha)
      (mul_le_mul_of_nonneg_left hη.2.1 hb)
    rwa [← add_mul, hab, one_mul] at h
  · change (a * θ 0 + b * η 0) * x₀ + (a * θ 1 + b * η 1) =
      (a * θ 2 + b * η 2) * e c₀
    linear_combination a * hθ.2.2.1 + b * hη.2.2.1
  · change (a * θ 0 + b * η 0) * x₁ + (a * θ 1 + b * η 1) =
      (a * θ 2 + b * η 2) * e c₁
    linear_combination a * hθ.2.2.2 + b * hη.2.2.2

/-- Renaming categorical classes requires transporting their explicitly
supplied coordinates. The fitted parameters and every corresponding score
then agree exactly, CM56; no invariance of an untransported numeric ID is used. -/
theorem relabeling {D : Type*} (π : C ≃ D) (e : C → ℝ) (x₀ x₁ : ℝ)
    (c₀ c₁ : C) (g x : ℝ) (c : C) :
    fitted (e ∘ π.symm) x₀ x₁ (π c₀) (π c₁) g = fitted e x₀ x₁ c₀ c₁ g ∧
    score (e ∘ π.symm) (fitted (e ∘ π.symm) x₀ x₁ (π c₀) (π c₁) g) x (π c) =
      score e (fitted e x₀ x₁ c₀ c₁ g) x c := by
  have h : fitted (e ∘ π.symm) x₀ x₁ (π c₀) (π c₁) g = fitted e x₀ x₁ c₀ c₁ g := by
    simp [fitted]
  refine ⟨h, ?_⟩
  rw [h]
  simp [score]

end Transformer.Grokking.FlatSpectrum.ConvexDepth.ClassGeometry
