module

public import Transformer.Grokking.FlatSpectrum.ConvexDepth.GatedContext

/-!
# Convex raw-margin constraints on actual context logits

Source: `review/convex/moments.md`, CM47. The reference scores, selected raw
labels and retention fraction are fixed. The candidate keeps a fraction of
every reference margin. This is a constraint on actual complete-head logits,
not a declaration that the reference labels are correct.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexDepth.MarginGuard

variable {X C : Type*}

/-- The selected raw label's score minus a competitor's score, CM47. -/
def gap (z : X → C → ℝ) (label : X → C) (x : X) (c : C) : ℝ :=
  z x (label x) - z x c

/-- All query/class halfspaces, with no target-label input, CM47. -/
def region (reference : X → C → ℝ) (label : X → C) (fraction : ℝ) :
    Set (X → C → ℝ) :=
  {z | ∀ x c, fraction * gap reference label x c ≤ gap z label x c}

/-- Fixed margin halfspaces form a convex set, for any retention fraction,
CM47. Positivity is needed later for strict decision preservation. -/
theorem region_convex (reference : X → C → ℝ) (label : X → C) (fraction : ℝ) :
    Convex ℝ (region reference label fraction) := by
  intro a ha b hb t u ht hu htu x c
  have h := add_le_add (mul_le_mul_of_nonneg_left (ha x c) ht)
    (mul_le_mul_of_nonneg_left (hb x c) hu)
  calc
    fraction * gap reference label x c =
        t * (fraction * gap reference label x c) +
          u * (fraction * gap reference label x c) := by rw [← add_mul, htu, one_mul]
    _ ≤ t * gap a label x c + u * gap b label x c := h
    _ = gap (t • a + u • b) label x c := by
      simp only [gap, Pi.add_apply, Pi.smul_apply, smul_eq_mul]
      ring

/-- A maximizing reference is feasible whenever the retained fraction is
at most one, CM47. This gives a concrete feasible point, not an optimizer. -/
theorem reference_mem (reference : X → C → ℝ) (label : X → C) {fraction : ℝ}
    (hone : fraction ≤ 1) (hmax : ∀ x c, reference x c ≤ reference x (label x)) :
    reference ∈ region reference label fraction := by
  intro x c
  have hgap : 0 ≤ gap reference label x c := sub_nonneg.mpr (hmax x c)
  have h := mul_nonneg (sub_nonneg.mpr hone) hgap
  nlinarith

variable {G J K : Type*} [Fintype G] [Fintype J] [Fintype K]

/-- Intersect CM46's local simplexes with actual raw-logit halfspaces,
CM47. The complete head and all gates/feature maps remain fixed. -/
def domain (gate : X → G → ℝ) (bank : J → K → X → ℝ) (base : J → ℝ)
    (radius : ℝ) (W : K → C → ℝ) (reference : G → J → ℝ)
    (label : X → C) (fraction : ℝ) : Set (G → J → ℝ) :=
  {a | a ∈ GatedContext.rows ∧
    headLogits (GatedContext.features gate bank base radius a) W ∈
      region (headLogits (GatedContext.features gate bank base radius reference) W)
        label fraction}

/-- CM46's actual affine logits pull the halfspaces back to a convex
coefficient domain, CM47. There is no learned product in this step. -/
theorem domain_convex (gate : X → G → ℝ) (bank : J → K → X → ℝ)
    (base : J → ℝ) (radius : ℝ) (W : K → C → ℝ) (reference : G → J → ℝ)
    (label : X → C) (fraction : ℝ) :
    Convex ℝ (domain gate bank base radius W reference label fraction) := by
  intro a ha b hb t u ht hu htu
  refine ⟨GatedContext.rows_convex ha.1 hb.1 ht hu htu, ?_⟩
  have h := (region_convex
    (headLogits (GatedContext.features gate bank base radius reference) W)
    label fraction) ha.2 hb.2 ht hu htu
  have heq : headLogits (GatedContext.features gate bank base radius (t • a + u • b)) W =
      t • headLogits (GatedContext.features gate bank base radius a) W +
        u • headLogits (GatedContext.features gate bank base radius b) W := by
    funext x
    exact GatedContext.logits_affine gate bank base radius a b W x t u htu
  change headLogits (GatedContext.features gate bank base radius (t • a + u • b)) W ∈ _
  rw [heq]
  exact h

/-- A feasible CM46 reference supplies a feasible guarded tensor when its
fixed labels maximize its actual scores, CM47. -/
theorem reference_mem_domain (gate : X → G → ℝ) (bank : J → K → X → ℝ)
    (base : J → ℝ) (radius : ℝ) (W : K → C → ℝ) (reference : G → J → ℝ)
    (label : X → C) {fraction : ℝ} (hone : fraction ≤ 1)
    (hrows : reference ∈ GatedContext.rows)
    (hmax : ∀ x c,
      headLogits (GatedContext.features gate bank base radius reference) W x c ≤
      headLogits (GatedContext.features gate bank base radius reference) W x (label x)) :
    reference ∈ domain gate bank base radius W reference label fraction := by
  exact ⟨hrows, reference_mem _ label hone hmax⟩

/-- Ordinary complete-vocabulary CE stays convex under the margin
constraints, CM47. No CE surrogate or target-dependent constraint is used. -/
theorem ce_convex [Fintype C] (gate : X → G → ℝ) (bank : J → K → X → ℝ)
    (base : J → ℝ) (radius : ℝ) (W : K → C → ℝ) (reference : G → J → ℝ)
    (label : X → C) (fraction : ℝ) (x : X) (y : C) :
    ConvexOn ℝ (domain gate bank base radius W reference label fraction)
      (fun a => Landscape.Neural.crossEntropy
        (headLogits (GatedContext.features gate bank base radius a) W x) y) := by
  exact (GatedContext.ce_convex gate bank base radius W x y).subset
    (fun _ ha => ha.1) (domain_convex gate bank base radius W reference label fraction)

/-- A separately fixed diagnostic bank may define the ordinary CE objective
while constraints use actual inference logits, CM47. For example, source
exclusion changes the fixed diagnostic rows, not the learned factors. -/
theorem diagnostic_ce_convex [Fintype C] (gate : X → G → ℝ)
    (bank lossBank : J → K → X → ℝ) (base : J → ℝ) (radius : ℝ)
    (W : K → C → ℝ) (reference : G → J → ℝ) (label : X → C)
    (fraction : ℝ) (x : X) (y : C) :
    ConvexOn ℝ (domain gate bank base radius W reference label fraction)
      (fun a => Landscape.Neural.crossEntropy
        (headLogits (GatedContext.features gate lossBank base radius a) W x) y) := by
  exact (GatedContext.ce_convex gate lossBank base radius W x y).subset
    (fun _ ha => ha.1) (domain_convex gate bank base radius W reference label fraction)

end Transformer.Grokking.FlatSpectrum.ConvexDepth.MarginGuard
