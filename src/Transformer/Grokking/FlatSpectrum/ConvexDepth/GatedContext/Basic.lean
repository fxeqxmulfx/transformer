module

public import Transformer.Grokking.FlatSpectrum.ConvexDepth.ContextMixture

/-!
# Anchored context mixtures with fixed gates

Source: `review/convex/moments.md`, CM46. Each local coefficient row is a
simplex. The input gates, anchor mixture and complete head remain fixed.
The interpolation radius restricts local changes without learning a gate.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexDepth.GatedContext

open scoped BigOperators

variable {G J K X C : Type*} [Fintype G] [Fintype J]

omit [Fintype G] in
/-- Every local mixture has nonnegative coefficients with mass one, CM46. -/
def rows : Set (G → J → ℝ) := {a | ∀ g, a g ∈ ContextMixture.simplex}

omit [Fintype G] in
/-- The whole collection of local mixtures is a convex domain, CM46. -/
theorem rows_convex : Convex ℝ (rows : Set (G → J → ℝ)) := by
  intro a ha b hb t u ht hu htu g
  exact ContextMixture.simplex_convex (ha g) (hb g) ht hu htu

omit [Fintype J] in
/-- Mix the fixed anchor with locally learned rows through fixed input gates,
CM46. The gate and radius are not additional learned factors. -/
noncomputable def weights (gate : X → G → ℝ) (base : J → ℝ) (radius : ℝ)
    (a : G → J → ℝ) (x : X) : J → ℝ :=
  (1 - radius) • base + radius • ∑ g, gate x g • a g

/-- Unit-mass gates and local rows retain total mass one for any radius,
CM46. Positivity is unnecessary for this exact identity. -/
theorem weights_sum (gate : X → G → ℝ) (base : J → ℝ) (radius : ℝ)
    (a : G → J → ℝ) (x : X) (hbase : ∑ j, base j = 1)
    (hgate : ∑ g, gate x g = 1) (ha : ∀ g, ∑ j, a g j = 1) :
    ∑ j, weights gate base radius a x j = 1 := by
  have hs : (∑ j, ∑ g, gate x g * a g j) = 1 := by
    rw [Finset.sum_comm]
    simp_rw [← Finset.mul_sum, ha, mul_one]
    exact hgate
  simp only [weights, Pi.add_apply, Pi.smul_apply, Finset.sum_apply,
    smul_eq_mul, Finset.sum_add_distrib, ← Finset.mul_sum, hbase, hs]
  ring

/-- A radius in [0,1] preserves the full simplex constraint pointwise,
CM46. Gates may overlap; no partition or orthogonal query matrix is assumed. -/
theorem weights_mem (gate : X → G → ℝ) {base : J → ℝ}
    (hbase : base ∈ ContextMixture.simplex) {radius : ℝ} (hzero : 0 ≤ radius)
    (hone : radius ≤ 1) {a : G → J → ℝ} (ha : a ∈ rows) (x : X)
    (hgate : gate x ∈ ContextMixture.simplex) :
    weights gate base radius a x ∈ ContextMixture.simplex := by
  constructor
  · intro j
    simp only [weights, Pi.add_apply, Pi.smul_apply, Finset.sum_apply, smul_eq_mul]
    apply add_nonneg (mul_nonneg (sub_nonneg.mpr hone) (hbase.1 j))
    apply mul_nonneg hzero
    exact Finset.sum_nonneg (fun g _ => mul_nonneg (hgate.1 g) ((ha g).1 j))
  · exact weights_sum gate base radius a x hbase.2 hgate.2 (fun g => (ha g).2)

omit [Fintype J] in
/-- With the anchor and gates fixed, effective weights depend affinely on
all local coefficients, CM46. The affine combination must have mass one. -/
theorem weights_affine (gate : X → G → ℝ) (base : J → ℝ) (radius : ℝ)
    (a b : G → J → ℝ) (x : X) (t u : ℝ) (htu : t + u = 1) :
    weights gate base radius (t • a + u • b) x =
      t • weights gate base radius a x + u • weights gate base radius b x := by
  funext j
  have hs : (∑ g, gate x g * (t * a g j + u * b g j)) =
      t * ∑ g, gate x g * a g j + u * ∑ g, gate x g * b g j := by
    rw [Finset.mul_sum, Finset.mul_sum, ← Finset.sum_add_distrib]
    apply Finset.sum_congr rfl
    intro g _
    ring
  simp only [weights, Pi.add_apply, Pi.smul_apply, Finset.sum_apply, smul_eq_mul]
  rw [hs]
  have hu : u = 1 - t := by linarith
  rw [hu]
  ring

omit [Fintype J] in
/-- The fixed anchor cancels from coefficient changes, CM46. -/
theorem weights_sub (gate : X → G → ℝ) (base : J → ℝ) (radius : ℝ)
    (a b : G → J → ℝ) (x : X) (j : J) :
    weights gate base radius b x j - weights gate base radius a x j =
      radius * ∑ g, gate x g * (b g j - a g j) := by
  simp only [weights, Pi.add_apply, Pi.smul_apply, Finset.sum_apply,
    smul_eq_mul, mul_sub, Finset.sum_sub_distrib]
  ring

/-- Actual context features, using the same bank at each gated query, CM46. -/
noncomputable def features (gate : X → G → ℝ) (bank : J → K → X → ℝ)
    (base : J → ℝ) (radius : ℝ) (a : G → J → ℝ) : K → X → ℝ :=
  fun k x => ContextMixture.features bank (weights gate base radius a x) k x

/-- Every bank preserves an observed row, so the anchored gated model does
too, CM46. Only the three mass-one identities are needed. -/
theorem features_at (gate : X → G → ℝ) (bank : J → K → X → ℝ)
    (source : K → X → ℝ) (base : J → ℝ) (radius : ℝ) (a : G → J → ℝ) (x : X)
    (hbase : ∑ j, base j = 1) (hgate : ∑ g, gate x g = 1)
    (ha : ∀ g, ∑ j, a g j = 1) (hbank : ∀ j k, bank j k x = source k x) (k : K) :
    features gate bank base radius a k x = source k x := by
  exact ContextMixture.features_at bank source (weights gate base radius a x)
    (weights_sum gate base radius a x hbase hgate ha) x hbank k

end Transformer.Grokking.FlatSpectrum.ConvexDepth.GatedContext
