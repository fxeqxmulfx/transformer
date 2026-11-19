module

public import Transformer.Grokking.Convex.ConvexDepth.GatedContext.Basic

/-!
# Complete products and bounds for anchored local contexts

Source: `review/convex/moments.md`, CM46. Actual logits use every original
real/imaginary coefficient. Fixed gates and a radius in [0,1] retain the
source feature bound; coefficient changes have an explicit local bound.
-/

@[expose] public section

namespace Transformer.Grokking.Convex.ConvexDepth.GatedContext

open scoped BigOperators

variable {G J K X C : Type*} [Fintype G] [Fintype J]

/-- The same source coordinate bound survives overlapping gates and a
fixed anchor, CM46. No normalization of the query matrix is introduced. -/
theorem features_bound (gate : X → G → ℝ) (bank : J → K → X → ℝ)
    {base : J → ℝ} (hbase : base ∈ ContextMixture.simplex) {radius : ℝ}
    (hzero : 0 ≤ radius) (hone : radius ≤ 1) {a : G → J → ℝ} (ha : a ∈ rows)
    (x : X) (hgate : gate x ∈ ContextMixture.simplex) {M : ℝ}
    (hbank : ∀ j k, |bank j k x| ≤ M) (k : K) :
    |features gate bank base radius a k x| ≤ M := by
  exact ContextMixture.features_bound bank
    (weights_mem gate hbase hzero hone ha x hgate) x hbank k

/-- Stochastic contexts and fixed confidence scales over a unitary complex
source retain the same bound under anchored local mixing, CM46. -/
theorem complex_features_bound {I : Type*} [Fintype I] [DecidableEq I]
    (U : Matrix I I ℂ) (hU : U ∈ Matrix.unitaryGroup I ℂ)
    (gate : X → G → ℝ) (w : J → X → I → ℝ) (scale : J → X → ℝ)
    {base : J → ℝ} (hbase : base ∈ ContextMixture.simplex) {radius : ℝ}
    (hzero : 0 ≤ radius) (hone : radius ≤ 1) {a : G → J → ℝ} (ha : a ∈ rows)
    (x : X) (hgate : gate x ∈ ContextMixture.simplex)
    (hw : ∀ j i, 0 ≤ w j x i) (hs : ∀ j, ∑ i, w j x i = 1)
    (hszero : ∀ j, 0 ≤ scale j x) (hsone : ∀ j, scale j x ≤ 1) (k : I × Bool) :
    |features gate (fun j => ContextScale.features (scale j)
      (ContextInterpolation.features (w j) (complexFeatures U))) base radius a k x| ≤ 1 := by
  exact ContextMixture.complex_features_bound U hU w scale
    (weights_mem gate hbase hzero hone ha x hgate) x hw hs hszero hsone k

variable [Fintype K]

/-- Expanding the complete finite products commutes a fixed gate with
the actual local logits, CM46. -/
theorem local_logits (gate : X → G → ℝ) (bank : J → K → X → ℝ)
    (a : G → J → ℝ) (W : K → C → ℝ) (x : X) :
    headLogits (ContextMixture.features bank (∑ g, gate x g • a g)) W x =
      ∑ g, gate x g • headLogits (ContextMixture.features bank (a g)) W x := by
  simp only [ContextMixture.logits]
  funext c
  simp only [Finset.sum_apply, Pi.smul_apply, smul_eq_mul, Finset.sum_mul, Finset.mul_sum]
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro g _
  apply Finset.sum_congr rfl
  intro j _
  ring

/-- Actual anchored products equal the fixed anchor plus gated local
products, CM46. There is no additional output decoder. -/
theorem logits (gate : X → G → ℝ) (bank : J → K → X → ℝ) (base : J → ℝ)
    (radius : ℝ) (a : G → J → ℝ) (W : K → C → ℝ) (x : X) :
    headLogits (features gate bank base radius a) W x =
      (1 - radius) • headLogits (ContextMixture.features bank base) W x +
      radius • ∑ g, gate x g • headLogits (ContextMixture.features bank (a g)) W x := by
  change headLogits (ContextMixture.features bank (weights gate base radius a x)) W x = _
  unfold weights
  rw [ContextMixture.logits_linear, local_logits]

/-- With gates, anchor, radius and head fixed, the whole logit vector is
affine in every local coefficient, CM46. -/
theorem logits_affine (gate : X → G → ℝ) (bank : J → K → X → ℝ) (base : J → ℝ)
    (radius : ℝ) (a b : G → J → ℝ) (W : K → C → ℝ) (x : X)
    (t u : ℝ) (htu : t + u = 1) :
    headLogits (features gate bank base radius (t • a + u • b)) W x =
      t • headLogits (features gate bank base radius a) W x +
      u • headLogits (features gate bank base radius b) W x := by
  change headLogits (ContextMixture.features bank (weights gate base radius (t • a + u • b) x)) W x = _
  rw [weights_affine gate base radius a b x t u htu, ContextMixture.logits_linear]
  rfl

/-- Every original coefficient remains charged to the same head budget,
CM46, even though different inputs use different local mixtures. -/
theorem logits_bound (gate : X → G → ℝ) (bank : J → K → X → ℝ)
    {base : J → ℝ} (hbase : base ∈ ContextMixture.simplex) {radius : ℝ}
    (hzero : 0 ≤ radius) (hone : radius ≤ 1) {a : G → J → ℝ} (ha : a ∈ rows)
    (x : X) (hgate : gate x ∈ ContextMixture.simplex) {M R : ℝ}
    (hM : 0 ≤ M) (hbank : ∀ j k, |bank j k x| ≤ M)
    {W : K → C → ℝ} (hW : W ∈ headBudget R) (c : C) :
    |headLogits (features gate bank base radius a) W x c| ≤ R * M := by
  apply headLogits_bound _ hW hM x
  exact features_bound gate bank hbase hzero hone ha x hgate hbank

/-- The actual logit change cancels the anchor and retains the declared
fixed gate and radius, CM46. -/
theorem logits_sub (gate : X → G → ℝ) (bank : J → K → X → ℝ) (base : J → ℝ)
    (radius : ℝ) (a b : G → J → ℝ) (W : K → C → ℝ) (x : X) (c : C) :
    headLogits (features gate bank base radius b) W x c -
      headLogits (features gate bank base radius a) W x c =
    radius * ∑ g, gate x g * (headLogits (ContextMixture.features bank (b g)) W x c -
      headLogits (ContextMixture.features bank (a g)) W x c) := by
  rw [logits, logits]
  simp only [Pi.add_apply, Pi.smul_apply, Finset.sum_apply, smul_eq_mul,
    mul_sub, Finset.sum_sub_distrib]
  ring

/-- The source bound controls local logit changes through the fixed radius
and gated L1 displacement, CM46. No gate normalization is needed here. -/
theorem logits_change_bound (gate : X → G → ℝ) (bank : J → K → X → ℝ)
    (base : J → ℝ) {radius : ℝ} (hzero : 0 ≤ radius) (a b : G → J → ℝ)
    (x : X) (hgate : ∀ g, 0 ≤ gate x g) {M R : ℝ} (hM : 0 ≤ M)
    (hbank : ∀ j k, |bank j k x| ≤ M) {W : K → C → ℝ}
    (hW : W ∈ headBudget R) (c : C) :
    |headLogits (features gate bank base radius b) W x c -
      headLogits (features gate bank base radius a) W x c| ≤
      radius * (R * M) * ∑ g, gate x g * ∑ j, |b g j - a g j| := by
  rw [logits_sub, abs_mul, abs_of_nonneg hzero]
  calc
    _ ≤ radius * ∑ g, |gate x g *
        (headLogits (ContextMixture.features bank (b g)) W x c -
          headLogits (ContextMixture.features bank (a g)) W x c)| :=
      mul_le_mul_of_nonneg_left (Finset.abs_sum_le_sum_abs _ _) hzero
    _ ≤ radius * ∑ g, gate x g * ((R * M) * ∑ j, |b g j - a g j|) := by
      apply mul_le_mul_of_nonneg_left _ hzero
      apply Finset.sum_le_sum
      intro g _
      rw [abs_mul, abs_of_nonneg (hgate g)]
      exact mul_le_mul_of_nonneg_left
        (ContextMixture.logits_change_bound bank (a g) (b g) x hM hbank hW c) (hgate g)
    _ = _ := by
      simp only [Finset.mul_sum]
      apply Finset.sum_congr rfl
      intro g _
      apply Finset.sum_congr rfl
      intro j _
      ring

end Transformer.Grokking.Convex.ConvexDepth.GatedContext
