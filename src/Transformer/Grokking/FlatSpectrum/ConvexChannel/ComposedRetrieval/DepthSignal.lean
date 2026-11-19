module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.ComposedRetrieval.Witness
public import Transformer.Grokking.GradientEvidence.ReadoutGradient
public import Transformer.Grokking.FlatSpectrum.Training.Logits

/-!
# When ordinary CE can train the second read

Source: `review/convex/moments.md`, CM114. Uniform relative scores erase
the effect of content routing and the depth gate. Away from uniform routing
an observed class contrast can distinguish depths when the other weights
are fixed. The CE derivative uses all original classes and the true label.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel.ComposedRetrieval

open scoped BigOperators
open Perspective Landscape.Neural

variable {O : Type*} {d n : ℕ}

/-- A spatially uniform learned relation gives a uniform destination law at
every query, even when the content match strongly prefers one key, CM114. -/
theorem jump_uniform (p : Parameters O d) (x : Fin n → O) (q : O) (r : ℝ)
    (hr : ∀ δ, p.relation δ = r) (k : Fin n) : jump p x q k = (n : ℝ)⁻¹ := by
  have hn := Nat.zero_lt_of_lt k.isLt
  have row (j : Fin n) : softmaxWeight (linkScore p j) k = (n : ℝ)⁻¹ := by
    simp [softmaxWeight, linkScore, hr, div_eq_mul_inv, Real.exp_ne_zero, Idx]
  simp only [jump, row, ← Finset.sum_mul, sum_softmaxWeight hn, one_mul]

/-- CM114. The common readout after a uniform jump, for any class and
coordinate-wise query rotation. -/
noncomputable def uniformRead (p : Parameters O d) (x : Fin n → O)
    (turn : Fin (d+1) → ℝ) (c : O) : ℝ :=
  ∑ k, (n : ℝ)⁻¹ * decode p (p.value (x k)) turn c

/-- The first read averages all visible values when link scores are constant, CM114. -/
theorem readOne_uniform (p : Parameters O d) (x : Fin n → O) (q c : O)
    (turn : Fin (d+1) → ℝ) (r : ℝ) (hr : ∀ δ, p.relation δ = r) :
    readOne p x q turn c = uniformRead p x turn c := by
  simp only [readOne, uniformRead, jump_uniform p x q r hr]

/-- The second content-dependent lookup yields exactly the same unrotated
read as the first at positive memory length under uniform links, CM114. -/
theorem readTwo_uniform (p : Parameters O d) (hn : 0 < n) (x : Fin n → O) (q c : O)
    (r : ℝ) (hr : ∀ δ, p.relation δ = r) : readTwo p x q c = uniformRead p x 0 c := by
  calc
    _ = ∑ j, jump p x q j * readOne p x (x j) 0 c := by
      simp only [readTwo, trail, Fintype.sum_prod_type, readOne, Finset.mul_sum, mul_assoc]
    _ = _ := by simp only [readOne_uniform p x _ c 0 r hr, ← Finset.sum_mul, sum_jump hn, one_mul]

/-- CM114. At uniform relative scores the complete logit retains the query
rotation and the product gate. Content routing and depth disappear exactly. -/
theorem logit_uniform (p : Parameters O d) (hn : 0 < n) (x : Fin n → O) (q c : O)
    (r : ℝ) (hr : ∀ δ, p.relation δ = r) :
    logit p x q c = blend p.product (uniformRead p x 0 c) (uniformRead p x (p.phase q) c) := by
  rw [logit, readOne_uniform p x q c 0 r hr, readTwo_uniform p hn x q c r hr,
    readOne_uniform p x q c (p.phase q) r hr]
  unfold blend
  ring

/-- Vary precisely the learned content and depth parameters while holding
the value, query rotation, relative table and product gate fixed, CM114. -/
def contentParameters (p : Parameters O d) (key : O → Fin (d+1) → ℝ)
    (bias : Option O → ℝ) (scale recency depth : ℝ) : Parameters O d :=
  { p with key, bias, scale, recency, depth }

/-- Any joint curve in the content and depth coordinates has zero actual CE
derivative at uniform links. The curve itself may be arbitrary, CM114. -/
theorem content_zero_derivative [Fintype O] (p : Parameters O d) (hn : 0 < n)
    (x : Fin n → O) (q y : O) (B r : ℝ) (hr : ∀ δ, p.relation δ = r)
    (key : ℝ → O → Fin (d+1) → ℝ) (bias : ℝ → Option O → ℝ)
    (scale recency depth : ℝ → ℝ) (a : ℝ) :
    HasDerivAt (fun s => crossEntropy (fun c => B * logit
      (contentParameters p (key s) (bias s) (scale s) (recency s) (depth s)) x q c) y) 0 a := by
  have he (s : ℝ) (c : O) : logit
      (contentParameters p (key s) (bias s) (scale s) (recency s) (depth s)) x q c =
      blend p.product (uniformRead p x 0 c) (uniformRead p x (p.phase q) c) :=
    logit_uniform _ hn x q c r hr
  simp_rw [he]
  exact hasDerivAt_const _ _

/-- CM114. Per-class change in the complete scaled logit per unit of learned depth. -/
noncomputable def depthDirection (p : Parameters O d) (x : Fin n → O) (q : O)
    (B : ℝ) (c : O) : ℝ := B * (1-p.product) * (readTwo p x q c - readOne p x q 0 c)

/-- The whole depth line is affine with other physical weights fixed, CM114. -/
theorem depth_line (p : Parameters O d) (x : Fin n → O) (q c : O) (B a : ℝ) :
    B * logit {p with depth := p.depth+a} x q c =
      B * logit p x q c + a * depthDirection p x q B c := by
  change B * blend p.product
    (blend (p.depth+a) (readOne p x q 0 c) (readTwo p x q c))
    (readOne p x q (p.phase q) c) = _
  unfold logit blend depthDirection
  ring

/-- Exact ordinary full-class CE slope for depth at the learned point, CM114. -/
noncomputable def depthSlope [Fintype O] (p : Parameters O d) (x : Fin n → O) (q y : O)
    (B : ℝ) : ℝ :=
  (∑ c, Geometry.probability (fun a => B * logit p x q a) c * depthDirection p x q B c)
    - depthDirection p x q B y

/-- The ordinary CE gradient compares the observed target's direction with
its full predictive expectation. A target above this mean favors increasing
the depth locally. Interior weights allow either feasible direction, CM114. -/
theorem depth_hasDerivAt [Fintype O] (p : Parameters O d) (x : Fin n → O) (q y : O) (B : ℝ) :
    HasDerivAt (fun a => crossEntropy (fun c => B * logit {p with depth := p.depth+a} x q c) y)
      (depthSlope p x q y B) 0 := by
  simp_rw [depth_line]
  exact GradientEvidence.readout_ce_line_hasDerivAt _ _ _

/-- At fixed other weights, the ordinary CE lies above its depth tangent
for every displacement, including feasible changes from the boundary, CM114. -/
theorem depth_tangent [Fintype O] (p : Parameters O d) (x : Fin n → O) (q y : O) (B a : ℝ) :
    crossEntropy (fun c => B * logit p x q c) y + a * depthSlope p x q y B ≤
      crossEntropy (fun c => B * logit {p with depth := p.depth+a} x q c) y := by
  simp_rw [depth_line]
  have h := Training.crossEntropy_first_order (fun c => B * logit p x q c)
    (fun c => B * logit p x q c + a * depthDirection p x q B c) y
  simpa only [add_sub_cancel_left, depthSlope, mul_sub, Finset.mul_sum, mul_left_comm] using h

/-- If the summed training slope is nonnegative, no feasible increase from
the current depth improves the training CE with other weights fixed, CM114.
Row indices retain original multiplicities and memory lengths. -/
theorem depth_increase_bound [Fintype O] {I : Type*} [Fintype I] {n : I → ℕ}
    (p : Parameters O d) (x : (i : I) → Fin (n i) → O) (q y : I → O) (B a : ℝ)
    (ha : 0 ≤ a) (hs : 0 ≤ ∑ i, depthSlope p (x i) (q i) (y i) B) :
    (∑ i, crossEntropy (fun c => B * logit p (x i) (q i) c) (y i)) ≤
      ∑ i, crossEntropy (fun c => B * logit {p with depth := p.depth+a} (x i) (q i) c) (y i) := by
  have h := Finset.sum_le_sum (s := Finset.univ) fun i _ =>
    depth_tangent p (x i) (q i) (y i) B a
  rw [Finset.sum_add_distrib, ← Finset.mul_sum] at h
  exact (le_add_of_nonneg_right (mul_nonneg ha hs)).trans h

/-- One nonconstant class-direction contrast on an observed prefix identifies
depth from that score contrast when the other weights are fixed, CM114. -/
theorem depth_identified (p : Parameters O d) (x : Fin n → O) (q c k : O) (B a b : ℝ)
    (hc : depthDirection p x q B c ≠ depthDirection p x q B k)
    (he : B * logit {p with depth := p.depth+a} x q c - B * logit {p with depth := p.depth+a} x q k =
      B * logit {p with depth := p.depth+b} x q c - B * logit {p with depth := p.depth+b} x q k) :
    a = b := by
  simp only [depth_line] at he
  by_contra hab
  exact (mul_ne_zero (sub_ne_zero.mpr hab) (sub_ne_zero.mpr hc)) (by linear_combination he)

/-- With distinguishable class directions at least one of their true
labels produces nonzero ordinary CE signal on this prefix, CM114. -/
theorem some_target_signal [Fintype O] (p : Parameters O d) (x : Fin n → O)
    (q c k : O) (B : ℝ) (hc : depthDirection p x q B c ≠ depthDirection p x q B k) :
    depthSlope p x q c B ≠ 0 ∨ depthSlope p x q k B ≠ 0 := by
  by_contra! h
  unfold depthSlope at h
  exact hc (by linarith [h.1,h.2])

/-- The CM113 repeated-key witness has distinct terminal and intermediate
class directions, so the nonuniform identification condition is inhabited. -/
theorem chain_depth_signal : depthDirection chain memory 0 1 1 < depthDirection chain memory 0 1 2 := by
  have hm := readTwo_margin chain memory 0 1 first second 1 (by norm_num) chain_separation.1
    chain_separation.2 (terminal_separation 1 (by decide))
  change 1-(1+2)*(first.error+second.error) ≤ readTwo chain memory 0 2 - readTwo chain memory 0 1 at hm
  have hp := chain_positive
  norm_num [guaranteed, chain] at hp
  have h1 := chain_identifies.1
  change (1:ℝ)*(1-1/32)*(readTwo chain memory 0 1-readOne chain memory 0 0 1) <
    1*(1-1/32)*(readTwo chain memory 0 2-readOne chain memory 0 0 2)
  linarith

example : jump {chain with relation := 0} memory 0 1 = (4:ℝ)⁻¹ :=
  jump_uniform _ _ _ 0 (fun _ => rfl) _
example : readOne {chain with relation := 0} memory 0 0 2 = uniformRead chain memory 0 2 :=
  readOne_uniform _ _ _ _ _ 0 (fun _ => rfl)
example : readTwo {chain with relation := 0} memory 0 2 = uniformRead chain memory 0 2 :=
  readTwo_uniform _ (by decide) _ _ _ 0 (fun _ => rfl)
example : logit {chain with relation := 0} memory 0 2 =
    blend chain.product (uniformRead chain memory 0 2) (uniformRead chain memory (chain.phase 0) 2) :=
  logit_uniform _ (by decide) _ _ _ 0 (fun _ => rfl)
example : HasDerivAt (fun s : ℝ => crossEntropy (fun c => 16 * logit
    (contentParameters {chain with relation := 0} (fun _ _ => s) (fun _ => s) s s s) memory 0 c) 2)
    0 0 := content_zero_derivative _ (by decide) _ _ _ _ 0 (fun _ => rfl) _ _ _ _ _ _
example (a b : ℝ) (h : logit {chain with depth := chain.depth+a} memory 0 1 -
    logit {chain with depth := chain.depth+a} memory 0 2 =
    logit {chain with depth := chain.depth+b} memory 0 1 - logit {chain with depth := chain.depth+b} memory 0 2) :
    a = b := depth_identified chain memory 0 1 2 1 a b chain_depth_signal.ne (by simpa using h)
example : depthSlope chain memory 0 1 1 ≠ 0 ∨ depthSlope chain memory 0 2 1 ≠ 0 :=
  some_target_signal chain memory 0 1 2 1 chain_depth_signal.ne
example : (1/8 : ℝ) = 1/8 := depth_identified chain memory 0 1 2 1 _ _ chain_depth_signal.ne rfl
example := depth_increase_bound {chain with product := 1} (fun _ : Unit => memory)
  (fun _ => 0) (fun _ => 2) 16 (1/8) (by norm_num) (by simp [depthSlope, depthDirection])

end Transformer.Grokking.FlatSpectrum.ConvexChannel.ComposedRetrieval
