module

public import Mathlib.Analysis.SpecialFunctions.Sigmoid
public import Transformer.Grokking.FlatSpectrum.ConvexChannel.ComposedRetrieval.DepthSignal

/-!
# Ordinary next-object loss in depth log odds

Source: `review/convex/moments.md`, CM115. A real training coordinate determines
the physical depth of CM113 through a sigmoid. The full CE slope in this
coordinate has the same sign as the physical slope, with a scale bounded by
one quarter. The CM114 tangent bound transfers with the physical displacement.
A class-score contrast that identifies physical depth also identifies its
finite log odds, with all other weights fixed.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel.ComposedRetrieval

open scoped BigOperators
open Landscape.Neural

variable {O : Type*} {d n : ℕ}

/-- CM115. Change the physical mixture through an unconstrained scalar. -/
noncomputable def atLogOdds (p : Parameters O d) (z : ℝ) : Parameters O d :=
  {p with depth := Real.sigmoid z}

/-- Every finite log odds assigns positive mass to both original reads, CM115. -/
theorem logOdds_interior (p : Parameters O d) (z : ℝ) :
    0 < (atLogOdds p z).depth ∧ (atLogOdds p z).depth < 1 :=
  ⟨Real.sigmoid_pos _, Real.sigmoid_lt_one _⟩

/-- Odds of nine to one give the exact predeclared initial physical depth, CM115. -/
theorem logOdds_initial (p : Parameters O d) : (atLogOdds p (Real.log 9)).depth = 9/10 := by
  norm_num [atLogOdds, Real.sigmoid_def, Real.exp_neg, Real.exp_log]

/-- CM115. Rate of change of the physical gate per unit of the raw parameter. -/
noncomputable def depthRate (z : ℝ) : ℝ := Real.sigmoid z * (1-Real.sigmoid z)

/-- The sigmoid rate is positive and bounded by one quarter everywhere, CM115. -/
theorem depthRate_bounds (z : ℝ) : 0 < depthRate z ∧ depthRate z ≤ 1/4 := by
  unfold depthRate
  exact ⟨mul_pos (Real.sigmoid_pos _) (sub_pos.mpr (Real.sigmoid_lt_one _)),
    by nlinarith [sq_nonneg (Real.sigmoid z - 1/2)]⟩

/-- At the initial odds, nine percent of the physical slope reaches the raw
coordinate before Adam's moment normalization, CM115. -/
theorem depthRate_initial : depthRate (Real.log 9) = 9/100 := by
  have h := logOdds_initial chain
  change Real.sigmoid (Real.log 9) = 9/10 at h
  norm_num [depthRate,h]

/-- The exact score change from a finite raw displacement is the learned
class direction times the difference of the two physical mixture weights.
This describes the two complete loss queries in the native study, CM115. -/
theorem logOdds_class_change (p : Parameters O d) (x : Fin n → O) (q c : O) (B z t : ℝ) :
    B * logit (atLogOdds p t) x q c = B * logit (atLogOdds p z) x q c +
      (Real.sigmoid t-Real.sigmoid z) * depthDirection (atLogOdds p z) x q B c := by
  simpa only [atLogOdds, add_sub_cancel] using
    depth_line (atLogOdds p z) x q c B (Real.sigmoid t-Real.sigmoid z)

/-- Uniform relative scores leave the ordinary loss flat even when the
initial depth is strictly positive and large. This property holds for every
observed label and finite log odds, CM115. -/
theorem logOdds_uniform_flat [Fintype O] (p : Parameters O d) (hn : 0 < n)
    (x : Fin n → O) (q y : O) (B r z : ℝ) (hr : ∀ δ, p.relation δ = r) :
    HasDerivAt (fun t => crossEntropy (fun c => B * logit (atLogOdds p t) x q c) y) 0 z := by
  exact content_zero_derivative p hn x q y B r hr (fun _ => p.key) (fun _ => p.bias)
    (fun _ => p.scale) (fun _ => p.recency) Real.sigmoid z

/-- Exact raw-coordinate derivative of the actual full-vocabulary CE, CM115. -/
theorem logOdds_hasDerivAt [Fintype O] (p : Parameters O d) (x : Fin n → O) (q y : O)
    (B z : ℝ) : HasDerivAt
      (fun t => crossEntropy (fun c => B * logit (atLogOdds p t) x q c) y)
      (depthSlope (atLogOdds p z) x q y B * depthRate z) z := by
  let pz := atLogOdds p z
  have ho : HasDerivAt
      (fun a => crossEntropy (fun c => B * logit {pz with depth := pz.depth+a} x q c) y)
      (depthSlope pz x q y B) (Real.sigmoid z - Real.sigmoid z) := by
    simpa only [sub_self] using depth_hasDerivAt pz x q y B
  have hi := (Real.hasDerivAt_sigmoid z).sub_const (Real.sigmoid z)
  simpa only [Function.comp_def, pz, atLogOdds, add_sub_cancel, depthRate] using ho.comp z hi

/-- The sign of a physical training signal is preserved at finite raw
coordinates. In particular a higher observed target direction still favors
increasing the gate, CM115. -/
theorem logOdds_slope_sign [Fintype O] (p : Parameters O d) (x : Fin n → O) (q y : O)
    (B z : ℝ) :
    (0 < depthSlope (atLogOdds p z) x q y B * depthRate z ↔
      0 < depthSlope (atLogOdds p z) x q y B) ∧
    (depthSlope (atLogOdds p z) x q y B * depthRate z < 0 ↔
      depthSlope (atLogOdds p z) x q y B < 0) := by
  refine ⟨mul_pos_iff_of_pos_right (depthRate_bounds z).1, ?_⟩
  simpa only [zero_mul] using
    mul_lt_mul_iff_left₀ (b := depthSlope (atLogOdds p z) x q y B) (c := 0) (depthRate_bounds z).1

/-- The raw CE slope has at most one quarter of the physical magnitude,
including all observed labels and both signs, CM115. -/
theorem logOdds_slope_bound [Fintype O] (p : Parameters O d) (x : Fin n → O) (q y : O)
    (B z : ℝ) : |depthSlope (atLogOdds p z) x q y B * depthRate z| ≤
      |depthSlope (atLogOdds p z) x q y B| / 4 := by
  rw [abs_mul, abs_of_pos (depthRate_bounds z).1]
  simpa only [div_eq_mul_inv, one_mul] using
    mul_le_mul_of_nonneg_left (depthRate_bounds z).2 (abs_nonneg (depthSlope (atLogOdds p z) x q y B))

/-- For any two raw values the tangent lower bound uses their *physical*
depth difference. This inequality gives the scope of the CM114 convexity
argument in the new coordinate, CM115. -/
theorem logOdds_loss_bound [Fintype O] (p : Parameters O d) (x : Fin n → O) (q y : O)
    (B z t : ℝ) :
    crossEntropy (fun c => B * logit (atLogOdds p z) x q c) y +
      (Real.sigmoid t - Real.sigmoid z) * depthSlope (atLogOdds p z) x q y B ≤
      crossEntropy (fun c => B * logit (atLogOdds p t) x q c) y := by
  simpa only [atLogOdds, add_sub_cancel] using
    depth_tangent (atLogOdds p z) x q y B (Real.sigmoid t - Real.sigmoid z)

/-- A nonnegative total physical slope on the original, possibly
different-length training rows rules out improvement at any larger raw
depth coordinate with the rest of the model fixed, CM115. -/
theorem logOdds_increase_bound [Fintype O] {I : Type*} [Fintype I] {n : I → ℕ}
    (p : Parameters O d) (x : (i : I) → Fin (n i) → O) (q y : I → O) (B z t : ℝ)
    (hzt : z ≤ t) (hs : 0 ≤ ∑ i, depthSlope (atLogOdds p z) (x i) (q i) (y i) B) :
    (∑ i, crossEntropy (fun c => B * logit (atLogOdds p z) (x i) (q i) c) (y i)) ≤
      ∑ i, crossEntropy (fun c => B * logit (atLogOdds p t) (x i) (q i) c) (y i) := by
  have ha := sub_nonneg.mpr (Real.sigmoid_strictMono.monotone hzt)
  simpa only [atLogOdds, add_sub_cancel] using
    depth_increase_bound (atLogOdds p z) x q y B _ ha hs

/-- A measured class contrast that separates physical depths identifies the
raw coordinate with the other weights fixed. The nondegeneracy condition
is inherited from the two physical reads, CM115. -/
theorem logOdds_identified (p : Parameters O d) (x : Fin n → O) (q c k : O) (B z t : ℝ)
    (hd : depthDirection p x q B c ≠ depthDirection p x q B k)
    (he : B * logit (atLogOdds p z) x q c - B * logit (atLogOdds p z) x q k =
      B * logit (atLogOdds p t) x q c - B * logit (atLogOdds p t) x q k) : z = t := by
  have h := depth_identified p x q c k B (Real.sigmoid z-p.depth) (Real.sigmoid t-p.depth) hd
    (by simpa only [atLogOdds, add_sub_cancel] using he)
  exact Real.sigmoid_strictMono.injective (by linarith)

example := logOdds_hasDerivAt chain memory 0 2 16 (Real.log 9)
example := logOdds_slope_bound chain memory 0 1 16 (Real.log 9)
example := logOdds_uniform_flat {chain with relation := 0} (by decide) memory 0 2 16 0
  (Real.log 9) (fun _ => rfl)
example : (0:ℝ) = 0 := logOdds_identified chain memory 0 1 2 1 0 0 chain_depth_signal.ne rfl
example := logOdds_increase_bound {chain with product := 1} (fun _ : Unit => memory)
  (fun _ => 0) (fun _ => 2) 16 0 (Real.log 9) (by positivity)
  (by simp [atLogOdds, depthSlope, depthDirection])

end Transformer.Grokking.FlatSpectrum.ConvexChannel.ComposedRetrieval
