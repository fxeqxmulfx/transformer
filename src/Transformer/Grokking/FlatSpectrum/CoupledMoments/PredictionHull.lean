module

public import Transformer.Grokking.FlatSpectrum.CoupledMoments.ScaledLoss
public import Transformer.Grokking.FlatSpectrum.ConvexChannel.Oracle

/-!
# A common convex class for different causal prediction families

Source: `review/flat_spectrum.md`, CM8. Predictions live in common probability coordinates
on the same input domain. The convex hull includes every expert without multiplying free
parameters. A feasible exact observed fit, or its ideal-loss excess, certifies the common
scaled CE objective. Neither certificate asserts recovery on unobserved inputs.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.CoupledMoments

variable {m n C K S : Type*}

/-- The convex hull in prediction coordinates, with fixed expert families (CM8). -/
def predictionHull (A : K → Set (Table m n C)) : Set (Table m n C) :=
  convexHull ℝ (⋃ k, A k)

/-- The common class is convex without a factorized mixture parametrization (CM8). -/
theorem predictionHull_convex (A : K → Set (Table m n C)) :
    Convex ℝ (predictionHull A) := convex_convexHull _ _

/-- Every feasible expert remains a feasible prediction in the enlarged class (CM8). -/
theorem mem_predictionHull (A : K → Set (Table m n C)) {k : K} {P : Table m n C}
    (hP : P ∈ A k) : P ∈ predictionHull A :=
  subset_convexHull ℝ _ (Set.mem_iUnion.mpr ⟨k, hP⟩)

variable [Fintype C] [Fintype S]

/-- A fixed visible-input lookup preserves every spectral table's probability rows
(CM8). No label values enter the lookup map. -/
theorem lookup_rows {X Y : Type*} (index : X → Y → m × n) {P : Table m n C}
    (hP : Rows P) : Rows (fun x y => P (index x y).1 (index x y).2) := by
  exact ⟨fun x y c => hP.1 _ _ c, fun x y => hP.2 _ _⟩

/-- Any fixed causal dictionary adapter maps a feasible joint channel into the same
probability-valued prediction coordinates (CM8). -/
theorem channel_prediction_rows {X Y Q V : Type*} [Fintype K] [Nonempty V]
    (query : X → Y → Q) (data : X → Y → K → V) {W : ConvexChannel.Channel Q K V C}
    (hW : ConvexChannel.Feasible W) :
    Rows (fun x y => ConvexChannel.logits W (query x y) (data x y)) := by
  exact ⟨fun x y c => ConvexChannel.logits_nonneg hW _ _ c,
    fun x y => ConvexChannel.sum_logits hW _ _⟩

/-- All common predictions retain the probability constraints (CM8). -/
theorem predictionHull_rows (A : K → Set (Table m n C))
    (hA : ∀ k, ∀ P ∈ A k, Rows P) {P : Table m n C} (hP : P ∈ predictionHull A) :
    Rows P := by
  apply convexHull_min (t := {Q : Table m n C | Rows Q}) ?_ rows_convex hP
  intro Q hQ
  obtain ⟨k, hk⟩ := Set.mem_iUnion.mp hQ
  exact hA k Q hk

/-- Fixed-scale ordinary CE remains convex on the enlarged class (CM8). -/
theorem predictionHull_loss_convex (A : K → Set (Table m n C)) (scale : ℝ)
    (cell : S → m × n) (labels : S → C) :
    ConvexOn ℝ (predictionHull A) (scaledLoss scale cell labels) := by
  refine ⟨predictionHull_convex A, ?_⟩
  intro P _ Q _ a b ha hb hab
  exact (scaled_loss_convex scale cell labels).2
    (Set.mem_univ _) (Set.mem_univ _) ha hb hab

variable [DecidableEq C]

/-- A feasible expert fitting observed labels minimizes the same CE over the full hull.
Membership is explicit; the scale must be shared by every expert (CM8). -/
theorem predictionHull_fit_minimum (A : K → Set (Table m n C))
    (hA : ∀ k, ∀ P ∈ A k, Rows P) {scale : ℝ} (hs : 0 ≤ scale)
    (cell : S → m × n) (labels : S → C) {k : K} {P : Table m n C}
    (hP : P ∈ A k) (hfit : ∀ s, P (cell s).1 (cell s).2 = ConvexRecall.targetLogits (labels s)) :
    P ∈ predictionHull A ∧ IsMinOn (scaledLoss scale cell labels) (predictionHull A) P := by
  refine ⟨mem_predictionHull A hP, ?_⟩
  intro Q hQ
  rw [scaled_loss_eq_ideal scale cell labels hfit]
  exact scaled_ideal_le hs cell labels (predictionHull_rows A hA hQ)

/-- The observed ideal-loss excess bounds suboptimality against every hull competitor,
even when candidate search has not solved the interior optimization (CM8). -/
theorem predictionHull_excess_le (A : K → Set (Table m n C))
    (hA : ∀ k, ∀ P ∈ A k, Rows P) {scale : ℝ} (hs : 0 ≤ scale)
    (cell : S → m × n) (labels : S → C) (P : Table m n C)
    {Q : Table m n C} (hQ : Q ∈ predictionHull A) :
    scaledLoss scale cell labels P - scaledLoss scale cell labels Q ≤
      scaledLoss scale cell labels P - scaledIdealLoss scale labels := by
  linarith [scaled_ideal_le hs cell labels (predictionHull_rows A hA hQ)]

private def witness : Table Bool Unit Bool := pointMass (fun x _ => x)
private def family (k : Bool) : Set (Table Bool Unit Bool) :=
  {if k then witness else uniform}

private theorem family_rows (k : Bool) (P : Table Bool Unit Bool) (hP : P ∈ family k) :
    Rows P := by
  simp only [family, Set.mem_singleton_iff] at hP
  subst P
  cases k
  · exact rows_uniform
  · exact rows_pointMass _

private theorem witness_mem : witness ∈ family true := by simp [family]

example := mem_predictionHull family witness_mem
example := predictionHull_rows family family_rows (mem_predictionHull family witness_mem)
example := predictionHull_fit_minimum family family_rows (by norm_num : (0 : ℝ) ≤ 10)
  (fun s : Bool => (s, ())) id witness_mem
  (by intro s; funext c; simp [witness, pointMass, ConvexRecall.targetLogits])
example := predictionHull_excess_le family family_rows (by norm_num : (0 : ℝ) ≤ 10)
  (fun s : Bool => (s, ())) id (uniform : Table Bool Unit Bool)
  (mem_predictionHull family witness_mem)

example := family_rows true witness witness_mem
example := lookup_rows (fun (x : Bool) (y : Bool) => (y, x))
  (rows_pointMass (fun (x : Bool) (y : Bool) => x != y))
example := channel_prediction_rows (fun (x : Bool) (y : Bool) => x && y)
  (fun (x : Bool) (y : Bool) (k : Bool) => (x != y) != k)
  (ConvexChannel.feasible_vertex (id : Bool → Bool) (fun _ _ => (id : Bool → Bool)))

end Transformer.Grokking.FlatSpectrum.CoupledMoments
