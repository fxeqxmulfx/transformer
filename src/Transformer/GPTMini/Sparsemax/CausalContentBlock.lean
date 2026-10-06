import Transformer.GPTMini.Sparsemax.CausalContentBits
import Transformer.GPTMini.Sparsemax.CubeContentBudget

/-!
# An expressive causal content block with a jointly affine learned forward

New block after arXiv:1602.02068v2, Eq. (1). The observed prefix is
encoded by position, presence and fixed token bits. Selected products of
these bits have learned vector coefficients. Affine readouts generate
actual Q/K routing and the original common values; the genuine sparsemax
forward equals this content-feature embedding on a convex domain.

This closes the earlier positional-only response restriction and removes
stored prototype records and nearest-prototype search. It covers one
jointly learned content-memory block. Full subset capacity is exponential;
K selected features have K*D learned coefficients. A learned arbitrary
token code, free physical embeddings, GPTMini normalization, a trainable
layer stack and FFN/language objectives are separate questions. A convex
criterion is preserved conditionally; no empirical Basis pass is claimed.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

open scoped BigOperators

/-- Genuine sparsemax/common-value predictions queried by actual causal data bits.
Source: the generated content architecture following sparsemax Eq. (1). -/
def causalContentForward {R T V B K D : ℕ} (encode : Fin V → Fin B → Bool)
    (tokens : Fin R → Fin T → Fin V) (rows : Fin R → Fin T)
    (features : Fin K → Finset (Fin T × Option (Fin B)))
    (offset gain : Fin T × Option (Fin B) → ℝ) (pick : Fin T × Option (Fin B) → Fin K)
    (channel : Fin D) (W : Matrix (Fin K) (Fin D) ℝ) : Matrix (Fin R) (Fin D) ℝ :=
  cubeContentLearnedForward (fun r => causalContentBits encode (tokens r) (rows r))
    features offset gain pick channel W

/-- Actual jointly learned attention/values produce the chosen nonlinear causal content features exactly.
Source: the proved physical sparsemax Eq. (1) forward, with fixed data encoding. -/
theorem causalContentForward_eq {R T V B K D : ℕ} (cap floor : ℝ) (encode : Fin V → Fin B → Bool)
    (tokens : Fin R → Fin T → Fin V) (rows : Fin R → Fin T)
    (features : Fin K → Finset (Fin T × Option (Fin B)))
    (offset gain : Fin T × Option (Fin B) → ℝ) (pick : Fin T × Option (Fin B) → Fin K)
    (channel : Fin D) (W : Matrix (Fin K) (Fin D) ℝ) (hf : 1 / 2 < floor)
    (hW : W ∈ cubeContentLearningDomain cap floor offset gain pick channel) :
    causalContentForward encode tokens rows features offset gain pick channel W =
      Matrix.of (fun r d => cubeContentOutput features W (causalContentBits encode (tokens r) (rows r)) d) :=
  cubeContentLearnedForward_eq cap floor _ features offset gain pick channel W hf hW

/-- A nonzero token-sensitive response inhabits all causal actual-forward premises. -/
example : causalContentForward (fixedBinaryContentCode 2 1) (fun _ : Fin 1 => fun _ : Fin 1 => (1 : Fin 2))
    (fun _ => 0) (fun _ : Fin 1 => {(0, some 0)}) (fun _ => (1 / 32 : ℝ))
    (fun _ => (1 / 32 : ℝ)) (fun _ => (0 : Fin 1)) 0 !![1] =
    Matrix.of (fun _ : Fin 1 => fun _ : Fin 1 => (-1 : ℝ)) := by
  rw [causalContentForward_eq 1 (3 / 4) _ _ _ _ _ _ _ _ _ (by norm_num)]
  · ext r d
    norm_num [cubeContentOutput, cubeContentCharacter, causalContentBits, fixedBinaryContentCode, cubeContentSign]
  · constructor
    · intro k d
      fin_cases k
      fin_cases d
      norm_num
    · constructor
      · intro i
        norm_num [cubeContentLearnedRoutes]
      · norm_num [cubeContentLearnedRoutes, Finset.sum_const, Fintype.card_prod]

/-- Hidden continuation changes cannot affect genuine learned predictions at any parameter assignment.
Source: explicit causal data mask before sparsemax Eq. (1), with no teacher routes. -/
theorem causalContentForward_causal {R T V B K D : ℕ} (encode : Fin V → Fin B → Bool)
    (tokens other : Fin R → Fin T → Fin V) (rows : Fin R → Fin T)
    (features : Fin K → Finset (Fin T × Option (Fin B)))
    (offset gain : Fin T × Option (Fin B) → ℝ) (pick : Fin T × Option (Fin B) → Fin K)
    (channel : Fin D) (W : Matrix (Fin K) (Fin D) ℝ)
    (h : ∀ r j, j ≤ rows r → tokens r j = other r j) :
    causalContentForward encode tokens rows features offset gain pick channel W =
      causalContentForward encode other rows features offset gain pick channel W := by
  have he : (fun r => causalContentBits encode (tokens r) (rows r)) =
      (fun r => causalContentBits encode (other r) (rows r)) := by
    funext r
    exact causalContentBits_causal encode (tokens r) (other r) (rows r) (h r)
  unfold causalContentForward
  rw [he]

/-- A real token change beyond the observed query inhabits all full-block causality premises. -/
example : causalContentForward (fixedBinaryContentCode 2 1) (fun _ : Fin 1 => fun _ : Fin 2 => (0 : Fin 2))
    (fun _ => 0) (fun _ : Fin 1 => {(0, some 0)}) (fun _ => (1 / 64 : ℝ))
    (fun _ => (1 / 64 : ℝ)) (fun _ => (0 : Fin 1)) 0 !![1] =
    causalContentForward (fixedBinaryContentCode 2 1)
      (fun _ : Fin 1 => fun j : Fin 2 => if j = 0 then 0 else 1)
      (fun _ => 0) (fun _ : Fin 1 => {(0, some 0)}) (fun _ => (1 / 64 : ℝ))
      (fun _ => (1 / 64 : ℝ)) (fun _ => (0 : Fin 1)) 0 !![1] := by
  apply causalContentForward_causal
  intro r j hj
  have he : j = 0 := by omega
  subst j
  rfl

/-- The complete genuine causal block is affine in its jointly learned state on the convex domain.
Source: content nonlinear features and exact learned sparsemax/value decoding following Eq. (1). -/
theorem causalContentForward_affine {R T V B K D : ℕ} (cap floor : ℝ) (encode : Fin V → Fin B → Bool)
    (tokens : Fin R → Fin T → Fin V) (rows : Fin R → Fin T)
    (features : Fin K → Finset (Fin T × Option (Fin B)))
    (offset gain : Fin T × Option (Fin B) → ℝ) (pick : Fin T × Option (Fin B) → Fin K)
    (channel : Fin D) (W Z : Matrix (Fin K) (Fin D) ℝ) (hf : 1 / 2 < floor)
    (hW : W ∈ cubeContentLearningDomain cap floor offset gain pick channel)
    (hZ : Z ∈ cubeContentLearningDomain cap floor offset gain pick channel)
    (a b : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hab : a + b = 1) :
    causalContentForward encode tokens rows features offset gain pick channel (a • W + b • Z) =
      a • causalContentForward encode tokens rows features offset gain pick channel W +
        b • causalContentForward encode tokens rows features offset gain pick channel Z :=
  cubeContentLearnedForward_affine cap floor _ features offset gain pick channel W Z hf hW hZ a b ha hb hab

/-- Distinct feasible learned token responses and a midpoint inhabit every causal joint-affinity premise. -/
example : causalContentForward (fixedBinaryContentCode 2 1) (fun _ : Fin 1 => fun _ : Fin 1 => (1 : Fin 2))
    (fun _ => 0) (fun _ : Fin 1 => {(0, some 0)}) (fun _ => (1 / 32 : ℝ))
    (fun _ => (1 / 32 : ℝ)) (fun _ => (0 : Fin 1)) 0 ((1 / 2 : ℝ) • !![0] + (1 / 2 : ℝ) • !![1]) =
      (1 / 2 : ℝ) • causalContentForward (fixedBinaryContentCode 2 1)
        (fun _ : Fin 1 => fun _ : Fin 1 => (1 : Fin 2)) (fun _ => 0)
        (fun _ : Fin 1 => {(0, some 0)}) (fun _ => (1 / 32 : ℝ))
        (fun _ => (1 / 32 : ℝ)) (fun _ => (0 : Fin 1)) 0 !![0] +
      (1 / 2 : ℝ) • causalContentForward (fixedBinaryContentCode 2 1)
        (fun _ : Fin 1 => fun _ : Fin 1 => (1 : Fin 2)) (fun _ => 0)
        (fun _ : Fin 1 => {(0, some 0)}) (fun _ => (1 / 32 : ℝ))
        (fun _ => (1 / 32 : ℝ)) (fun _ => (0 : Fin 1)) 0 !![1] := by
  apply causalContentForward_affine 1 (3 / 4) _ _ _ _ _ _ _ _ _ _ (by norm_num)
  · constructor
    · intro k d
      fin_cases k
      fin_cases d
      norm_num
    · constructor
      · intro i
        norm_num [cubeContentLearnedRoutes]
      · norm_num [cubeContentLearnedRoutes, Finset.sum_const, Fintype.card_prod]
  · constructor
    · intro k d
      fin_cases k
      fin_cases d
      norm_num
    · constructor
      · intro i
        norm_num [cubeContentLearnedRoutes]
      · norm_num [cubeContentLearnedRoutes, Finset.sum_const, Fintype.card_prod]
  all_goals norm_num

/-- Any supplied convex output criterion remains convex in the entire causal-block learned state.
Source: genuine joint forward affinity after sparsemax Eq. (1); a nonlinear FFN is not included. -/
theorem causalContentCriterion_convex {R T V B K D : ℕ} (cap floor : ℝ) (encode : Fin V → Fin B → Bool)
    (tokens : Fin R → Fin T → Fin V) (rows : Fin R → Fin T)
    (features : Fin K → Finset (Fin T × Option (Fin B)))
    (offset gain : Fin T × Option (Fin B) → ℝ) (pick : Fin T × Option (Fin B) → Fin K)
    (channel : Fin D) (criterion : Matrix (Fin R) (Fin D) ℝ → ℝ) (hf : 1 / 2 < floor)
    (hcriterion : ConvexOn ℝ Set.univ criterion) :
    ConvexOn ℝ (cubeContentLearningDomain cap floor offset gain pick channel)
      (fun W => criterion (causalContentForward encode tokens rows features offset gain pick channel W)) := by
  refine ⟨cubeContentLearningDomain_convex cap floor offset gain pick channel, ?_⟩
  intro W hW Z hZ a b ha hb hab
  change criterion (causalContentForward encode tokens rows features offset gain pick channel (a • W + b • Z)) ≤
    a * criterion (causalContentForward encode tokens rows features offset gain pick channel W) +
      b * criterion (causalContentForward encode tokens rows features offset gain pick channel Z)
  rw [causalContentForward_affine cap floor encode tokens rows features offset gain pick channel W Z hf hW hZ a b ha hb hab]
  exact hcriterion.2 (Set.mem_univ _) (Set.mem_univ _) ha hb hab

/-- Ordinary finite prediction error supplies a genuine nonconstant convex criterion instance. -/
example : ConvexOn ℝ (cubeContentLearningDomain 1 (3 / 4)
    (fun _ : Fin 1 × Option (Fin 1) => (1 / 32 : ℝ)) (fun _ => (1 / 32 : ℝ))
    (fun _ => (0 : Fin 1)) (0 : Fin 1))
    (fun W => matrixOutputError !![0] (causalContentForward (fixedBinaryContentCode 2 1)
      (fun _ : Fin 1 => fun _ : Fin 1 => (1 : Fin 2)) (fun _ => 0)
      (fun _ : Fin 1 => {(0, some 0)}) (fun _ => (1 / 32 : ℝ))
      (fun _ => (1 / 32 : ℝ)) (fun _ => (0 : Fin 1)) 0 W)) :=
  causalContentCriterion_convex _ _ _ _ _ _ _ _ _ _ _ (by norm_num) (matrixOutputError_convex _)

end Transformer.GPTMini.Sparsemax
