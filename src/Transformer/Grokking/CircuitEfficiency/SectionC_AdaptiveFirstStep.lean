import Transformer.Grokking.CircuitEfficiency.SectionC_SubweightGradient
import Transformer.Grokking.AdamW.FirstStep

/-!
# Native adaptive first update of the fixed-table product model

Sources: Varma et al., arXiv:2309.02390v1, section 3 and appendix C,
product-subweight CE; native AdamW at lab commit 43d4d66, PyTorch
2.14.1's documented zero-moment first update, formalized in
Transformer.Grokking.AdamW.FirstStep. Use actual unpenalized CE partial
derivatives and decoupled parameter decay, instead of replacing that
algorithm by the source's coupled-penalty gradient descent.

The source's seeded zero-product initialization has the same initial
CE gradients as its r>1 coupled budget, but adaptive normalization
changes the product growth. Derive the exact finite first products,
with all competing classes retained. These are exact-real update
identities from zero moments, not persistent-moment convergence,
rounded GPTMini kernel outputs or a general grokking prediction.
-/

namespace Transformer.Grokking.CircuitEfficiency

open Transformer.Grokking.AdamW

/-- Magnitude of the actual finite-class CE derivative at tied logits.
Source: arXiv:2309.02390v1, appendix C's CE, derived in table_train_ce_deriv. -/
noncomputable def initialCEGradientMagnitude (remaining : ℕ) : ℝ :=
  ((remaining : ℝ) + 1) / ((remaining : ℝ) + 2)

/-- The CE gradient magnitude is positive for every q>=2. Source:
arXiv:2309.02390v1, appendix C; a real competing class remains present. -/
theorem initial_ce_gradient_magnitude_pos (remaining : ℕ) :
    0 < initialCEGradientMagnitude remaining := by
  unfold initialCEGradientMagnitude
  positivity

/-- Apply the documented native first update to the actual four CE
partials, with zero moments and parameter decay. Sources: arXiv:2309.02390v1,
appendix C's product CE; AdamW at 43d4d66. Setting the coupled multiplier
to zero retains plain CE; exponent two has no remaining penalty effect. -/
noncomputable def subweightAdamWFirstStep (remaining : ℕ)
    (b1 b2 eps decay rate : ℝ) (z : (ℝ × ℝ) × (ℝ × ℝ)) : (ℝ × ℝ) × (ℝ × ℝ) :=
  let g := subweightGradient remaining 0 0 0 2 z.1.1 z.1.2 z.2.1 z.2.2
  ((firstUpdate b1 b2 eps decay rate z.1.1 g.1.1,
    firstUpdate b1 b2 eps decay rate z.1.2 g.1.2),
   (firstUpdate b1 b2 eps decay rate z.2.1 g.2.1,
    firstUpdate b1 b2 eps decay rate z.2.2 g.2.2))

/-- Candidate first product, proved equal to the native updated factors
below. Sources: actual product CE in arXiv:2309.02390v1, appendix C,
and native AdamW at 43d4d66; the decay factor acts on the seeded partner. -/
noncomputable def adaptiveSeedProduct (remaining : ℕ) (eps decay rate seed : ℝ) : ℝ :=
  rate * (1 - rate * decay) * initialCEGradientMagnitude remaining * seed ^ 2 /
    (initialCEGradientMagnitude remaining * seed + eps)

/-- Adaptive normalization of the initial negative CE coordinate gradient.
Sources: arXiv:2309.02390v1, appendix C, seeded product; native AdamW
at 43d4d66. This algebraic identity permits any real epsilon; the later
positive-product and ratio claims require the native positive epsilon. -/
theorem native_seeded_first_coordinate (remaining : ℕ)
    (b1 b2 eps decay rate seed : ℝ) (h1 : b1 ≠ 1) (h2 : b2 ≠ 1) (hs : 0 ≤ seed) :
    firstUpdate b1 b2 eps decay rate 0 (-seed * initialCEGradientMagnitude remaining) =
      rate * initialCEGradientMagnitude remaining * seed /
        (initialCEGradientMagnitude remaining * seed + eps) := by
  have hmu := initial_ce_gradient_magnitude_pos remaining
  have hn : -seed * initialCEGradientMagnitude remaining ≤ 0 := by nlinarith
  unfold firstUpdate
  rw [firstDirection_eq b1 b2 eps _ h1 h2, abs_of_nonpos hn]
  ring

example : (9 / 10 : ℝ) ≠ 1 ∧ (49 / 50 : ℝ) ≠ 1 ∧ 0 ≤ (1 / 200 : ℝ) := by norm_num

/-- A zero CE coordinate gradient leaves only decoupled parameter decay.
Source: native AdamW at 43d4d66, with zero initial moment buffers. -/
theorem native_zero_gradient_update (b1 b2 eps decay rate p : ℝ)
    (h1 : b1 ≠ 1) (h2 : b2 ≠ 1) :
    firstUpdate b1 b2 eps decay rate p 0 = (1 - rate * decay) * p := by
  unfold firstUpdate
  rw [firstDirection_eq b1 b2 eps 0 h1 h2]
  simp

example : (9 / 10 : ℝ) ≠ 1 ∧ (49 / 50 : ℝ) ≠ 1 := by norm_num

/-- Both seeded circuits receive the native finite first update, rather
than the source's GD update. Sources: arXiv:2309.02390v1, appendix C's
initialization, and native AdamW at 43d4d66; all four gradients were derived. -/
theorem subweight_adamw_initial_step (remaining : ℕ)
    (b1 b2 eps decay rate genSeed memSeed : ℝ)
    (h1 : b1 ≠ 1) (h2 : b2 ≠ 1) (hg : 0 ≤ genSeed) (hm : 0 ≤ memSeed) :
    subweightAdamWFirstStep remaining b1 b2 eps decay rate ((0, genSeed), (0, memSeed)) =
      ((rate * initialCEGradientMagnitude remaining * genSeed /
          (initialCEGradientMagnitude remaining * genSeed + eps), (1 - rate * decay) * genSeed),
       (rate * initialCEGradientMagnitude remaining * memSeed /
          (initialCEGradientMagnitude remaining * memSeed + eps), (1 - rate * decay) * memSeed)) := by
  unfold subweightAdamWFirstStep
  rw [subweight_gradient_initial remaining 0 0 0 2 genSeed memSeed (by norm_num)]
  dsimp
  change ((firstUpdate b1 b2 eps decay rate 0 (-genSeed * initialCEGradientMagnitude remaining),
    firstUpdate b1 b2 eps decay rate genSeed 0),
    (firstUpdate b1 b2 eps decay rate 0 (-memSeed * initialCEGradientMagnitude remaining),
    firstUpdate b1 b2 eps decay rate memSeed 0)) = _
  rw [native_seeded_first_coordinate remaining b1 b2 eps decay rate genSeed h1 h2 hg,
    native_seeded_first_coordinate remaining b1 b2 eps decay rate memSeed h1 h2 hm,
    native_zero_gradient_update b1 b2 eps decay rate genSeed h1 h2,
    native_zero_gradient_update b1 b2 eps decay rate memSeed h1 h2]

example : (9 / 10 : ℝ) ≠ 1 ∧ (49 / 50 : ℝ) ≠ 1 ∧ 0 ≤ (1 / 200 : ℝ) ∧ 0 ≤ (1 : ℝ) := by
  norm_num

/-- The algebraic product formula equals actual native updated factor
products. Sources: arXiv:2309.02390v1, appendix C's product coordinates,
and native AdamW at 43d4d66; no prescribed successful trajectory is used. -/
theorem subweight_adamw_initial_products (remaining : ℕ)
    (b1 b2 eps decay rate genSeed memSeed : ℝ)
    (h1 : b1 ≠ 1) (h2 : b2 ≠ 1) (hg : 0 ≤ genSeed) (hm : 0 ≤ memSeed) :
    let next := subweightAdamWFirstStep remaining b1 b2 eps decay rate ((0, genSeed), (0, memSeed))
    (next.1.1 * next.1.2, next.2.1 * next.2.2) =
      (adaptiveSeedProduct remaining eps decay rate genSeed, adaptiveSeedProduct remaining eps decay rate memSeed) := by
  rw [subweight_adamw_initial_step remaining b1 b2 eps decay rate genSeed memSeed h1 h2 hg hm]
  dsimp [adaptiveSeedProduct]
  congr 1 <;> ring

example : (9 / 10 : ℝ) ≠ 1 ∧ (49 / 50 : ℝ) ≠ 1 ∧ 0 ≤ (1 / 200 : ℝ) ∧ 0 ≤ (1 : ℝ) := by
  norm_num

/-- A native positive rate, positive epsilon and positive remaining
decay factor give a positive first product for every positive seed.
Source: native AdamW at 43d4d66 applied to appendix C's actual CE. -/
theorem adaptive_seed_product_pos (remaining : ℕ) (eps decay rate seed : ℝ)
    (he : 0 < eps) (heta : 0 < rate) (hd : 0 < 1 - rate * decay) (hs : 0 < seed) :
    0 < adaptiveSeedProduct remaining eps decay rate seed := by
  have hmu := initial_ce_gradient_magnitude_pos remaining
  unfold adaptiveSeedProduct
  positivity

example : 0 < (1 / 100000000 : ℝ) ∧ 0 < (1 / 1000 : ℝ) ∧
    0 < 1 - (1 / 1000 : ℝ) * (1 / 10) ∧ 0 < (1 / 200 : ℝ) := by norm_num

/-- If both Gen factors are zero, adaptive normalization also leaves
them zero on the first native update from zero moments. Sources:
arXiv:2309.02390v1, section 3's product circuit, and native AdamW at
43d4d66. This does not reset moments or assert a repeated-step law. -/
theorem zero_gen_seed_native_first_step (remaining : ℕ)
    (b1 b2 eps decay rate memSeed : ℝ) (h1 : b1 ≠ 1) (h2 : b2 ≠ 1) (hm : 0 ≤ memSeed) :
    (subweightAdamWFirstStep remaining b1 b2 eps decay rate ((0, 0), (0, memSeed))).1 = (0, 0) := by
  rw [subweight_adamw_initial_step remaining b1 b2 eps decay rate 0 memSeed h1 h2 (by norm_num) hm]
  simp

example : (9 / 10 : ℝ) ≠ 1 ∧ (49 / 50 : ℝ) ≠ 1 ∧ 0 ≤ (1 : ℝ) := by norm_num

end Transformer.Grokking.CircuitEfficiency
