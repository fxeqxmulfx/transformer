import Transformer.Grokking.CircuitEfficiency.SectionC_AdaptiveFirstStep

/-!
# The source's squared seed suppression does not transfer to AdamW

Sources: Varma et al., arXiv:2309.02390v1, section 3, Slow vs fast
learning, appendix C's seeds and finite-class CE; native AdamW at lab
commit 43d4d66, PyTorch 2.14.1 first update. Compare actual product
growth with zero moment buffers and a positive remaining decay factor.

For 0<genSeed<memSeed and positive epsilon, the native first product
ratio is strictly between the squared seed ratio of ordinary GD and
the unsquared seed ratio. Adaptive normalization attenuates the
source's initialization-based slowness, but still initially favors Mem.
The exact ratio is genSeed^2*(mu*memSeed+eps) divided by
memSeed^2*(mu*genSeed+eps), where mu=(q-1)/q.
No repeated-step convergence, learned-circuit decomposition or delayed
crossing for GPTMini follows from this exact-real first-step comparison.
-/

namespace Transformer.Grokking.CircuitEfficiency

/-- Common rate/decay factors cancel, leaving the seed-dependent native
ratio. Sources: arXiv:2309.02390v1, appendix C's product coordinates,
and native AdamW at 43d4d66; positivity prevents vanishing denominators. -/
theorem adaptive_seed_product_ratio (remaining : ℕ) (eps decay rate genSeed memSeed : ℝ)
    (he : 0 < eps) (heta : 0 < rate) (hd : 0 < 1 - rate * decay)
    (hg : 0 ≤ genSeed) (hm : 0 < memSeed) :
    adaptiveSeedProduct remaining eps decay rate genSeed /
      adaptiveSeedProduct remaining eps decay rate memSeed =
        genSeed ^ 2 * (initialCEGradientMagnitude remaining * memSeed + eps) /
          (memSeed ^ 2 * (initialCEGradientMagnitude remaining * genSeed + eps)) := by
  have hmu := initial_ce_gradient_magnitude_pos remaining
  have hgden : initialCEGradientMagnitude remaining * genSeed + eps ≠ 0 := by positivity
  have hmden : initialCEGradientMagnitude remaining * memSeed + eps ≠ 0 := by positivity
  unfold adaptiveSeedProduct
  field_simp

example : 0 < (1 / 100000000 : ℝ) ∧ 0 < (1 / 1000 : ℝ) ∧
    0 < 1 - (1 / 1000 : ℝ) * (1 / 10) ∧ 0 ≤ (1 / 200 : ℝ) ∧ 0 < (1 : ℝ) := by norm_num

/-- Strictly bound the adaptive product ratio by the squared and
unsquared seed ratios. Sources: arXiv:2309.02390v1, section 3's squared
GD suppression, compared with native AdamW at 43d4d66; positive epsilon
is retained for the strict upper bound. -/
theorem adaptive_seed_ratio_bounds (remaining : ℕ) (eps decay rate genSeed memSeed : ℝ)
    (he : 0 < eps) (heta : 0 < rate) (hd : 0 < 1 - rate * decay)
    (hg : 0 < genSeed) (hgm : genSeed < memSeed) :
    (genSeed / memSeed) ^ 2 < adaptiveSeedProduct remaining eps decay rate genSeed /
      adaptiveSeedProduct remaining eps decay rate memSeed ∧
    adaptiveSeedProduct remaining eps decay rate genSeed /
      adaptiveSeedProduct remaining eps decay rate memSeed < genSeed / memSeed := by
  have hm : 0 < memSeed := hg.trans hgm
  have hmu := initial_ce_gradient_magnitude_pos remaining
  have hgden : 0 < initialCEGradientMagnitude remaining * genSeed + eps := by positivity
  have hden : 0 < memSeed ^ 2 * (initialCEGradientMagnitude remaining * genSeed + eps) := by positivity
  rw [adaptive_seed_product_ratio remaining eps decay rate genSeed memSeed he heta hd hg.le hm, div_pow]
  constructor
  · apply (div_lt_div_iff₀ (sq_pos_of_pos hm) hden).mpr
    have hgap : 0 < genSeed ^ 2 * memSeed ^ 2 * initialCEGradientMagnitude remaining * (memSeed - genSeed) := by
      positivity
    nlinarith [hgap]
  · apply (div_lt_div_iff₀ hden hm).mpr
    have hgap : 0 < genSeed * memSeed * eps * (memSeed - genSeed) := by positivity
    nlinarith [hgap]

example : 0 < (1 / 100000000 : ℝ) ∧ 0 < (1 / 1000 : ℝ) ∧
    0 < 1 - (1 / 1000 : ℝ) * (1 / 10) ∧ 0 < (1 / 200 : ℝ) ∧ (1 / 200 : ℝ) < 1 := by norm_num

/-- Transfer those bounds to actual native updated factor products,
using the proved CE gradient and first-step algorithm. Sources:
arXiv:2309.02390v1, appendix C, and native AdamW at 43d4d66. -/
theorem native_initial_seed_ratio_bounds (remaining : ℕ)
    (b1 b2 eps decay rate genSeed memSeed : ℝ)
    (h1 : b1 ≠ 1) (h2 : b2 ≠ 1) (he : 0 < eps) (heta : 0 < rate)
    (hd : 0 < 1 - rate * decay) (hg : 0 < genSeed) (hgm : genSeed < memSeed) :
    let next := subweightAdamWFirstStep remaining b1 b2 eps decay rate ((0, genSeed), (0, memSeed))
    (genSeed / memSeed) ^ 2 < (next.1.1 * next.1.2) / (next.2.1 * next.2.2) ∧
      (next.1.1 * next.1.2) / (next.2.1 * next.2.2) < genSeed / memSeed := by
  have hm : 0 < memSeed := hg.trans hgm
  have hproducts := subweight_adamw_initial_products remaining b1 b2 eps decay rate genSeed memSeed h1 h2 hg.le hm.le
  have hgen := congrArg Prod.fst hproducts
  have hmem := congrArg Prod.snd hproducts
  dsimp only at hgen hmem ⊢
  rw [hgen, hmem]
  exact adaptive_seed_ratio_bounds remaining eps decay rate genSeed memSeed he heta hd hg hgm

example : (9 / 10 : ℝ) ≠ 1 ∧ (49 / 50 : ℝ) ≠ 1 ∧ 0 < (1 / 100000000 : ℝ) ∧
    0 < (1 / 1000 : ℝ) ∧ 0 < 1 - (1 / 1000 : ℝ) * (1 / 10) ∧
    0 < (1 / 200 : ℝ) ∧ (1 / 200 : ℝ) < 1 := by norm_num

/-- With the source's q=113 and 0.005/1 seeds, the lab's native first
update suppresses Gen by a factor between 200 and 201, rather than
40000. Sources: arXiv:2309.02390v1, appendix C's seeds/class count,
and native hyperparameters at 43d4d66; these are combined explicitly. -/
theorem source_seeds_native_ratio_range :
    let next := subweightAdamWFirstStep 111 (9 / 10) (49 / 50)
      (1 / 100000000) (1 / 10) (1 / 1000) ((0, 1 / 200), (0, 1))
    (1 / 201 : ℝ) < (next.1.1 * next.1.2) / (next.2.1 * next.2.2) ∧
      (next.1.1 * next.1.2) / (next.2.1 * next.2.2) < (1 / 200 : ℝ) := by
  rw [subweight_adamw_initial_step 111 (9 / 10) (49 / 50) (1 / 100000000) (1 / 10)
    (1 / 1000) (1 / 200) 1 (by norm_num) (by norm_num) (by norm_num) (by norm_num)]
  norm_num [initialCEGradientMagnitude]

/-- Adaptive normalization still gives initial train fit and test failure
for the unequal positive seeds. Sources: arXiv:2309.02390v1, section 3's
lookup tables and native AdamW at 43d4d66; slower emergence after this
first update is not asserted without persistent-moment dynamics. -/
theorem native_initial_train_fit_test_failure (remaining : ℕ)
    (b1 b2 eps decay rate genSeed memSeed : ℝ)
    (h1 : b1 ≠ 1) (h2 : b2 ≠ 1) (he : 0 < eps) (heta : 0 < rate)
    (hd : 0 < 1 - rate * decay) (hg : 0 < genSeed) (hgm : genSeed < memSeed) :
    let next := subweightAdamWFirstStep remaining b1 b2 eps decay rate ((0, genSeed), (0, memSeed))
    Transformer.Grokking.NaiveLoss.StrictCorrect
      (trainTableLogits remaining (next.1.1 * next.1.2) (next.2.1 * next.2.2)) 0 ∧
    ¬ Transformer.Grokking.NaiveLoss.StrictCorrect
      (heldoutTableLogits remaining (next.1.1 * next.1.2) (next.2.1 * next.2.2)) 0 := by
  have hm : 0 < memSeed := hg.trans hgm
  have hproducts := subweight_adamw_initial_products remaining b1 b2 eps decay rate genSeed memSeed h1 h2 hg.le hm.le
  have hgen := congrArg Prod.fst hproducts
  have hmem := congrArg Prod.snd hproducts
  dsimp only at hgen hmem ⊢
  rw [hgen, hmem]
  have hp := adaptive_seed_product_pos remaining eps decay rate genSeed he heta hd hg
  have hq := adaptive_seed_product_pos remaining eps decay rate memSeed he heta hd hm
  have hratio := (adaptive_seed_ratio_bounds remaining eps decay rate genSeed memSeed he heta hd hg hgm).2
  have hs : genSeed / memSeed < 1 := (div_lt_iff₀ hm).mpr (by simpa using hgm)
  have hi : adaptiveSeedProduct remaining eps decay rate genSeed < adaptiveSeedProduct remaining eps decay rate memSeed := by
    have hh := (div_lt_iff₀ hq).mp (hratio.trans hs)
    simpa using hh
  exact ⟨(train_table_strict_correct_iff remaining _ _).mpr (by linarith), heldout_table_incorrect remaining _ _ hi⟩

example : (9 / 10 : ℝ) ≠ 1 ∧ (49 / 50 : ℝ) ≠ 1 ∧ 0 < (1 / 100000000 : ℝ) ∧
    0 < (1 / 1000 : ℝ) ∧ 0 < 1 - (1 / 1000 : ℝ) * (1 / 10) ∧
    0 < (1 / 200 : ℝ) ∧ (1 / 200 : ℝ) < 1 := by norm_num

/-- The same source seeds instantiate the actual table decisions,
as well as a product ratio, using the actual updated factors. -/
example :
    let next := subweightAdamWFirstStep 111 (9 / 10) (49 / 50)
      (1 / 100000000) (1 / 10) (1 / 1000) ((0, 1 / 200), (0, 1))
    Transformer.Grokking.NaiveLoss.StrictCorrect
      (trainTableLogits 111 (next.1.1 * next.1.2) (next.2.1 * next.2.2)) 0 ∧
    ¬ Transformer.Grokking.NaiveLoss.StrictCorrect
      (heldoutTableLogits 111 (next.1.1 * next.1.2) (next.2.1 * next.2.2)) 0 := by
  exact native_initial_train_fit_test_failure 111 (9 / 10) (49 / 50)
    (1 / 100000000) (1 / 10) (1 / 1000) (1 / 200) 1
    (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num)

end Transformer.Grokking.CircuitEfficiency
