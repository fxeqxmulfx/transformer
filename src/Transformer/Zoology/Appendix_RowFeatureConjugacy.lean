/-
# Row factors are conjugates of feature factors

Arora et al., arXiv:2312.04927v1, Appendix `prop: butterfly-hyena`.
The exact bit exchange transports a row partner into a feature partner,
while permuting the actual coefficients by the same coordinate map.
This yields a 35-layer factor without additional rows or features.
-/

import Transformer.Zoology.Appendix_RowFeatureExchange

noncomputable section

namespace Transformer.Zoology

/-- The coordinate permutation commutes with pointwise linear combination.
Source: Appendix `prop: butterfly-hyena`, transport of diagonal coefficients. -/
theorem rowFeatureExchange_linear {p k : ℕ} (r : Fin p) (f : Fin k)
    (a b v w : RealSequence (butterflyWidth p) (butterflyWidth k)) :
    rowFeatureExchange r f (fun i q => a i q * v i q + b i q * w i q) =
      fun i q => rowFeatureExchange r f a i q * rowFeatureExchange r f v i q +
        rowFeatureExchange r f b i q * rowFeatureExchange r f w i q := by
  funext i q
  simp only [rowFeatureExchange]
  split_ifs <;> rfl

/-- Exchanging row and feature digits turns their partner reads into one another.
Source: Appendix `prop: butterfly-hyena`, same-shape binary routing. -/
theorem rowFeatureExchange_partner {p k : ℕ} (r : Fin p) (f : Fin k)
    (u : RealSequence (butterflyWidth p) (butterflyWidth k)) :
    rowFeatureExchange r f (fun i q => u i (butterflyToggleIndex k f.val q)) =
      fun i q => rowFeatureExchange r f u (butterflyToggleIndex p r.val i) q := by
  funext i q
  cases hr : butterflyToggleUpper p r.val i <;>
    cases hf : butterflyToggleUpper k f.val q <;>
      simp [rowFeatureExchange, hr, hf, butterflyToggleUpper_toggle,
        butterflyToggleIndex_involutive p r.val i,
        butterflyToggleIndex_involutive k f.val q]

/-- The permuted feature factor is exactly the original row factor.
Source: Appendix `prop: butterfly-hyena`, arbitrary coefficient diagonals. -/
theorem rowFeatureExchange_conjugacy {p k : ℕ} (r : Fin p) (f : Fin k)
    (a b u : RealSequence (butterflyWidth p) (butterflyWidth k)) :
    rowFeatureExchange r f (fun i q =>
      rowFeatureExchange r f a i q * rowFeatureExchange r f u i q +
        rowFeatureExchange r f b i q *
          rowFeatureExchange r f u i (butterflyToggleIndex k f.val q)) =
      fun i q => a i q * u i q + b i q * u (butterflyToggleIndex p r.val i) q := by
  rw [rowFeatureExchange_linear, rowFeatureExchange_involutive,
    rowFeatureExchange_involutive, rowFeatureExchange_involutive,
    rowFeatureExchange_partner, rowFeatureExchange_involutive]

/-- Compile a row factor by exact bit exchange, feature computation, and
inverse exchange. Source: Appendix `prop: butterfly-hyena`, constant-depth factor. -/
def inPlaceRowFactorNetwork {p k : ℕ} (r : Fin p) (f : Fin k)
    (a b : RealSequence (butterflyWidth p) (butterflyWidth k)) :
    CyclicKCoyoteNetwork (butterflyWidth p) k 1 :=
  ((rowFeatureExchangeNetwork r f).append
    (featurePairMatrixNetwork f (featureStagePairMatrix f
      (rowFeatureExchange r f a) (rowFeatureExchange r f b)))).append
    (rowFeatureExchangeNetwork r f)

/-- Every row-axis factor has exact original-layout semantics.
Source: Appendix `prop: butterfly-hyena`, part (1), including singular blocks. -/
theorem inPlaceRowFactorNetwork_correct {p k : ℕ} (r : Fin p) (f : Fin k)
    (a b u : RealSequence (butterflyWidth p) (butterflyWidth k)) :
    (inPlaceRowFactorNetwork r f a b).run u =
      fun i q => a i q * u i q + b i q * u (butterflyToggleIndex p r.val i) q := by
  rw [inPlaceRowFactorNetwork, CyclicKCoyoteNetwork.run_append,
    CyclicKCoyoteNetwork.run_append, rowFeatureExchangeNetwork_correct,
    featurePairMatrixNetwork_correct, featureStagePairMatrix_correct,
    rowFeatureExchangeNetwork_correct, rowFeatureExchange_conjugacy]

/-- Exact row-factor depth is 35, independent of n,d and the paired row digit.
Source: Appendix `prop: butterfly-hyena`, constant-depth factor component. -/
theorem inPlaceRowFactorNetwork_layerCount {p k : ℕ} (r : Fin p) (f : Fin k)
    (a b : RealSequence (butterflyWidth p) (butterflyWidth k)) :
    (inPlaceRowFactorNetwork r f a b).layerCount = 35 := by
  rw [inPlaceRowFactorNetwork, CyclicKCoyoteNetwork.layerCount_append,
    CyclicKCoyoteNetwork.layerCount_append, rowFeatureExchangeNetwork_layerCount,
    featurePairMatrixNetwork_layerCount]

end Transformer.Zoology
