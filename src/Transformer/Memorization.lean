/-
# How much do language models memorize?

Formalization of arXiv:2505.24832v3, Morris, Sitawarin, Guo,
Kokhlikyan, Suh, Rush, Chaudhuri, Mahloujifar (June 2025).
The PDF, TeX source, and numerical tables are stored locally under
papers/arXiv-2505.24832.

Coverage:
* Section 2.1: actual Shannon mutual information and conditional mutual
  information in bits; intended/unintended decomposition; nonnegativity;
  the trained-model entropy upper bound.
* Proposition 1 and Appendix A.6: super-additivity for finite datasets
  conditionally independent given the ground-truth model. The appendix's
  conditional independence qualification is explicit, and the result is
  proved without assuming an entropy inequality as a hidden premise.
* Section 2.2, Definitions 2 and 3: genuine shortest bit-program lengths,
  existence and minimality, missing descriptions represented by `none`,
  signed algorithmic information, and a documented data/model argument-order
  correction. The decoding semantics are explicit inputs, not new axioms.
* Proposition 4: a proved counterexample under Definition 2's stated
  arbitrary-computational-model generality. One uniform n-bit sample and
  identity training have n bits of Shannon memorization, while an actual
  computable reversible literal interpreter has zero algorithmic
  memorization. No length-independent comparison constant exists in that
  generality. A universal prefix-free-machine comparison is not asserted;
  it requires additional hypotheses and coding/distribution constants.
* Section 2.3: the likelihood estimator and the coding correction: pointwise
  maximum likelihood need not be normalized, while an equal mixture costs
  at most one extra ideal bit relative to that maximum.
* Definition 5 and Section 3: capacity as a supremum over input probability
  distributions; agreement with an attained maximum; observational lower
  bounds; finite-model-state and binary-storage upper bounds; nonnegative
  intended memorization for deterministic learning.
* Section 3.2: exact uniform-token dataset entropy N*S*log₂(V), including
  the paper's 704 bits per default synthetic sequence. All sixteen Table 1
  rows certify the stated precision trend and rounded mean bpp values.
* Section 5.2: the exact printed sigmoid scaling formula and coefficients,
  its actual large-dataset limit above 0.835 rather than 0.5, and its
  increasing rather than decreasing dataset-size dependence. All six
  Table 2 rows are retained, including a 9.31-point validation discrepancy;
  "generally within 1.5 points" cannot be read as a universal bound.

The finite-alphabet Shannon results apply directly to bounded-length,
finite-vocabulary datasets and finite-precision model encodings. The
empirical tables certify printed rounded measurements, not retraining,
population scaling guarantees, grokking, or all experimental plot curves.
The approximate 3.6-bpp observation is not promoted to a universal law.
No new sorry or extra axioms are introduced.

Finite entropy support is the required 27-module subset of teorth/pfr,
copied with its Apache 2.0 license in third_party/Entropy and built against
the repository's existing Lean/Mathlib 4.34.1. No external package is added.
scripts/memorization_data.py regenerates the numerical modules using exact
printed rational values and source SHA256 hashes; --check verifies them.
scripts/MemorizationAxioms.lean audits all paper and vendored entropy
declarations, including their transitive proof dependencies.
-/

import Transformer.Memorization.Section2_Proposition1
import Transformer.Memorization.Section2_Proposition4
import Transformer.Memorization.Section2_Compression
import Transformer.Memorization.Section3_Capacity
import Transformer.Memorization.Section3_PrecisionResults
import Transformer.Memorization.Section5_Membership
import Transformer.Memorization.Section5_ValidationResults
