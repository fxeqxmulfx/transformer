/-
# Deep double descent

Formalization of arXiv:1912.02292v1, Nakkiran, Kaplun, Bansal, Yang,
Barak, Sutskever, "Deep Double Descent: Where Bigger Models and More Data
Hurt". The PDF, TeX, and author-provided experimental data are preserved
locally in papers/arXiv-1912.02292v1.

Coverage:
* Section 2, Definition 1: actual iid empirical and population risks;
  extended-natural EMC, agreement with every existing finite maximum,
  tolerance monotonicity, and complexity improvement under training-risk
  improvement. The unrestricted maximum may fail to exist.
* Sections 1, 2, and 6: a gradient-descent counterexample to unconditional
  monotonicity of EMC in training time, and actual learners showing that
  higher EMC alone does not imply smaller population test risk, even
  arbitrarily far above the sample count.
* Section 4: uniform incorrect-label noise and its exact affine effect
  on classification risk. A sign error in the authors' plotting notebook
  is recorded in a proved counterexample.
* Sections 5--7: finite double-descent predicates and exact numerical
  witnesses from the published IWSLT/WMT model-size logs, a ResNet epoch
  trajectory, and both fixed-size Transformer sample-count logs. In each
  sample log, multiplying 4000 samples by 4.5 worsens recorded test loss.
* Section 8: finite-horizon optimal early stopping, attainment, and its
  monotonicity in the horizon. It need not remove model-wise double descent.
* Appendix B.3: token indexing, autoregressive likelihood, and nonnegative
  token NLL with correct infinite loss at probability zero.
* Appendix D: exact linear interpolation needs sufficient dimension and
  full rank for uniqueness. Zero-target fixed-feature regression has
  infinite EMC, so feature count is not a universal identity for EMC.

Hypothesis 1 remains an informal scientific conjecture: the paper supplies
no definitions of natural procedures, admissible perturbations, or critical
width. Its unrestricted EMC-based interpretation is refuted rather than
asserted under an impossible sorry. The numerical theorems certify the
published records; they do not prove population guarantees for retraining,
all architectures, optimizer robustness, or all the Appendix E experiments.
No new sorry, axioms, or external package dependencies are introduced.

The data extraction script uses exact CSV decimals and exact binary64
fractions, records source hashes, and supports --check. The data's plotting
labels and duplicated training-error column are documented where relevant.
-/

import Transformer.DoubleDescent.Section2_EffectiveComplexity
import Transformer.DoubleDescent.Section2_Hypothesis
import Transformer.DoubleDescent.Section4_LabelNoise
import Transformer.DoubleDescent.Section5_TranslationResults
import Transformer.DoubleDescent.Section6_ResNetResults
import Transformer.DoubleDescent.Section7_TranslationResults
import Transformer.DoubleDescent.Section8_EarlyStopping
import Transformer.DoubleDescent.AppendixB_Translation
import Transformer.DoubleDescent.AppendixD_Interpolation
import Transformer.DoubleDescent.AppendixD_FourierFeatures
