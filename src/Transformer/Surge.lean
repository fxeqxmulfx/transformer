/-
Formalization of:
  Li, Zhao, Zhang, Sun, Wu, Jiao, Wang, Liu, Fang, Xue, Tao, Cui, Wang,
  "Surge Phenomenon in Optimal Learning Rate and Batch Size Scaling",
  arXiv:2405.14578v5.

Deviations from the source, each recorded in the docstring of the file that
makes it:
* the loss along an update is the quadratic model of arXiv:1812.06162, as the
  paper evaluates it, and Lemma 1 needs `tr(H cov(V)) + E[V]ᵀHE[V] > 0`, which
  it leaves implicit (`Section2_Lemma1`);
* the limit of eq. (29) needs `G_t ≠ 0`, and fails without it
  (`SectionA_SignUpdate`);
* eq. (36) needs the coordinates of `G_est` independent: the per-sample gradients of all
  the parameters are taken independent (`SectionC_SignMoments`);
* Theorem 2, as Lemma 1, needs a positive denominator, which a positive semidefinite
  nonzero Hessian gives (`Section2_Theorem2`);
* the "≈" of eq. (10) is not an equality; it holds as a ratio tending to `1` as `B → 0`
  and as `B → ∞` (`Section2_SignApprox`);
* the "≈" of eqs. (13) and (39), for `B ≪ πσ_i²/(2μ_i²)`, holds as a ratio tending to `1` as
  `B → 0`, the one of eq. (39) for `μ_i ≠ 0` (`Section2_SmallBatchLimit`).
-/

import Transformer.Surge.Section2_Lemma1
import Transformer.Surge.Section2_PeakRate
import Transformer.Surge.Section2_SignApprox
import Transformer.Surge.Section2_SmallBatch
import Transformer.Surge.Section2_SmallBatchLimit
import Transformer.Surge.Section2_Theorem2
import Transformer.Surge.SectionA_AdamMoments
import Transformer.Surge.SectionA_SignUpdate
import Transformer.Surge.SectionC_SignMoments
