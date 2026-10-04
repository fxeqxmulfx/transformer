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
  `B → 0`, the one of eq. (39) for `μ_i ≠ 0` (`Section2_SmallBatchLimit`);
* eq. (11) holds when the `𝓔_i(B)` share a profile, `𝓔_i(B) = f(B)μ_i/σ_i`, as in the
  linearization of eq. (39), for which eq. (12) is proved; the "≈" of eqs. (3) and (4) holds as
  a ratio tending to `1` as `B → 0` (`Section2_RateForm`);
* eq. (17) of Theorem 4 needs every `μ_i ≠ 0`, which its condition `B ≫ πσ_i²/(2μ_i²)` presumes
  without stating it, and fails without it (`Section2_LargeBatch`);
* the "≈" of eq. (18) of Theorem 5, for `B ≪ πσ_i²/(2μ_i²)`, holds as a ratio tending to `1` as
  `B → 0` (`Section2_LossDrop`);
* the "≈" of `B_noise ≈ B_crit` in eq. (21) is an equality at constant `μ, σ, H`, where `B_peak`
  is the balance point; in general `B_crit = E_min/S_min` is the mean of `B_noise` over the run.
  The fit `B_crit ≈ B_*/L^{1/α_B}` of eq. (21), from Kaplan et al., is a hypothesis
  (`Section2_Tradeoff`);
* eq. (22) holds with `B_crit = E_min/S_min` for `B_noise`, as eq. (21) allows, the two
  coinciding at constant `μ, σ, H`; the expectations of eqs. (23) and (24) are means over a
  finite grid of batches (`Section3_Estimation`).

Not transcribed, deliberately: the experiments of §3.1, §3.3 and Appendix H, the figures, and
the restatements of §1, §4, §5 and §6; the remark after Theorem 4 that late in training its limit
"is more likely to exceed the local maximum", a heuristic; the grid searches and curve fits of
§3.2, of which the identities behind eqs. (22) to (24) are proved.
-/

import Transformer.Surge.Section2_LargeBatch
import Transformer.Surge.Section2_Lemma1
import Transformer.Surge.Section2_LossDrop
import Transformer.Surge.Section2_PeakRate
import Transformer.Surge.Section2_RateForm
import Transformer.Surge.Section2_SignApprox
import Transformer.Surge.Section2_SmallBatch
import Transformer.Surge.Section2_SmallBatchLimit
import Transformer.Surge.Section2_Theorem2
import Transformer.Surge.Section2_Tradeoff
import Transformer.Surge.Section3_Estimation
import Transformer.Surge.SectionA_AdamMoments
import Transformer.Surge.SectionA_SignUpdate
import Transformer.Surge.SectionC_SignMoments
