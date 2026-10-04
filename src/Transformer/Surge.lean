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
  (`SectionA_SignUpdate`).
-/

import Transformer.Surge.Section2_Lemma1
import Transformer.Surge.SectionA_AdamMoments
import Transformer.Surge.SectionA_SignUpdate
