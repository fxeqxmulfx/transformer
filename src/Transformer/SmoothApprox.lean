/-
# Smooth approximation and smooth test functions

`SmoothApprox/Basic.lean` approximates continuous and `C¹` functions with
compact support on a finite-dimensional space by smooth ones with compact
support, uniformly and in the `C¹` norm, by mollification.
`SmoothApprox/Measure.lean` uses this to let smooth test functions determine
probability measures carried by a ball and to transfer weak continuity of a
curve of measures from smooth to continuous test functions.

Use: the weak formulation `d:solconti` of arXiv:2305.05465v6, §6, which tests
against `C_c^∞`.
-/

import Transformer.SmoothApprox.Basic
import Transformer.SmoothApprox.Measure
