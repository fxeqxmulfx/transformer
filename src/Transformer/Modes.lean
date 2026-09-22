/-
Formalization of:
  Geshkovski, Rigollet, Sun,
  "On the number of modes of Gaussian kernel density estimators",
  arXiv:2412.09080v3.

Not transcribed, deliberately:
* the intermediate displays of §5.4, whose exponents are unspecified `O(1)`s;
  they are steps of the proof of `lem:error-higher`, which is stated;
* the conjecture of §6 on `d`-dimensional KDEs, whose regime of `β` and `n`
  the source does not give, so there is no statement to write down.
-/

import Transformer.Modes.Growth
import Transformer.Modes.Section1_KDE
import Transformer.Modes.Section1_Mammen
import Transformer.Modes.Section1_Main
import Transformer.Modes.Section1_Belt
import Transformer.Modes.Section1_Sketch
import Transformer.Modes.Section2_KacRice
import Transformer.Modes.Section2_RandomLine
import Transformer.Modes.Section2_Field
import Transformer.Modes.Section2_Degenerate
import Transformer.Modes.Section2_MainForm
import Transformer.Modes.Section2_Gt
import Transformer.Modes.Section2_GaussianInt
import Transformer.Modes.Section5_GaussMoments
import Transformer.Modes.Section5_Moments
import Transformer.Modes.Section5_MomentsAsymp
import Transformer.Modes.Section5_CovAsymp
import Transformer.Modes.Section2_MomentsP
import Transformer.Modes.Section2_MomentsPCov
import Transformer.Modes.Section2_PhiT
import Transformer.Modes.Section2_PhiTQuot
import Transformer.Modes.Section2_PhiTUniform
import Transformer.Modes.Section2_PhiTAsymp
import Transformer.Modes.Section2_PhiTDelta
import Transformer.Modes.Section2_MainIntPhi
import Transformer.Modes.Section2_IntPhiB
import Transformer.Modes.Section3_Hermite
import Transformer.Modes.Section3_Cumulants
import Transformer.Modes.Section3_ExpMoments
import Transformer.Modes.Section3_MixedMoments
import Transformer.Modes.Section3_LogDeriv
import Transformer.Modes.Section3_PowDeriv
import Transformer.Modes.Section3_ScalarMGF
import Transformer.Modes.Section3_ScaledSumMGF
import Transformer.Modes.Section3_CumulantMoment
import Transformer.Modes.Section3_BR
import Transformer.Modes.Section3_Edgeworth
import Transformer.Modes.Section3_SigmaPos
import Transformer.Modes.Section3_Standardized
import Transformer.Modes.Section3_ChangeOfVar
import Transformer.Modes.Section3_ErrorThird
import Transformer.Modes.Section3_ErrorHigher
import Transformer.Modes.Section3_ErrorKR
import Transformer.Modes.Section4_KacRiceAppl
import Transformer.Modes.Section4_ScaleSpace
import Transformer.Modes.Section4_Tail
import Transformer.Modes.Section5_PtBdd
import Transformer.Modes.Section5_PtBddFourier
import Transformer.Modes.Section5_PtBddOne
