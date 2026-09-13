/-
Formalization of seven papers on the mathematics of Transformers:

1. Geshkovski, Letrouit, Polyanskiy, Rigollet — arXiv:2312.10794v5
   "A mathematical perspective on Transformers".

2. Geshkovski, Koubbi, Polyanskiy, Rigollet — arXiv:2410.06833v1
   "Dynamic metastability in the self-attention model".

3. Geshkovski, Karagodin, Polyanskiy, Rigollet — arXiv:2411.04551v3
   "Measure-to-measure interpolation using Transformers".

4. Karagodin, Polyanskiy, Rigollet — arXiv:2411.04990v2
   "Clustering in Causal Attention Masking".

5. Karagodin, Polyanskiy, Rigollet — arXiv:2510.22026v2
   "Normalization in Attention Dynamics".

6. Geshkovski, Polyanskiy, Rigollet — arXiv:2512.01868v4
   "The Mean-Field Dynamics of Transformers" (survey).

7. Zhai — arXiv:2603.09078v1
   "Exclusive Self Attention".

Alongside these, `Transformer.ALM` formalizes the paraboloid-lifted lookup
used by the append-only lookup machine (Percepta, transformer-vm).

This top-level module re-exports the formalization of every section of all
papers.  Each per-paper subdirectory mirrors the structure of the source
manuscript, with one file per section.  Statements that have full proofs in
the paper are spelled out (modulo `sorry` for analytic machinery that is
beyond the scope of this formalization); definitions and equation displays
are translated directly.
-/

import Transformer.Basic
import Transformer.ALM
import Transformer.Section1_IPS
import Transformer.Section2_FlowMap
import Transformer.Section3_SmallBeta
import Transformer.Section4_LargeBeta
import Transformer.Section5_HighD
import Transformer.Section6_Circle
import Transformer.Section7_BBGKY
import Transformer.Section8_General
import Transformer.Section9_Approximation
import Transformer.AppendixA_Beta0
import Transformer.AppendixB_BetaInterval
import Transformer.AppendixC_BetaTiny
import Transformer.AppendixD_PhaseTransition
import Transformer.Metastability
import Transformer.Interpolation
import Transformer.Causal
import Transformer.Normalization
import Transformer.MeanField
import Transformer.XSA
import Transformer.GPTMini
