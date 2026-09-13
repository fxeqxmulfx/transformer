/-
Formalization of eight papers on the mathematics of Transformers:

1. Weiss, Goldberg, Yahav — arXiv:2106.06981v2
   "Thinking Like Transformers".

2. Geshkovski, Letrouit, Polyanskiy, Rigollet — arXiv:2312.10794v5
   "A mathematical perspective on Transformers".

3. Geshkovski, Koubbi, Polyanskiy, Rigollet — arXiv:2410.06833v1
   "Dynamic metastability in the self-attention model".

4. Geshkovski, Karagodin, Polyanskiy, Rigollet — arXiv:2411.04551v3
   "Measure-to-measure interpolation using Transformers".

5. Karagodin, Polyanskiy, Rigollet — arXiv:2411.04990v2
   "Clustering in Causal Attention Masking".

6. Karagodin, Polyanskiy, Rigollet — arXiv:2510.22026v2
   "Normalization in Attention Dynamics".

7. Geshkovski, Polyanskiy, Rigollet — arXiv:2512.01868v4
   "The Mean-Field Dynamics of Transformers" (survey).

8. Zhai — arXiv:2603.09078v1
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
import Transformer.RASP
import Transformer.Perspective
import Transformer.Metastability
import Transformer.Interpolation
import Transformer.Causal
import Transformer.Normalization
import Transformer.MeanField
import Transformer.XSA
import Transformer.GPTMini
