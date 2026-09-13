/-
# What Algorithms can Transformers Learn?

Formalization of Zhou, Bradley, Littwin, Razin, Saremi, Susskind, Bengio,
Nakkiran — arXiv:2310.16028v1, "What Algorithms can Transformers Learn?  A
Study in Length Generalization" (ICLR 2024).

The paper's thesis is that a transformer length-generalizes on exactly those
tasks that can be written as a short program in RASP-L, a causal and
"learnable" restriction of the RASP of `Transformer.RASP`.  Its own
mathematical content is small and sits in the appendices: the conjecture
(§3), the RASP-L core (Appendix C, Listing 3), the attention constructions
that compile an aggregation (Appendix C), and the two lemmas behind §4's
min-degree separation (Appendices D and E).

The index restrictions that make RASP-L *learnable* are, by the paper's own
statement, "an open question for future work"; they are not formalized here,
and the core is given with its causal `select` and nothing more.

| Module | Contents |
| --- | --- |
| `RASPL.Defs` | the RASP-L core of Listing 3: causal `select`, the aggregations, `kqv` |
| `RASPL.Attention` | one-hot encoding, the Constructability Lemma, and max as a shifted mean |
| `RASPL.Conjecture` | Realizability, Simplicity, Diversity, and the conjecture as a hypothesis |
| `RASPL.Fourier` | Fourier analysis on the hypercube, which §4's argument needs |
| `RASPL.Restrict` | fixing one coordinate, and what it does to the coefficients |
| `RASPL.Degree` | the degree profile, and Lemma E.1: restriction strictly lowers it |
| `RASPL.MinDegree` | Lemma 4.1: a min-degree interpolator ignores coordinates fixed on the train set |
| `RASPL.Gotu` | the one-line AND program, and that min-degree interpolation misses it |
-/

import Transformer.RASPL.Defs
import Transformer.RASPL.Attention
import Transformer.RASPL.Conjecture
import Transformer.RASPL.Fourier
import Transformer.RASPL.Restrict
import Transformer.RASPL.Degree
import Transformer.RASPL.MinDegree
import Transformer.RASPL.Gotu
