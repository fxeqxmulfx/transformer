module

public import Transformer.CausalRecall.Basic
public import Transformer.CausalRecall.Summary
public import Transformer.CausalRecall.Recurrent
public import Transformer.CausalRecall.MQAR

/-!
# Recall through a query-free summary

In no paper. A causal model that reads the context through a summary formed
before the query arrives answers the query from that summary alone. Zoology
(arXiv:2312.04927v1, Appendix `app:retnet-proof`, `cor: space-ar`) bounds such
recall by the index argument for one recurrent state.

`Transformer.CausalRecall.Basic` counts the dictionaries a message can answer,
exactly and in part. `Transformer.CausalRecall.Summary` defines causal models
whose every layer reads the states before a position through a summary in a
set `R`, proves that the output at an appended token depends only on it and
the `L` summaries of the context, and so bounds their recall by `|R|^L`.
`Transformer.CausalRecall.Recurrent` places finite-state recurrent networks in
the class. `Transformer.CausalRecall.MQAR` turns the bound into one on Zoology's
aligned multi-query recall: deciding whether `v` answers the last query over
`c` tokens needs `2^{c-1} ≤ |R|^L`, and few sets of keys are decided right on
many queries.
-/
