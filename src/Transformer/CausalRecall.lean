module

public import Transformer.CausalRecall.Basic
public import Transformer.CausalRecall.Summary
public import Transformer.CausalRecall.Recurrent
public import Transformer.CausalRecall.MQAR
public import Transformer.CausalRecall.Separation
public import Transformer.CausalRecall.QueryFirst

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
many queries. `Transformer.CausalRecall.Separation` separates from the class
the matcher of one layer and width `5`, which compares a query with a key: at
every depth and summary width, some length has the matcher decide every
instance, while each model of the class decides the index instances right on
three quarters of the queries for fewer than half of the sets of keys.

The bound is on the order of the query and the context, not on recurrence.
`Transformer.CausalRecall.QueryFirst` gives a recurrent network that reads the
query before the bindings and holds the key it waits for or the value it has
found: `|K| + |V|` states recall every dictionary, at least `max(|K|, |V|)` are
needed, and the same tokens with the query last need `|V|^|K|`.
-/
