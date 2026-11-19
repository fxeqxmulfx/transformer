module

public import Transformer.CausalRecall.Basic

/-!
# Recall through a query-free summary

In no paper. A causal model that reads the context through a summary formed
before the query arrives answers the query from that summary alone. Zoology
(arXiv:2312.04927v1, Appendix `app:retnet-proof`, `cor: space-ar`) bounds such
recall by the index argument for one recurrent state. This development counts
the dictionaries a summary can answer, exactly and in part
(`Transformer.CausalRecall.Basic`).
-/
