module

public import Transformer.Basis.Encoding.Basic
public import Transformer.Basis.Encoding.Recall
public import Transformer.Basis.Encoding.Tasks

/-!
# Coverage of the actual model input by the raw Basis grammars

Source: Basis vocabulary/context recipes and synthetic raw-input parsers
at cbafbe9. Every valid prefix is nonempty, encodable and within its model
context. This includes all raw key/value records and both parity positions.
-/

@[expose] public section
