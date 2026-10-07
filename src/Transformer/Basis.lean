import Transformer.Basis.Basic
import Transformer.Basis.Depth
import Transformer.Basis.Recall
import Transformer.Basis.Parity
import Transformer.Basis.Tasks
import Transformer.Basis.Requirements
import Transformer.Basis.Encoding
import Transformer.Basis.RecallAnswer
import Transformer.Basis.RecallParsed
import Transformer.Basis.RecallTablePositions

/-!
# Basis on List Int

Raw integer-token semantics for depth E_2/E_4, easy/hard MQAR and
no-scratchpad parity. Each total function preserves its input and appends
one answer; two calls generate parity's answer and EOS. SolvesTask states
agreement on the supervised grammar within the actual context cap.

Source: domain/basis.py and infrastructure/benchmarks/synthetic/ at cbafbe9.
The real model adapter and prediction certificates are exported by
Transformer.GPTMini.TokenInterface. Necessary answer distinctions cover
depth order, raw MQAR binding/overwrites and parity bits. Every supervised
prefix is proved to be nonempty, encodable and within its actual context.
Raw recall parser inversion identifies the final adjacent matching write,
including unrelated later records and hard-mode overwrites.
The complete parser decomposition also retains all earlier/later table
and query/filler alphabet checks needed by the actual raw encoder.
Exact serialization indices prove value/key adjacency and bound every
matching record index by the parser's selected chronological last write.
-/
