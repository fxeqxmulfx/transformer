/-
# GGUF tensor formats

The tensor types llama.cpp stores model weights in, decoded from their bytes as
`ggml/src/ggml-quants.c` decodes them (llama.cpp master, commit `335b21f`).

| Module | Contents |
| --- | --- |
| `GGUF.Basic` | bytes and bit fields; IEEE binary16, bfloat16, binary32 |
| `GGUF.Nearest` | the dead zone of round-to-nearest, and the context length of softmax weights stored in a float |
-/

import Transformer.GGUF.Basic
import Transformer.GGUF.Nearest
