module

public import Transformer.GainMatroid.Section1_Projective
public import Transformer.GainMatroid.Section1_Elementary
public import Transformer.GainMatroid.Section1_Prop11
public import Transformer.GainMatroid.Section1_Canonical
public import Transformer.GainMatroid.Section2_Gains
public import Transformer.GainMatroid.Section2_Walks
public import Transformer.GainMatroid.Section2_Cycles
public import Transformer.GainMatroid.Section4_Switching
public import Transformer.GainMatroid.Section4_Diagonal

/-!
# Matrix representations of frame and lifted-graphic matroids

Funk, Pivotto and Slilaty, *Matrix representations of frame and lifted-graphic matroids correspond
to gain functions* (arXiv:1609.05574v5). The modules follow the sections of the manuscript.

`Section1_Projective` gives the row-space form of projective equivalence, `Section1_Elementary` the
source's elementary operations, and `Section1_Prop11` proves that the two agree and Proposition 1.1.
`Section1_Canonical` defines oriented graphs and the canonical frame and lift matrices.
`Section2_Gains` defines switching and scaling of gain functions, `Section2_Walks` walks and paths,
and `Section2_Cycles` cycles and thetas.
`Section4_Switching` proves Proposition 4.1 and `Section4_Diagonal` proves Lemma 4.2.
-/
