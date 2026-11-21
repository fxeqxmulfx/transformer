module

public import Transformer.GainMatroid.Section1_Projective
public import Transformer.GainMatroid.Section1_Elementary
public import Transformer.GainMatroid.Section1_Prop11
public import Transformer.GainMatroid.Section1_Canonical
public import Transformer.GainMatroid.Section1_Conjectures
public import Transformer.GainMatroid.Section2_Gains
public import Transformer.GainMatroid.Section2_Walks
public import Transformer.GainMatroid.Section2_Cycles
public import Transformer.GainMatroid.Section2_Balance
public import Transformer.GainMatroid.Section2_Biased
public import Transformer.GainMatroid.Section2_BiasedExamples
public import Transformer.GainMatroid.Section2_Forests
public import Transformer.GainMatroid.Section2_Normalized
public import Transformer.GainMatroid.Section2_Minors
public import Transformer.GainMatroid.Section2_Connectivity
public import Transformer.GainMatroid.Section2_Classes
public import Transformer.GainMatroid.Section4_Switching
public import Transformer.GainMatroid.Section4_Diagonal
public import Transformer.GainMatroid.Section4_Rows
public import Transformer.GainMatroid.Section4_Graphs
public import Transformer.GainMatroid.Section4_Relations
public import Transformer.GainMatroid.Section4_LiftRows
public import Transformer.GainMatroid.Section4_Small
public import Transformer.GainMatroid.Section4_SmallExamples

/-!
# Matrix representations of frame and lifted-graphic matroids

Funk, Pivotto and Slilaty, *Matrix representations of frame and lifted-graphic matroids correspond
to gain functions* (arXiv:1609.05574v5). The modules follow the sections of the manuscript.

`Section1_Projective` gives the row-space form of projective equivalence, `Section1_Elementary` the
source's elementary operations, and `Section1_Prop11` proves that the two agree and Proposition 1.1.
`Section1_Canonical` defines oriented graphs and the canonical frame and lift matrices.
`Section1_Conjectures` proves that Conjectures 1 and 2 fail as literally stated because of joints.
`Section2_Gains` defines switching and scaling of gain functions, `Section2_Walks` walks and paths,
and `Section2_Cycles` cycles and thetas. `Section2_Balance` gives the gain of a walk and proves that
balance of a cycle does not depend on the walk, `Section2_Biased` defines biased graphs and `B_γ`,
and `Section2_BiasedExamples` applies them to three parallel links. `Section2_Forests` proves
Proposition 2.1 and `Section2_Normalized` Proposition 2.2. `Section2_Minors` defines deletion,
contraction and induced gain functions and proves Proposition 2.3. `Section2_Connectivity` defines
separations and `k`-connected graphs and extends forests to maximal ones. `Section2_Classes`
defines balanced, almost-balanced, properly unbalanced and tangled biased graphs and joints.
`Section4_Switching` proves Proposition 4.1 and `Section4_Diagonal` proves Lemma 4.2.
`Section4_Rows` names what deleting a vertex leaves of a graph, `Section4_Graphs` defines `2C₃`, `K₄`
and the tube, and `Section4_Relations` and `Section4_LiftRows` read a projective equivalence off the rows of `T`.
`Section4_Small` deduces Lemmas 4.3 to 4.10, and `Section4_SmallExamples` meets their hypotheses.
-/
