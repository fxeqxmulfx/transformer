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
public import Transformer.GainMatroid.Section2_Matroids
public import Transformer.GainMatroid.Section2_LinkMinors
public import Transformer.GainMatroid.Section2_InducedRealization
public import Transformer.GainMatroid.Section2_InducedMinors
public import Transformer.GainMatroid.Section2_JointContractions
public import Transformer.GainMatroid.Section2_MatrixMinors
public import Transformer.GainMatroid.Section2_MatrixContractions
public import Transformer.GainMatroid.Section2_MatrixJoints
public import Transformer.GainMatroid.Section2_MatrixEquivalence
public import Transformer.GainMatroid.Section2_DeltaY
public import Transformer.GainMatroid.Section2_DeltaYExamples
public import Transformer.GainMatroid.Section2_CycleDegrees
public import Transformer.GainMatroid.Section2_FiniteMatroids
public import Transformer.GainMatroid.Section2_MatroidExchange
public import Transformer.GainMatroid.Section2_ExchangeGraphs
public import Transformer.GainMatroid.Section2_ExchangeMatroids
public import Transformer.GainMatroid.Section2_DeltaYMatroids
public import Transformer.GainMatroid.Section2_Representable
public import Transformer.GainMatroid.Section2_RepresentationExchange
public import Transformer.GainMatroid.Section2_ExchangeMatrices
public import Transformer.GainMatroid.Section2_GainExchange
public import Transformer.GainMatroid.Section2_ExchangeForms
public import Transformer.GainMatroid.Section2_TriadForms
public import Transformer.GainMatroid.Section2_TriadColumns
public import Transformer.GainMatroid.Section2_K4Forms
public import Transformer.GainMatroid.Section2_K4Matrix
public import Transformer.GainMatroid.Section2_ExchangedMatrices
public import Transformer.GainMatroid.Section2_BiasedMatroids
public import Transformer.GainMatroid.Section2_FullRankLift
public import Transformer.GainMatroid.Section2_UnbalancingClasses
public import Transformer.GainMatroid.Section2_UnbalancingRelation
public import Transformer.GainMatroid.Section4_Switching
public import Transformer.GainMatroid.Section4_Diagonal
public import Transformer.GainMatroid.Section4_Rows
public import Transformer.GainMatroid.Section4_Graphs
public import Transformer.GainMatroid.Section4_Relations
public import Transformer.GainMatroid.Section4_LiftRows
public import Transformer.GainMatroid.Section4_Small
public import Transformer.GainMatroid.Section4_SmallExamples
public import Transformer.GainMatroid.Section3_Classes
public import Transformer.GainMatroid.Section3_SmallGraphs
public import Transformer.GainMatroid.Section3_SmallExamples
public import Transformer.GainMatroid.Section5_ProperlyUnbalanced
public import Transformer.GainMatroid.Section1_ThreeVertices
public import Transformer.GainMatroid.Section1_Fan
public import Transformer.GainMatroid.Section1_Theorem1
public import Transformer.GainMatroid.Section4_TwoVertices
public import Transformer.GainMatroid.Section4_U24
public import Transformer.GainMatroid.Section4_TwoVertexCircuits
public import Transformer.GainMatroid.Section4_U24Matroid

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
`Section2_Matroids` defines frame and lift circuits, matroid connectivity and uniform matroids.
`Section2_LinkMinors` defines link minors, their induced gains and the two contractions of a joint.
`Section2_InducedRealization` lifts cycles of a link minor, and `Section2_InducedMinors` and
`Section2_JointContractions` prove Proposition 2.4. `Section2_MatrixMinors` defines `A / e` and
proves Lemma 2.5 (i), `Section2_MatrixContractions` proves it for links and refutes the printed
frame case, `Section2_MatrixJoints` treats joints, and `Section2_MatrixEquivalence` proves that
projective equivalence passes to minors of a matrix.
`Section2_DeltaY` defines the Δ-Y and Y-Δ exchanges of biased graphs and states Propositions 2.6
and 2.7, and `Section2_DeltaYExamples` shows that the printed `∇_Y B` can contain a non-cycle.
`Section2_CycleDegrees` proves that a vertex of a loopless graph meets a cycle in zero or two edges
and lists the seven cycles of `K₄`. `Section2_FiniteMatroids` builds finite matroids from their
independent sets and decides their flats. `Section2_MatroidExchange` defines `M(K₄)` and the Δ-Y and
Y-Δ exchanges of matroids, `Section2_ExchangeGraphs` and `Section2_ExchangeMatroids` compute them on
small graphs, and `Section2_DeltaYMatroids` states Proposition 2.8 and shows that part (ii) needs
`Y` to be a triad. `Section2_Representable` defines the matrix representations of a matroid and
their projective classes, and `Section2_RepresentationExchange` refutes Proposition 2.9 as printed
and states it for a coindependent triangle. `Section2_ExchangeMatrices` defines `I(K₄)` and the
exchanges of matrices and proves Proposition 2.11 for identity gains on the exchanged edges.
`Section2_GainExchange` refutes Proposition 2.10 (i) as printed and states it for gain functions
that are the identity on the triangle. `Section2_ExchangeForms` proves that projectively equivalent
matrices represent the same matroid and brings a represented triangle to the last three columns of
`I(K₄)`. `Section2_TriadForms` gives the row vector of a cocircuit, `Section2_TriadColumns` refutes
the printed claim that every triad takes the first three columns of `I(K₄)`, and `Section2_K4Forms`
proves it for an independent triad. `Section2_K4Matrix` proves that every representation of `M(K₄)`
is projectively equivalent to `I(K₄)`, and `Section2_ExchangedMatrices` states that the exchanged
matrices represent the exchanged matroids. `Section2_BiasedMatroids` states what the source cites
from Zaslavsky on `F(G, B)`, `L(G, B)`, their canonical matrices and minors, and proves that the
biased graph determines them and that they agree without vertex-disjoint unbalanced cycles.
`Section2_FullRankLift` deletes one row per component of a canonical lift matrix, and shows that the
rows left are independent with an unbalanced cycle and need not be without one.
`Section2_UnbalancingClasses` defines the unbalancing classes at a balancing vertex of an
almost-balanced biased graph, and `Section2_UnbalancingRelation` states what §2.5 claims of them.
`Section4_Switching` proves Proposition 4.1 and `Section4_Diagonal` proves Lemma 4.2.
`Section4_Rows` names what deleting a vertex leaves of a graph, `Section4_Graphs` defines `2C₃`,
`K₄` and the tube, and `Section4_Relations` and `Section4_LiftRows` read a projective equivalence
off the rows of `T`. `Section4_Small` deduces Lemmas 4.3 to 4.10, and `Section4_SmallExamples` meets
their hypotheses. `Section3_Classes` reads off cycles of two and three edges, `Section3_SmallGraphs`
decides which biased `2C₃`, tubes and `K₄` are properly unbalanced, and `Section3_SmallExamples`
gives concrete ones. `Section5_ProperlyUnbalanced` proves Theorem 5.1 from the rows of `T`.
`Section1_ThreeVertices` counts the edges of circuits on three vertices, `Section1_Fan` realizes
`U_{3,5}` as a frame matroid on three vertices, and `Section1_Theorem1` proves from it that
Theorem 1 (i) fails as printed on an almost-balanced biased graph. `Section4_TwoVertices` treats
the columns of loops and the cycles on two vertices, and `Section4_U24` proves Lemmas 4.11 and 4.12.
`Section4_TwoVertexCircuits` counts the edges of circuits on two vertices, and `Section4_U24Matroid`
proves that `𝖴₂` and `𝖴₃` represent `U_{2,4}`.
-/
