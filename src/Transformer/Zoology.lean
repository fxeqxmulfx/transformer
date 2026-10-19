module

public import Transformer.Zoology.Section3_MQAR
public import Transformer.Zoology.Section4_Coyote
public import Transformer.Zoology.Appendix_Attention
public import Transformer.Zoology.Appendix_SoftmaxRecall
public import Transformer.Zoology.Appendix_Shift
public import Transformer.Zoology.Appendix_Network
public import Transformer.Zoology.Appendix_RetNet
public import Transformer.Zoology.Appendix_CircuitDefs
public import Transformer.Zoology.Section5_SelectiveAttention
public import Transformer.Zoology.Section4_PairedRecall
public import Transformer.Zoology.Appendix_Butterfly
public import Transformer.Zoology.Appendix_ButterflyTree
public import Transformer.Zoology.Appendix_Kaleidoscope
public import Transformer.Zoology.Appendix_PaddedModel
public import Transformer.Zoology.Appendix_ShiftUp
public import Transformer.Zoology.Appendix_RunningSum
public import Transformer.Zoology.Appendix_RunningSumBlocks
public import Transformer.Zoology.Appendix_TwoLayerAttention
public import Transformer.Zoology.Appendix_AttentionMatrices
public import Transformer.Zoology.Appendix_CircuitPrimitives
public import Transformer.Zoology.Appendix_KaleidoscopeLinear
public import Transformer.Zoology.Appendix_KaleidoscopeMatrix
public import Transformer.Zoology.Appendix_KCoyote
public import Transformer.Zoology.Appendix_KCoyoteZero
public import Transformer.Zoology.Appendix_KCoyotePrimitives
public import Transformer.Zoology.Appendix_KCoyoteGates
public import Transformer.Zoology.Appendix_CircuitSemantics
public import Transformer.Zoology.Appendix_CircuitEvaluation
public import Transformer.Zoology.Appendix_KNetwork
public import Transformer.Zoology.Appendix_SmallCircuits
public import Transformer.Zoology.Appendix_DependentDistance
public import Transformer.Zoology.Appendix_DependentDistanceMany
public import Transformer.Zoology.Appendix_RequiredDistances
public import Transformer.Zoology.Appendix_PositionMask
public import Transformer.Zoology.Appendix_RawDistanceOffset
public import Transformer.Zoology.Appendix_Autocorrelation
public import Transformer.Zoology.Appendix_SequentialMQAR
public import Transformer.Zoology.Appendix_CircuitMemory
public import Transformer.Zoology.Appendix_CircuitGateLayers
public import Transformer.Zoology.Appendix_CircuitGateCorrectness
public import Transformer.Zoology.Appendix_CircuitPrefix
public import Transformer.Zoology.Appendix_CircuitSimulation
public import Transformer.Zoology.Appendix_CircuitBatchDefs
public import Transformer.Zoology.Appendix_CircuitBatchCorrectness
public import Transformer.Zoology.Appendix_CircuitParallelPrefix
public import Transformer.Zoology.Appendix_CircuitParallel
public import Transformer.Zoology.Appendix_CircuitCutWidth
public import Transformer.Zoology.Appendix_CyclicCircuitNetwork
public import Transformer.Zoology.Appendix_CyclicCircuitInput
public import Transformer.Zoology.Appendix_CyclicCircuitOutput
public import Transformer.Zoology.Appendix_CyclicCircuitSimulation
public import Transformer.Zoology.Appendix_CyclicCircuitEquivalence
public import Transformer.Zoology.Appendix_CircuitResourceBound
public import Transformer.Zoology.Appendix_CausalCircuitLimit
public import Transformer.Zoology.Appendix_ShiftSumLayout
public import Transformer.Zoology.Appendix_ShiftSumCopy
public import Transformer.Zoology.Appendix_ShiftSumSimulation
public import Transformer.Zoology.Appendix_CyclicKNetwork
public import Transformer.Zoology.Appendix_ShiftSumK
public import Transformer.Zoology.Appendix_BlockButterflyIndex
public import Transformer.Zoology.Appendix_BlockButterflyCoyote
public import Transformer.Zoology.Appendix_ShiftSumProgram
public import Transformer.Zoology.Appendix_DiagonalLinearLayout
public import Transformer.Zoology.Appendix_DiagonalLinearSimulation
public import Transformer.Zoology.Appendix_DiagonalLinearK
public import Transformer.Zoology.Appendix_ButterflyIdentity
public import Transformer.Zoology.Appendix_ButterflyToggle
public import Transformer.Zoology.Appendix_ButterflyToggleShift
public import Transformer.Zoology.Appendix_BinaryButterflyStage
public import Transformer.Zoology.Appendix_BinaryButterflyProgram
public import Transformer.Zoology.Appendix_BinaryButterflyTranspose
public import Transformer.Zoology.Appendix_ButterflyProgramMatrix
public import Transformer.Zoology.Appendix_RowMajorKaleidoscope
public import Transformer.Zoology.Appendix_ExpandedRowMajorCoyote
public import Transformer.Zoology.Appendix_RowMajorCoyoteResources
public import Transformer.Zoology.Appendix_ButterflyTreeLevels
public import Transformer.Zoology.Appendix_ButterflyTreeProduct
public import Transformer.Zoology.Appendix_ButterflyGridLayout
public import Transformer.Zoology.Appendix_ButterflyGridToggle
public import Transformer.Zoology.Appendix_ButterflyGridVectors
public import Transformer.Zoology.Appendix_ButterflyGridStages
public import Transformer.Zoology.Appendix_KaleidoscopeGrid
public import Transformer.Zoology.Appendix_KaleidoscopeCast
public import Transformer.Zoology.Appendix_ButterflyGridPadding
public import Transformer.Zoology.Appendix_ExpandedKaleidoscopeGrid
public import Transformer.Zoology.Appendix_ButterflyPairOrientation
public import Transformer.Zoology.Appendix_SingleLevelButterfly
public import Transformer.Zoology.Appendix_InPlaceLinear
public import Transformer.Zoology.Appendix_PairMatrixFactorization
public import Transformer.Zoology.Appendix_FeaturePairAction
public import Transformer.Zoology.Appendix_FeaturePairPrimitives
public import Transformer.Zoology.Appendix_FeaturePairShears
public import Transformer.Zoology.Appendix_FeaturePairMatrix
public import Transformer.Zoology.Appendix_ControlledFeatureSwap
public import Transformer.Zoology.Appendix_RowFeatureExchange
public import Transformer.Zoology.Appendix_RowFeatureConjugacy
public import Transformer.Zoology.Appendix_InPlaceButterflyProgram
public import Transformer.Zoology.Appendix_InPlaceWeightWidths
public import Transformer.Zoology.Appendix_ExactKaleidoscopeCoyote
public import Transformer.Zoology.Appendix_ExactKaleidoscopeParameters

/-!
# Zoology: Measuring and Improving Recall in Efficient Language Models

Formalization of Arora et al., arXiv:2312.04927v1. The modules follow the
main sections and theoretical appendix of the manuscript. This is partial:
the width-sensitive K-matrix circuit resource bound, the parallel MQAR
algorithm and its Coyote resource bounds, the randomized RetNet lower
bound, and the empirical claims remain to be formalized. A general
arithmetic DAG is simulated on a one-token ordinary-Coyote feature-memory
encoding with `2 * depth` layers and `inputs + size` feature width. A cyclic
Coyote construction now preserves the original `n × d` external layout with
`2 * depth + 6` layers and inner feature width `n*d + size`. The compressed
K-matrix and cut-width resource bounds need further work.
Primary-form butterfly factors now have an exact three-layer K-constrained
compiler preserving feature width. Products share 9n workspace rows;
primary-form width-w expanded K products use 9en rows, exactly d features,
and 6w log₂(end) layers, with exact stored parameter counts. An in-place
compiler for d ≥ 2 realizes the existing expanded recursive K class with
exactly en rows, d features, and at most 70w log₂(end) layers. Every actual
projection has hierarchy width one and expansion two; scalar storage is
proved from the stored layer data. The width-sensitive circuit factorization
remains open. Recursive butterfly trees and all existing BB*, Kaleidoscope,
and ExpandedKaleidoscope objects now have a proved transport to the original
mixed n × d layout. Binary partner indices, actual scalar matrix transpose,
hierarchy width, expansion padding and cropping are preserved; compiled
K networks realize those existing objects with unchanged feature width.
The causal-only variant cannot simulate even a one-gate future-reading
circuit; the cyclic variant can.
The input-dependent MQAR files
prove exact lookup for covered shifts and identify missing steps in the
paper's raw-position projections and autocorrelation selector.
-/

@[expose] public section
