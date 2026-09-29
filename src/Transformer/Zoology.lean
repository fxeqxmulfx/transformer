/-
# Zoology: Measuring and Improving Recall in Efficient Language Models

Formalization of Arora et al., arXiv:2312.04927v1.  The modules follow the
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

import Transformer.Zoology.Section3_MQAR
import Transformer.Zoology.Section4_Coyote
import Transformer.Zoology.Appendix_Attention
import Transformer.Zoology.Appendix_SoftmaxRecall
import Transformer.Zoology.Appendix_Shift
import Transformer.Zoology.Appendix_Network
import Transformer.Zoology.Appendix_RetNet
import Transformer.Zoology.Appendix_CircuitDefs
import Transformer.Zoology.Section5_SelectiveAttention
import Transformer.Zoology.Section4_PairedRecall
import Transformer.Zoology.Appendix_Butterfly
import Transformer.Zoology.Appendix_ButterflyTree
import Transformer.Zoology.Appendix_Kaleidoscope
import Transformer.Zoology.Appendix_PaddedModel
import Transformer.Zoology.Appendix_ShiftUp
import Transformer.Zoology.Appendix_RunningSum
import Transformer.Zoology.Appendix_RunningSumBlocks
import Transformer.Zoology.Appendix_TwoLayerAttention
import Transformer.Zoology.Appendix_AttentionMatrices
import Transformer.Zoology.Appendix_CircuitPrimitives
import Transformer.Zoology.Appendix_KaleidoscopeLinear
import Transformer.Zoology.Appendix_KaleidoscopeMatrix
import Transformer.Zoology.Appendix_KCoyote
import Transformer.Zoology.Appendix_KCoyoteZero
import Transformer.Zoology.Appendix_KCoyotePrimitives
import Transformer.Zoology.Appendix_KCoyoteGates
import Transformer.Zoology.Appendix_CircuitSemantics
import Transformer.Zoology.Appendix_CircuitEvaluation
import Transformer.Zoology.Appendix_KNetwork
import Transformer.Zoology.Appendix_SmallCircuits
import Transformer.Zoology.Appendix_DependentDistance
import Transformer.Zoology.Appendix_DependentDistanceMany
import Transformer.Zoology.Appendix_RequiredDistances
import Transformer.Zoology.Appendix_PositionMask
import Transformer.Zoology.Appendix_RawDistanceOffset
import Transformer.Zoology.Appendix_Autocorrelation
import Transformer.Zoology.Appendix_SequentialMQAR
import Transformer.Zoology.Appendix_CircuitMemory
import Transformer.Zoology.Appendix_CircuitGateLayers
import Transformer.Zoology.Appendix_CircuitGateCorrectness
import Transformer.Zoology.Appendix_CircuitPrefix
import Transformer.Zoology.Appendix_CircuitSimulation
import Transformer.Zoology.Appendix_CircuitBatchDefs
import Transformer.Zoology.Appendix_CircuitBatchCorrectness
import Transformer.Zoology.Appendix_CircuitParallelPrefix
import Transformer.Zoology.Appendix_CircuitParallel
import Transformer.Zoology.Appendix_CircuitCutWidth
import Transformer.Zoology.Appendix_CyclicCircuitNetwork
import Transformer.Zoology.Appendix_CyclicCircuitInput
import Transformer.Zoology.Appendix_CyclicCircuitOutput
import Transformer.Zoology.Appendix_CyclicCircuitSimulation
import Transformer.Zoology.Appendix_CyclicCircuitEquivalence
import Transformer.Zoology.Appendix_CircuitResourceBound
import Transformer.Zoology.Appendix_CausalCircuitLimit
import Transformer.Zoology.Appendix_ShiftSumLayout
import Transformer.Zoology.Appendix_ShiftSumCopy
import Transformer.Zoology.Appendix_ShiftSumSimulation
import Transformer.Zoology.Appendix_CyclicKNetwork
import Transformer.Zoology.Appendix_ShiftSumK
import Transformer.Zoology.Appendix_BlockButterflyIndex
import Transformer.Zoology.Appendix_BlockButterflyCoyote
import Transformer.Zoology.Appendix_ShiftSumProgram
import Transformer.Zoology.Appendix_DiagonalLinearLayout
import Transformer.Zoology.Appendix_DiagonalLinearSimulation
import Transformer.Zoology.Appendix_DiagonalLinearK
import Transformer.Zoology.Appendix_ButterflyIdentity
import Transformer.Zoology.Appendix_ButterflyToggle
import Transformer.Zoology.Appendix_ButterflyToggleShift
import Transformer.Zoology.Appendix_BinaryButterflyStage
import Transformer.Zoology.Appendix_BinaryButterflyProgram
import Transformer.Zoology.Appendix_BinaryButterflyTranspose
import Transformer.Zoology.Appendix_ButterflyProgramMatrix
import Transformer.Zoology.Appendix_RowMajorKaleidoscope
import Transformer.Zoology.Appendix_ExpandedRowMajorCoyote
import Transformer.Zoology.Appendix_RowMajorCoyoteResources
import Transformer.Zoology.Appendix_ButterflyTreeLevels
import Transformer.Zoology.Appendix_ButterflyTreeProduct
import Transformer.Zoology.Appendix_ButterflyGridLayout
import Transformer.Zoology.Appendix_ButterflyGridToggle
import Transformer.Zoology.Appendix_ButterflyGridVectors
import Transformer.Zoology.Appendix_ButterflyGridStages
import Transformer.Zoology.Appendix_KaleidoscopeGrid
import Transformer.Zoology.Appendix_KaleidoscopeCast
import Transformer.Zoology.Appendix_ButterflyGridPadding
import Transformer.Zoology.Appendix_ExpandedKaleidoscopeGrid
import Transformer.Zoology.Appendix_ButterflyPairOrientation
import Transformer.Zoology.Appendix_SingleLevelButterfly
import Transformer.Zoology.Appendix_InPlaceLinear
import Transformer.Zoology.Appendix_PairMatrixFactorization
import Transformer.Zoology.Appendix_FeaturePairAction
import Transformer.Zoology.Appendix_FeaturePairPrimitives
import Transformer.Zoology.Appendix_FeaturePairShears
import Transformer.Zoology.Appendix_FeaturePairMatrix
import Transformer.Zoology.Appendix_ControlledFeatureSwap
import Transformer.Zoology.Appendix_RowFeatureExchange
import Transformer.Zoology.Appendix_RowFeatureConjugacy
import Transformer.Zoology.Appendix_InPlaceButterflyProgram
import Transformer.Zoology.Appendix_InPlaceWeightWidths
import Transformer.Zoology.Appendix_ExactKaleidoscopeCoyote
import Transformer.Zoology.Appendix_ExactKaleidoscopeParameters
