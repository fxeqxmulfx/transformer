module

public import Transformer.SequenceModel.Pooling
public import Transformer.SequenceModel.RotarySpectrum
public import Transformer.SequenceModel.Transfer.Alignment
public import Transformer.SequenceModel.Attention.FiniteStep
public import Transformer.SequenceModel.Attention.ZeroRun
public import Transformer.SequenceModel.Attention.Expressivity.Existence
public import Transformer.SequenceModel.Attention.Finite.Witnesses
public import Transformer.SequenceModel.Attention.Grid.Model
public import Transformer.SequenceModel.Information.Distribution
public import Transformer.SequenceModel.Resources
public import Transformer.SequenceModel.Attention.Stability
public import Transformer.SequenceModel.Attention.Execution
public import Transformer.SequenceModel.Attention.Spectral
public import Transformer.SequenceModel.Attention.Backend
public import Transformer.SequenceModel.Attention.Retrieval
public import Transformer.SequenceModel.Attention.CenteredXSA
public import Transformer.SequenceModel.Attention.ConstantTokens
public import Transformer.SequenceModel.Attention.ConstantPrefixes

/-!
# General positional inputs and normalized multi-variable attention

Source: `review/flat_spectrum.md`, *General positional models*, G1--G5,
*Transformer learning bridges*, R1, S1--S2, T1--T4, E1--E8, and
*Finite sequence capacity and information limits*, F1--F6, N1--N2, and
*Prescribed RoPE capacity*, P1--P6.
The sampled source dynamics apply to arbitrary finite parameterized predictors.
Arbitrary rotations, per-pair RoPE frequencies, other positional mixers and
finite input tuples are included; fixed 90 degrees is only one special case.

The concrete attention model follows the user's diff_gpt at b98c665b, retaining
embedding/final RMSNorm, Peri-RMSNorm on both branches, Q/K RMSNorm, arbitrary
rotary phases, causal softmax, clipped XSA, SwiGLU and tied output. Inverse
coordinate maps prove that every trainable matrix entry participates.

Local analyticity is proved from actual XSA activation regularity. Actual
CE gradients and Adam preconditioning derive cross-Jacobian transfer and
finite prediction bounds. A finite source-shaped Transformer represents
every binary finite task, including powered division, with a strict margin.
Every positive finite arity also has a finite strict classifier with the
prescribed integer-position geometric RoPE grid and the original input tokens.
The final operand serves as readout; all normalizers and tied weights remain.
The earlier chosen-phase construction also covers zero operands with its
explicit appended readout token. RoPE spectrum and zero-initialization
counterexamples rule out unconditional spectral invariance and learning
guarantees. Neither capacity proof gives a floating-point conditioning bound
or convergence from typical random initialization.
Sampled-label indistinguishability and a finite-distribution unseen-label
bound identify information limits independently of architecture and optimizer size.
Finite parameter alphabets additionally give a necessary joint depth, width
and stored precision budget for universal classification; this is not a
task-specific lower bound for powered modular division. The normalized rounded
source's counting-logic compilers are in `Transformer.CRASP.Normalized` and
`Transformer.CRASP.Rotary`. Quantitative phase and attention-rounding bounds
keep saturation and underflow assumptions explicit. A strict exact-source
classifier at any fixed finite context admits a periodic frequency replacement
with unchanged weights, width and depth; its context-dependent period does
not transfer a uniform language depth lower bound.
Recovered powered-division feature scores also compile into the actual
normalized source, preserving the entire score table up to positive row
scaling. Their proved margin gives a positive local execution allowance.
Under explicit numerical certificates, a fixed RoPE-aware counting family
agrees with every executed class decision; see *Unified spectral transformer
certificate*, J1--J7. This uses ideal feature recovery and finite interpolation,
and does not establish optimizer convergence or efficient arithmetic circuits.
Concrete backend certificates in *Concrete backend contracts*, K1--K6,
derive primitive error bounds from algebraic and finite polynomial checks.
Captured CPU/CUDA outputs satisfy them by ordinary Lean proofs. A complete
compiled source invocation satisfies every ModelAccurate check (L1--L3).
Its unavoidable residual error proves that the current uniform global bound
cannot meet any attainable decision margin (L4--L5); this identifies the next
numerical gap without asserting source decision agreement or uniform backend
guarantees.
-/
