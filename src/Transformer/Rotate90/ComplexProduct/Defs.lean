module

public import Transformer.Rotate90.Defs
public import Transformer.Grokking.FlatSpectrum.FeatureCodes.NonuniqueRecovery
public import Transformer.Grokking.FlatSpectrum.FeatureCodes.Margins

/-!
# The native complex-product Rotate90 model

Source: `review/flat_spectrum.md`, *Complex-product Rotate90 bridge*, CP1;
`rust_projects/grokking/experiments/rotate90_complex`, implementation b70811e.
This is the plain product arm in exact arithmetic: shared E, fixed quarter-turn,
all four RMSNorms, complex-linear projections, product divided by sqrt(2), and
one unscaled residual branch. The real expansion is defined independently.
Coordinates are logical pairs; the Rust hidden buffer stores its real and
imaginary halves separately. This change of indexing is not a PTX verification.
Floating-point execution and optimizer convergence are not definitions here.
-/

@[expose] public section
noncomputable section

namespace Transformer.Rotate90.ComplexProduct

open Landscape.Neural
open scoped BigOperators

/-- Adjacent real/imaginary coordinates of a complex vector (CP1). -/
def realify {K : Type*} (z : K → ℂ) : Channels K → ℝ :=
  fun i => if i.2 = 0 then (z i.1).re else (z i.1).im

/-- Input-major real expansion of a complex-linear matrix (CP1). -/
def realMatrix {K H : Type*} (A : K → H → ℂ) : Channels K → Channels H → ℝ :=
  fun i j => if j.2 = 0 then
    (if i.2 = 0 then (A i.1 j.1).re else -(A i.1 j.1).im)
  else (if i.2 = 0 then (A i.1 j.1).im else (A i.1 j.1).re)

/-- The four independent parameter arrays of the native model (CP1). -/
structure Weights (V K H : Type*) where
  embedding : V → K → ℂ
  gate : K → H → ℂ
  value : K → H → ℂ
  output : H → K → ℂ

variable {V K H : Type*} [Fintype K] [Fintype H]

/-- Ordinary row-vector complex matrix multiplication (CP1). -/
def denseC (A : K → H → ℂ) (z : K → ℂ) (j : H) : ℂ := ∑ i, z i * A i j

/-- RMS averages over the 2|K| real coordinates, as the Rust kernel does (CP1). -/
def energy (z : K → ℂ) : ℝ := (∑ k, Complex.normSq (z k)) / (2 * Fintype.card K)

/-- The positive-epsilon normalization denominator, retaining the real-coordinate factor two. -/
def denominator (eps : ℝ) (z : K → ℂ) : ℝ := Real.sqrt (eps + energy z)

/-- Complex notation for exactly the native real RMSNorm (CP1). -/
def norm (eps : ℝ) (z : K → ℂ) (k : K) : ℂ := z k / (denominator eps z : ℂ)

/-- The plain complex product, including its frozen variance calibration (CP1). -/
def product (g u : H → ℂ) (j : H) : ℂ := g j * u j / (Real.sqrt 2 : ℂ)

/-- Independent real implementation of the native four-product kernel (CP1). -/
def realProduct (g u : Channels H → ℝ) (j : Channels H) : ℝ :=
  (if j.2 = 0 then g (j.1, 0) * u (j.1, 0) - g (j.1, 1) * u (j.1, 1)
   else g (j.1, 0) * u (j.1, 1) + g (j.1, 1) * u (j.1, 0)) / Real.sqrt 2

/-- Complex product branch before output normalization, with actual trainable matrices. -/
def branch (w : Weights V K H) (z : K → ℂ) : K → ℂ :=
  denseC w.output (product (denseC w.gate z) (denseC w.value z))

/-- The one-block complex residual state, with three of the four RMSNorms (CP1). -/
def state (eps : ℝ) (w : Weights V K H) (a b : V) : K → ℂ :=
  let x := norm eps (fun k => w.embedding a k + Complex.I * w.embedding b k)
  fun k => x k + norm eps (branch w (norm eps x)) k

/-- A real Hermitian readout from the same embedding used at the inputs (CP1). -/
def readout (E : V → K → ℂ) (z : K → ℂ) (q : V) : ℝ :=
  ∑ k, (z k * star (E q k)).re

/-- The fourth RMSNorm and tied readout of the complex model (CP1). -/
def logit (eps : ℝ) (w : Weights V K H) (a b q : V) : ℝ :=
  readout w.embedding (norm eps (state eps w a b)) q

/-- The independently expanded three-matrix real branch (CP1). -/
def realBranch (w : Weights V K H) (x : Channels K → ℝ) : Channels K → ℝ :=
  dense (realMatrix w.output)
    (realProduct (dense (realMatrix w.gate) x) (dense (realMatrix w.value) x))

/-- The native residual formula, retaining the shared embedding and all input RMSNorms. -/
def realState (eps : ℝ) (w : Weights V K H) (a b : V) : Channels K → ℝ :=
  let x := rmsNorm eps (tiedInput (fun q => realify (w.embedding q)) a b)
  fun i => x i + rmsNorm eps (realBranch w (rmsNorm eps x)) i

/-- The actual expanded real model's normalized, tied output logits (CP1). -/
def realLogit (eps : ℝ) (w : Weights V K H) (a b q : V) : ℝ :=
  tiedReadout (fun q => realify (w.embedding q)) (rmsNorm eps (realState eps w a b)) q

end Transformer.Rotate90.ComplexProduct
