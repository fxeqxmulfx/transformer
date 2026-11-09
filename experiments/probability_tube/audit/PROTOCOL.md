# Checkpoint probability audit implementation

This specifies the implementation of the parent protocol before inspecting
training probabilities or running the constructor. Keep all 144 anchors and
all declared horizons. Source recreation is a separate operation. Construction
plus verification has one cumulative 1800-second budget; compilation is excluded.

The constructor opens only each seed's identity and the checkpoint at the
requested anchor. It regenerates the exact training input list from the pinned
source and validates its identity. It may execute the old model on all 4704
training inputs. It cannot open training reports, traces, later checkpoints,
gradients or later outcomes. Freeze each forecast payload and its SHA-256, then
the complete construction report, before a separate verifier reads future data.

Use the committed EM1–EM5 formulas, every expanded parameter coordinate, saved
moments, and the known FP32 schedule. Old matrix operator caps are Frobenius
norms. Batch FP64 matrix products cache the old intermediate norms; this does
not change the scalar encoder-radius formula. Keep the existing warm-cone
relative allowance 1e-6 and the per-coordinate execution guard
`8*f32::EPSILON*(1+max_parameter+eta*max_direction*(1+1e-6))` from
`future_forcing`. Propagate this guard through every block, including repeated
matrix entries. These are conditional numerical allowances, not proved CUDA
rounding bounds. The verifier checks actual endpoint displacement against them.

For each input, bound every unused-versus-rival contrast using the largest
old contrast norm over those pairs, largest summed row-motion radius and that
input's full feature-motion radius. Keep all 97 classes in its old softmax.
Apply the sharp uniform-contrast ceiling to the largest old unused-class
probability for that input, then take the maximum over all 4704 inputs. This
uses a conservative shared contrast budget; it does not select a competitor
or discard denominator terms. Record each input's ceiling and feature radius.

For positive horizons the execution-contrast change allowance is
`(2*old_contrast_norm+row_motion)*1e-3+4e-3`. This follows from the existing
feature-error allowance 1e-3 and readout-dot error allowance 1e-3, applied at
both endpoints. At horizon zero the probability equals the measured old
probability. Separately check both execution allowances at old and later
checkpoints. Keep the probability comparison allowance 1e-8 and report maximum
excesses before tolerance. FP64 evaluation is not outward-rounded certification.

Reuse the isotropic and checkpoint-fixed feature metric from `ce_metric`.
For every horizon retain q=1, the old q without motion (invalid as a forecast),
and the transported q, together with their raw fixed-reference CE coefficients.
The ordinary feature energy cap is `128*(1+5e-6)`; the ordinary readout-energy
conversion includes the largest metric weight. Compare the full saved
parameter/moment potential at every endpoint. At horizon one independently
recompute the actual old minibatch partial CE, use the source GPU contraction
and clipping norm, and retain the full `ce_metric` signed decomposition.
Do not interpret a partial-reference condition as full AdamW nonincrease.

The verifier checks forecast hashes and reconstructs each constructor payload
using the same old-only interface before opening its future checkpoints.
For every endpoint, check full parameter displacement, feature displacement,
execution allowances, contrast increase and unused-class probabilities on all
training inputs. Keep every error, missing endpoint, cap and increasing full
potential. A useful horizon is a declared positive horizon whose transported
raw CE condition passes; zero is an admissible result. Finite endpoint checks
do not assert verification of every intervening floating-point operation.
