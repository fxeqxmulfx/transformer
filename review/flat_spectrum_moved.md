# Sections moved out of the flat-spectrum note

On 2026-11-19 the sections of [`flat_spectrum.md`](flat_spectrum.md) on other subjects moved,
unchanged, to the files below. A citation of `review/flat_spectrum.md` with one of these
sections or labels refers to the file named here.

| Sections | Labels | Now in |
| --- | --- | --- |
| *Actual network expressivity for every exponent*, *General positional models*, *Transformer learning bridges*, *Finite sequence capacity and information limits*, *Prescribed RoPE capacity*, *Depth, width and precision*, *Uniform nonperiodic RoPE*, *Quantitative RoPE approximation*, *Local attention routing* | G1–G5, R1, S1–S2, T1–T4, E1–E8, N1–N2, F1–F6, P1–P6, R1–R5, U1–U4, A1–A6, LR1–LR4 | [`actual_network/sequence_models.md`](actual_network/sequence_models.md) |
| *Full-stack rounding certificates*, *Primitive execution certificates*, *Unified spectral transformer certificate*, *Concrete backend contracts*, *Complete captured execution*, *Trace-dependent execution bounds* | B1–B9, C1–C8, J1–J7, K1–K6, L1–L5, M1–M5 | [`actual_network/execution.md`](actual_network/execution.md) |
| *Training-state transport*, *Loss-controlled geometry*, *Probability transfer*, *Signed directional transport*, *Residual learning dynamics*, *Adaptive interaction control*, *Actual single-update evidence*, *Joint difficult-query control*, *Feature forecasts*, *Forecast tightening*, *Training-to-forecast bridge* | TT1–TT5, LG1–LG5, PT1–PT5, ST1–ST5, RD1–RD5, AI1–AI5, JC1–JC4, FF1–FF3, FT1–FT3, TF1–TF4 | [`actual_network/transport.md`](actual_network/transport.md) |
| *Observable spectral geometry*, *Class-contrast discovery and persistence*, *Forward–Forward reformulation*, *Main-model AdamW motion*, *Observed finite-window persistence*, *Future forcing from checkpoint moments*, *Loss-specific momentum control*, *Checkpoint-fixed feature metrics*, *Prospective probability control*, *Full encoder motion*, *Pairwise encoder amplification*, *Query-weighted feature covariance* | OG1–OG5, CG1–CG6, LF1–LF8, AD1–AD5, DP1–DP3, FF1–FF4, CL1–CL4, FM1–FM4, PM1–PM4, EM1–EM5, PA1–PA2, QG1–QG4 | [`actual_network/dynamics.md`](actual_network/dynamics.md) |
| *Alias symmetry and training*, *Sampled updates can break aliases*, *Learning transfer*, *Optimizer memory*, *Gradient forcing* | S1–S7, T1–T6, M1–M6, F1–F4 | [`schedule_free/training.md`](schedule_free/training.md) |
| *Convex answer-loss recall* through *Convex spectral upper bounds* | CR, CC1–CC8, B1–B3, MF, CFt, CF, O, CI, SB | [`convex/channels.md`](convex/channels.md) |
| *Coupling complex class moments* | CM1–CM120 | [`convex/moments.md`](convex/moments.md) |
| *The wrap of a step is a comparison*, *The step is the phase of a sum of two phases*, *The rate of a recipe*, *A comparison of two free scores*, *The numbers of the tokens*, *The encoder: steps and levels*, *The carry of the codec*, *The decoder holds the level* | CW1–CW10, CH1–CH9, RR1–RR7, CS1–CS7, NT1–NT8, EN1–EN6, CC1–CC7, DL1–DL6 | [`diffgpt/steps.md`](diffgpt/steps.md) |
| Two paragraphs of *Experimental corrections to learning implications*, on the residual audit and the depth-attention toy | — | the end of [`residual_connections.md`](residual_connections.md) |

Some letters name items in more than one file: B, F, M, S and T in `actual_network/`,
`schedule_free/` and `convex/`, CC in `diffgpt/` and `convex/`, and FF in *Feature forecasts*
and in *Future forcing from checkpoint moments*. The section a citation names decides. The
R1–R4 of *Flat kernels of any shape* stay in `flat_spectrum.md`.

Their Lean left `Transformer.Grokking.FlatSpectrum` the same day. The DiffGPT modules are
in `Transformer.Grokking.DiffGPT`. The convex ones are in `Transformer.Grokking.Convex`,
with those of [`convex_depth.md`](convex_depth.md) and [`convex_causal.md`](convex_causal.md);
only B1–B3 of `convex/channels.md` stay in `Transformer.Grokking.FlatSpectrum`.
