/-
# IC-EoT: scalar parameter accounting

arXiv:2603.22095v2, §3.2, Eqs. (14)–(27), and §4.3.2, Table 3.
The equation-defined independent-parameter architecture does not yield the
IC-EoT counts printed in Table 3. The difference is recorded explicitly;
no claim is made about an unavailable implementation's parameter tying.
-/

import Transformer.ICEoT.Section4_ReportedModels

namespace Transformer.ICEoT

/-- Query/key matrices, latent bias, gate/value diagonals, gate bias,
projection, two feed-forward matrices and their biases; §3.2, Eqs. (15)–(25).
Fixed activations have no trainable parameters. -/
def blockParameterCount (m h heads ff : ℕ) : ℕ :=
  heads * (2 * m * h + 4 * h) + m * heads * h + m +
    m * ff + ff + ff * m + m

/-- Independent scalar parameters in the exact encoder equations:
embedding/bias, all blocks, readout/bias; §3.2, Eqs. (14)–(27).
The positional encoding is fixed and excluded. -/
def encoderParameterCount (din m h heads ff out depth : ℕ) : ℕ :=
  din * m + m + depth * blockParameterCount m h heads ff + m * out + out

/-- Every trainable-parameter count as reported in Table 3, §4.3.2. -/
def reportedParameterCount : Testbed → Architecture → ℕ
  | .apartment, .iceot => 23563
  | .apartment, .iclstm => 22511
  | .apartment, .eot => 27595
  | .apartment, .lstm => 75839
  | .office, .iceot => 25488
  | .office, .iclstm => 29436
  | .office, .eot => 28880
  | .office, .lstm => 85774

/-- With the stated Table 3 dimensions and Table 2 selective inputs,
Eqs. (14)–(27) have 31,179 and 33,104 independent scalar parameters.
Table 3 instead prints 23,563 and 25,488. This is a mismatch between
the written architecture and the reported counts, conditional on each
listed matrix/diagonal/bias being independently trainable as parameterized
here. Additional unreported tying or omission could explain an actual
implementation's counts; the experiment itself is not refuted.
Source: arXiv:2603.22095v2, §3.2 and §4.3.2, Tables 2, 3. -/
theorem equation_parameter_counts_disagree :
    encoderParameterCount 19 64 64 1 128 11 1 = 31179 ∧
    encoderParameterCount 44 64 64 1 128 16 1 = 33104 ∧
    encoderParameterCount 19 64 64 1 128 11 1 ≠ reportedParameterCount .apartment .iceot ∧
    encoderParameterCount 44 64 64 1 128 16 1 ≠ reportedParameterCount .office .iceot := by decide

/-- The testbed-dependent increase of 1,925 parameters does agree with
the embedding/readout size change; §4.3.2, Table 3. -/
theorem reported_parameter_increment :
    reportedParameterCount .office .iceot - reportedParameterCount .apartment .iceot =
      encoderParameterCount 44 64 64 1 128 16 1 - encoderParameterCount 19 64 64 1 128 11 1 := by
  decide

/-- The reported conventional LSTM has more trainable parameters than
the shared-matrix IC-LSTM in both testbeds; §4.3.2, Table 3. -/
theorem reported_lstm_parameter_comparison (t : Testbed) :
    reportedParameterCount t .iclstm < reportedParameterCount t .lstm := by cases t <;> decide

end Transformer.ICEoT
