/-
# Muon — the distributed resource calculation

arXiv:2502.16982, §2.3, “Analysis”. The source's communication bound is a
byte-count model without an additional tensor-parallel gather. It is not
a theorem about wall-clock latency. The factor-of-two memory saving is
for momentum buffers, excluding shared master weights and workspaces.
-/

import Transformer.Muon.Section2_Models

namespace Transformer.Muon

/-- Momentum-buffer storage for Muon, arXiv:2502.16982, §2.3, “Memory Usage”.
There is one real-word buffer for each local parameter entry. -/
def muonMomentumBytes (entries bytesPerWord : ℕ) : ℕ := entries * bytesPerWord

/-- AdamW has two momentum buffers of the same shape and precision,
arXiv:2502.16982, §2.3, “Memory Usage”. -/
def adamMomentumBytes (entries bytesPerWord : ℕ) : ℕ := 2 * entries * bytesPerWord

/-- Muon's momentum-buffer memory is half AdamW's for the same number of
entries and precision, arXiv:2502.16982, §2.3, “Memory Usage”. -/
theorem momentum_memory_half (entries bytesPerWord : ℕ) :
    2 * muonMomentumBytes entries bytesPerWord = adamMomentumBytes entries bytesPerWord := by
  simp [muonMomentumBytes, adamMomentumBytes, Nat.mul_assoc]

/-- bf16 gather payload is half fp32 payload for the same entries,
arXiv:2502.16982, §2.3, “Communication Overhead”. -/
theorem bf16_payload_half (entries : ℕ) : 2 * (2 * entries) = 4 * entries := by omega

/-- Byte workload for the source's two fp32 collectives and extra bf16
gather, arXiv:2502.16982, §2.3, “Communication Overhead”. -/
def muonCommunication (baseEntries gatheredEntries : ℝ) : ℝ :=
  4 * baseEntries + 2 * gatheredEntries + 4 * baseEntries

/-- Byte workload for the two baseline fp32 collectives,
arXiv:2502.16982, §2.3, “Communication Overhead”. -/
def adamCommunication (baseEntries : ℝ) : ℝ := 4 * baseEntries + 4 * baseEntries

/-- The workload ratio is in `(1,1.25]` under the source's stated payload
comparison and without the extra TP gather mentioned in its footnote.
Source: arXiv:2502.16982, §2.3, “Communication Overhead”. -/
theorem communication_ratio_bounds (base gathered : ℝ) (hbase : 0 < base)
    (hgathered : 0 < gathered) (hsize : gathered ≤ base) :
    1 < muonCommunication base gathered / adamCommunication base ∧
      muonCommunication base gathered / adamCommunication base ≤ 5 / 4 := by
  have hd : 0 < adamCommunication base := by unfold adamCommunication; positivity
  constructor
  · rw [lt_div_iff₀ hd]
    unfold muonCommunication adamCommunication
    linarith
  · rw [div_le_iff₀ hd]
    unfold muonCommunication adamCommunication
    linarith

/-- The communication hypotheses are satisfiable,
arXiv:2502.16982, §2.3: equal, positive baseline and gather payloads. -/
example : (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧ (1 : ℝ) ≤ 1 := by norm_num

/-- The source's worst-case `4+2+4` versus `4+4` comparison gives exactly
`1.25`, arXiv:2502.16982, §2.3, “Communication Overhead”. -/
theorem communication_ratio_max (base : ℝ) (hbase : 0 < base) :
    muonCommunication base base / adamCommunication base = 5 / 4 := by
  unfold muonCommunication adamCommunication
  field_simp
  ring

/-- A positive workload exists, arXiv:2502.16982, §2.3. -/
example : (0 : ℝ) < 1 := by norm_num

/-- A shared positive master-weight allocation prevents the *total* optimizer
memory from being halved. The source's valid half-memory claim is about
momentum buffers only. Source: arXiv:2502.16982, §2.3, “Memory Usage”. -/
theorem total_memory_not_half (shared momentum : ℕ) (hshared : 0 < shared) :
    2 * (shared + momentum) ≠ shared + 2 * momentum := by omega

/-- The shared allocation can be positive, arXiv:2502.16982, §2.3. -/
example : 0 < (1 : ℕ) := by norm_num

end Transformer.Muon
