module

public import Transformer.MHC.Section4_Regularized

/-!
# DeepSeek V4.1: delayed input mixing in Single-Pass mHC

DeepSeek-V4.1-Flash technical report, §2.4.1, Eqs. (2)–(6), downloaded
as `papers/DeepSeek-V4.1-Flash/DeepSeek_V41_Tech_Report.pdf`.
The shift uses the previous block's read coefficients with the current
streams. It changes the model's function; fusion then implements the
shifted function. The initial read coefficients are an explicit input.
-/

@[expose] public section

open scoped BigOperators
open Transformer.Residual

namespace Transformer.DeepSeekV41

variable {ι : Type*} [Fintype ι]
variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

/-- Token-wise coefficients `(A_l, B_l, C_l)`. These fields are numerical
data, not assumptions about their constraint or how they are predicted.
Source: DeepSeek-V4.1-Flash technical report, §2.4.1, Eq. (2). -/
structure Coefficients (ι : Type*) where
  read : ι → ℝ
  skip : Matrix ι ι ℝ
  write : ι → ℝ

/-- Coefficients of the released four-stream kernel, in the orientation
needed by Eq. (6). Sources: pinned `model.py`, `hc_mixes`/`hc_post`, and
`kernel.py`, `hc_split_sinkhorn_kernel`. As in GLM Flash, the combination
matrix is transposed and the read gate includes epsilon. The read map
produced here is consumed by the next sublayer in Single-Pass mHC. -/
noncomputable def implementationCoefficients (ε : ℝ) (a c : Fin 4 → ℝ)
    (S : Matrix (Fin 4) (Fin 4) ℝ) : Coefficients (Fin 4) :=
  ⟨fun i => MHC.gate (a i) + ε, (MHC.implementationComb ε S).transpose,
    fun i => MHC.writeGate (c i)⟩

/-- Original mHC: the current token state predicts all three maps.
Source: DeepSeek-V4.1-Flash technical report, §2.4.1, Eq. (2). -/
noncomputable def currentStep (predict : (ι → E) → Coefficients ι)
    (F : E → E) (x : ι → E) : ι → E :=
  hyperStep (predict x).read (predict x).write (predict x).skip F x

/-- Single-Pass mHC reads using `A_(l-1)` while its skip and write maps
still come from `H(X_l)`. Source: DeepSeek-V4.1-Flash technical report,
§2.4.1, Eq. (6). -/
noncomputable def singlePassStep (predict : (ι → E) → Coefficients ι)
    (previousRead : ι → ℝ) (F : E → E) (x : ι → E) : ι → E :=
  hyperStep previousRead (predict x).write (predict x).skip F x

/-- A depth recursion carrying the previous predictor's read coefficients.
The pair is `(current streams, read coefficients for the next block)`.
Source: DeepSeek-V4.1-Flash technical report, §2.4.1, Eq. (6). -/
noncomputable def singlePassRun (predict : ℕ → (ι → E) → Coefficients ι)
    (F : ℕ → E → E) (initialRead : ι → ℝ) (x : ι → E) :
    ℕ → ((ι → E) × (ι → ℝ))
  | 0 => (x, initialRead)
  | t + 1 =>
    let previous := singlePassRun predict F initialRead x t
    (singlePassStep (predict t) previous.2 (F t) previous.1,
      (predict t previous.1).read)

/-- The carried read map comes from the previous block's state, rather than
from the updated streams. Source: the coefficient dependency in Eq. (6). -/
theorem singlePassRun_read (predict : ℕ → (ι → E) → Coefficients ι)
    (F : ℕ → E → E) (initialRead : ι → ℝ) (x : ι → E) (t : ℕ) :
    (singlePassRun predict F initialRead x (t + 1)).2 =
      (predict t (singlePassRun predict F initialRead x t).1).read := by
  rfl

/-- Equality with the unshifted block holds if the two read maps coincide.
This is a sufficient condition, not an unconditional equivalence.
Source: comparison of Eqs. (2) and (6) in §2.4.1. -/
theorem singlePass_eq_current (predict : (ι → E) → Coefficients ι)
    (a : ι → ℝ) (F : E → E) (x : ι → E) (ha : a = (predict x).read) :
    singlePassStep predict a F x = currentStep predict F x := by
  unfold singlePassStep currentStep
  rw [ha]

/-- Equal read maps are realizable with a nonzero positive read gate. -/
example : (fun _ : Fin 4 => (1 / 2 : ℝ)) =
    (Coefficients.mk (fun _ : Fin 4 => (1 / 2 : ℝ)) 1 (fun _ => 1)).read := rfl

/-- The shift can change the function even with an identity skip map and
strictly positive read/write gates. Thus Mega-mHC's implementation fusion
must not be confused with equivalence to the original architecture.
Source: DeepSeek-V4.1-Flash technical report, §2.4.1, Eqs. (2), (6). -/
theorem shifted_read_changes_output :
    ∃ (predict : (Fin 1 → ℝ) → Coefficients (Fin 1)) (a : Fin 1 → ℝ)
      (x : Fin 1 → ℝ),
      (predict x).skip ∈ doublyStochastic ℝ (Fin 1) ∧
      (∀ i, 0 < a i ∧ 0 < (predict x).read i ∧ 0 < (predict x).write i) ∧
      singlePassStep predict a id x ≠ currentStep predict id x := by
  let predict : (Fin 1 → ℝ) → Coefficients (Fin 1) :=
    fun _ => ⟨fun _ => 1 / 2, 1, fun _ => 1⟩
  refine ⟨predict, fun _ => 1 / 4, fun _ => 1, ?_, ?_, ?_⟩
  · exact (doublyStochastic ℝ (Fin 1)).one_mem
  · norm_num [predict]
  · intro h
    have h0 := congrFun h 0
    norm_num [singlePassStep, currentStep, predict, hyperStep, mix,
      weightedSum, Fin.sum_univ_one] at h0

/-- Original activation reads plus writes, including branch pre-norm.
Source: DeepSeek-V4.1-Flash technical report, §2.4.1, after Eq. (5). -/
def originalTraffic (n d : ℕ) : ℕ := (4 * n + 4) * d

/-- Traffic after fusing update and prediction, with current read maps.
Source: DeepSeek-V4.1-Flash technical report, §2.4.1, two-pass analysis. -/
def twoPassTraffic (n d : ℕ) : ℕ := (3 * n + 2) * d

/-- Single-Pass mHC activation traffic, counted in scalar elements.
Source: DeepSeek-V4.1-Flash technical report, §2.4.1, lower bound and Mega-mHC. -/
def singlePassTraffic (n d : ℕ) : ℕ := (2 * n + 2) * d

/-- The stated traffic count is exactly half of the original count. This
is arithmetic under the source's cost model, not a GPU timing theorem.
Source: DeepSeek-V4.1-Flash technical report, §2.4.1. -/
theorem singlePass_halves_traffic (n d : ℕ) :
    originalTraffic n d = 2 * singlePassTraffic n d := by
  unfold originalTraffic singlePassTraffic
  ring

/-- Relative to two-pass fusion, the read shift saves one full stream read.
Source: DeepSeek-V4.1-Flash technical report, §2.4.1. -/
theorem singlePass_saves_stream_read (n d : ℕ) :
    singlePassTraffic n d + n * d = twoPassTraffic n d := by
  unfold singlePassTraffic twoPassTraffic
  ring

/-- Four-stream counts for the released V4.1 configuration: 20d, 14d, 10d.
Source: §2.4.1 and pinned `papers/DeepSeek-V4.1-Flash/config.json`, `hc_mult=4`. -/
theorem four_stream_traffic (d : ℕ) :
    originalTraffic 4 d = 20 * d ∧ twoPassTraffic 4 d = 14 * d ∧
      singlePassTraffic 4 d = 10 * d := by
  norm_num [originalTraffic, twoPassTraffic, singlePassTraffic]

/-- At the released hidden width 5120 the three cost models count 102400,
71680 and 51200 scalar activation transfers per token, respectively.
Source: §2.4.1 and pinned `config.json`, `hidden_size=5120`. -/
theorem released_width_traffic :
    originalTraffic 4 5120 = 102400 ∧ twoPassTraffic 4 5120 = 71680 ∧
      singlePassTraffic 4 5120 = 51200 := by
  norm_num [originalTraffic, twoPassTraffic, singlePassTraffic]

end Transformer.DeepSeekV41
