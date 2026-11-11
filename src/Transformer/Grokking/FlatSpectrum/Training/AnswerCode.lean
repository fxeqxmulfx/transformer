module

public import Mathlib.Analysis.SpecialFunctions.Complex.CircleAddChar
public import Transformer.Grokking.FlatSpectrum.Training.Margin

/-!
# The probes of an answer code

Source: `review/flat_spectrum.md`, *Training with cross-entropy* (added), item XE8.
The kernel of the note shows the answer of a task at each cell as the value `φ(c)` of an answer
code `φ : C → ℂ`. A sample `s` sees the cell `cell s`, and its class `c` has the probe
`φ(c) e_{cell s}` (`codeProbe`), so that the logit of `c` is `Re (conj φ(c) X_{cell s})`
(`pairing_codeProbe`): the nearest-codeword decoder of the note read as a softmax. The probes
vanish off the seen cells (`codeProbe_apply_of_notMem`) and have the norm of their codeword
(`frobeniusSq_codeProbe`).

If the code is unimodular and `K` shows the codeword of the label at every seen cell, the gap
of the sample `s` against the class `c` at `K` is `|φ(y s) − φ(c)|² / 2` (`gap_codeProbe`). So
with an injective code `K` has a positive margin (`margin_codeProbe_pos`): it classifies every
sample, by half the squared distance of two codewords.

The witnesses: the `1 × 2` kernel `(1, 1)` with the code `±1` on its first cell, and the kernel
of `a³ + a b` for `p = 3` with the code `ψ` on the cell `(0, 0)`.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Training

open Matrix
open scoped ComplexConjugate

variable {m n S C : Type*} [Fintype m] [Fintype n] [DecidableEq m] [DecidableEq n]

/-- The probes of the answer code `φ` on the cells `cell s`: the class `c` of the sample `s`
has the probe `φ(c) e_{cell s}`, item XE8. -/
def codeProbe (φ : C → ℂ) (cell : S → m × n) (s : S) (c : C) : Matrix m n ℂ :=
  Matrix.single (cell s).1 (cell s).2 (φ c)

/-- `⟨a e_ij, X⟩ = Re (conj a · X_ij)`. -/
lemma pairing_single (i : m) (j : n) (a : ℂ) (X : Matrix m n ℂ) :
    pairing (Matrix.single i j a) X = (conj a * X i j).re := by
  rw [pairing, trace_conjTranspose_mul, Finset.sum_eq_single_of_mem i (Finset.mem_univ i),
    Finset.sum_eq_single_of_mem j (Finset.mem_univ j), Matrix.single_apply_same,
    Complex.star_def]
  · intro j' _ hj
    rw [Matrix.single_apply_of_ne (h := fun h => hj h.2.symm), star_zero, zero_mul]
  · intro i' _ hi
    exact Finset.sum_eq_zero fun j' _ => by
      rw [Matrix.single_apply_of_ne (h := fun h => hi h.1.symm), star_zero, zero_mul]

lemma frobeniusSq_single (i : m) (j : n) (a : ℂ) :
    frobeniusSq (Matrix.single i j a) = ‖a‖ ^ 2 := by
  rw [frobeniusSq, Finset.sum_eq_single_of_mem i (Finset.mem_univ i),
    Finset.sum_eq_single_of_mem j (Finset.mem_univ j), Matrix.single_apply_same]
  · intro j' _ hj
    rw [Matrix.single_apply_of_ne (h := fun h => hj h.2.symm), norm_zero]
    norm_num
  · intro i' _ hi
    exact Finset.sum_eq_zero fun j' _ => by
      rw [Matrix.single_apply_of_ne (h := fun h => hi h.1.symm), norm_zero]
      norm_num

omit [DecidableEq m] [DecidableEq n] in
/-- `⟨K, X⟩ = Σ_ij Re (conj K_ij X_ij)`. -/
lemma pairing_eq_sum_re (K X : Matrix m n ℂ) :
    pairing K X = ∑ i, ∑ j, (conj (K i j) * X i j).re := by
  rw [pairing, trace_conjTranspose_mul, Complex.re_sum]
  exact Finset.sum_congr rfl fun i _ => Complex.re_sum _ _

variable {φ : C → ℂ} {cell : S → m × n} {y : S → C} {K : Matrix m n ℂ}

/-- The logit of the class `c` at the sample `s`: `Re (conj φ(c) · X_{cell s})`. -/
lemma pairing_codeProbe (φ : C → ℂ) (cell : S → m × n) (s : S) (c : C) (X : Matrix m n ℂ) :
    pairing (codeProbe φ cell s c) X = (conj (φ c) * X (cell s).1 (cell s).2).re :=
  pairing_single _ _ _ _

omit [Fintype m] [Fintype n] in
/-- The probes of an answer code vanish off the seen cells. -/
lemma codeProbe_apply_of_notMem {i : m} {j : n} (h : (i, j) ∉ Set.range cell) (s : S) (c : C) :
    codeProbe φ cell s c i j = 0 :=
  Matrix.single_apply_of_ne _ _ _ _ _ fun h' => h ⟨s, Prod.ext h'.1 h'.2⟩

lemma frobeniusSq_codeProbe (φ : C → ℂ) (cell : S → m × n) (s : S) (c : C) :
    frobeniusSq (codeProbe φ cell s c) = ‖φ c‖ ^ 2 :=
  frobeniusSq_single _ _ _

/-- **The gaps of a kernel that shows the labels** (`review/flat_spectrum.md`, item XE8): for a
unimodular code `φ` and `K = φ(y s)` at the cell of every sample `s`, the gap of `s` against
the class `c` is `|φ(y s) − φ(c)|² / 2`. -/
theorem gap_codeProbe (hφ : ∀ c, ‖φ c‖ = 1) (hK : ∀ s, K (cell s).1 (cell s).2 = φ (y s))
    (s : S) (c : C) : gap (codeProbe φ cell) y K s c = ‖φ (y s) - φ c‖ ^ 2 / 2 := by
  have h1 : (conj (φ (y s)) * φ (y s)).re = 1 := by
    rw [Complex.conj_mul', hφ]
    norm_num
  rw [gap, pairing_codeProbe, pairing_codeProbe, hK, h1, ← Complex.normSq_eq_norm_sq,
    Complex.normSq_sub, Complex.normSq_eq_norm_sq, Complex.normSq_eq_norm_sq, hφ, hφ,
    mul_comm (conj (φ c))]
  ring

/-- **A kernel that shows the labels with an injective unimodular code has a positive margin**
(`review/flat_spectrum.md`, item XE8). -/
theorem margin_codeProbe_pos [Fintype S] [Fintype C] [Nonempty S] [Nontrivial C]
    (hφ : ∀ c, ‖φ c‖ = 1) (hinj : Function.Injective φ)
    (hK : ∀ s, K (cell s).1 (cell s).2 = φ (y s)) : 0 < margin (codeProbe φ cell) y K := by
  obtain ⟨s, c, hc, hg⟩ := exists_gap_eq_margin (codeProbe φ cell) y K
  rw [← hg, gap_codeProbe hφ hK]
  exact div_pos (pow_pos (norm_pos_iff.mpr (sub_ne_zero.mpr fun e => hc (hinj e).symm)) 2)
    two_pos

/-! The witnesses: the code `±1` of `Bool`, and the code `ψ` of `ZMod 3`. -/

/-- The code `±1` of the two classes. -/
def signCode (c : Bool) : ℂ := if c then 1 else -1

lemma norm_signCode (c : Bool) : ‖signCode c‖ = 1 := by
  cases c <;> simp [signCode]

lemma injective_signCode : Function.Injective signCode := by
  intro a b h
  cases a <;> cases b <;> simp_all [signCode] <;> norm_num at h

example := pairing_single (0 : Fin 1) (0 : Fin 2) 1 (Matrix.of ![![(1 : ℂ), 1]])

example := frobeniusSq_single (0 : Fin 1) (0 : Fin 2) 1

example := codeProbe_apply_of_notMem (φ := signCode) (cell := fun _ : Fin 1 => ((0 : Fin 1),
  (0 : Fin 2))) (i := 0) (j := 1) (by simp) 0 true

/-- The kernel `(1, 1)` with its first cell seen and labeled `true`: the gap is `2`. -/
example := gap_codeProbe (cell := fun _ : Fin 1 => ((0 : Fin 1), (0 : Fin 2)))
  (y := fun _ => true) (K := Matrix.of ![![(1 : ℂ), 1]]) norm_signCode (fun _ => by
    simp [signCode]) 0 false

example := margin_codeProbe_pos (cell := fun _ : Fin 1 => ((0 : Fin 1), (0 : Fin 2)))
  (y := fun _ => true) (K := Matrix.of ![![(1 : ℂ), 1]]) norm_signCode injective_signCode
  fun _ => by simp [signCode]

/-- The kernel of `a³ + a b` for `p = 3` shows the answers of its task with the code `ψ`. -/
example := margin_codeProbe_pos (φ := (ZMod.stdAddChar : AddChar (ZMod 3) ℂ))
  (cell := fun _ : Fin 1 => ((0 : ZMod 3), (0 : ZMod 3))) (y := fun _ => 0)
  (K := cubicKernel 3) (fun c => AddChar.norm_apply _ c) ZMod.injective_stdAddChar
  fun _ => by simp [cubicKernel, taskKernel]

end Transformer.Grokking.FlatSpectrum.Training
