module

public import Mathlib.Topology.Instances.Matrix
public import Transformer.Grokking.FlatSpectrum.Training.Lifted

/-!
# Continuity of the pieces of the weight-decayed loss

Source: `review/flat_spectrum.md`, *Training reaches the optimal fit* (added), items WD4 and
WD5, the limits their proofs take. The squared Frobenius norm, the trace pairing, the stacked
factors, the lifted gradient and the product `U Vᴴ` are continuous, so they pass to the limit of
a convergent run.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Training

open Matrix Filter Topology

variable {m n k : Type*} [Fintype m] [Fintype n]

lemma continuous_frobeniusSq : Continuous (frobeniusSq : Matrix m n ℂ → ℝ) :=
  continuous_finsetSum _ fun i _ => continuous_finsetSum _ fun j _ =>
    ((continuous_id.matrix_elem i j).norm.pow 2)

/-- The trace pairing of two continuous families is continuous. -/
lemma continuous_pairing {X : Type*} [TopologicalSpace X] {f g : X → Matrix m n ℂ}
    (hf : Continuous f) (hg : Continuous g) : Continuous fun x => pairing (f x) (g x) :=
  Complex.continuous_re.comp (hf.matrix_conjTranspose.matrix_mul hg).matrix_trace

example := continuous_pairing (continuous_id (X := Matrix (Fin 1) (Fin 1) ℂ)) continuous_const
  (g := fun _ => 1)

omit [Fintype m] [Fintype n] in
lemma continuous_stack : Continuous fun UV : Matrix m k ℂ × Matrix n k ℂ => stack UV.1 UV.2 := by
  refine continuous_matrix fun i j => ?_
  rcases i with i | i
  · exact continuous_fst.matrix_elem i j
  · exact continuous_snd.matrix_elem i j

omit [Fintype m] [Fintype n] in
lemma continuous_lift [DecidableEq m] [DecidableEq n] (lam : ℝ) :
    Continuous (lift lam : Matrix m n ℂ → Matrix (m ⊕ n) (m ⊕ n) ℂ) :=
  Continuous.matrix_fromBlocks continuous_const continuous_id continuous_id.matrix_conjTranspose
    continuous_const

omit [Fintype m] [Fintype n] in
/-- The product `U Vᴴ` is continuous in the factors. -/
lemma continuous_mul_conjTranspose [Fintype k] :
    Continuous fun UV : Matrix m k ℂ × Matrix n k ℂ => UV.1 * UV.2ᴴ :=
  continuous_fst.matrix_mul continuous_snd.matrix_conjTranspose

/-- A convergent sequence of matrices has bounded Frobenius norms. -/
lemma exists_frobeniusSq_le_of_tendsto {W : ℕ → Matrix m n ℂ} {Winf : Matrix m n ℂ}
    (h : Tendsto W atTop (𝓝 Winf)) : ∃ B : ℝ, ∀ t, frobeniusSq (W t) ≤ B := by
  obtain ⟨B, hB⟩ := ((continuous_frobeniusSq.tendsto Winf).comp h).bddAbove_range
  exact ⟨B, fun t => hB (Set.mem_range_self t)⟩

example := exists_frobeniusSq_le_of_tendsto
  (tendsto_const_nhds (x := (1 : Matrix (Fin 1) (Fin 1) ℂ)) (f := atTop (α := ℕ)))

/-- If `W_t → W∞` and the differences `W_t − W∞` are multiplied by a convergent family, the
limit equation `W∞ = A∞ W∞` of a recursion `W_{t+1} = A_t W_t` holds. -/
lemma eq_mul_of_tendsto_of_succ {p : Type*} [Fintype p] {A : ℕ → Matrix p p ℂ}
    {Ainf : Matrix p p ℂ} (hA : Tendsto A atTop (𝓝 Ainf)) {W : ℕ → Matrix p k ℂ}
    {Winf : Matrix p k ℂ} (hW : Tendsto W atTop (𝓝 Winf)) (hrec : ∀ t, W (t + 1) = A t * W t) :
    Winf = Ainf * Winf := by
  have h1 : Tendsto (fun t => W (t + 1)) atTop (𝓝 Winf) := hW.comp (tendsto_add_atTop_nat 1)
  have h2 : Tendsto (fun t => A t * W t) atTop (𝓝 (Ainf * Winf)) :=
    ((continuous_fst.matrix_mul continuous_snd).tendsto (Ainf, Winf)).comp (hA.prodMk_nhds hW)
  exact tendsto_nhds_unique h1 (by simpa only [hrec] using h2)

example := eq_mul_of_tendsto_of_succ (k := Fin 1)
  (tendsto_const_nhds (x := (1 : Matrix (Fin 1) (Fin 1) ℂ)) (f := atTop (α := ℕ)))
  (tendsto_const_nhds (x := (1 : Matrix (Fin 1) (Fin 1) ℂ)) (f := atTop (α := ℕ)))
  fun _ => (Matrix.one_mul _).symm

end Transformer.Grokking.FlatSpectrum.Training
