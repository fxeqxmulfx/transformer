module

public import Transformer.Optimization.PAGE.FiniteSum
public import Mathlib.Topology.MetricSpace.Pseudo.Pi

/-! Continuity of the actual PAGE iterates jointly in the constant step and
starting weights. Uniformity over all sampling decisions on a finite horizon
uses finite branching and does not assume a particular refresh probability. -/

@[expose] public section

namespace Transformer.Optimization.PAGE

open Metric Set

variable {I E : Type*} [Fintype I] [NormedAddCommGroup E] [InnerProductSpace ℝ E] [CompleteSpace E]

theorem continuous_run (loss : I → E → ℝ) (hg : ∀ i, Continuous (gradient (loss i)))
    (path : ℕ → Option I) (t : ℕ) :
    Continuous (fun z : ℝ × E => (run loss z.1 z.2 path t).weights) ∧
    Continuous (fun z : ℝ × E => (run loss z.1 z.2 path t).estimate) := by
  have hf : Continuous (full loss) := (continuous_finsetSum _ (fun i _ => hg i)).const_smul _
  induction t with
  | zero => exact ⟨continuous_snd, hf.comp continuous_snd⟩
  | succ t ih =>
    obtain ⟨hx, hv⟩ := ih
    have hy : Continuous (fun z : ℝ × E => (run loss z.1 z.2 path t).weights -
        z.1 • (run loss z.1 z.2 path t).estimate) := hx.sub (continuous_fst.smul hv)
    refine ⟨hy, ?_⟩
    cases h : path t with
    | none =>
      simp only [run, step, h]
      exact hf.comp hy
    | some i =>
      simp only [run, step, h]
      exact hv.add (((hg i).comp hy).sub ((hg i).comp hx))

private theorem run_zero_rate (loss : I → E → ℝ) (x : E) (d : ℕ → Option I) (t : ℕ) :
    (run loss 0 x d t).weights = x := by
  induction t with
  | zero => rfl
  | succ t ih => simpa only [run, step, zero_smul, sub_zero] using ih

/-- Finite-horizon stability at any reference rate, simultaneously for all
sampling paths. The hypothesis can describe margins on any set of inputs. -/
theorem isOpen_all_prefix (loss : I → E → ℝ) (hg : ∀ i, Continuous (gradient (loss i)))
    (K : ℕ) {G : Set E} (hG : IsOpen G) :
    IsOpen {z : ℝ × E | ∀ path : ℕ → Option I, ∀ t ≤ K, (run loss z.1 z.2 path t).weights ∈ G} := by
  let ext (p : Fin K → Option I) := fun j : ℕ => if hj : j < K then p ⟨j,hj⟩ else none
  have he : {z : ℝ × E | ∀ path : ℕ → Option I, ∀ t ≤ K, (run loss z.1 z.2 path t).weights ∈ G} =
      ⋂ p : Fin K → Option I, ⋂ t : Fin (K+1), {z : ℝ × E | (run loss z.1 z.2 (ext p) t).weights ∈ G} := by
    ext z
    simp only [mem_ofPred_eq, mem_iInter]
    refine ⟨fun h p t => h (ext p) t (by omega), ?_⟩
    intro h path t ht
    have heq := run_prefix loss z.1 z.2 t (a := path) (b := ext (fun j => path j))
      (fun j hj => by simp [ext, show j < K by omega])
    rw [heq]
    exact h (fun j => path j) ⟨t,by omega⟩
  rw [he]
  exact isOpen_iInter_of_finite fun p => isOpen_iInter_of_finite fun t =>
    hG.preimage (continuous_run loss hg (ext p) t).1

/-- Every start sufficiently near an interior point stays in the same open
set for K updates when the positive constant step is sufficiently small. -/
theorem stays_open_finite (loss : I → E → ℝ) (hg : ∀ i, Continuous (gradient (loss i)))
    (K : ℕ) {G : Set E} (hG : IsOpen G) {x : E} (hx : x ∈ G) :
    ∃ δ > 0, ∀ η : ℝ, |η| < δ → ∀ y ∈ ball x δ, ∀ path : ℕ → Option I, ∀ t ≤ K,
      (run loss η y path t).weights ∈ G := by
  have h := isOpen_all_prefix loss hg K hG
  obtain ⟨δ,hδ,hb⟩ := isOpen_iff.mp h (0,x) (fun d t _ => by rwa [run_zero_rate])
  refine ⟨δ,hδ,fun η hη y hy => @hb (η,y) ?_⟩
  simpa [mem_ball, Prod.dist_eq, Real.dist_eq] using max_lt hη hy

example (x : ℝ) (K : ℕ) := stays_open_finite (I := Unit) (fun _ (_ : ℝ) => 0)
  (by intro i; simp [continuous_const]) K isOpen_univ (mem_univ x)

end Transformer.Optimization.PAGE
