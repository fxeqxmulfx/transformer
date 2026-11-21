module

public import Transformer.Grokking.FlatSpectrum.BilinearKernel
public import Transformer.Grokking.FlatSpectrum.TablePart.FlatCycles

/-!
# PT6: the same part, against M

Source: `review/flat_spectrum.md`, *What a part of the table determines*, PT6. Let
`f = g(u) + h(v) + uᵀ M v` with `w ↦ wᵀ M` injective, and `K = [ψ(f)]`. On a proper part `Ω` whose
graph is connected and whose twists span all matrices, the values of `f` on `Ω` leave one function
of the flat class, and by Theorem A in the form NV1 the minimum of M from `Ω` is not `K`
(`determined_and_not_returned`). With `M = 0` the kernel has rank one on any nonempty rows and
columns (`rank_submatrix_bilinearKernel_zero`), and Theorem A does not speak about it.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.TablePart

open Matrix

variable {p : ℕ} [Fact p.Prime] {ι κ : Type*} [Fintype ι] [Fintype κ]

/-- The table of `g(u) + h(v) + uᵀ M v`. -/
def flatTable (g : (ι → ZMod p) → ZMod p) (h : (κ → ZMod p) → ZMod p) (M : Matrix ι κ (ZMod p)) :
    (ι → ZMod p) × (κ → ZMod p) → ZMod p :=
  fun c => g c.1 + h c.2 + c.1 ⬝ᵥ M *ᵥ c.2

/-- The table of `g(u) + h(v) + uᵀ M v` is a member of the flat class. -/
lemma flatTable_mem (g : (ι → ZMod p) → ZMod p) (h : (κ → ZMod p) → ZMod p)
    (M : Matrix ι κ (ZMod p)) : flatTable g h M ∈ flatClass (ZMod p) ι κ :=
  mem_flatClass.2 ⟨g, h, M, fun _ _ => rfl⟩

/-- **With `M = 0` the kernel has rank one**, on any nonempty rows and columns
(`review/flat_spectrum.md`, PT6): `ψ(g u + h v) = ψ(g u) ψ(h v)`. -/
theorem rank_submatrix_bilinearKernel_zero {ι' κ' : Type*} [Fintype ι'] [Fintype κ']
    [Nonempty ι'] [Nonempty κ'] (g : (ι → ZMod p) → ZMod p) (h : (κ → ZMod p) → ZMod p)
    (r : ι' → ι → ZMod p) (c : κ' → κ → ZMod p) :
    ((bilinearKernel p g h (0 : Matrix ι κ (ZMod p))).submatrix r c).rank = 1 := by
  classical
  set A := (bilinearKernel p g h (0 : Matrix ι κ (ZMod p))).submatrix r c with hA
  have hvec : A = vecMulVec (fun i => (ZMod.stdAddChar (g (r i)) : ℂ))
      (fun j => ZMod.stdAddChar (h (c j))) := by
    ext i j
    simp [hA, bilinearKernel, vecMulVec_apply, AddChar.map_add_eq_mul]
  refine le_antisymm (hvec ▸ rank_vecMulVec_le _ _) (Nat.one_le_iff_ne_zero.2 fun h0 => ?_)
  have hr : LinearMap.range A.mulVecLin = ⊥ := Submodule.finrank_eq_zero.1 h0
  have hz : A.mulVecLin = 0 := LinearMap.range_eq_bot.1 hr
  obtain ⟨i⟩ := ‹Nonempty ι'›
  obtain ⟨j⟩ := ‹Nonempty κ'›
  have hij : A i j = 0 := by
    simpa using congrFun (congrArg (fun f => f (Pi.single j 1)) hz) i
  exact bilinearKernel_ne_zero p g h 0 _ _ hij

/-- Rows and columns of `ZMod 3`, with `g` and `h` zero. -/
example : ((bilinearKernel 3 (fun _ => 0) (fun _ => 0)
    (0 : Matrix (Fin 1) (Fin 1) (ZMod 3))).submatrix
      (id : (Fin 1 → ZMod 3) → Fin 1 → ZMod 3) (id : (Fin 1 → ZMod 3) → Fin 1 → ZMod 3)).rank = 1 :=
  rank_submatrix_bilinearKernel_zero _ _ _ _

variable [DecidableEq ι] [DecidableEq κ]

/-- **The same part, against M** (`review/flat_spectrum.md`, PT6). On a proper part with a connected
graph and twists that span all matrices, the data leave one function of the class, and M does not
return its kernel. -/
theorem determined_and_not_returned (g : (ι → ZMod p) → ZMod p) (h : (κ → ZMod p) → ZMod p)
    {M : Matrix ι κ (ZMod p)} (hM : Function.Injective M.vecMul)
    {Ω : Set ((ι → ZMod p) × (κ → ZMod p))} (hΩ : Ω ≠ Set.univ)
    (hc : (patternGraph Ω).Connected) (ht : twistSpan Ω = ⊤) :
    (∀ f ∈ flatClass (ZMod p) ι κ, (∀ c ∈ Ω, f c = flatTable g h M c) → f = flatTable g h M) ∧
      ¬ IsMinOn nuclearNorm {X | ∀ ij ∈ Ω, X ij.1 ij.2 = bilinearKernel p g h M ij.1 ij.2}
        (bilinearKernel p g h M) :=
  ⟨fun f hf hag => (determines_flatClass_iff Ω).2 ⟨hc, ht⟩ f hf _ (flatTable_mem g h M) hag,
    not_isMinOn_bilinearKernel p g h hM hΩ⟩

/-- The hypotheses hold at `p = 3` on the table without its cell `(0, 0)`. -/
example : ¬ IsMinOn nuclearNorm
    {X | ∀ ij ∈ ({c | c ≠ (0, 0)} : Set ((Fin 1 → ZMod 3) × (Fin 1 → ZMod 3))),
      X ij.1 ij.2 = bilinearKernel 3 (fun _ => 0) (fun _ => 0)
        (1 : Matrix (Fin 1) (Fin 1) (ZMod 3)) ij.1 ij.2}
    (bilinearKernel 3 (fun _ => 0) (fun _ => 0) (1 : Matrix (Fin 1) (Fin 1) (ZMod 3))) := by
  have hv (u : Fin 1 → ZMod 3) : (u, (fun _ => 1 : Fin 1 → ZMod 3)) ≠ (0, 0) :=
    fun h => one_ne_zero (congrFun (Prod.ext_iff.1 h).2 0)
  have hu (v : Fin 1 → ZMod 3) : ((fun _ => 1 : Fin 1 → ZMod 3), v) ≠ (0, 0) :=
    fun h => one_ne_zero (congrFun (Prod.ext_iff.1 h).1 0)
  have h2 (v : Fin 1 → ZMod 3) : ((fun _ => 2 : Fin 1 → ZMod 3), v) ≠ (0, 0) :=
    fun h => absurd (show (2 : ZMod 3) = 0 from congrFun (Prod.ext_iff.1 h).1 0) (by decide)
  refine (determined_and_not_returned (fun _ => 0) (fun _ => 0)
    (fun a b h => by simpa using h) (fun h => ?_) ?_ ?_).2
  · exact (h ▸ Set.mem_univ ((0, 0) : (Fin 1 → ZMod 3) × (Fin 1 → ZMod 3)) :
      ((0, 0) : (Fin 1 → ZMod 3) × (Fin 1 → ZMod 3)) ∈ {c | c ≠ (0, 0)}) rfl
  · exact connected_of_hub (fun _ => 1) (fun u => ⟨fun _ => 1, hu _, hv u⟩) fun v => ⟨_, hu v⟩
  · exact twistSpan_eq_top_of_fourCycle (u := fun _ => 1) (u' := fun _ => 2) (v := fun _ => 1)
      (v' := fun _ => 2) (fun h => absurd (show (1 : ZMod 3) = 2 from congrFun h 0) (by decide))
      (fun h => absurd (show (1 : ZMod 3) = 2 from congrFun h 0) (by decide))
      (hu _) (h2 _) (h2 _) (hu _)

end Transformer.Grokking.FlatSpectrum.TablePart
