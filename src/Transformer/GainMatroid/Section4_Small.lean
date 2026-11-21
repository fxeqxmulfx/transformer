module

public import Transformer.GainMatroid.Section4_LiftRows

/-!
# Lemmas 4.3 to 4.10: `2C₃`, `K₄` and the tube

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §4.1. On these three graphs, a gain function with no
balanced 2-cycle (no balanced triangle on `K₄`) determines its canonical matrix up to switching, and
up to switching and scaling in the lift case, and no canonical frame matrix is projectively
equivalent to a canonical lift matrix.

The source treats each graph by its own matrix computation, and obtains `K₄` from `2C₃` by a
`Δ`-`Y` exchange. Here all eight lemmas are instances of the row theorems `frame_projEquiv_iff`,
`lift_projEquiv_iff` and `not_projEquiv_frame_lift`. They hold for every orientation of the graphs,
and in Lemmas 4.5 and 4.8 the hypothesis on the lift gain function `ψ` is not needed.

The source's proofs have three slips that do not affect the statements. Lemma 4.4 calls `tᵢ` the
`i`th column of `T` where the row is meant, Lemma 4.9 repeats `t₁·e₅` where `t₁·e₆` is meant, and
Lemma 4.10 writes `φ(e₅) = ψ(e₄) = 0` where `ψ(e₅)` is meant.
-/

@[expose] public section

namespace Transformer.GainMatroid

variable {F Γ : Type*} [Field F]

namespace OrientedGraph

section Criteria

variable {V E : Type*} [DecidableEq V] {G : OrientedGraph V E}

/-- If parallel links always have different gains, then no 2-cycle is balanced. -/
@[to_additive encard_ne_two_of_addParallel
  /-- If parallel links always have different gains, then no 2-cycle is balanced. -/]
theorem encard_ne_two_of_parallel [Group Γ] {γ : E → Γ}
    (h : ∀ e f u v, e ≠ f → u ≠ v → G.Joins e u v → G.Joins f u v →
      G.dart γ e u ≠ G.dart γ f u) :
    ∀ C ∈ G.balancedCycles γ, C.encard ≠ 2 := by
  rintro C ⟨u, W, hW, rfl, hg⟩ hC
  rw [hW.encard_edgeSet] at hC
  have hlen : W.edges.length = 2 := by exact_mod_cast hC
  cases W with
  | nil => simp [Walk.edges] at hlen
  | cons e he p =>
    rename_i v
    cases p with
    | nil => simp [Walk.edges] at hlen
    | cons f hf q =>
      cases q with
      | cons g hg' r => simp [Walk.edges] at hlen
      | nil =>
        have hne : e ≠ f := by simpa [Walk.edges] using hW.edges_nodup
        have huv : u ≠ v := by
          have := hW.support_tail_nodup
          simp only [Walk.support, List.tail_cons, List.nodup_cons, List.mem_singleton] at this
          exact fun h => this.1 h.symm
        rw [Walk.gain_cons, Walk.gain_cons, Walk.gain_nil, mul_one,
          dart_symm γ hf.symm huv, mul_inv_eq_one] at hg
        exact h e f u v hne huv he hf.symm hg

/-- If every triangle has non-identity gain, then no 3-cycle is balanced. -/
@[to_additive encard_ne_three_of_addTriangle
  /-- If every triangle has nonzero gain, then no 3-cycle is balanced. -/]
theorem encard_ne_three_of_triangle [Group Γ] {γ : E → Γ}
    (h : ∀ a b c x y z, a ≠ b → a ≠ c → b ≠ c → x ≠ y → x ≠ z → y ≠ z → G.Joins a x y →
      G.Joins b y z → G.Joins c z x → G.dart γ a x * (G.dart γ b y * G.dart γ c z) ≠ 1) :
    ∀ C ∈ G.balancedCycles γ, C.encard ≠ 3 := by
  rintro C ⟨u, W, hW, rfl, hg⟩ hC
  rw [hW.encard_edgeSet] at hC
  have hlen : W.edges.length = 3 := by exact_mod_cast hC
  cases W with
  | nil => simp [Walk.edges] at hlen
  | cons a ha p =>
    cases p with
    | nil => simp [Walk.edges] at hlen
    | cons b hb q =>
      cases q with
      | nil => simp [Walk.edges] at hlen
      | cons c hc r =>
        cases r with
        | cons d hd s => simp [Walk.edges] at hlen
        | nil =>
          have hE : a ≠ b ∧ a ≠ c ∧ b ≠ c := by
            have := hW.edges_nodup
            simp only [Walk.edges, List.nodup_cons, List.mem_cons, List.not_mem_nil, or_false,
              not_or] at this
            exact ⟨this.1.1, this.1.2, this.2.1⟩
          have hV := hW.support_tail_nodup
          simp only [Walk.support, List.tail_cons, List.nodup_cons, List.mem_cons,
            List.not_mem_nil, or_false, not_or] at hV
          rw [Walk.gain_cons, Walk.gain_cons, Walk.gain_cons, Walk.gain_nil, mul_one] at hg
          exact h a b c u _ _ hE.1 hE.2.1 hE.2.2 (Ne.symm hV.1.2) (Ne.symm hV.2.1) hV.1.1 ha hb hc
            hg

end Criteria

section Lemmas

/-- **Lemma 4.3.** On `2C₃`, a gain function with no balanced 2-cycle is switching equivalent to
`ψ` exactly when their canonical frame matrices are projectively equivalent. -/
theorem twoC3_switchEquiv_iff {G : OrientedGraph (Fin 3) (Fin 6)} (hG : G.IsReorientationOf twoC3)
    {φ ψ : Fin 6 → Fˣ} (h : ∀ C ∈ G.balancedCycles φ, C.encard ≠ 2) :
    SwitchEquiv G φ ψ ↔ ProjEquiv (frameMatrix G φ) (frameMatrix G ψ) :=
  (frame_projEquiv_iff (hG.loopless twoC3_loopless) (hG.connectedOff twoC3_star)
    (hasNoBalancingVertex_of_twoC3 hG h)).symm

/-- **Lemma 4.4.** On `2C₃`, an additive gain function with no balanced 2-cycle is
switching-and-scaling equivalent to `ψ` exactly when their canonical lift matrices are
projectively equivalent. -/
theorem twoC3_switchScaleEquiv_iff {G : OrientedGraph (Fin 3) (Fin 6)}
    (hG : G.IsReorientationOf twoC3) {φ ψ : Fin 6 → F}
    (h : ∀ C ∈ G.addBalancedCycles φ, C.encard ≠ 2) :
    SwitchScaleEquiv G φ ψ ↔ ProjEquiv (liftMatrix G φ) (liftMatrix G ψ) :=
  (lift_projEquiv_iff (hG.loopless twoC3_loopless) (hG.connectedOff twoC3_star)
    (hasNoAddBalancingVertex_of_twoC3 hG h)).symm

/-- **Lemma 4.5.** On `2C₃`, the canonical frame matrix of a gain function with no balanced 2-cycle
is projectively equivalent to no canonical lift matrix. The source also assumes that `ψ` yields no
balanced 2-cycle. -/
theorem twoC3_not_projEquiv {G : OrientedGraph (Fin 3) (Fin 6)} (hG : G.IsReorientationOf twoC3)
    {φ : Fin 6 → Fˣ} (h : ∀ C ∈ G.balancedCycles φ, C.encard ≠ 2) (ψ : Fin 6 → F) :
    ¬ProjEquiv (frameMatrix G φ) (liftMatrix G ψ) :=
  not_projEquiv_frame_lift (hG.loopless twoC3_loopless) (hG.connectedOff twoC3_star)
    (hasNoBalancingVertex_of_twoC3 hG h) ψ

/-- **Lemma 4.6.** On `K₄`, a gain function with no balanced triangle is switching equivalent to
`ψ` exactly when their canonical frame matrices are projectively equivalent. -/
theorem K4_switchEquiv_iff {G : OrientedGraph (Fin 4) (Fin 6)} (hG : G.IsReorientationOf K4)
    {φ ψ : Fin 6 → Fˣ} (h : ∀ C ∈ G.balancedCycles φ, C.encard ≠ 3) :
    SwitchEquiv G φ ψ ↔ ProjEquiv (frameMatrix G φ) (frameMatrix G ψ) :=
  (frame_projEquiv_iff (hG.loopless K4_loopless) (hG.connectedOff K4_star)
    (hasNoBalancingVertex_of_K4 hG h)).symm

/-- **Lemma 4.7.** On `K₄`, an additive gain function with no balanced triangle is
switching-and-scaling equivalent to `ψ` exactly when their canonical lift matrices are projectively
equivalent. -/
theorem K4_switchScaleEquiv_iff {G : OrientedGraph (Fin 4) (Fin 6)} (hG : G.IsReorientationOf K4)
    {φ ψ : Fin 6 → F} (h : ∀ C ∈ G.addBalancedCycles φ, C.encard ≠ 3) :
    SwitchScaleEquiv G φ ψ ↔ ProjEquiv (liftMatrix G φ) (liftMatrix G ψ) :=
  (lift_projEquiv_iff (hG.loopless K4_loopless) (hG.connectedOff K4_star)
    (hasNoAddBalancingVertex_of_K4 hG h)).symm

/-- **Lemma 4.8.** On `K₄`, the canonical frame matrix of a gain function with no balanced triangle
is projectively equivalent to no canonical lift matrix. The source also assumes that `ψ` yields no
balanced triangle. -/
theorem K4_not_projEquiv {G : OrientedGraph (Fin 4) (Fin 6)} (hG : G.IsReorientationOf K4)
    {φ : Fin 6 → Fˣ} (h : ∀ C ∈ G.balancedCycles φ, C.encard ≠ 3) (ψ : Fin 6 → F) :
    ¬ProjEquiv (frameMatrix G φ) (liftMatrix G ψ) :=
  not_projEquiv_frame_lift (hG.loopless K4_loopless) (hG.connectedOff K4_star)
    (hasNoBalancingVertex_of_K4 hG h) ψ

/-- **Lemma 4.9.** On the tube, a gain function with no balanced 2-cycle is switching equivalent to
`ψ` exactly when their canonical frame matrices are projectively equivalent. -/
theorem tube_switchEquiv_iff {G : OrientedGraph (Fin 4) (Fin 6)} (hG : G.IsReorientationOf tube)
    {φ ψ : Fin 6 → Fˣ} (h : ∀ C ∈ G.balancedCycles φ, C.encard ≠ 2) :
    SwitchEquiv G φ ψ ↔ ProjEquiv (frameMatrix G φ) (frameMatrix G ψ) :=
  (frame_projEquiv_iff (hG.loopless tube_loopless) (hG.connectedOff tube_star)
    (hasNoBalancingVertex_of_tube hG h)).symm

/-- **Lemma 4.10.** On the tube, an additive gain function with no balanced 2-cycle is
switching-and-scaling equivalent to `ψ` exactly when their canonical lift matrices are projectively
equivalent. -/
theorem tube_switchScaleEquiv_iff {G : OrientedGraph (Fin 4) (Fin 6)}
    (hG : G.IsReorientationOf tube) {φ ψ : Fin 6 → F}
    (h : ∀ C ∈ G.addBalancedCycles φ, C.encard ≠ 2) :
    SwitchScaleEquiv G φ ψ ↔ ProjEquiv (liftMatrix G φ) (liftMatrix G ψ) :=
  (lift_projEquiv_iff (hG.loopless tube_loopless) (hG.connectedOff tube_star)
    (hasNoAddBalancingVertex_of_tube hG h)).symm

end Lemmas

end OrientedGraph

end Transformer.GainMatroid
