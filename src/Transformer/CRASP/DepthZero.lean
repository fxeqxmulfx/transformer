/-
# What a depth-0 formula reads

arXiv:2506.16055v3, "Knee-Deep in C-RASP: A Transformer Depth Hierarchy"
(COLM 2025), Appendix C.1, the depth-0 cases of the proof of
`lem:TLCP_commutative`.

A formula of depth `0` counts nothing: its terms are built from `1` and `+`.
At a position it reads the letter there and the values of its Parikh numerical
predicates, and nothing else (`Form.sat_eq_of_depth_eq_zero`).  A PNP reads the
Parikh vector of the whole string and the position, so two strings with the
same Parikh vector agree on it position by position
(`Form.sat_eq_of_mem_pnps`).
-/

import Transformer.CRASP.Parikh

namespace Transformer
namespace CRASP

universe u

variable {σ : Type u}

mutual

/-- Every PNP of a formula is a PNP (§4.3, "the PNPs of `φ`"). -/
theorem Form.exists_eq_pnp_of_mem_pnps :
    ∀ (φ : Form σ) {ψ : Form σ}, ψ ∈ φ.pnps → ∃ π, ψ = .pnp π
  | .sym _, _, h => by rw [Form.pnps] at h; exact absurd h List.not_mem_nil
  | .lt t₁ t₂, _, h => by
      rw [Form.pnps, List.mem_append] at h
      exact h.elim t₁.exists_eq_pnp_of_mem_pnps t₂.exists_eq_pnp_of_mem_pnps
  | .neg φ, _, h => by rw [Form.pnps] at h; exact φ.exists_eq_pnp_of_mem_pnps h
  | .and φ₁ φ₂, _, h => by
      rw [Form.pnps, List.mem_append] at h
      exact h.elim φ₁.exists_eq_pnp_of_mem_pnps φ₂.exists_eq_pnp_of_mem_pnps
  | .pnp π, _, h => by rw [Form.pnps, List.mem_singleton] at h; exact ⟨π, h⟩

/-- Every PNP of a term is a PNP (§4.3, "the PNPs of `φ`"). -/
theorem Term.exists_eq_pnp_of_mem_pnps :
    ∀ (t : Term σ) {ψ : Form σ}, ψ ∈ t.pnps → ∃ π, ψ = .pnp π
  | .countL φ, _, h => by rw [Term.pnps] at h; exact φ.exists_eq_pnp_of_mem_pnps h
  | .countR φ, _, h => by rw [Term.pnps] at h; exact φ.exists_eq_pnp_of_mem_pnps h
  | .add t₁ t₂, _, h => by
      rw [Term.pnps, List.mem_append] at h
      exact h.elim t₁.exists_eq_pnp_of_mem_pnps t₂.exists_eq_pnp_of_mem_pnps
  | .one, _, h => by rw [Term.pnps] at h; exact absurd h List.not_mem_nil

end

/-- The hypotheses of `Form.exists_eq_pnp_of_mem_pnps` and
`Term.exists_eq_pnp_of_mem_pnps` are satisfiable: a PNP is among its own PNPs,
and among those of a count of it. -/
example :
    (Form.pnp fun _ _ => true : Form σ) ∈ (Form.pnp fun _ _ => true : Form σ).pnps ∧
      (Form.pnp fun _ _ => true : Form σ) ∈ (Term.countL (.pnp fun _ _ => true) : Term σ).pnps := by
  rw [Term.pnps, Form.pnps]
  exact ⟨List.mem_singleton_self _, List.mem_singleton_self _⟩

variable [DecidableEq σ]

/-- A term of depth `0` is built from `1` and `+`, so its value depends on
neither the string nor the position (Definition `def:TLC_depth`). -/
theorem Term.val_eq_of_depth_eq_zero (w v : List σ) (i j : ℕ) :
    ∀ t : Term σ, t.depth = 0 → t.val w i = t.val v j
  | .countL _, h => by rw [Term.depth] at h; omega
  | .countR _, h => by rw [Term.depth] at h; omega
  | .add t₁ t₂, h => by
      rw [Term.depth] at h
      rw [Term.val, Term.val, t₁.val_eq_of_depth_eq_zero w v i j (by omega),
        t₂.val_eq_of_depth_eq_zero w v i j (by omega)]
  | .one, _ => rfl

/-- The hypothesis of `Term.val_eq_of_depth_eq_zero` is satisfiable: `1 + 1`. -/
example : (Term.add .one .one : Term σ).depth = 0 := rfl

/-- **What a depth-0 formula reads.**  Its truth value at a position depends
only on the letter there and on the values of its PNPs (Definitions
`def:TLC_semantics` and `def:TLC_depth`; Appendix C.1, the depth-0 cases of the
proof of `lem:TLCP_commutative`). -/
theorem Form.sat_eq_of_depth_eq_zero {w v : List σ} {i j : ℕ} (hsym : w[i - 1]? = v[j - 1]?) :
    ∀ φ : Form σ, φ.depth = 0 → (∀ ψ ∈ φ.pnps, ψ.sat w i = ψ.sat v j) → φ.sat w i = φ.sat v j
  | .sym a, _, _ => by rw [Form.sat, Form.sat, hsym]
  | .lt t₁ t₂, h, _ => by
      rw [Form.depth] at h
      rw [Form.sat, Form.sat, t₁.val_eq_of_depth_eq_zero w v i j (by omega),
        t₂.val_eq_of_depth_eq_zero w v i j (by omega)]
  | .neg φ, h, hp => by
      rw [Form.depth] at h
      rw [Form.pnps] at hp
      rw [Form.sat, Form.sat, φ.sat_eq_of_depth_eq_zero hsym h hp]
  | .and φ₁ φ₂, h, hp => by
      rw [Form.depth] at h
      rw [Form.pnps] at hp
      rw [Form.sat, Form.sat,
        φ₁.sat_eq_of_depth_eq_zero hsym (by omega) fun ψ hψ => hp ψ (List.mem_append_left _ hψ),
        φ₂.sat_eq_of_depth_eq_zero hsym (by omega) fun ψ hψ => hp ψ (List.mem_append_right _ hψ)]
  | .pnp π, _, hp => hp (.pnp π) (by rw [Form.pnps]; exact List.mem_singleton_self _)

/-- The hypotheses of `Form.sat_eq_of_depth_eq_zero` are satisfiable: `Q_a`
at the first position of `a` and of `ab`. -/
example (a b : σ) :
    [a][1 - 1]? = [a, b][1 - 1]? ∧ (Form.sym a : Form σ).depth = 0 ∧
      ∀ ψ ∈ (Form.sym a : Form σ).pnps, ψ.sat [a] 1 = ψ.sat [a, b] 1 := by
  refine ⟨rfl, rfl, fun ψ hψ => ?_⟩
  rw [Form.pnps] at hψ
  exact absurd hψ List.not_mem_nil

/-- A PNP of a formula reads the same at the same position of two strings with
the same Parikh vector (§2.3, Definition `def:PNP`). -/
theorem Form.sat_eq_of_mem_pnps (φ : Form σ) {ψ : Form σ} (hψ : ψ ∈ φ.pnps) {w v : List σ}
    (h : parikh w = parikh v) (i : ℕ) : ψ.sat w i = ψ.sat v i := by
  obtain ⟨π, rfl⟩ := φ.exists_eq_pnp_of_mem_pnps hψ
  rw [Form.sat, Form.sat, h]

/-- The hypotheses of `Form.sat_eq_of_mem_pnps` are satisfiable: a PNP of
itself, on `ab` and `ba`. -/
example :
    (Form.pnp fun _ _ => true : Form Bool) ∈ (Form.pnp fun _ _ => true : Form Bool).pnps ∧
      parikh [true, false] = parikh [false, true] := by
  rw [Form.pnps]
  exact ⟨List.mem_singleton_self _, funext fun a => by cases a <;> rfl⟩

end CRASP
end Transformer
