/-
# Depth-1 formulas are commutative on the middle

arXiv:2506.16055v3, "Knee-Deep in C-RASP: A Transformer Depth Hierarchy"
(COLM 2025), §4.3 and Appendix C.1: `lem:TLCP_commutative`.

Two strings `λ u ϱ` and `λ u' ϱ` that match an affix restriction and share a
Parikh vector differ only by a permutation of their middles.  The paper builds
a permutation of positions and shows that a depth-0 formula reads the same at
a position of one string and at its image in the other.  Here a count `◁#[χ]`
at the last position is split into the prefix, the middle and the suffix
instead: the prefix and the suffix carry the same letters in both strings, and
the middle count is a count over the letters of `u`, the same as over those of
`u'` (`Term.val_countL_length_eq`).  The rest is an induction on the formula at
the last position.

The last position lies in the suffix `ϱ` as long as `ϱ` is never empty: there
`▷#[χ]` sees that one position, and `Q_σ` reads the same letter in both
strings.
-/

import Transformer.CRASP.Middle

namespace Transformer
namespace CRASP

universe u

variable {σ : Type u} [DecidableEq σ]

/-- **A depth-1 count at the last position** (Appendix C.1, proof of
`lem:TLCP_commutative`: "the `◁#` terms count all positions").  For a depth-0
`χ` whose PNPs are constant on the middle, `◁#[χ]` takes the same value at the
last position of `λ(n⃗) u ϱ(n⃗)` and of `λ(n⃗) v ϱ(n⃗)` when both have Parikh
vector `n⃗`. -/
theorem Term.val_countL_length_eq (A : Affix σ) {n : PVec σ} {u v : List σ} {χ : Form σ}
    (hχ : χ.depth = 0) (hpnp : PnpsConstantOn χ A.middle)
    (hu : parikh (A.pre n ++ u ++ A.suf n) = n) (hv : parikh (A.pre n ++ v ++ A.suf n) = n) :
    (Term.countL χ).val (A.pre n ++ v ++ A.suf n) (A.pre n ++ v ++ A.suf n).length =
      (Term.countL χ).val (A.pre n ++ u ++ A.suf n) (A.pre n ++ u ++ A.suf n).length := by
  have hpar := hv.trans hu.symm
  have hperm : v.Perm u := perm_of_parikh_eq (parikh_eq_of_parikh_append_eq hpar)
  have hlen : v.length = u.length := hperm.length_eq
  have hmid : ∀ {x : List σ}, x.Perm u → parikh (A.pre n ++ x ++ A.suf n) = n →
      (List.range' 1 x.length).countP (fun j => χ.sat (A.pre n ++ x ++ A.suf n) ((A.pre n).length + j)) =
        x.countP fun a => χ.sat (A.pre n ++ u ++ A.suf n) ((A.pre n).length + (u.idxOf a + 1)) :=
    fun hx hxn => countP_range'_eq_countP _ _ _ fun _ _ hj hja =>
      Form.sat_middle A hχ hpnp hu hxn hj hja (hx.mem_iff.mp (List.mem_of_getElem? hja))
  rw [Term.val, Term.val, ← List.countP_eq_length_filter, ← List.countP_eq_length_filter]
  simp only [List.length_append, countP_range'_add]
  rw [hmid hperm hv, hmid (List.Perm.refl u) hu, hperm.countP_eq, hlen]
  congr 1
  · congr 1
    refine List.countP_congr fun j hj => iff_of_eq (congrArg (· = true) ?_)
    rw [List.mem_range'_1] at hj
    refine Form.sat_eq_of_depth_eq_zero ?_ χ hχ fun ψ hψ => χ.sat_eq_of_mem_pnps hψ hpar j
    rw [List.append_assoc, List.append_assoc, List.getElem?_append_left (by omega),
      List.getElem?_append_left (by omega)]
  · refine List.countP_congr fun j hj => iff_of_eq (congrArg (· = true) ?_)
    rw [List.mem_range'_1] at hj
    refine Form.sat_eq_of_depth_eq_zero ?_ χ hχ fun ψ hψ => χ.sat_eq_of_mem_pnps hψ hpar _
    rw [List.getElem?_append_right (l₁ := A.pre n ++ v) (by simp only [List.length_append]; omega),
      List.getElem?_append_right (l₁ := A.pre n ++ u) (by simp only [List.length_append]; omega)]
    simp only [List.length_append, hlen]

/-- The hypotheses of `Term.val_countL_length_eq` are satisfiable: `Q_a` under
the trivial affix restriction, on `ab` and `ba`. -/
example :
    (Form.sym false : Form Bool).depth = 0 ∧
      PnpsConstantOn (Form.sym false : Form Bool) (⟨fun _ => [], fun _ => []⟩ : Affix Bool).middle ∧
      parikh ((⟨fun _ => [], fun _ => []⟩ : Affix Bool).pre (parikh [false, true]) ++ [false, true] ++
        (⟨fun _ => [], fun _ => []⟩ : Affix Bool).suf (parikh [false, true])) = parikh [false, true] ∧
      parikh ((⟨fun _ => [], fun _ => []⟩ : Affix Bool).pre (parikh [false, true]) ++ [true, false] ++
        (⟨fun _ => [], fun _ => []⟩ : Affix Bool).suf (parikh [false, true])) = parikh [false, true] := by
  refine ⟨rfl, fun ψ hψ => ?_, rfl, funext fun a => by cases a <;> rfl⟩
  rw [Form.pnps] at hψ
  exact absurd hψ List.not_mem_nil

mutual

/-- **A depth-1 formula at the last position** (Appendix C.1, proof of
`lem:TLCP_commutative`).  Two strings of the same length and Parikh vector,
with the same last letter, on which every `◁#[χ]` for a depth-0 `χ` with PNPs
constant on `I` takes the same value at the last position, satisfy the same
depth-1 formulas with PNPs constant on `I`. -/
theorem Form.sat_length_eq_of_depth_le_one {I : IntervalFamily σ} {w w' : List σ}
    (hlen : w.length = w'.length) (hpar : parikh w = parikh w')
    (hlast : w[w.length - 1]? = w'[w'.length - 1]?)
    (hcount : ∀ χ : Form σ, χ.depth = 0 → PnpsConstantOn χ I →
      (Term.countL χ).val w w.length = (Term.countL χ).val w' w'.length) :
    ∀ φ : Form σ, φ.depth ≤ 1 → PnpsConstantOn φ I → φ.sat w w.length = φ.sat w' w'.length
  | .sym _, _, _ => by rw [Form.sat, Form.sat, hlast]
  | .lt t₁ t₂, h, hp => by
      rw [Form.depth, max_le_iff] at h
      rw [PnpsConstantOn, Form.pnps] at hp
      rw [Form.sat, Form.sat,
        Term.val_length_eq_of_depth_le_one hlen hpar hlast hcount t₁ h.1 fun ψ hψ =>
          hp ψ (List.mem_append_left _ hψ),
        Term.val_length_eq_of_depth_le_one hlen hpar hlast hcount t₂ h.2 fun ψ hψ =>
          hp ψ (List.mem_append_right _ hψ)]
  | .neg φ, h, hp => by
      rw [Form.depth] at h
      rw [PnpsConstantOn, Form.pnps] at hp
      rw [Form.sat, Form.sat, Form.sat_length_eq_of_depth_le_one hlen hpar hlast hcount φ h hp]
  | .and φ₁ φ₂, h, hp => by
      rw [Form.depth, max_le_iff] at h
      rw [PnpsConstantOn, Form.pnps] at hp
      rw [Form.sat, Form.sat,
        Form.sat_length_eq_of_depth_le_one hlen hpar hlast hcount φ₁ h.1 fun ψ hψ =>
          hp ψ (List.mem_append_left _ hψ),
        Form.sat_length_eq_of_depth_le_one hlen hpar hlast hcount φ₂ h.2 fun ψ hψ =>
          hp ψ (List.mem_append_right _ hψ)]
  | .pnp _, _, _ => by rw [Form.sat, Form.sat, hpar, hlen]

/-- **A depth-1 term at the last position** (Appendix C.1, proof of
`lem:TLCP_commutative`: "the `▷#` terms only count the last position"), under
the hypotheses of `Form.sat_length_eq_of_depth_le_one`. -/
theorem Term.val_length_eq_of_depth_le_one {I : IntervalFamily σ} {w w' : List σ}
    (hlen : w.length = w'.length) (hpar : parikh w = parikh w')
    (hlast : w[w.length - 1]? = w'[w'.length - 1]?)
    (hcount : ∀ χ : Form σ, χ.depth = 0 → PnpsConstantOn χ I →
      (Term.countL χ).val w w.length = (Term.countL χ).val w' w'.length) :
    ∀ t : Term σ, t.depth ≤ 1 → (∀ ψ ∈ t.pnps, ConstantOn ψ I) →
      t.val w w.length = t.val w' w'.length
  | .countL φ, h, hp => by
      rw [Term.depth] at h
      rw [Term.pnps] at hp
      exact hcount φ (by omega) hp
  | .countR φ, h, hp => by
      rw [Term.depth] at h
      rw [Term.pnps] at hp
      rw [Term.val, Term.val, Nat.add_sub_cancel_left, Nat.add_sub_cancel_left, List.range'_one,
        List.range'_one, ← List.countP_eq_length_filter, ← List.countP_eq_length_filter,
        List.countP_singleton, List.countP_singleton,
        Form.sat_length_eq_of_depth_le_one hlen hpar hlast hcount φ (by omega) hp]
  | .add t₁ t₂, h, hp => by
      rw [Term.depth, max_le_iff] at h
      rw [Term.pnps] at hp
      rw [Term.val, Term.val,
        Term.val_length_eq_of_depth_le_one hlen hpar hlast hcount t₁ h.1 fun ψ hψ =>
          hp ψ (List.mem_append_left _ hψ),
        Term.val_length_eq_of_depth_le_one hlen hpar hlast hcount t₂ h.2 fun ψ hψ =>
          hp ψ (List.mem_append_right _ hψ)]
  | .one, _, _ => rfl

end

/-- The hypotheses of `Form.sat_length_eq_of_depth_le_one` and
`Term.val_length_eq_of_depth_le_one` are satisfiable: one string twice, with
`▷#[Q_a] < 1` and `▷#[Q_a]`. -/
example (w : List σ) (a : σ) (I : IntervalFamily σ) :
    w.length = w.length ∧ parikh w = parikh w ∧ w[w.length - 1]? = w[w.length - 1]? ∧
      (∀ χ : Form σ, χ.depth = 0 → PnpsConstantOn χ I →
        (Term.countL χ).val w w.length = (Term.countL χ).val w w.length) ∧
      (Form.lt (.countR (.sym a)) .one : Form σ).depth ≤ 1 ∧
      PnpsConstantOn (Form.lt (.countR (.sym a)) .one : Form σ) I ∧
      (Term.countR (.sym a) : Term σ).depth ≤ 1 ∧
      ∀ ψ ∈ (Term.countR (.sym a) : Term σ).pnps, ConstantOn ψ I := by
  refine ⟨rfl, rfl, rfl, fun _ _ _ => rfl, le_rfl, fun ψ hψ => ?_, le_rfl, fun ψ hψ => ?_⟩ <;>
    simp [Form.pnps, Term.pnps] at hψ

/-- **Lemma `lem:TLCP_commutative` (Commutativity of depth 1).**  For a
depth-1 formula of `TL[◁#,▷#]^P` — and so in particular of `TL[◁#]^P` — and an
affix restriction whose suffix is never empty, if the PNPs of the formula are
constant on the middle then the language it defines is commutative on the
middle (§4.3; proof in Appendix C.1).

The nonempty-suffix hypothesis is what stops `Q_σ` from reading the last
position, which is the position the whole formula is judged at. -/
theorem commutativeOnMiddle_of_mem_TLCP_one (φ : Form σ) (hφ : φ ∈ TLCP σ 1)
    (A : Affix σ) (hsuf : ∀ n : PVec σ, 1 ≤ (A.suf n).length)
    (hpnp : PnpsConstantOn φ A.middle) :
    CommutativeOnMiddle φ.lang A := by
  rintro w w' ⟨u, hu⟩ ⟨u', hu'⟩ hn
  obtain ⟨n, hw⟩ : ∃ n, parikh w = n := ⟨_, rfl⟩
  have hw' : parikh w' = n := hn.symm.trans hw
  rw [hw] at hu
  rw [hw'] at hu'
  subst hu hu'
  have hperm := perm_of_parikh_eq (parikh_eq_of_parikh_append_eq (hw.trans hw'.symm))
  refine iff_of_eq (congrArg (· = true) (Form.sat_length_eq_of_depth_le_one ?_ (hw.trans hw'.symm)
    ?_ (fun χ hχ hχp => Term.val_countL_length_eq A hχ hχp hw' hw) φ hφ hpnp))
  · simp only [List.length_append, hperm.length_eq]
  · rw [getElem?_length_sub_one_append _ (hsuf n), getElem?_length_sub_one_append _ (hsuf n)]

/-- The hypotheses of `commutativeOnMiddle_of_mem_TLCP_one` are satisfiable:
the formula `◁#[Q_a] < 1` has depth 1 and no PNPs, and the affix restriction
that pins a single `a` at the end has a nonempty suffix everywhere. -/
example (a : σ) :
    (Form.isZero (.countL (.sym a)) ∈ TLCP σ 1) ∧
      (∀ n : PVec σ, 1 ≤ ((⟨fun _ => [], fun _ => [a]⟩ : Affix σ).suf n).length) ∧
      PnpsConstantOn (Form.isZero (.countL (.sym a)))
        (⟨fun _ => [], fun _ => [a]⟩ : Affix σ).middle := by
  refine ⟨Nat.le_refl 1, fun _ => Nat.le_refl 1, ?_⟩
  intro ψ hψ
  simp [Form.isZero, Form.pnps, Term.pnps] at hψ

end CRASP
end Transformer
