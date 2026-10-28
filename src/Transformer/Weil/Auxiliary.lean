module

public import Transformer.Weil.Independence

/-!
# Derivatives of Stepanov's auxiliary polynomial

Let `f ∈ F[X]` have degree `m ≥ 1`. By Leibniz, `E^{(l)}(f^c) = f^{c − l} G_l` with
`deg G_l ≤ l (m − 1)`, so for `k ≤ c`, `f^k E^{(k)}(s f^c) = f^c σ_k(s)` with
`σ_k(s) = ∑_{i ≤ k} f^i G_{k − i} E^{(i)}(s)`. For `k < q = |F|` the factors `X^{q j}` pass
through `E^{(k)}`, so the auxiliary polynomial `R = ∑_j (r_j + s_j f^c) X^{q j}` has
`f^k E^{(k)} R = ∑_j (f^k E^{(k)}(r_j) + f^c σ_k(s_j)) X^{q j}`. At a point `x` with
`f(x)^c = a`, where `x^q = x`, this takes the value of
`T_k = ∑_j (f^k E^{(k)}(r_j) + a σ_k(s_j)) X^j`, a polynomial of degree at most
`d + k (m − 1) + J − 1` when every `r_j` and `s_j` has degree at most `d`.
-/

@[expose] public section

namespace Transformer.Weil

open Polynomial Finset

variable {F : Type*} [Field F]

/-- `E^{(l)}(f^c) = f^{c − l} G_l` with `deg G_l ≤ l (m − 1)`, `m = deg f`. -/
theorem exists_hasseDeriv_pow_eq {f : F[X]} (hf : f ≠ 0) (c : ℕ) :
    ∃ G : ℕ → F[X], ∀ l, hasseDeriv l (f ^ c) = f ^ (c - l) * G l ∧
      (G l).natDegree ≤ l * (f.natDegree - 1) := by
  have h : ∀ l, ∃ G : F[X], hasseDeriv l (f ^ c) = f ^ (c - l) * G := fun l =>
    pow_sub_dvd_hasseDeriv_pow f c l
  choose G hG using h
  refine ⟨G, fun l => ⟨hG l, ?_⟩⟩
  by_cases h0 : G l = 0
  · simp [h0]
  have h1 := natDegree_hasseDeriv_le (f ^ c) l
  rw [hG l, natDegree_mul (pow_ne_zero _ hf) h0, natDegree_pow, natDegree_pow] at h1
  have h2 : l * (f.natDegree - 1) = l * f.natDegree - l := Nat.mul_sub_one l _
  rcases le_total l c with hlc | hlc
  · have : (c - l) * f.natDegree + l * f.natDegree = c * f.natDegree := by
      rw [← add_mul, Nat.sub_add_cancel hlc]
    omega
  · have h3 : (c - l) * f.natDegree = 0 := by rw [Nat.sub_eq_zero_of_le hlc, zero_mul]
    have h4 : c * f.natDegree ≤ l * f.natDegree := Nat.mul_le_mul_right _ hlc
    omega

example := exists_hasseDeriv_pow_eq (F := ZMod 3) X_ne_zero 2

/-- `σ_k(s) = ∑_{i ≤ k} f^i G_{k − i} E^{(i)}(s)`, the cofactor of `f^c` in
`f^k E^{(k)}(s f^c)`. -/
noncomputable def sigma (f : F[X]) (G : ℕ → F[X]) (k : ℕ) : F[X] →ₗ[F] F[X] :=
  ∑ i ∈ range (k + 1), LinearMap.mulLeft F (f ^ i * G (k - i)) ∘ₗ hasseDeriv i

theorem sigma_apply (f : F[X]) (G : ℕ → F[X]) (k : ℕ) (s : F[X]) :
    sigma f G k s = ∑ i ∈ range (k + 1), f ^ i * G (k - i) * hasseDeriv i s := by
  simp [sigma, mul_assoc]

/-- For `k ≤ c`, `f^k E^{(k)}(s f^c) = f^c σ_k(s)`. -/
theorem pow_mul_hasseDeriv_mul_pow {f : F[X]} {c k : ℕ} (hk : k ≤ c) {G : ℕ → F[X]}
    (hG : ∀ l, hasseDeriv l (f ^ c) = f ^ (c - l) * G l) (s : F[X]) :
    f ^ k * hasseDeriv k (s * f ^ c) = f ^ c * sigma f G k s := by
  rw [hasseDeriv_mul, Finset.Nat.sum_antidiagonal_eq_sum_range_succ_mk, sigma_apply,
    Finset.mul_sum, Finset.mul_sum]
  refine Finset.sum_congr rfl fun i hi => ?_
  have hi : i ≤ k := Nat.lt_succ_iff.mp (Finset.mem_range.mp hi)
  dsimp only
  rw [hG, show f ^ c = f ^ (c - (k - i)) * f ^ (k - i) by
      rw [← pow_add, Nat.sub_add_cancel (by omega)],
    show f ^ k = f ^ i * f ^ (k - i) by rw [← pow_add, Nat.add_sub_cancel' hi]]
  ring

example := pow_mul_hasseDeriv_mul_pow (F := ZMod 3) (f := X) (c := 2) (k := 1) (by decide)
  (G := fun l => (exists_hasseDeriv_pow_eq (F := ZMod 3) X_ne_zero 2).choose l)
  (fun l => ((exists_hasseDeriv_pow_eq (F := ZMod 3) X_ne_zero 2).choose_spec l).1) X

/-- If `deg P ≤ i + n`, then `deg (P E^{(i)}(s)) ≤ deg s + n`. -/
theorem natDegree_mul_hasseDeriv_le {P : F[X]} {i n : ℕ} (hP : P.natDegree ≤ i + n)
    (s : F[X]) : (P * hasseDeriv i s).natDegree ≤ s.natDegree + n := by
  rcases lt_or_ge s.natDegree i with h | h
  · rw [hasseDeriv_eq_zero_of_lt_natDegree s i h, mul_zero, natDegree_zero]
    exact Nat.zero_le _
  · have := natDegree_hasseDeriv_le s i
    exact natDegree_mul_le.trans (by omega)

example := natDegree_mul_hasseDeriv_le (F := ZMod 3) (P := X) (i := 1) (n := 0) (by simp) X

/-- `deg (f^k E^{(k)}(r)) ≤ deg r + k (m − 1)`. -/
theorem natDegree_pow_mul_hasseDeriv_le {f : F[X]} (hf : 1 ≤ f.natDegree) (k : ℕ)
    (r : F[X]) : (f ^ k * hasseDeriv k r).natDegree ≤ r.natDegree + k * (f.natDegree - 1) := by
  refine natDegree_mul_hasseDeriv_le (natDegree_pow_le.trans ?_) r
  obtain ⟨n, hn⟩ : ∃ n, f.natDegree = n + 1 := ⟨_, (Nat.succ_pred_eq_of_pos hf).symm⟩
  rw [hn, Nat.add_sub_cancel, mul_add, mul_one, add_comm]

example := natDegree_pow_mul_hasseDeriv_le (F := ZMod 3) (f := X) (by simp) 1 X

/-- `deg σ_k(s) ≤ deg s + k (m − 1)`. -/
theorem natDegree_sigma_le {f : F[X]} (hf : 1 ≤ f.natDegree) {G : ℕ → F[X]}
    (hG : ∀ l, (G l).natDegree ≤ l * (f.natDegree - 1)) (k : ℕ) (s : F[X]) :
    (sigma f G k s).natDegree ≤ s.natDegree + k * (f.natDegree - 1) := by
  rw [sigma_apply]
  refine natDegree_sum_le_of_forall_le _ _ fun i hi => natDegree_mul_hasseDeriv_le ?_ s
  have hi : i ≤ k := Nat.lt_succ_iff.mp (Finset.mem_range.mp hi)
  refine natDegree_mul_le.trans ?_
  have h1 : (f ^ i).natDegree ≤ i * f.natDegree := natDegree_pow_le
  have h2 := hG (k - i)
  obtain ⟨n, hn⟩ : ∃ n, f.natDegree = n + 1 := ⟨_, (Nat.succ_pred_eq_of_pos hf).symm⟩
  rw [hn, Nat.add_sub_cancel] at h2 ⊢
  rw [hn] at h1
  have h3 : i * (n + 1) = i * n + i := by ring
  have h4 : (k - i) * n + i * n = k * n := by rw [← add_mul, Nat.sub_add_cancel hi]
  omega

example := natDegree_sigma_le (F := ZMod 3) (f := X) (by simp) (G := fun _ => 0)
  (fun _ => by simp) 1 X

/-- `deg T_k ≤ d + k (m − 1) + J − 1` when every `r_j` and `s_j` has degree at most `d`. -/
theorem natDegree_sum_pow_mul_hasseDeriv_le {f : F[X]} (hf : 1 ≤ f.natDegree)
    {G : ℕ → F[X]} (hG : ∀ l, (G l).natDegree ≤ l * (f.natDegree - 1)) (k : ℕ) (a : F)
    {J d : ℕ} {r s : Fin J → F[X]} (hr : ∀ j, (r j).natDegree ≤ d)
    (hs : ∀ j, (s j).natDegree ≤ d) :
    (∑ j, (f ^ k * hasseDeriv k (r j) + a • sigma f G k (s j)) * X ^ (j : ℕ)).natDegree ≤
      d + k * (f.natDegree - 1) + (J - 1) := by
  refine natDegree_sum_le_of_forall_le _ _ fun j _ =>
    natDegree_mul_le.trans (add_le_add ((natDegree_add_le _ _).trans (max_le ?_ ?_)) ?_)
  · exact (natDegree_pow_mul_hasseDeriv_le hf k _).trans (by have := hr j; omega)
  · exact (natDegree_smul_le _ _).trans
      ((natDegree_sigma_le hf hG k _).trans (by have := hs j; omega))
  · exact (natDegree_X_pow_le _).trans (by have := j.2; omega)

example := natDegree_sum_pow_mul_hasseDeriv_le (F := ZMod 3) (f := X) (by simp)
  (G := fun _ => 0) (fun _ => by simp) 1 1 (J := 1) (d := 1) (r := fun _ => X)
  (s := fun _ => X) (fun _ => by simp) (fun _ => by simp)

variable [Fintype F]

/-- For `k < q` and `k ≤ c`, `f^k E^{(k)}(∑_j (r_j + s_j f^c) X^{q j}) =
∑_j (f^k E^{(k)}(r_j) + f^c σ_k(s_j)) X^{q j}`. -/
theorem pow_mul_hasseDeriv_sum {f : F[X]} {c k : ℕ} (hkq : k < Fintype.card F) (hkc : k ≤ c)
    {G : ℕ → F[X]} (hG : ∀ l, hasseDeriv l (f ^ c) = f ^ (c - l) * G l) {J : ℕ}
    (r s : Fin J → F[X]) :
    f ^ k * hasseDeriv k (∑ j, (r j + s j * f ^ c) * X ^ (Fintype.card F * (j : ℕ))) =
      ∑ j, (f ^ k * hasseDeriv k (r j) + f ^ c * sigma f G k (s j)) *
        X ^ (Fintype.card F * (j : ℕ)) := by
  rw [map_sum, Finset.mul_sum]
  refine Finset.sum_congr rfl fun j _ => ?_
  rw [hasseDeriv_mul_X_pow_card_mul _ _ hkq, map_add, ← pow_mul_hasseDeriv_mul_pow hkc hG]
  ring

example := pow_mul_hasseDeriv_sum (F := ZMod 3) (f := X) (c := 1) (k := 1) (by simp) le_rfl
  (G := (exists_hasseDeriv_pow_eq (F := ZMod 3) X_ne_zero 1).choose)
  (fun l => ((exists_hasseDeriv_pow_eq (F := ZMod 3) X_ne_zero 1).choose_spec l).1)
  (J := 1) (fun _ => X) (fun _ => X)

/-- At `x` with `f(x)^c = a`, `f^k E^{(k)} R` takes the value of `T_k`. -/
theorem eval_pow_mul_hasseDeriv_sum {f : F[X]} {c k : ℕ} (hkq : k < Fintype.card F)
    (hkc : k ≤ c) {G : ℕ → F[X]} (hG : ∀ l, hasseDeriv l (f ^ c) = f ^ (c - l) * G l)
    {J : ℕ} (r s : Fin J → F[X]) {a x : F} (hx : f.eval x ^ c = a) :
    (f ^ k * hasseDeriv k (∑ j, (r j + s j * f ^ c) * X ^ (Fintype.card F * (j : ℕ)))).eval x =
      (∑ j, (f ^ k * hasseDeriv k (r j) + a • sigma f G k (s j)) * X ^ (j : ℕ)).eval x := by
  rw [pow_mul_hasseDeriv_sum hkq hkc hG]
  simp only [eval_finsetSum, eval_mul, eval_add, eval_pow, eval_smul, eval_X, smul_eq_mul, hx,
    pow_mul, FiniteField.pow_card]

example := eval_pow_mul_hasseDeriv_sum (F := ZMod 3) (f := X) (c := 1) (k := 1) (by simp) le_rfl
  (G := (exists_hasseDeriv_pow_eq (F := ZMod 3) X_ne_zero 1).choose)
  (fun l => ((exists_hasseDeriv_pow_eq (F := ZMod 3) X_ne_zero 1).choose_spec l).1)
  (J := 1) (fun _ => X) (fun _ => X) (a := 1) (x := 1) (by simp)

end Transformer.Weil
