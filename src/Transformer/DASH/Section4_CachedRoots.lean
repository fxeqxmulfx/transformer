/-
# DASH — inverse-root buffers between refreshes

arXiv:2602.02016v2, §4. A refresh frequency `f>0` recomputes at multiples
of `f` and reuses the last computed inverse root at all intervening steps.
-/

import Transformer.DASH.Section4_Blocking

namespace Transformer.DASH

/-- The last inverse-root refresh performed through step `t`,
arXiv:2602.02016v2, §4, preconditioner update frequency. -/
def lastRootRefresh (f : ℕ) : ℕ → ℕ
  | 0 => 0
  | t + 1 => if (t + 1) % f = 0 then t + 1 else lastRootRefresh f t

/-- Inverse-root buffer initialized at step zero and refreshed every `f` steps.
Source: arXiv:2602.02016v2, §4, stored inverse-root tensors. -/
def cachedRoot {α β : Type*} (solve : α → β) (A : ℕ → α) (f : ℕ) : ℕ → β
  | 0 => solve (A 0)
  | t + 1 => if (t + 1) % f = 0 then solve (A (t + 1)) else cachedRoot solve A f t

/-- The stored buffer is exactly the solver output from the latest refresh.
It is not silently treated as the root of the current preconditioner.
Source: arXiv:2602.02016v2, §4, using stored buffers between calls. -/
theorem cachedRoot_at_last_refresh {α β : Type*} (solve : α → β) (A : ℕ → α)
    (f t : ℕ) : cachedRoot solve A f t = solve (A (lastRootRefresh f t)) := by
  induction t with
  | zero => rfl
  | succ t ih => simp only [cachedRoot, lastRootRefresh]; split <;> simp_all

/-- The most recent refresh is the current step rounded down to a multiple of
the positive frequency. Source: arXiv:2602.02016v2, §4, inverse-root frequency. -/
theorem lastRootRefresh_eq (f : ℕ) (hf : 0 < f) (t : ℕ) :
    lastRootRefresh f t = t - t % f := by
  induction t with
  | zero => simp [lastRootRefresh]
  | succ t ih =>
    by_cases h : (t + 1) % f = 0
    · simp [lastRootRefresh, h]
    · rw [lastRootRefresh, ite_eq_right h, ih]
      have hr := Nat.mod_lt t hf
      have hrel : (t + 1) % f = (t % f + 1) % f := by rw [Nat.mod_add_mod]
      have hsmall : t % f + 1 < f := by
        by_contra hn
        have heq : t % f + 1 = f := by omega
        rw [heq, Nat.mod_self] at hrel
        exact h hrel
      rw [Nat.mod_eq_of_lt hsmall] at hrel
      have hle := Nat.mod_le t f
      omega

/-- A positive frequency witnesses the refresh assumptions,
arXiv:2602.02016v2, §4. -/
example : 0 < (10 : ℕ) := by norm_num

/-- The root buffer is fewer than `f` steps old.
Source: arXiv:2602.02016v2, §4, intervening steps between inverse-root calls. -/
theorem cachedRoot_age (f : ℕ) (hf : 0 < f) (t : ℕ) :
    t - lastRootRefresh f t < f := by
  rw [lastRootRefresh_eq f hf]
  have hle := Nat.mod_le t f
  have hlt := Nat.mod_lt t hf
  omega

/-- Age-bound assumptions are satisfiable, arXiv:2602.02016v2, §4. -/
example : 0 < (1 : ℕ) := by norm_num

/-- Frequency one always uses a freshly computed root.
Source: arXiv:2602.02016v2, §4, experiments with `f=1`. -/
theorem cachedRoot_frequency_one {α β : Type*} (solve : α → β) (A : ℕ → α) (t : ℕ) :
    cachedRoot solve A 1 t = solve (A t) := by
  have heq : lastRootRefresh 1 t = t := by
    simpa only [Nat.mod_one, Nat.sub_zero] using lastRootRefresh_eq 1 (by norm_num) t
  rw [cachedRoot_at_last_refresh, heq]

/-- Batched root caching equals individual caching coordinate by coordinate.
This proves that stacking and stored buffers preserve the same root histories,
without a claim about hardware running time.
Source: arXiv:2602.02016v2, §4, stacked inverse-root buffers. -/
theorem cachedRoot_batch {ι α β : Type*} (solve : α → β) (A : ℕ → ι → α) (f t : ℕ) :
    cachedRoot (fun X i => solve (X i)) A f t =
      fun i => cachedRoot solve (fun k => A k i) f t := by
  funext i
  rw [cachedRoot_at_last_refresh, cachedRoot_at_last_refresh]

end Transformer.DASH
