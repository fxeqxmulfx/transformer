module

public import Transformer.Weil.HasseDeriv
public import Transformer.Weil.Independence
public import Transformer.Weil.Auxiliary
public import Transformer.Weil.LinearSystem
public import Transformer.Weil.Stepanov
public import Transformer.Weil.Hyperelliptic
public import Transformer.Weil.TracePoly
public import Transformer.Weil.TraceIndependence
public import Transformer.Weil.TraceSystem
public import Transformer.Weil.TraceStepanov
public import Transformer.Weil.ArtinSchreier
public import Transformer.Weil.Transfer
public import Transformer.Weil.PowerSums
public import Transformer.Weil.RootSum
public import Transformer.Weil.LFunction
public import Transformer.Weil.Divisors
public import Transformer.Weil.NewtonIdentity
public import Transformer.Weil.FrobeniusOrbit
public import Transformer.Weil.FieldSum

/-!
# Stepanov's method for `y² = f(x)` and `y^p − y = f(x)`

Over a finite field `F` of odd order `q`, for `f ∈ F[X]` of odd degree `m`, Stepanov's method
counts the points with `f(x)^{(q−1)/2} = a` with an auxiliary polynomial
`R = ∑_{j < J} (r_j + s_j f^{(q−1)/2}) X^{q j}` that vanishes to order `ℓ` at each of them:

* `HasseDeriv`: Hasse derivatives; `X^{q j}` passes through `E^{(k)}` for `k < q`, and vanishing
  to order `ℓ` on `S` forces `ℓ |S| ≤ deg R`;
* `Independence`: `R ≠ 0` unless every `r_j`, `s_j` is `0`, for `deg r_j, deg s_j < K` with
  `m + 2K ≤ q + 1`;
* `Auxiliary`: at a point with `f(x)^{(q−1)/2} = a`, `f^k E^{(k)} R` takes the value of a
  polynomial `T_k`, linear in the `r_j`, `s_j`;
* `LinearSystem`: Siegel's lemma makes every `T_k`, `k < ℓ`, vanish when
  `ℓ (K + J + (ℓ − 1)(m − 1)) < 2 J K`;
* `Stepanov`: then `ℓ · #{x : f(x)^{(q−1)/2} = a} ≤ K − 1 + m (q − 1)/2 + q (J − 1)`;
* `Hyperelliptic`: for `q ≥ 64` and `8 m ≤ q`, `#{x : f(x)^{(q−1)/2} = a} ≤ q/2 + 3 m √q`,
  `|∑_x χ₂(f(x))| ≤ 7 m √q` and `|#{(x, y) : y² = f(x)} − q| ≤ 7 m √q`.

For `q = p^ν` and `f` of degree `m ≥ 1` prime to `p`, the same method counts the points with
`Tr(f(x)) = c` through the trace polynomial `g = T_ν(f) = ∑_{k < ν} f^{p^k}` and
`R = ∑_{i < p, j < J} r_{ij} g^i X^{q j}`:

* `TracePoly`: `g(x) = Tr(f(x))`, `deg g = m p^{ν − 1}`, and on `{g = c}` every
  `E^{(b)}(g^i)` agrees with a polynomial of degree `≤ (m − 1) b`;
* `TraceIndependence`: `R ≠ 0` unless every `r_{ij}` is `0`, for `deg r_{ij} < K ≤ p^{ν − 1}`;
* `TraceSystem`: on `{g = c}`, `E^{(k)} R` takes the value of a polynomial `T_k`, linear in the
  `r_{ij}`, of degree `< K + J + (m − 1)(ℓ − 1)`;
* `TraceStepanov`: if `ℓ (K + J + (m − 1)(ℓ − 1)) < p J K`, then
  `ℓ · #{x : g(x) = c} ≤ K − 1 + (p − 1) m p^{ν − 1} + q (J − 1)`;
* `ArtinSchreier`: for `q ≥ 5`, `#{x : Tr(f(x)) = c} ≤ q/p + 4 m √q`, and
  `|∑_x ψ(Tr f(x))| ≤ 4 p m √q` for a nontrivial additive character `ψ` of `𝔽_p`.

Toward the `L`-function of `y^p − y = f(x)`:

* `Transfer`: if `‖∑_{ω ∈ s} ω^n‖ ≤ M R^n` for every large `n`, then every `ω ∈ s` has
  `‖ω‖ ≤ R`;
* `PowerSums`: the power sums `p_j = ∑_i β_i^j` of the roots of `g`, from its coefficients as
  those of `−X A′/A` (`A` the reverse of `g`): Newton's recursion, `p_j(g h) = p_j(g) + p_j(h)`,
  `p_j(X − β) = β^j`;
* `RootSum`: `∑_i f(β_i) = ∑_j f_j p_j(g)` without a splitting field; adding `t X^{n − m}` to a
  monic `g` of degree `n ≥ m = deg f` moves it by `−m f_m t`;
* `LFunction`: the coefficients `c_n = ∑_{g monic, deg g = n} ψ(∑_i f(β_i))` of the
  `L`-function, `c_0 = 1`, and `c_n = 0` for `n ≥ m` when `ψ ≠ 1` and `m` is prime to `p`;
* `Divisors`: for monic `g` of degree `≤ N`, `∑_{P^e ∣ g} deg P = deg g` over monic irreducible
  `P` and `1 ≤ e ≤ N`, and `∑_{deg g = n, P^e ∣ g} λ(g) = λ(P)^e c_{n − e deg P}`;
* `NewtonIdentity`: `n c_n = ∑_{1 ≤ d ≤ n} S_d c_{n − d}`, with
  `S_d = ∑_{e deg P = d} deg P · λ(P)^e`;
* `FrobeniusOrbit`: for `x` in an extension `L` of degree `d`, with minimal polynomial `P` of
  degree `k`, `P = ∏_{i < k} (X − x^{q^i})` and `Tr_{L/K} f(x) = (d/k) ∑_i f(β_i)`;
* `FieldSum`: `S_d = ∑_{x ∈ L} ψ(Tr_{L/K} f(x))` for `L` of degree `d`.

The proofs are written from the mathematics (S. A. Stepanov, 1969; W. M. Schmidt, *Equations
over Finite Fields*, Ch. I); no other formalization is used.
-/
