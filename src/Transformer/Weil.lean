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

The proofs are written from the mathematics (S. A. Stepanov, 1969; W. M. Schmidt, *Equations
over Finite Fields*, Ch. I); no other formalization is used.
-/
