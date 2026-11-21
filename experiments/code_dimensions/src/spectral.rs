//! The complex SVD by one-sided Jacobi, the projection onto the tangent space of the rank-`r`
//! matrices, and the first-order test of an interpolant. Matrices are column-major.
use num_complex::Complex64 as C;

const ZERO: C = C::new(0.0, 0.0);
const SWEEPS: usize = 60;
/// Columns count as orthogonal below this cosine.
const ORTHOGONAL: f64 = 1e-14;
/// A singular value below this fraction of the largest counts as zero.
const RANK_TOLERANCE: f64 = 1e-9;
const CG_TOLERANCE: f64 = 1e-10;
const CG_ITERATIONS: usize = 5000;

/// `A = U diag(σ) Vᴴ` for the `rank` nonzero singular values, largest first.
pub struct Svd {
    pub sigma: Vec<f64>,
    pub rank: usize,
    pub u: Vec<C>,
    pub v: Vec<C>,
}

/// Rotate columns `x`, `y` by the Jacobi rotation with cosine `c`, sine `s` and phase `w`.
fn rotate(x: &mut [C], y: &mut [C], c: f64, s: f64, w: C) {
    for (xi, yi) in x.iter_mut().zip(y.iter_mut()) {
        let turned = *yi * w;
        let new_x = *xi * c - turned * s;
        *yi = *xi * s + turned * c;
        *xi = new_x;
    }
}

/// The SVD of an `m × n` matrix. O(sweeps · n² · (m + n)) time, O(n²) extra memory.
pub fn svd(mut a: Vec<C>, m: usize, n: usize) -> Svd {
    let mut v = vec![ZERO; n * n];
    for i in 0..n {
        v[i * n + i] = C::new(1.0, 0.0);
    }
    let scale: f64 = a.iter().map(C::norm_sqr).sum();
    for _ in 0..SWEEPS {
        let mut rotated = false;
        for j in 1..n {
            let (left, right) = a.split_at_mut(j * m);
            let (vl, vr) = v.split_at_mut(j * n);
            for i in 0..j {
                let x = &mut left[i * m..(i + 1) * m];
                let y = &mut right[..m];
                let (mut alpha, mut beta, mut gamma) = (0.0, 0.0, ZERO);
                for (xi, yi) in x.iter().zip(y.iter()) {
                    alpha += xi.norm_sqr();
                    beta += yi.norm_sqr();
                    gamma += xi.conj() * yi;
                }
                let g = gamma.norm();
                if alpha.min(beta) <= 1e-26 * scale || g <= ORTHOGONAL * (alpha * beta).sqrt() {
                    continue;
                }
                rotated = true;
                let zeta = (beta - alpha) / (2.0 * g);
                let t = zeta.signum() / (zeta.abs() + (1.0 + zeta * zeta).sqrt());
                let c = 1.0 / (1.0 + t * t).sqrt();
                let w = gamma.conj() / g;
                rotate(x, y, c, c * t, w);
                rotate(&mut vl[i * n..(i + 1) * n], &mut vr[..n], c, c * t, w);
            }
        }
        if !rotated {
            break;
        }
    }
    let norms: Vec<f64> = a
        .chunks_exact(m)
        .map(|col| col.iter().map(C::norm_sqr).sum::<f64>().sqrt())
        .collect();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_unstable_by(|&i, &j| norms[j].total_cmp(&norms[i]));
    let sigma: Vec<f64> = order.iter().map(|&k| norms[k]).collect();
    let cutoff = RANK_TOLERANCE * sigma[0];
    let rank = sigma.iter().take_while(|&&s| s > cutoff).count();
    let mut u = Vec::with_capacity(m * rank);
    let mut kept = Vec::with_capacity(n * rank);
    for &k in &order[..rank] {
        u.extend(a[k * m..(k + 1) * m].iter().map(|z| z / norms[k]));
        kept.extend_from_slice(&v[k * n..(k + 1) * n]);
    }
    Svd { sigma, rank, u, v: kept }
}

/// `A B` for `A` of `rows × inner` and `B` of `inner × cols`. O(rows · inner · cols).
fn mul(a: &[C], rows: usize, inner: usize, b: &[C], cols: usize) -> Vec<C> {
    let mut out = vec![ZERO; rows * cols];
    for (col, b_col) in out.chunks_exact_mut(rows).zip(b.chunks_exact(inner)) {
        for (a_col, &weight) in a.chunks_exact(rows).zip(b_col) {
            for (o, x) in col.iter_mut().zip(a_col) {
                *o += x * weight;
            }
        }
    }
    out
}

/// `Aᴴ B` for `A` of `rows × acols` and `B` of `rows × bcols`. O(rows · acols · bcols).
fn mul_conj_left(a: &[C], rows: usize, acols: usize, b: &[C], bcols: usize) -> Vec<C> {
    let mut out = vec![ZERO; acols * bcols];
    for (col, b_col) in out.chunks_exact_mut(acols).zip(b.chunks_exact(rows)) {
        for (o, a_col) in col.iter_mut().zip(a.chunks_exact(rows)) {
            *o = a_col.iter().zip(b_col).map(|(x, y)| x.conj() * y).sum();
        }
    }
    out
}

/// `A Bᴴ` for `A` of `rows × r` and `B` of `cols × r`. O(rows · r · cols).
fn mul_conj_right(a: &[C], rows: usize, b: &[C], cols: usize) -> Vec<C> {
    let mut out = vec![ZERO; rows * cols];
    for (a_col, b_col) in a.chunks_exact(rows).zip(b.chunks_exact(cols)) {
        for (col, &weight) in out.chunks_exact_mut(rows).zip(b_col) {
            let weight = weight.conj();
            for (o, x) in col.iter_mut().zip(a_col) {
                *o += x * weight;
            }
        }
    }
    out
}

/// `P_T(Z) = U Uᴴ Z + Z V Vᴴ - U Uᴴ Z V Vᴴ` for an `n × n` matrix `Z`. O(n² r) time and O(n r)
/// extra memory; at full rank `T` is the whole space.
fn project(svd: &Svd, n: usize, z: &[C]) -> Vec<C> {
    let r = svd.rank;
    if r == n {
        return z.to_vec();
    }
    let uz = mul_conj_left(&svd.u, n, r, z, n);
    let zv = mul(z, n, n, &svd.v, r);
    let uzv = mul(&uz, r, n, &svd.v, r);
    let corner = mul_conj_right(&uzv, r, &svd.v, n);
    let inner: Vec<C> = uz.iter().zip(&corner).map(|(x, y)| x - y).collect();
    let mut out = mul(&svd.u, n, r, &inner, n);
    let right = mul_conj_right(&zv, n, &svd.v, n);
    for (o, x) in out.iter_mut().zip(&right) {
        *o += x;
    }
    out
}

fn norm_sqr(z: &[C]) -> f64 {
    z.iter().map(C::norm_sqr).sum()
}

/// The first-order test of `M = U S Vᴴ` on a split.
pub struct FirstOrder {
    /// `‖R‖ / ‖G‖`, with `G = U S^{-1/3} Vᴴ` and `R` its component in the tangent directions
    /// that vanish on the training cells.
    pub ratio: f64,
    /// `‖R‖` on the training cells, relative to `‖G‖`; zero up to the solver's tolerance.
    pub on_train: f64,
    pub iterations: usize,
}

/// Solve `P_Ω P_T P_Ω w = P_Ω G` by conjugate gradients, then `R = G - P_T w`. The system is
/// consistent because `G` lies in `T`. O(iterations · n² r) time, O(n²) memory.
pub fn first_order(svd: &Svd, n: usize, train: &[bool]) -> FirstOrder {
    let scaled: Vec<C> = svd
        .u
        .chunks_exact(n)
        .zip(&svd.sigma)
        .flat_map(|(col, s)| col.iter().map(move |z| z * s.powf(-1.0 / 3.0)))
        .collect();
    let g = mul_conj_right(&scaled, n, &svd.v, n);
    let restrict = |z: &mut [C]| {
        for (x, &on) in z.iter_mut().zip(train) {
            if !on {
                *x = ZERO;
            }
        }
    };
    let mut residual = g.clone();
    restrict(&mut residual);
    let target = norm_sqr(&residual);
    let mut direction = residual.clone();
    let mut w = vec![ZERO; n * n];
    let mut rr = target;
    let mut iterations = 0;
    while iterations < CG_ITERATIONS && rr > CG_TOLERANCE * CG_TOLERANCE * target {
        let mut image = project(svd, n, &direction);
        restrict(&mut image);
        let curvature: f64 = direction.iter().zip(&image).map(|(d, h)| (d.conj() * h).re).sum();
        if curvature <= 1e-30 * target {
            break;
        }
        let step = rr / curvature;
        for (x, d) in w.iter_mut().zip(&direction) {
            *x += d * step;
        }
        for (x, h) in residual.iter_mut().zip(&image) {
            *x -= h * step;
        }
        let next = norm_sqr(&residual);
        let beta = next / rr;
        for (d, x) in direction.iter_mut().zip(&residual) {
            *d = x + *d * beta;
        }
        rr = next;
        iterations += 1;
    }
    let image = project(svd, n, &w);
    let mut r: Vec<C> = g.iter().zip(&image).map(|(x, y)| x - y).collect();
    let size = norm_sqr(&g);
    let ratio = (norm_sqr(&r) / size).sqrt();
    restrict(&mut r);
    FirstOrder { ratio, on_train: (norm_sqr(&r) / size).sqrt(), iterations }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn svd_reconstructs_a_rank_two_matrix() {
        let (x, y) = (C::new(1.0, 2.0), C::new(-0.5, 0.25));
        let a = vec![x, y, x + y, y, x * y, y - x, x + x, y + y, x * 2.0 + y * 2.0];
        let out = svd(a.clone(), 3, 3);
        let scaled: Vec<C> = out
            .u
            .chunks_exact(3)
            .zip(&out.sigma)
            .flat_map(|(col, s)| col.iter().map(move |z| z * s))
            .collect();
        let back = mul_conj_right(&scaled, 3, &out.v, 3);
        let error: f64 = a.iter().zip(&back).map(|(p, q)| (p - q).norm_sqr()).sum();
        assert!(error < 1e-20, "error {error}");
    }

    #[test]
    fn projection_is_idempotent_and_fixes_the_factors() {
        let (x, y) = (C::new(1.0, 2.0), C::new(-0.5, 0.25));
        let a = vec![x, y, y, x * x, x, y * y, y, y, x];
        let m = svd(a.clone(), 3, 3);
        let reduced = Svd {
            sigma: m.sigma[..1].to_vec(),
            rank: 1,
            u: m.u[..3].to_vec(),
            v: m.v[..3].to_vec(),
        };
        let once = project(&reduced, 3, &a);
        let twice = project(&reduced, 3, &once);
        let error: f64 = once.iter().zip(&twice).map(|(p, q)| (p - q).norm_sqr()).sum();
        assert!(error < 1e-20, "error {error}");
    }

    #[test]
    fn a_full_rank_table_keeps_the_heldout_cells() {
        // The 2 × 2 DFT: full rank, flat, so `R` is `G` on the cell outside the training set.
        let s = C::new(1.0, 0.0);
        let a = vec![s, s, s, -s];
        let m = svd(a, 2, 2);
        let out = first_order(&m, 2, &[true, true, true, false]);
        assert!((out.ratio - 0.5).abs() < 1e-9, "ratio {}", out.ratio);
        assert!(out.on_train < 1e-9);
    }
}
