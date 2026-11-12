//! Complex f64 LAPACK SVD and BLAS multiplication, using column-major storage.
//! SB1's operator ball is projected by clipping singular values, not by imposing flatness.
use crate::{C, Result};
use serde::{Deserialize, Serialize};
use std::ffi::{c_char, c_int, c_void};

#[link(name = "openblas")]
unsafe extern "C" {
    fn zgesvd_(
        jobu: *const c_char,
        jobvt: *const c_char,
        m: *const c_int,
        n: *const c_int,
        a: *mut C,
        lda: *const c_int,
        s: *mut f64,
        u: *mut C,
        ldu: *const c_int,
        vt: *mut C,
        ldvt: *const c_int,
        work: *mut C,
        lwork: *const c_int,
        rwork: *mut f64,
        info: *mut c_int,
        jobu_len: usize,
        jobvt_len: usize,
    );
    fn cblas_zgemm(
        order: c_int,
        ta: c_int,
        tb: c_int,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: *const c_void,
        a: *const c_void,
        lda: c_int,
        b: *const c_void,
        ldb: c_int,
        beta: *const c_void,
        c: *mut c_void,
        ldc: c_int,
    );
}

pub struct Svd {
    pub values: Vec<f64>,
    u: Vec<C>,
    vh: Vec<C>,
}

pub fn svd(x: &[C], size: usize) -> Result<Svd> {
    if size == 0
        || size > 4096
        || x.len() != size * size
        || x.iter().any(|z| !z.re.is_finite() || !z.im.is_finite())
    {
        return Err("invalid square SVD input".into());
    }
    let n = size as c_int;
    let mut a = x.to_vec();
    let mut result = Svd {
        values: vec![0.; size],
        u: vec![C::default(); x.len()],
        vh: vec![C::default(); x.len()],
    };
    let mut rw = vec![0.; 5 * size];
    let mut query = [C::default()];
    let mut info = 0;
    let job = b'A' as c_char;
    // LAPACK receives only pointers to repr(C) Complex64, never complex values by value.
    // All arrays and leading dimensions match the validated square input. Trailing
    // size_t character lengths implement the gfortran ABI of the system library.
    unsafe {
        zgesvd_(
            &job,
            &job,
            &n,
            &n,
            a.as_mut_ptr(),
            &n,
            result.values.as_mut_ptr(),
            result.u.as_mut_ptr(),
            &n,
            result.vh.as_mut_ptr(),
            &n,
            query.as_mut_ptr(),
            &-1,
            rw.as_mut_ptr(),
            &mut info,
            1,
            1,
        );
    }
    if info != 0 || !query[0].re.is_finite() || query[0].re < 1. || query[0].re > c_int::MAX as f64
    {
        return Err(format!("SVD workspace query failed: {info}").into());
    }
    let len = query[0].re.ceil() as c_int;
    let mut work = vec![C::default(); len as usize];
    a.copy_from_slice(x);
    unsafe {
        zgesvd_(
            &job,
            &job,
            &n,
            &n,
            a.as_mut_ptr(),
            &n,
            result.values.as_mut_ptr(),
            result.u.as_mut_ptr(),
            &n,
            result.vh.as_mut_ptr(),
            &n,
            work.as_mut_ptr(),
            &len,
            rw.as_mut_ptr(),
            &mut info,
            1,
            1,
        );
    }
    if info != 0 || result.values.iter().any(|s| !s.is_finite() || *s < 0.) {
        return Err(format!("SVD failed: {info}").into());
    }
    Ok(result)
}

impl Svd {
    pub fn reconstruct(mut self, radius: f64) -> Vec<C> {
        let size = self.values.len();
        for (col, &sigma) in self.values.iter().enumerate() {
            for row in 0..size {
                self.u[row + size * col] *= sigma.min(radius);
            }
        }
        let n = size as c_int;
        let (one, zero) = (C::new(1., 0.), C::default());
        let mut out = vec![zero; size * size];
        // CBLAS enum values from the installed cblas.h: column major, no transpose.
        // Both operands and the output are disjoint square column-major arrays.
        unsafe {
            cblas_zgemm(
                102,
                111,
                111,
                n,
                n,
                n,
                (&one as *const C).cast(),
                self.u.as_ptr().cast(),
                n,
                self.vh.as_ptr().cast(),
                n,
                (&zero as *const C).cast(),
                out.as_mut_ptr().cast(),
                n,
            );
        }
        out
    }
}

pub fn max_norm(x: &[C]) -> f64 {
    x.iter().map(|z| z.norm()).fold(0., f64::max)
}

pub fn frobenius_distance(x: &[C], y: &[C]) -> f64 {
    x.iter()
        .zip(y)
        .map(|(a, b)| (*a - *b).norm_sqr())
        .sum::<f64>()
        .sqrt()
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Projection {
    pub iterations: usize,
    pub capped: bool,
    pub change: f64,
    pub radial_shrink: f64,
}

pub fn project(
    input: &[C],
    n: usize,
    radius: f64,
    cap: usize,
    tol: f64,
) -> Result<(Vec<C>, Projection)> {
    if !radius.is_finite() || radius <= 0. || cap == 0 || !tol.is_finite() || tol <= 0. {
        return Err("invalid projection configuration".into());
    }
    let mut x = input.to_vec();
    let mut p = vec![C::default(); x.len()];
    let mut q = p.clone();
    let mut report = Projection {
        iterations: 0,
        capped: true,
        change: f64::INFINITY,
        radial_shrink: 1.,
    };
    for iteration in 1..=cap {
        let mut y = vec![C::default(); x.len()];
        for j in 0..x.len() {
            let z = x[j] + p[j];
            y[j] = z / z.norm().max(1.);
            p[j] = z - y[j];
            y[j] += q[j];
        }
        let decomposition = svd(&y, n)?;
        let next = if decomposition.values[0] > radius {
            decomposition.reconstruct(radius)
        } else {
            y.clone()
        };
        for j in 0..x.len() {
            q[j] = y[j] - next[j];
        }
        report.change = x
            .iter()
            .zip(&next)
            .map(|(a, b)| (*a - *b).norm())
            .fold(0., f64::max);
        report.iterations = iteration;
        x = next;
        if report.change <= tol && max_norm(&x) <= 1. + tol {
            report.capped = false;
            break;
        }
    }
    let norm = svd(&x, n)?.values[0];
    report.radial_shrink = max_norm(&x).max(norm / radius).max(1.);
    for z in &mut x {
        *z /= report.radial_shrink;
    }
    Ok((x, report))
}
