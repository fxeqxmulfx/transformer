//! Checked sizes around native OpenBLAS operations; diagnostic FP64, not intervals.
use num_complex::Complex64;

#[link(name = "openblas")]
unsafe extern "C" {
    fn openblas_set_num_threads(n: i32);
    fn cblas_dgemm(
        layout: i32,
        ta: i32,
        tb: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: f64,
        a: *const f64,
        lda: i32,
        b: *const f64,
        ldb: i32,
        beta: f64,
        c: *mut f64,
        ldc: i32,
    );
    fn dposv_(
        uplo: *const u8,
        n: *const i32,
        nrhs: *const i32,
        a: *mut f64,
        lda: *const i32,
        b: *mut f64,
        ldb: *const i32,
        info: *mut i32,
    );
    fn zgesvd_(
        jobu: *const u8,
        jobvt: *const u8,
        m: *const i32,
        n: *const i32,
        a: *mut Complex64,
        lda: *const i32,
        s: *mut f64,
        u: *mut Complex64,
        ldu: *const i32,
        vt: *mut Complex64,
        ldvt: *const i32,
        work: *mut Complex64,
        lwork: *const i32,
        rwork: *mut f64,
        info: *mut i32,
    );
}

pub fn threads() {
    unsafe {
        openblas_set_num_threads(4);
    }
}

pub fn linear(x: &[f64], rows: usize, input: usize, w: &[f64], output: usize) -> Vec<f64> {
    assert_eq!(x.len(), rows * input);
    assert_eq!(w.len(), output * input);
    let mut y = vec![0.0; rows * output];
    unsafe {
        cblas_dgemm(
            101,
            111,
            112,
            rows as i32,
            output as i32,
            input as i32,
            1.0,
            x.as_ptr(),
            input as i32,
            w.as_ptr(),
            input as i32,
            0.0,
            y.as_mut_ptr(),
            output as i32,
        );
    }
    y
}

pub fn ridge(x: &[f64], y: &[f64], rows: usize, d: usize, k: usize) -> (Vec<f64>, f64) {
    assert_eq!(x.len(), rows * d);
    assert_eq!(y.len(), rows * k);
    let mut gram = vec![0.0; d * d];
    let mut cross = vec![0.0; d * k];
    unsafe {
        cblas_dgemm(
            101,
            112,
            111,
            d as i32,
            d as i32,
            rows as i32,
            1.0,
            x.as_ptr(),
            d as i32,
            x.as_ptr(),
            d as i32,
            0.0,
            gram.as_mut_ptr(),
            d as i32,
        );
        cblas_dgemm(
            101,
            112,
            111,
            d as i32,
            k as i32,
            rows as i32,
            1.0,
            x.as_ptr(),
            d as i32,
            y.as_ptr(),
            k as i32,
            0.0,
            cross.as_mut_ptr(),
            k as i32,
        );
    }
    let penalty = 1e-8 * (0..d).map(|j| gram[j * d + j]).sum::<f64>() / d as f64;
    assert!(penalty > 0.0);
    for j in 0..d {
        gram[j * d + j] += penalty;
    }
    let mut weights = vec![0.0; d * k];
    for j in 0..d {
        for c in 0..k {
            weights[c * d + j] = cross[j * k + c];
        }
    }
    let mut info = 0;
    unsafe {
        dposv_(
            b"U".as_ptr(),
            &(d as i32),
            &(k as i32),
            gram.as_mut_ptr(),
            &(d as i32),
            weights.as_mut_ptr(),
            &(d as i32),
            &mut info,
        );
    }
    assert_eq!(info, 0, "ridge Cholesky failed");
    assert!(weights.iter().all(|x| x.is_finite()));
    (weights, penalty)
}

pub fn singular_values(x: &[Complex64], n: usize) -> Vec<f64> {
    assert_eq!(x.len(), n * n);
    let mut a = vec![Complex64::default(); n * n];
    for i in 0..n {
        for j in 0..n {
            a[j * n + i] = x[i * n + j];
        }
    }
    let mut s = vec![0.0; n];
    let mut rwork = vec![0.0; 5 * n];
    let mut dummy_u = [Complex64::default()];
    let mut dummy_v = [Complex64::default()];
    let mut work = vec![Complex64::default()];
    let mut info = 0;
    let call = |work: &mut [Complex64],
                size: i32,
                a: &mut [Complex64],
                s: &mut [f64],
                rwork: &mut [f64],
                u: &mut [Complex64],
                v: &mut [Complex64],
                info: &mut i32| unsafe {
        zgesvd_(
            b"N".as_ptr(),
            b"N".as_ptr(),
            &(n as i32),
            &(n as i32),
            a.as_mut_ptr(),
            &(n as i32),
            s.as_mut_ptr(),
            u.as_mut_ptr(),
            &1,
            v.as_mut_ptr(),
            &1,
            work.as_mut_ptr(),
            &size,
            rwork.as_mut_ptr(),
            info,
        );
    };
    call(
        &mut work,
        -1,
        &mut a,
        &mut s,
        &mut rwork,
        &mut dummy_u,
        &mut dummy_v,
        &mut info,
    );
    assert_eq!(info, 0);
    let size = work[0].re as i32;
    work.resize(size as usize, Complex64::default());
    call(
        &mut work,
        size,
        &mut a,
        &mut s,
        &mut rwork,
        &mut dummy_u,
        &mut dummy_v,
        &mut info,
    );
    assert_eq!(info, 0, "complex SVD failed");
    assert!(s.iter().all(|s| s.is_finite() && *s >= 0.0));
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rectangular_gemm_and_complex_svd_match_closed_forms() {
        assert_eq!(
            linear(
                &[1., 2., 3., 4., 5., 6.],
                2,
                3,
                &[1., 0., -1., 0., 2., 0.],
                2
            ),
            [-2., 4., -2., 10.]
        );
        let s = singular_values(
            &[
                Complex64::new(0., 3.),
                Complex64::default(),
                Complex64::default(),
                Complex64::new(-4., 0.),
            ],
            2,
        );
        assert!((s[0] - 4.).abs() < 1e-12 && (s[1] - 3.).abs() < 1e-12);
    }
    #[test]
    fn ridge_solves_a_known_linear_map() {
        let x = [1., 0., 0., 1., 1., 1., 2., -1.];
        let y = linear(&x, 4, 2, &[2., -3.], 1);
        let (w, _) = ridge(&x, &y, 4, 2, 1);
        assert!((w[0] - 2.).abs() < 1e-6 && (w[1] + 3.).abs() < 1e-6);
    }
}
