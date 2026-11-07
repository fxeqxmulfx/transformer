//! OpenBLAS operations from flat_approx 9328bc83; SPD solve replaces its fixed ridge fit.
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

pub fn solve_spd(matrix: &[f64], rhs: &[f64], d: usize) -> Vec<f64> {
    assert!(d > 0 && rhs.len().is_multiple_of(d));
    assert_eq!(matrix.len(), d * d);
    let k = rhs.len() / d;
    let mut gram = matrix.to_vec();
    let mut solution = rhs.to_vec();
    let mut info = 0;
    unsafe {
        dposv_(
            b"U".as_ptr(),
            &(d as i32),
            &(k as i32),
            gram.as_mut_ptr(),
            &(d as i32),
            solution.as_mut_ptr(),
            &(d as i32),
            &mut info,
        );
    }
    assert_eq!(info, 0, "transport Cholesky failed");
    assert!(solution.iter().all(|x| x.is_finite()));
    solution
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
    fn spd_solver_checks_two_independent_right_sides() {
        let matrix = [4., 1., 1., 3.];
        let solution = solve_spd(&matrix, &[9., 7., -1., 8.], 2);
        let reconstructed = linear(&solution, 2, 2, &matrix, 2);
        assert!(
            reconstructed
                .iter()
                .zip([9., 7., -1., 8.])
                .all(|(x, y)| (x - y).abs() < 1e-12)
        );
        assert!((solution[0] - 20. / 11.).abs() < 1e-12);
        assert!((solution[1] - 19. / 11.).abs() < 1e-12);
    }
}
