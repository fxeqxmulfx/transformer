//! Full complex thin SVD, adapted from residual_dynamics/algebra.rs.
//! LAPACK zgesvd retains every returned singular direction, including zeros.
//! Thin output is padded to the complete query width by the model recovery.
use crate::{C, Result};

#[link(name = "openblas")]
unsafe extern "C" {
    fn openblas_set_num_threads(n: i32);
    fn zgesvd_(
        jobu: *const u8,
        jobvt: *const u8,
        m: *const i32,
        n: *const i32,
        a: *mut C,
        lda: *const i32,
        s: *mut f64,
        u: *mut C,
        ldu: *const i32,
        vt: *mut C,
        ldvt: *const i32,
        work: *mut C,
        lwork: *const i32,
        rwork: *mut f64,
        info: *mut i32,
    );
}

pub struct Svd {
    pub values: Vec<f64>,
    /// Column-major q by rank; rank = min(q, stored columns), no threshold.
    pub u: Vec<C>,
    /// Column-major rank by stored columns, already conjugate transposed.
    pub vt: Vec<C>,
}

pub fn set_threads(n: i32) -> Result<()> {
    if !(1..=8).contains(&n) {
        return Err("invalid BLAS thread count".into());
    }
    unsafe { openblas_set_num_threads(n) };
    Ok(())
}

pub fn svd(q: usize, columns: usize, input: &[C]) -> Result<Svd> {
    if q == 0
        || columns == 0
        || q.checked_mul(columns) != Some(input.len())
        || input.iter().any(|x| !x.re.is_finite() || !x.im.is_finite())
    {
        return Err("invalid complete SVD input".into());
    }
    let m = i32::try_from(q)?;
    let n = i32::try_from(columns)?;
    let rank = q.min(columns);
    let ldvt = i32::try_from(rank)?;
    let mut a = input.to_vec();
    let mut values = vec![0.; rank];
    let mut u = vec![C::default(); q * rank];
    let mut vt = vec![C::default(); rank * columns];
    let mut rwork = vec![0.; 5 * rank];
    let mut query = [C::default()];
    let mut info = 0;
    unsafe {
        zgesvd_(
            b"S".as_ptr(),
            b"S".as_ptr(),
            &m,
            &n,
            a.as_mut_ptr(),
            &m,
            values.as_mut_ptr(),
            u.as_mut_ptr(),
            &m,
            vt.as_mut_ptr(),
            &ldvt,
            query.as_mut_ptr(),
            &-1,
            rwork.as_mut_ptr(),
            &mut info,
        )
    };
    if info != 0 || !query[0].re.is_finite() || query[0].re < 1. || query[0].re > i32::MAX as f64 {
        return Err("LAPACK workspace query failed".into());
    }
    let size = query[0].re.ceil() as i32;
    let mut work = vec![C::default(); size as usize];
    a.copy_from_slice(input);
    unsafe {
        zgesvd_(
            b"S".as_ptr(),
            b"S".as_ptr(),
            &m,
            &n,
            a.as_mut_ptr(),
            &m,
            values.as_mut_ptr(),
            u.as_mut_ptr(),
            &m,
            vt.as_mut_ptr(),
            &ldvt,
            work.as_mut_ptr(),
            &size,
            rwork.as_mut_ptr(),
            &mut info,
        )
    };
    if info != 0
        || values.iter().any(|&s| !s.is_finite() || s < 0.)
        || u.iter()
            .chain(&vt)
            .any(|x| !x.re.is_finite() || !x.im.is_finite())
    {
        return Err("complete complex SVD failed".into());
    }
    Ok(Svd { values, u, vt })
}
