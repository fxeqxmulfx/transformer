//! Small Hermitian matrices, in column-major storage; Frobenius real coordinates.
//! ZHEEV interface: https://www.netlib.org/lapack/explore-html/d8/d1c/group__heev_gadbb2b87ce42e51fdaac3228a857a58c8.html
//! ABI follows the existing spectral crate's OpenBLAS/gfortran wrapper, unchanged.
use crate::{C, Result};
use std::ffi::{c_char, c_int};

#[link(name = "openblas")]
unsafe extern "C" {
    fn zheev_(
        job: *const c_char,
        uplo: *const c_char,
        n: *const c_int,
        a: *mut C,
        lda: *const c_int,
        w: *mut f64,
        work: *mut C,
        lwork: *const c_int,
        rwork: *mut f64,
        info: *mut c_int,
        job_len: usize,
        uplo_len: usize,
    );
}

pub struct Eigen {
    /// Ascending eigenvalues and matching orthonormal columns.
    pub values: Vec<f64>,
    pub vectors: Vec<C>,
}

pub fn eigen(a: &[C], size: usize) -> Result<Eigen> {
    if size == 0
        || size > 194
        || a.len() != size * size
        || a.iter().any(|z| !z.re.is_finite() || !z.im.is_finite())
    {
        return Err("invalid bounded Hermitian eigensolve input".into());
    }
    let scale = a.iter().map(|z| z.norm()).fold(1., f64::max);
    for j in 0..size {
        for i in 0..size {
            if (a[i + size * j] - a[j + size * i].conj()).norm() > 1e-11 * scale {
                return Err("eigensolve input is not Hermitian".into());
            }
        }
    }
    let n = size as c_int;
    let (job, uplo) = (b'V' as c_char, b'U' as c_char);
    let mut out = Eigen {
        values: vec![0.; size],
        vectors: a.to_vec(),
    };
    let mut rw = vec![0.; (3 * size).max(1)];
    let mut query = [C::default()];
    let mut info = 0;
    // All pointers have the documented dimensions. Complex64 is repr(C);
    // the two trailing character lengths use the installed gfortran ABI.
    unsafe {
        zheev_(
            &job,
            &uplo,
            &n,
            out.vectors.as_mut_ptr(),
            &n,
            out.values.as_mut_ptr(),
            query.as_mut_ptr(),
            &-1,
            rw.as_mut_ptr(),
            &mut info,
            1,
            1,
        );
    }
    if info != 0 || !query[0].re.is_finite() || query[0].re < 1. || query[0].re > 1_000_000. {
        return Err(format!("Hermitian workspace query failed: {info}").into());
    }
    let len = query[0].re.ceil() as c_int;
    let mut work = vec![C::default(); len as usize];
    out.vectors.copy_from_slice(a);
    unsafe {
        zheev_(
            &job,
            &uplo,
            &n,
            out.vectors.as_mut_ptr(),
            &n,
            out.values.as_mut_ptr(),
            work.as_mut_ptr(),
            &len,
            rw.as_mut_ptr(),
            &mut info,
            1,
            1,
        );
    }
    if info != 0
        || out.values.iter().any(|x| !x.is_finite())
        || out
            .vectors
            .iter()
            .any(|z| !z.re.is_finite() || !z.im.is_finite())
    {
        return Err(format!("Hermitian eigensolve failed: {info}").into());
    }
    Ok(out)
}

pub fn reconstruct(e: &Eigen, values: &[f64]) -> Vec<C> {
    let n = e.values.len();
    assert_eq!(values.len(), n);
    let mut a = vec![C::default(); n * n];
    for (k, &v) in values.iter().enumerate() {
        for j in 0..n {
            for i in 0..n {
                a[i + n * j] += e.vectors[i + n * k] * e.vectors[j + n * k].conj() * v;
            }
        }
    }
    a
}

pub fn density(a: &[C], n: usize) -> Result<Vec<C>> {
    let e = eigen(a, n)?;
    let mut sum = 0.;
    let mut theta = 0.;
    for (j, &v) in e.values.iter().rev().enumerate() {
        sum += v;
        let t = (sum - 1.) / (j + 1) as f64;
        if v > t {
            theta = t;
        }
    }
    Ok(reconstruct(
        &e,
        &e.values
            .iter()
            .map(|v| (v - theta).max(0.))
            .collect::<Vec<_>>(),
    ))
}

/// Coordinates for E_ii, (E_ij+E_ji)/sqrt(2), i(E_ij-E_ji)/sqrt(2), i<j.
pub fn pack(a: &[C], n: usize) -> Vec<f64> {
    assert_eq!(a.len(), n * n);
    let mut v: Vec<_> = (0..n).map(|i| a[i + n * i].re).collect();
    for j in 0..n {
        for i in 0..j {
            v.push(std::f64::consts::SQRT_2 * a[i + n * j].re);
            v.push(std::f64::consts::SQRT_2 * a[i + n * j].im);
        }
    }
    v
}
pub fn unpack(v: &[f64], n: usize) -> Vec<C> {
    assert_eq!(v.len(), n * n);
    let mut a = vec![C::default(); n * n];
    for i in 0..n {
        a[i + n * i] = C::new(v[i], 0.);
    }
    let mut k = n;
    for j in 0..n {
        for i in 0..j {
            a[i + n * j] = C::new(v[k], v[k + 1]) / std::f64::consts::SQRT_2;
            a[j + n * i] = a[i + n * j].conj();
            k += 2;
        }
    }
    a
}
pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}
pub fn distance(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// Re/Im of B M B* at a specified observed row pair, represented as Tr(C M).
pub fn objective(b: &[C], n: usize, r: usize, x: usize, y: usize, imag: bool) -> Vec<f64> {
    let mut c = vec![C::default(); r * r];
    for j in 0..r {
        for i in 0..r {
            let t = b[y + n * i].conj() * b[x + n * j];
            let th = b[x + n * i].conj() * b[y + n * j];
            c[i + r * j] = if imag {
                (t - th) * C::new(0., -0.5)
            } else {
                (t + th) * 0.5
            };
        }
    }
    pack(&c, r)
}

pub fn reading(b: &[C], n: usize, r: usize, m: &[C], x: usize, y: usize) -> C {
    let mut out = C::default();
    for j in 0..r {
        for i in 0..r {
            out += b[x + n * i] * m[i + r * j] * b[y + n * j].conj();
        }
    }
    out
}
