//! Thin rectangular real SVD of centered training states; native LAPACK dgesvd.
#[link(name = "openblas")]
unsafe extern "C" {
    fn dgesvd_(
        jobu: *const u8,
        jobvt: *const u8,
        m: *const i32,
        n: *const i32,
        a: *mut f64,
        lda: *const i32,
        s: *mut f64,
        u: *mut f64,
        ldu: *const i32,
        vt: *mut f64,
        ldvt: *const i32,
        work: *mut f64,
        lwork: *const i32,
        info: *mut i32,
    );
}

pub struct Svd {
    pub values: Vec<f64>,
    pub left: Vec<f64>,
    pub right: Vec<f64>,
}

pub fn thin(x: &[f64], rows: usize, cols: usize) -> Svd {
    assert!(rows > 0 && cols > 0 && x.len() == rows * cols);
    assert!(x.iter().all(|v| v.is_finite()));
    let rank = rows.min(cols);
    let mut a: Vec<_> = (0..cols)
        .flat_map(|j| (0..rows).map(move |i| x[i * cols + j]))
        .collect();
    let mut values = vec![0.; rank];
    let mut left = vec![0.; rows * rank];
    let mut right = vec![0.; rank * cols];
    let mut work = vec![0.];
    let mut info = 0;
    unsafe {
        dgesvd_(
            b"S".as_ptr(),
            b"S".as_ptr(),
            &(rows as i32),
            &(cols as i32),
            a.as_mut_ptr(),
            &(rows as i32),
            values.as_mut_ptr(),
            left.as_mut_ptr(),
            &(rows as i32),
            right.as_mut_ptr(),
            &(rank as i32),
            work.as_mut_ptr(),
            &-1,
            &mut info,
        );
    }
    assert_eq!(info, 0, "SVD workspace query failed");
    let size = work[0] as i32;
    work.resize(size as usize, 0.);
    unsafe {
        dgesvd_(
            b"S".as_ptr(),
            b"S".as_ptr(),
            &(rows as i32),
            &(cols as i32),
            a.as_mut_ptr(),
            &(rows as i32),
            values.as_mut_ptr(),
            left.as_mut_ptr(),
            &(rows as i32),
            right.as_mut_ptr(),
            &(rank as i32),
            work.as_mut_ptr(),
            &size,
            &mut info,
        );
    }
    assert_eq!(info, 0, "training-state SVD failed");
    assert!(values.iter().all(|v| v.is_finite() && *v >= 0.));
    let left = (0..rows)
        .flat_map(|i| {
            (0..rank).map({
                let left = &left;
                move |j| left[i + j * rows]
            })
        })
        .collect();
    let right = (0..rank)
        .flat_map(|i| {
            (0..cols).map({
                let right = &right;
                move |j| right[i + j * rank]
            })
        })
        .collect();
    Svd {
        values,
        left,
        right,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rectangular_svd_reconstructs_known_matrices() {
        for (rows, cols, x) in [
            (3, 2, vec![3., 0., 0., 4., 0., 0.]),
            (2, 3, vec![3., 0., 0., 0., 4., 0.]),
        ] {
            let s = thin(&x, rows, cols);
            assert!((s.values[0] - 4.).abs() < 1e-12 && (s.values[1] - 3.).abs() < 1e-12);
            let rank = rows.min(cols);
            for i in 0..rows {
                for k in 0..cols {
                    let actual: f64 = (0..rank)
                        .map(|j| s.left[i * rank + j] * s.values[j] * s.right[j * cols + k])
                        .sum();
                    assert!((actual - x[i * cols + k]).abs() < 1e-12);
                }
            }
            for i in 0..rank {
                for j in 0..rank {
                    let inner: f64 = (0..cols)
                        .map(|k| s.right[i * cols + k] * s.right[j * cols + k])
                        .sum();
                    assert!((inner - f64::from(i == j)).abs() < 1e-12);
                }
            }
        }
    }
}
