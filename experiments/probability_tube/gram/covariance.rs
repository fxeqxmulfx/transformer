//! QG3 covariance C=T^T diag(q) T. No future-state or file-access interface.
//! For n rows, gamma_(2n+2) covers preweighting and any ordinary FP64 dot
//! reduction. Cauchy-Schwarz bounds sum_i q_i |T_ik T_il| by sqrt(d_k d_l).
//! The additive floor covers subnormal operations under the checked ranges.
use crate::{OldBound, Result, add, div, mul, require, sqrt};

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
}
pub fn single_thread() {
    // Set before any parallel work; the audit itself is single threaded.
    unsafe { openblas_set_num_threads(1) }
}
pub struct Covariance {
    rows: usize,
    width: usize,
    diagonal_upper: Vec<f64>,
    entry_abs_upper: Vec<f64>,
    pub maximum_entry_error: f64,
}
impl Covariance {
    pub fn new(features: &[f64], q: &[f64], width: usize) -> Result<Self> {
        let n = q.len();
        require(
            (1..=1_000_000).contains(&n)
                && (1..=1024).contains(&width)
                && features.len() == n * width,
            "covariance shape",
        )?;
        require(
            q.iter().all(|v| v.is_finite() && *v > 0. && *v <= 1.)
                && features.iter().all(|v| v.is_finite() && v.abs() <= 1e6),
            "covariance input range",
        )?;
        let mut diagonal_upper = vec![0.; width];
        let weighted: Vec<_> = features
            .iter()
            .enumerate()
            .map(|(i, &x)| {
                let row_q = q[i / width];
                let k = i % width;
                diagonal_upper[k] = add(diagonal_upper[k], mul(row_q, mul(x, x)));
                row_q * x
            })
            .collect();
        let mut covariance = vec![0.; width * width];
        // SAFETY: the checked row-major n*width arrays and width*width output
        // match T^T*(q*T), with valid leading dimensions and bounded i32 sizes.
        unsafe {
            cblas_dgemm(
                101,
                112,
                111,
                width as i32,
                width as i32,
                n as i32,
                1.,
                features.as_ptr(),
                width as i32,
                weighted.as_ptr(),
                width as i32,
                0.,
                covariance.as_mut_ptr(),
                width as i32,
            );
        }
        require(
            covariance.iter().all(|v| v.is_finite()),
            "nonfinite BLAS covariance",
        )?;
        let operations = (2 * n + 2) as f64;
        let nu = mul(operations, f64::EPSILON / 2.);
        let gamma = div(nu, (1. - nu).next_down());
        let maximum_feature = features.iter().map(|v| v.abs()).fold(0., f64::max);
        let underflow = mul(mul(f64::MIN_POSITIVE, operations), add(1., maximum_feature));
        let mut maximum_entry_error = 0f64;
        let entry_abs_upper = covariance
            .iter()
            .enumerate()
            .map(|(i, &entry)| {
                let absolute_sum = sqrt(mul(diagonal_upper[i / width], diagonal_upper[i % width]));
                let error = add(mul(gamma, absolute_sum), underflow);
                maximum_entry_error = maximum_entry_error.max(error);
                add(entry.abs(), error)
            })
            .collect();
        Ok(Self {
            rows: n,
            width,
            diagonal_upper,
            entry_abs_upper,
            maximum_entry_error,
        })
    }
    /// Full feature Gram: all off-diagonal correlations remain present.
    pub fn bound(&self, metric: &[f64]) -> Result<OldBound> {
        require(
            metric.len() == self.width
                && metric
                    .iter()
                    .all(|v| v.is_finite() && *v > 0. && *v <= 1e12),
            "metric range",
        )?;
        let energy = self
            .entry_abs_upper
            .iter()
            .enumerate()
            .fold(0., |acc, (i, &v)| {
                add(
                    acc,
                    mul(
                        mul(metric[i / self.width], metric[i % self.width]),
                        mul(v, v),
                    ),
                )
            });
        let trace = self
            .diagonal_upper
            .iter()
            .zip(metric)
            .fold(0., |acc, (&d, &r)| add(acc, mul(d, r)));
        let gram_coefficient = div(sqrt(energy), self.rows as f64);
        let trace_coefficient = div(trace, self.rows as f64);
        require(
            gram_coefficient.is_finite() && trace_coefficient.is_finite(),
            "nonfinite Gram bound",
        )?;
        Ok(OldBound {
            gram_energy_upper: energy,
            gram_coefficient,
            trace_coefficient,
            selected: gram_coefficient.min(trace_coefficient),
        })
    }
}
