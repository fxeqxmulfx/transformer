//! Fixed ordinal kernels, independent of raw target labels. Affine/quadratic
//! descriptors and Gaussian distances are generic input structure, not a task
//! answer formula. CM51 bounds the signed maps, not their target residuals.
use crate::*;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Spec {
    Baseline,
    Polynomial {
        degree: u32,
        axis: [usize; 2],
        ridge: f64,
    },
    Gaussian {
        length: f64,
        axis: [usize; 2],
        ridge: f64,
    },
}

pub fn bank() -> Vec<Spec> {
    let mut result = vec![Spec::Baseline];
    for axis in [[1, 1], [4, 1], [1, 4]] {
        for ridge in [0.001, 0.1] {
            for degree in [1, 2] {
                result.push(Spec::Polynomial {
                    degree,
                    axis,
                    ridge,
                });
            }
            for length in [1. / 32., 1. / 16., 1. / 8., 1. / 4., 1. / 2., 1.] {
                result.push(Spec::Gaussian {
                    length,
                    axis,
                    ridge,
                });
            }
        }
    }
    result
}

impl Spec {
    pub fn value(&self, a: [f64; 2], b: [f64; 2]) -> f64 {
        match *self {
            Self::Baseline => 0.,
            Self::Polynomial { degree, axis, .. } => {
                (1. + axis[0] as f64 * a[0] * b[0] + axis[1] as f64 * a[1] * b[1])
                    .powi(degree as i32)
            }
            Self::Gaussian { length, axis, .. } => {
                let d =
                    axis[0] as f64 * (a[0] - b[0]).powi(2) + axis[1] as f64 * (a[1] - b[1]).powi(2);
                1. + (-d / (2. * length * length)).exp()
            }
        }
    }
    fn ridge(&self) -> Result<f64> {
        match *self {
            Self::Polynomial { ridge, degree, .. }
                if (1..=2).contains(&degree) && ridge > 0. && ridge.is_finite() =>
            {
                Ok(ridge)
            }
            Self::Gaussian { ridge, length, .. }
                if ridge > 0. && ridge.is_finite() && length > 0. && length.is_finite() =>
            {
                Ok(ridge)
            }
            _ => Err("invalid fitted kernel specification".into()),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FitReport {
    pub block: usize,
    pub bank: usize,
    pub observations: usize,
    pub matrix_sha256: Option<String>,
    pub inverse_sha256: Option<String>,
    pub max_inverse_residual: Option<f64>,
    pub refusal: Option<String>,
}

pub(crate) struct Fit {
    points: Vec<[f64; 2]>,
    spec: Spec,
    inverse: Vec<f64>,
}

pub fn point(n: usize, cell: usize) -> [f64; 2] {
    [
        (cell % n) as f64 / (n - 1) as f64,
        (cell / n) as f64 / (n - 1) as f64,
    ]
}

impl Fit {
    pub(crate) fn new(
        points: Vec<[f64; 2]>,
        spec: Spec,
        work: &mut Work,
    ) -> Result<(Self, String, f64)> {
        let n = points.len();
        if n == 0 || n > solver::MAX_OBSERVATIONS || points.iter().flatten().any(|v| !v.is_finite())
        {
            return Err("kernel block coordinates or size refused".into());
        }
        let ridge = spec.ridge()?;
        let mut matrix = vec![0.; n * n];
        for i in 0..n {
            for j in 0..n {
                matrix[i * n + j] =
                    spec.value(points[i], points[j]) + if i == j { ridge } else { 0. };
            }
        }
        work.kernel_matrices += 1;
        // Charge full solves and residual products even if the solver refuses.
        work.system_solves += n;
        work.inverse_residual_products += (n as u64).pow(3);
        let matrix_hash = digest(&matrix)?;
        let (inverse, residual) = solver::inverse(&matrix, n)?;
        Ok((
            Self {
                points,
                spec,
                inverse,
            },
            matrix_hash,
            residual,
        ))
    }

    pub(crate) fn excluded(&self, state: usize, work: &mut Work) -> Result<Vec<f64>> {
        let n = self.points.len();
        if state >= n {
            return Err("excluded kernel source outside block".into());
        }
        let diagonal = self.inverse[state * n + state];
        if diagonal <= 0. || !diagonal.is_finite() {
            return Err("invalid inverse diagonal".into());
        }
        let mut weights: Vec<_> = (0..n)
            .map(|i| {
                if i == state {
                    0.
                } else {
                    -self.inverse[state * n + i] / diagonal
                }
            })
            .collect();
        work.excluded_state_rows += 1;
        work.normalizations += 1;
        solver::normalize(&mut weights)?;
        Ok(weights)
    }

    pub(crate) fn query(&self, point: [f64; 2], work: &mut Work) -> Result<Vec<f64>> {
        let n = self.points.len();
        let k: Vec<_> = self
            .points
            .iter()
            .map(|&p| self.spec.value(p, point))
            .collect();
        let mut weights: Vec<_> = (0..n)
            .map(|i| (0..n).map(|j| self.inverse[i * n + j] * k[j]).sum())
            .collect();
        work.normalizations += 1;
        solver::normalize(&mut weights)?;
        Ok(weights)
    }
}

pub(crate) type Fits = Vec<Vec<Option<Fit>>>;

pub(crate) fn fit_bank(
    parent: &parent::Model,
    bank: &[Spec],
    active: &[bool],
    work: &mut Work,
) -> Result<(Fits, Vec<FitReport>)> {
    let source = parent.source();
    let mut fits = Vec::with_capacity(source.blocks.len());
    let mut reports = Vec::new();
    for (block, b) in source.blocks.iter().enumerate() {
        let mut local = Vec::with_capacity(bank.len());
        for (index, spec) in bank.iter().enumerate() {
            if index == 0 || !active[block] {
                local.push(None);
                continue;
            }
            let points = b
                .observed
                .iter()
                .map(|&cell| point(source.n, cell))
                .collect();
            match Fit::new(points, spec.clone(), work) {
                Ok((fit, matrix, residual)) => {
                    reports.push(FitReport {
                        block,
                        bank: index,
                        observations: b.observed.len(),
                        matrix_sha256: Some(matrix),
                        inverse_sha256: Some(digest(&fit.inverse)?),
                        max_inverse_residual: Some(residual),
                        refusal: None,
                    });
                    local.push(Some(fit));
                }
                Err(error) => {
                    reports.push(FitReport {
                        block,
                        bank: index,
                        observations: b.observed.len(),
                        matrix_sha256: None,
                        inverse_sha256: None,
                        max_inverse_residual: None,
                        refusal: Some(error.to_string()),
                    });
                    local.push(None);
                }
            }
        }
        fits.push(local);
    }
    Ok((fits, reports))
}
