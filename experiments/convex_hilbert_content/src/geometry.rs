//! CM108 MatrixBound.operator_norm and normalize, at886e7d05, in f64.
//! This is a feasibility rule, not a closest-point or rounding certificate.
use crate::{C, Result};
use rotary::atom::{Matrix, State};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Geometry {
    Old,
    Hilbert,
}
impl Geometry {
    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "old" => Ok(Self::Old),
            "hilbert" => Ok(Self::Hilbert),
            _ => Err("unknown geometry".into()),
        }
    }
    pub fn state_norm(self, z: State) -> f64 {
        match self {
            Self::Old => z[0].norm().max(z[1].norm()),
            Self::Hilbert => z[0].norm().hypot(z[1].norm()),
        }
    }
    pub fn row_norm(self, z: State) -> f64 {
        match self {
            Self::Old => z[0].norm() + z[1].norm(),
            Self::Hilbert => self.state_norm(z),
        }
    }
    pub fn matrix_norm(self, m: Matrix) -> Result<f64> {
        match self {
            Self::Old => Ok(self.row_norm(m[0]).max(self.row_norm(m[1]))),
            Self::Hilbert => hilbert_norm(m),
        }
    }
    pub fn initial(self, mut z: State) -> Result<State> {
        finite(z)?;
        match self {
            Self::Old => {
                for x in &mut z {
                    *x /= x.norm().max(1.);
                }
            }
            Self::Hilbert => {
                let d = self.state_norm(z).max(1.);
                for x in &mut z {
                    *x /= d;
                }
            }
        }
        Ok(z)
    }
    pub fn output(self, z: State) -> Result<State> {
        match self {
            Self::Old => readouts::projection::project(z),
            Self::Hilbert => self.initial(z),
        }
    }
    pub fn matrix(self, mut m: Matrix) -> Result<Matrix> {
        match self {
            Self::Old => {
                m[0] = readouts::projection::project(m[0])?;
                m[1] = readouts::projection::project(m[1])?;
            }
            Self::Hilbert => {
                let d = hilbert_norm(m)?.max(1.);
                for x in m.iter_mut().flatten() {
                    *x /= d;
                }
            }
        }
        Ok(m)
    }
}
pub fn finite(z: State) -> Result<()> {
    if z.iter().any(|x| !x.re.is_finite() || !x.im.is_finite()) {
        return Err("nonfinite complete complex coordinate".into());
    }
    Ok(())
}
pub fn hilbert_norm(m: Matrix) -> Result<f64> {
    finite(m[0])?;
    finite(m[1])?;
    let a = m[0][0].norm_sqr() + m[1][0].norm_sqr();
    let d = m[0][1].norm_sqr() + m[1][1].norm_sqr();
    let q: C = m[0][0].conj() * m[0][1] + m[1][0].conj() * m[1][1];
    let n = ((a + d + (a - d).hypot(2. * q.norm())) / 2.).sqrt();
    if !n.is_finite() {
        return Err("nonfinite complete Hilbert norm".into());
    }
    Ok(n)
}
