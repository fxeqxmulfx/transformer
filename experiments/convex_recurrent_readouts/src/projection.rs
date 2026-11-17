//! CM102 at9d1ce0fb: exact complete complex row projection in real arithmetic.
//! The native f64 evaluation keeps both stored coordinates, including exact zeros.
//! It preserves phases and implements all magnitude/zero/boundary cases; no
//! feature, rank, class or coordinate is removed from subsequent training.
use crate::{C, Result};
use rotary::atom::State;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Rule {
    #[default]
    Radial,
    Euclidean,
}

/// CM102 Projection.magnitudes, resize and project; inside rows return directly.
pub fn project(z: State) -> Result<State> {
    let r = [z[0].norm(), z[1].norm()];
    if !r[0].is_finite()
        || !r[1].is_finite()
        || !(r[0] + r[1]).is_finite()
        || (0..2).any(|i| r[i] == 0. && z[i] != C::default())
    {
        return Err("nonfinite or underflowed complete projection input".into());
    }
    if r[0] + r[1] <= 1. {
        return Ok(z);
    }
    let s = if r[0] - r[1] >= 1. {
        [1., 0.]
    } else if r[1] - r[0] >= 1. {
        [0., 1.]
    } else {
        [(r[0] - r[1] + 1.) / 2., (r[1] - r[0] + 1.) / 2.]
    };
    let mut point = [C::default(); 2];
    for i in 0..2 {
        if s[i] < 0. || s[i] > r[i] {
            return Err("invalid complete projected magnitude".into());
        }
        if r[i] != 0. {
            point[i] = z[i] * (s[i] / r[i]);
        } else if s[i] != 0. {
            return Err("nonzero magnitude at zero complete coordinate".into());
        }
    }
    if point[0].norm() + point[1].norm() > 1. + 2e-12 {
        return Err("complete Euclidean row budget exceeded".into());
    }
    Ok(point)
}

#[cfg(test)]
mod tests;
