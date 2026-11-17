//! Complete initial/matrix/frequency numerical proposal coordinates, CM97/98.
//! Row radial projection enforces the explicit maximum-norm operator budget.
use crate::{
    C, Result,
    atom::{Atom, Family, State},
};
use baseline::random::Random;

pub fn disk(z: C) -> Result<C> {
    let n = z.norm();
    if !n.is_finite() {
        return Err("nonfinite initial proposal".into());
    }
    Ok(if n > 1. { z / n } else { z })
}
pub fn row_ball(u: State) -> Result<State> {
    let n = u[0].norm() + u[1].norm();
    if !n.is_finite() {
        return Err("nonfinite row proposal".into());
    }
    Ok(if n > 1. { [u[0] / n, u[1] / n] } else { u })
}
#[derive(Clone, Copy)]
pub(crate) enum Coordinate {
    Initial(usize),
    Matrix(usize, usize, usize),
    Frequency(usize),
}
impl Coordinate {
    pub fn kind(self) -> usize {
        match self {
            Self::Initial(_) => 0,
            Self::Matrix(..) => 1,
            Self::Frequency(_) => 2,
        }
    }
    pub fn apply(self, atom: &Atom, delta: C) -> Result<Atom> {
        let mut a = atom.clone();
        match self {
            Self::Initial(i) => a.initial[i] = disk(a.initial[i] + delta)?,
            Self::Matrix(o, i, j) => {
                a.transition[o][i][j] += delta;
                a.transition[o][i] = row_ball(a.transition[o][i])?;
            }
            Self::Frequency(i) => {
                let f = a
                    .frequency
                    .as_mut()
                    .ok_or("frequency absent from plain family")?;
                f[i] += delta.re;
                if !f[i].is_finite() {
                    return Err("nonfinite frequency proposal".into());
                }
            }
        }
        Ok(a)
    }
}
pub(crate) fn coordinates(observed: &[bool], family: Family, skip_absent: bool) -> Vec<Coordinate> {
    let mut c = vec![Coordinate::Initial(0), Coordinate::Initial(1)];
    for (o, &present) in observed.iter().enumerate() {
        if skip_absent && !present {
            continue;
        }
        for i in 0..2 {
            for j in 0..2 {
                c.push(Coordinate::Matrix(o, i, j));
            }
        }
    }
    if family == Family::Rotary {
        c.push(Coordinate::Frequency(0));
        c.push(Coordinate::Frequency(1));
    }
    c
}
pub(crate) fn random_start(objects: usize, family: Family, rng: &mut Random) -> Atom {
    let unit = |rng: &mut Random| {
        let z = C::new(rng.gauss(0., 1.), rng.gauss(0., 1.));
        if z.norm() == 0. {
            C::new(1., 0.)
        } else {
            z / z.norm()
        }
    };
    let mut a = Atom::zero(objects, family);
    for z in &mut a.initial {
        *z = unit(rng);
    }
    for matrix in &mut a.transition {
        let column = rng.below(2) as usize;
        matrix[0][column] = unit(rng);
        matrix[1][1 - column] = unit(rng);
    }
    // Consume identical RNG draws in both families to match content starts.
    let frequency = [rng.gauss(0., 1.), rng.gauss(0., 1.)];
    if family == Family::Rotary {
        a.frequency = Some(frequency);
    }
    a
}
