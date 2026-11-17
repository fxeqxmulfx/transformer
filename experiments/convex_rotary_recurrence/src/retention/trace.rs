//! Complete complex execution and an independent real four-coordinate check.
//! Real conversion follows tests/geometry.rs at534a258f, without complex products.
use crate::{
    C, Result,
    atom::{Atom, State, operator},
};
use serde_json::{Value, json};

pub(super) fn norm(state: State) -> f64 {
    state[0].norm().max(state[1].norm())
}
pub(super) fn values(state: State) -> [f64; 4] {
    [state[0].re, state[0].im, state[1].re, state[1].im]
}
pub(super) fn bits(value: f64) -> u64 {
    if value == 0. { 0 } else { value.to_bits() }
}
pub(super) fn states(atom: &Atom, prefix: &[i64]) -> Result<Vec<State>> {
    let phases = atom.phases(prefix.len())?;
    let mut state = atom.initial;
    let mut result = vec![state];
    for (t, &object) in prefix.iter().enumerate() {
        let matrix = *atom
            .transition
            .get(usize::try_from(object)?)
            .ok_or("invalid object")?;
        state = if let Some(phases) = &phases {
            let r = phases[t];
            let inner = operator(matrix, [r[0].conj() * state[0], r[1].conj() * state[1]]);
            [r[0] * inner[0], r[1] * inner[1]]
        } else {
            operator(matrix, state)
        };
        result.push(state);
    }
    if state != atom.state(prefix)? {
        return Err("state trace differs from actual Atom.state".into());
    }
    Ok(result)
}
fn rotate(z: &mut [f64; 4], frequency: [f64; 2], position: f64) {
    for j in 0..2 {
        let (s, c) = (frequency[j] * position).sin_cos();
        let (x, y) = (z[2 * j], z[2 * j + 1]);
        z[2 * j] = c * x - s * y;
        z[2 * j + 1] = s * x + c * y;
    }
}
pub(super) fn independent_real(atom: &Atom, prefix: &[i64]) -> Result<[f64; 4]> {
    let mut z = values(atom.initial);
    for (t, &o) in prefix.iter().enumerate() {
        if let Some(f) = atom.frequency {
            rotate(&mut z, f, -(t as f64));
        }
        let m = atom
            .transition
            .get(usize::try_from(o)?)
            .ok_or("invalid object")?;
        let mut matrix = [[0.; 4]; 4];
        for i in 0..2 {
            for j in 0..2 {
                matrix[2 * i][2 * j] = m[i][j].re;
                matrix[2 * i][2 * j + 1] = -m[i][j].im;
                matrix[2 * i + 1][2 * j] = m[i][j].im;
                matrix[2 * i + 1][2 * j + 1] = m[i][j].re;
            }
        }
        z = std::array::from_fn(|i| matrix[i].iter().zip(z).map(|(u, v)| u * v).sum());
        if let Some(f) = atom.frequency {
            rotate(&mut z, f, t as f64);
        }
    }
    Ok(z)
}

pub(super) fn statistics(xs: &[f64]) -> Value {
    json!({"count":xs.len(),"minimum":xs.iter().copied().reduce(f64::min),
        "maximum":xs.iter().copied().reduce(f64::max),
        "mean":if xs.is_empty() { None } else { Some(xs.iter().sum::<f64>()/xs.len() as f64) },
        "exact_zeros":xs.iter().filter(|&&x| x==0.).count()})
}
pub(super) fn variation(xs: &[State]) -> Value {
    let mut mean = [C::default(); 2];
    for x in xs {
        for j in 0..2 {
            mean[j] += x[j] / xs.len() as f64;
        }
    }
    let variance = xs
        .iter()
        .map(|x| (x[0] - mean[0]).norm_sqr() + (x[1] - mean[1]).norm_sqr())
        .sum::<f64>()
        / xs.len() as f64;
    json!({"mean_state":mean,"mean_squared_distance_from_mean":variance,
        "exactly_constant":xs.windows(2).all(|pair| pair[0]==pair[1]),
        "end_max_norm":statistics(&xs.iter().map(|&x| norm(x)).collect::<Vec<_>>())})
}
