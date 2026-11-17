//! Approximate learned-component proposals on the CM92 convex joint domain.
//! Alternating complex contractions search the full factors; they are not an
//! exact global linear minimization oracle. No convergence certificate is claimed.
//! CM94 skips only exactly zero full-class rows during repeated contractions.
use crate::{C, Result, model::Atom};
use baseline::random::Random;
use bilinear::domain::Shape;

pub struct Gradient {
    pub shape: Shape,
    // CM88 real input rows give the full complex gradient (g,-g). This stores
    // g at every original (class,query,context), including unobserved zeros.
    pub values: Vec<f64>,
}

impl Gradient {
    pub fn zero(shape: Shape) -> Result<Self> {
        let count = shape
            .objects
            .checked_mul(shape.queries)
            .and_then(|n| n.checked_mul(shape.columns))
            .ok_or("complete gradient dimensions overflow")?;
        Ok(Self {
            shape,
            values: vec![0.; count],
        })
    }

    pub fn index(&self, c: usize, q: usize, d: usize) -> usize {
        (d * self.shape.queries + q) * self.shape.objects + c
    }

    pub fn value(&self, atom: &Atom) -> f64 {
        self.value_rows(atom, &self.active_rows())
    }

    /// Scan every class of every row afresh, without a magnitude threshold.
    pub(crate) fn active_rows(&self) -> Vec<usize> {
        self.values
            .chunks_exact(self.shape.objects)
            .enumerate()
            .filter_map(|(i, row)| row.iter().any(|&v| v != 0.).then_some(i))
            .collect()
    }

    pub(crate) fn value_rows(&self, atom: &Atom, rows: &[usize]) -> f64 {
        rows.iter()
            .map(|&dq| {
                let row = &self.values[dq * self.shape.objects..(dq + 1) * self.shape.objects];
                let q = dq % self.shape.queries;
                let d = dq / self.shape.queries;
                let u: C = row.iter().zip(&atom.output).map(|(&g, &u)| u * g).sum();
                (C::new(1., 1.) * u * atom.query[q] * atom.context[d]).re
            })
            .sum()
    }

    pub(crate) fn contract(&self, atom: &Atom, axis: usize, rows: &[usize]) -> Vec<C> {
        let s = self.shape;
        let length = [s.objects, s.queries, s.columns][axis];
        let mut result = vec![C::default(); length];
        for &dq in rows {
            let row = &self.values[dq * s.objects..(dq + 1) * s.objects];
            let q = dq % s.queries;
            let d = dq / s.queries;
            if axis == 0 {
                let factor = C::new(1., 1.) * atom.query[q] * atom.context[d];
                for (out, &g) in result.iter_mut().zip(row) {
                    *out += factor * g;
                }
            } else {
                let u: C = row.iter().zip(&atom.output).map(|(&g, &u)| u * g).sum();
                if axis == 1 {
                    result[q] += C::new(1., 1.) * u * atom.context[d];
                } else {
                    result[d] += C::new(1., 1.) * u * atom.query[q];
                }
            }
        }
        result
    }

    pub fn propose(
        &self,
        rng: &mut Random,
        restarts: usize,
        sweeps: usize,
        previous: Option<&Atom>,
    ) -> Result<(Atom, f64)> {
        if restarts == 0 || sweeps == 0 {
            return Err("empty component search".into());
        }
        let rows = self.active_rows();
        let mut best = Atom::zero(self.shape);
        let mut best_value = 0.;
        for restart in 0..(restarts + usize::from(previous.is_some())) {
            let mut atom = if restart == restarts {
                previous.ok_or("missing previous proposal")?.clone()
            } else {
                Atom {
                    output: random_vector(self.shape.objects, rng)?,
                    query: random_vector(self.shape.queries, rng)?,
                    context: random_vector(self.shape.columns, rng)?,
                }
            };
            for _ in 0..sweeps {
                for axis in 0..3 {
                    let mut vector = self.contract(&atom, axis, &rows);
                    for z in &mut vector {
                        *z = -z.conj();
                    }
                    normalize(&mut vector)?;
                    match axis {
                        0 => atom.output = vector,
                        1 => atom.query = vector,
                        _ => atom.context = vector,
                    }
                }
            }
            atom.validate(self.shape)?;
            let value = self.value_rows(&atom, &rows);
            if !value.is_finite() {
                return Err("nonfinite component objective".into());
            }
            if value < best_value {
                best = atom;
                best_value = value;
            }
        }
        Ok((best, best_value))
    }
}

pub fn normalize(vector: &mut [C]) -> Result<()> {
    let squared: f64 = vector.iter().map(C::norm_sqr).sum();
    if !squared.is_finite() {
        return Err("nonfinite factor normalization".into());
    }
    if squared == 0. {
        if vector.iter().any(|&z| z != C::default()) {
            return Err("factor norm underflow".into());
        }
        return Ok(());
    }
    let norm = squared.sqrt();
    for z in vector {
        *z /= norm;
    }
    Ok(())
}

fn random_vector(n: usize, rng: &mut Random) -> Result<Vec<C>> {
    let mut v: Vec<_> = (0..n)
        .map(|_| C::new(rng.gauss(0., 1.), rng.gauss(0., 1.)))
        .collect();
    normalize(&mut v)?;
    Ok(v)
}
