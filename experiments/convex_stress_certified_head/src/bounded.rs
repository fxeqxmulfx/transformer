//! Full real products with outward bounds, independently reordered complex
//! products, and a strict common-numerator winner certificate (CM63).
use crate::{interval::Interval as I, *};

pub(crate) fn products(e: &Evaluator<'_>, rows: &[Features]) -> (Vec<f64>, Vec<I>, f64, I) {
    let v = e.forecast.classes;
    let mut z = vec![0.; v];
    let mut bound = vec![I::point(0.); v];
    let (mut n, mut nb) = (0., I::point(0.));
    for (p, h) in rows.iter().enumerate() {
        for x in h.iter().map(|x| x.abs()).chain(std::iter::once(1.)) {
            n += x;
            nb = nb.add(I::point(x));
        }
        for k in 0..16 {
            let (mut phi, mut pb) = (0., I::point(0.));
            for (s, x) in row(h, k / 8).into_iter().enumerate() {
                let u = source(k % 8, s);
                phi += x * u;
                pb = pb.add(I::point(x).scale(u));
            }
            for c in 0..v {
                let w = e.coefficients[(p * 16 + k) * v + c];
                z[c] += phi * w;
                bound[c] = bound[c].add(pb.scale(w));
            }
        }
    }
    (
        z,
        bound,
        n.max(1.),
        I {
            lo: nb.lo.max(1.),
            hi: nb.hi.max(1.),
        },
    )
}

fn complex(e: &Evaluator<'_>, rows: &[Features]) -> Vec<f64> {
    let v = e.forecast.classes;
    (0..v)
        .map(|c| {
            let mut z = 0.;
            for (p, h) in rows.iter().enumerate() {
                for b in 0..2 {
                    for f in 0..4 {
                        let (mut re, mut im) = (0., 0.);
                        for (s, x) in row(h, b).into_iter().enumerate() {
                            let phase = (s * f) % 4;
                            re += x * [0.5, 0., -0.5, 0.][phase];
                            im += x * [0., 0.5, 0., -0.5][phase];
                        }
                        let k = (p * 16 + b * 8 + f * 2) * v + c;
                        z += re * e.coefficients[k] - im * e.coefficients[k + v];
                    }
                }
            }
            z
        })
        .collect()
}

pub(crate) fn evaluate(e: &Evaluator<'_>, rows: &[Features]) -> Option<Scores> {
    let (real, bounds, n, nb) = products(e, rows);
    if real.iter().any(|x| !x.is_finite())
        || !nb.contains(n)
        || bounds.iter().zip(&real).any(|(b, x)| !b.contains(*x))
    {
        return None;
    }
    let best = joint::argmax(&real);
    if bounds
        .iter()
        .enumerate()
        .any(|(c, b)| c != best && b.hi >= bounds[best].lo)
    {
        return None;
    }
    let other = complex(e, rows);
    if other.iter().any(|x| !x.is_finite()) || joint::argmax(&other) != best {
        return None;
    }
    let logits: Vec<_> = real.iter().map(|x| (x - real[best]) / n).collect();
    let independent: Vec<_> = other.iter().map(|x| (x - other[best]) / n).collect();
    let mut width = 0.0f64;
    let mut residual = 0.0f64;
    for (c, b) in bounds.iter().enumerate() {
        let bound = if c == best {
            I::point(0.)
        } else {
            b.sub(bounds[best]).divide_positive(nb)
        };
        if !bound.contains(logits[c]) || !bound.contains(independent[c]) {
            return None;
        }
        width = width.max((bound.hi - bound.lo).next_up());
        residual = residual.max((logits[c] - independent[c]).abs());
    }
    if width > 1e-7 || residual > 1e-7 || joint::argmax(&logits) != best {
        return None;
    }
    Some(Scores {
        logits,
        predicted: best,
        maximizers: vec![best],
        method: Method::SeparatedEnclosures,
        independent_residual: residual,
        logit_enclosure_width: width,
        target_realizability_certified: false,
    })
}
